"""Training-only instance pairing. No model parameters or external annotations."""
from collections import defaultdict
import hashlib
import json
import math
import random

import torch
import torch.nn.functional as F


def validate_pair_config(args):
    enabled = getattr(args, 'enable_pair_sampling', False)
    weight = getattr(args, 'pair_loss_weight', 0.)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('pair_loss_weight must be finite and nonnegative')
    if weight and not enabled:
        raise ValueError('pair_loss_weight requires --enable_pair_sampling')
    if getattr(args, 'export_val_predictions', False) and getattr(args, 'distributed', False):
        raise ValueError('Prediction export initially supports single-process evaluation only')
    if not enabled:
        return
    if getattr(args, 'distributed', False):
        raise ValueError('Pair sampling initially supports single-process training only')
    if args.dataset not in ('unc', 'unc+', 'gref', 'gref_umd'):
        raise ValueError('Pair sampling supports RefCOCO family only')
    for name in ('enable_mrefm', 'enable_ref_mim', 'enable_ref_mlm', 'enable_ptp_aux',
                 'enable_syntax_bias', 'use_contrastive_loss', 'enable_seg_mask', 'use_mask_loss'):
        if getattr(args, name, False):
            raise ValueError('Initial pair ablation does not support --' + name)
    if not math.isfinite(args.pair_margin) or args.pair_margin < 0 or args.pair_min_patches < 1:
        raise ValueError('pair_margin must be finite and nonnegative; pair_min_patches >= 1')


def box_iou_xywh(a, b):
    intersection = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(
        0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    union = a[2]*a[3] + b[2]*b[3] - intersection
    return intersection / union if union > 0 else 1.


class PairIndex:
    """Conservative geometric candidates, NOT verified same-category instances.

    RefCOCO caches lack instance IDs. Exact box duplicates, high-overlap boxes,
    invalid boxes and identical normalized phrases cannot be negatives.
    """
    def __init__(self, records, max_iou=0.5):
        if not 0 <= max_iou < 1:
            raise ValueError('pair max_iou must be in [0, 1)')
        self.records = records
        self.max_iou = max_iou
        self.partners = [[] for _ in records]
        groups = defaultdict(list)
        for i, record in enumerate(records):
            if len(record) != 5:
                raise ValueError('PairIndex requires RefCOCO-family 5-field cache records')
            box = [float(v) for v in record[2]]
            if len(box) != 4 or not all(math.isfinite(v) for v in box) or min(box[2:]) <= 0:
                continue
            groups[str(record[0])].append(i)
        for ids in groups.values():
            for offset, i in enumerate(ids):
                for j in ids[offset+1:]:
                    a, b = records[i], records[j]
                    if ' '.join(a[3].lower().split()) == ' '.join(b[3].lower().split()):
                        continue
                    if box_iou_xywh(a[2], b[2]) > max_iou:
                        continue
                    self.partners[i].append(j)
                    self.partners[j].append(i)

    def audit(self):
        pairable = sum(bool(p) for p in self.partners)
        canonical = [[str(r[0]), [float(v) for v in r[2]], r[3]] for r in self.records]
        digest = hashlib.sha256(json.dumps(canonical, ensure_ascii=False,
                                          separators=(',', ':')).encode('utf-8')).hexdigest()
        return dict(expressions=len(self.records), images=len({r[0] for r in self.records}),
                    geometric_targets=len({(r[0], tuple(r[2])) for r in self.records}),
                    pairable_expressions=pairable,
                    pairable_fraction=pairable / max(1, len(self.records)),
                    candidate_pairs=sum(map(len, self.partners)) // 2,
                    max_iou=self.max_iou, records_sha256=digest,
                    verified_instance_ids=False, same_category_verified=False)


class PairBatchSampler:
    """Greedy pairs without replacement. Flattened batch size and budget fixed.

    Each index appears at most once per epoch; only the final incomplete batch
    is dropped. Pair fraction is an upper bound on paired sample slots.
    """
    def __init__(self, index, batch_size, seed=42, pair_fraction=0.5):
        if batch_size < 2 or not 0 < pair_fraction <= 1:
            raise ValueError('batch_size >= 2 and 0 < pair_fraction <= 1 required')
        self.max_pairs = int(batch_size * pair_fraction) // 2
        if self.max_pairs < 1:
            raise ValueError('pair_fraction allocates fewer than two sample slots')
        self.index, self.batch_size = index, batch_size
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.index.records) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.index.records)))
        rng.shuffle(order)
        available = set(order)
        cursor = 0
        for _ in range(len(self)):
            batch, slots, pairs = [], 0, 0
            while slots < self.batch_size:
                while order[cursor] not in available:
                    cursor += 1
                i = order[cursor]
                cursor += 1
                available.remove(i)
                candidates = []
                if pairs < self.max_pairs and slots + 2 <= self.batch_size:
                    candidates = [j for j in self.index.partners[i] if j in available]
                if candidates:
                    j = rng.choice(candidates)
                    available.remove(j)
                    batch.append((i, j))
                    pairs += 1
                    slots += 2
                else:
                    batch.append(i)
                    slots += 1
            yield batch

    def manifest(self):
        batches = list(self)
        used = [i for b in batches for key in b
                for i in (key if isinstance(key, tuple) else (key,))]
        return dict(epoch=self.epoch, seed=self.seed, batch_size=self.batch_size,
                    batches=batches, samples=len(used), unique_samples=len(set(used)),
                    dropped_indices=sorted(set(range(len(self.index.records))) - set(used)),
                    pairs=sum(isinstance(key, tuple) for b in batches for key in b))


def pair_ranking_loss(scores, masks, padding, pairs, margin=0.1, min_patches=1):
    """Mean hinge on raw cosine scores in non-overlapping GT regions.

    masks: B,1,H,W; padding: B,H,W, True means padding. Coordinates are shared
    within a pair. Area pooling keeps small valid regions; padded patches and
    shared target pixels are excluded before pooling. Returns differentiable
    zero when no comparison is valid.
    """
    if not math.isfinite(margin) or margin < 0 or min_patches < 1:
        raise ValueError('finite nonnegative margin and min_patches >= 1 required')
    grid = math.isqrt(scores.shape[-1])
    if grid * grid != scores.shape[-1]:
        raise ValueError('scores must have a square patch grid')
    values, gaps = [], []
    masks = masks[:, 0].bool()
    padding = padding.bool()
    for a, b in pairs:
        valid_pixels = ~(padding[a] | padding[b])
        region_a = masks[a] & ~masks[b] & valid_pixels
        region_b = masks[b] & ~masks[a] & valid_pixels
        # Completely ignore a patch if any part covers padding.
        valid_patches = F.adaptive_max_pool2d((~valid_pixels).float()[None, None],
                                             (grid, grid)).flatten() == 0
        weights = [F.adaptive_avg_pool2d(r.float()[None, None], (grid, grid)).flatten()
                   * valid_patches for r in (region_a, region_b)]
        if any(int((w > 0).sum()) < min_patches for w in weights):
            continue
        # Coarse patches containing both instances cannot discriminate them.
        shared = (weights[0] > 0) & (weights[1] > 0)
        weights = [w.masked_fill(shared, 0) for w in weights]
        if any(int((w > 0).sum()) < min_patches for w in weights):
            continue
        wa, wb = [w / w.sum() for w in weights]
        for sample, positive, negative in ((a, wa, wb), (b, wb, wa)):
            gap = (scores[sample].float() * (positive - negative)).sum()
            gaps.append(gap.detach())
            values.append(F.relu(margin - gap))
    loss = torch.stack(values).mean() if values else scores.float().sum() * 0.
    stats = dict(pair_valid_comparisons=len(values), pair_candidate_pairs=len(pairs),
                 pair_score_gap=float(torch.stack(gaps).mean()) if gaps else 0.)
    return loss, stats

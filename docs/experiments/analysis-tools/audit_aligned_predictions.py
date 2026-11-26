"""Exploratory aligned REC prediction diagnostics, with image-cluster bootstrap.

No category labels inferred from phrases. Other-target confusion is a proxy
using only annotated targets in this validation set. Not a new benchmark.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import zipfile

import numpy as np


def read(path):
    if '::' in path:
        archive, member = path.split('::', 1)
        with zipfile.ZipFile(archive) as z:
            data = z.read(member).decode()
    else:
        data = Path(path).read_text()
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def xyxy(box):
    x, y, w, h = box
    return np.array([x-w/2, y-h/2, x+w/2, y+h/2])


def iou(a, b):
    a, b = xyxy(a), xyxy(b)
    a = np.clip(a, 0, 1)
    inter = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:]-a[:2]).prod()+np.maximum(0, b[2:]-b[:2]).prod()-inter
    return float(inter/max(1e-12, union))


def validate_recorded_ious(records):
    # The evaluator clips predicted xyxy and may convert fp16 model outputs
    # before promoting arithmetic with fp32 targets. Reconstruct both modes.
    import torch
    def convert(x):
        cx, cy, w, h = x.unbind(-1)
        return torch.stack([cx-.5*w, cy-.5*h, cx+.5*w, cy+.5*h], -1)
    gt = convert(torch.tensor([r['target_box_cxcywh'] for r in records], dtype=torch.float32))
    expected = np.array([r['iou'] for r in records])
    errors = {}
    for name, dtype in [('fp16', torch.float16), ('fp32', torch.float32)]:
        pred = convert(torch.tensor([r['pred_box_cxcywh'] for r in records], dtype=dtype))
        # torch2.0 CPU lacks half clamp; 0/1 clipping commutes with this cast.
        pred = pred.float().clamp(0, 1).to(dtype)
        intersection = (torch.minimum(pred[:, 2:], gt[:, 2:])-torch.maximum(pred[:, :2], gt[:, :2])).clamp(min=0).prod(1)
        pa = (pred[:, 2]-pred[:, 0])*(pred[:, 3]-pred[:, 1])
        ga = (gt[:, 2]-gt[:, 0])*(gt[:, 3]-gt[:, 1])
        actual = (intersection/(pa+ga-intersection+1e-16)).numpy()
        errors[name] = float(np.max(np.abs(actual-expected)))
    mode = min(errors, key=errors.get)
    assert errors[mode] < 5e-5, errors
    return dict(mode=mode, max_abs_error=errors[mode])


def analyze(reference, candidate):
    assert len(reference) == len(candidate) and reference
    assert len({r['sample_index'] for r in reference}) == len(reference)
    for a, b in zip(reference, candidate):
        for key in ['sample_index', 'image_id', 'phrase', 'coordinate_frame']:
            assert a[key] == b[key], key
        np.testing.assert_allclose(a['target_box_cxcywh'], b['target_box_cxcywh'], rtol=0, atol=1e-7)
        for r in [a, b]:
            assert bool(r['iou'] >= .5) == bool(r['hit_at_0_5'])
    verification = [validate_recorded_ious(r) for r in [reference, candidate]]
    ah = np.array([r['hit_at_0_5'] for r in reference], dtype=int)
    bh = np.array([r['hit_at_0_5'] for r in candidate], dtype=int)
    groups = defaultdict(list)
    for k, r in enumerate(reference):
        groups[r['image_id']].append(k)
    sizes = np.array([len(ids) for ids in groups.values()])
    changes = np.array([(bh[ids]-ah[ids]).sum() for ids in groups.values()])
    rng = np.random.RandomState(20261124)
    samples = rng.randint(len(groups), size=(2000, len(groups)))
    boot = changes[samples].sum(1)/sizes[samples].sum(1)*100
    result = dict(n=len(reference), images=len(groups), evaluator_reconstruction=verification, reference_accuracy=ah.mean()*100,
                  candidate_accuracy=bh.mean()*100, delta_pp=(bh-ah).mean()*100,
                  corrected=int(((ah == 0) & (bh == 1)).sum()),
                  regressed=int(((ah == 1) & (bh == 0)).sum()),
                  image_bootstrap_delta_95ci_pp=np.quantile(boot, [.025, .975]).tolist(),
                  training_seed_count=1, exploratory=True,
                  uncertainty_scope='validation image sampling, not training-seed variability')
    strata = defaultdict(list)
    for k, r in enumerate(reference):
        area = r['target_box_cxcywh'][2]*r['target_box_cxcywh'][3]
        length = len(r['phrase'].split())
        strata['padded_area_lt_1pct' if area < .01 else 'padded_area_1to10pct' if area < .1 else 'padded_area_ge_10pct'].append(k)
        strata['words_le3' if length <= 3 else 'words_4to8' if length <= 8 else 'words_gt8'].append(k)
    result['strata'] = {name: dict(n=len(ids), reference_accuracy=float(ah[ids].mean()*100),
                                  candidate_accuracy=float(bh[ids].mean()*100),
                                  delta_pp=float((bh[ids]-ah[ids]).mean()*100))
                         for name, ids in strata.items()}
    confusion = [0, 0]
    pairs, both_correct, collapse = 0, [0, 0], [0, 0]
    for ids in groups.values():
        targets = {tuple(reference[k]['target_box_cxcywh']) for k in ids}
        for which, records in enumerate([reference, candidate]):
            for k in ids:
                r = records[k]
                if not r['hit_at_0_5'] and any(iou(r['pred_box_cxcywh'], box) >= .5
                    and iou(r['target_box_cxcywh'], box) <= .5 for box in targets):
                    confusion[which] += 1
        for offset, a in enumerate(ids):
            for b in ids[offset+1:]:
                if reference[a]['phrase'].strip().lower() == reference[b]['phrase'].strip().lower():
                    continue
                if iou(reference[a]['target_box_cxcywh'], reference[b]['target_box_cxcywh']) > .5:
                    continue
                pairs += 1
                for which, records in enumerate([reference, candidate]):
                    both_correct[which] += bool(records[a]['hit_at_0_5'] and records[b]['hit_at_0_5'])
                    collapse[which] += iou(records[a]['pred_box_cxcywh'], records[b]['pred_box_cxcywh']) >= .5
    result['other_annotated_target_confusion_proxy'] = dict(reference=confusion[0], candidate=confusion[1],
        limitation='incomplete target inventory; not verified same-category mistakes')
    result['geometric_pair_diagnostic'] = dict(description_pairs=pairs,
        both_correct_pct=[v/max(1, pairs)*100 for v in both_correct],
        overlapping_predictions_pct=[v/max(1, pairs)*100 for v in collapse],
        limitation='correlated expression pairs, unverified semantic exclusivity; descriptive only')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reference', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = analyze(read(a.reference), read(a.candidate))
    Path(a.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

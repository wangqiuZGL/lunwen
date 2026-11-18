"""Optional wrapper; base Dataset and ordinary collate remain unchanged."""
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.misc import collate_fn


class PairedDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            return [self.base[key]]
        if len(key) != 2:
            raise ValueError('A pair requires exactly two sample indices')
        rows = [self.base.pull_item(i) for i in key]
        if rows[0][0] != rows[1][0]:
            raise ValueError('Paired targets must belong to the same image')
        boxes = torch.stack([row[3].clone() for row in rows])
        union = torch.cat((boxes[:, :2].min(0).values, boxes[:, 2:].max(0).values))
        # One transform invocation for both targets. Concatenated text makes
        # the original directional-word crop guard consider both descriptions.
        transformed = self.base.transform(dict(
            img=rows[0][1], box=union, pair_boxes=boxes,
            text=' '.join(row[2].lower() for row in rows),
            obj_mask=torch.cat([row[5] for row in rows], dim=0)))
        samples = []
        for k, row in enumerate(rows):
            samples.append((transformed['img'], transformed['mask'].numpy(),
                            np.array([], dtype=int), np.array([], dtype=int),
                            transformed['pair_boxes'][k].numpy(), row[0], row[2].lower(),
                            row[4], transformed['obj_mask'][k:k+1].numpy().astype(int)))
        return samples


def paired_collate(groups):
    flat, pairs = [], []
    for samples in groups:
        if len(samples) == 2:
            pairs.append((len(flat), len(flat) + 1))
        elif len(samples) != 1:
            raise ValueError('Expected singleton or pair')
        flat.extend(samples)
    return (*collate_fn(flat), {'pairs': pairs})

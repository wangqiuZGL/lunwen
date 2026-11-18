"""Sample training images only; preserve official validation/test annotations."""
import argparse
import hashlib
import json
import math
import random
import shutil
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dataset', default='unc+')
    parser.add_argument('--fraction', default=0.2, type=float)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError('fraction must be in (0, 1]')
    source = args.source / args.dataset
    output = args.output / args.dataset
    if output.resolve() == source.resolve():
        raise ValueError('Never overwrite original annotations')
    train_file = source / (args.dataset + '_train.pth')
    records = torch.load(train_file, map_location='cpu')
    image_ids = sorted({row[0] for row in records})
    count = max(1, math.ceil(len(image_ids) * args.fraction))
    selected = sorted(random.Random(args.seed).sample(image_ids, count))
    selected_set = set(selected)
    subset = [row for row in records if row[0] in selected_set]
    for split in ('val', 'test', 'testA', 'testB'):
        path = source / f'{args.dataset}_{split}.pth'
        if path.exists():
            heldout = torch.load(path, map_location='cpu')
            if selected_set & {row[0] for row in heldout}:
                raise ValueError('Image overlap with held-out split: ' + split)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(subset, output / train_file.name)
    hashes = {}
    for path in source.glob('*.pth'):
        if path.name in (train_file.name, f'{args.dataset}_trainval.pth'):
            continue
        shutil.copy2(path, output / path.name)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(dataset=args.dataset, fraction=args.fraction, seed=args.seed,
                    original_images=len(image_ids), selected_images=count,
                    original_expressions=len(records), selected_expressions=len(subset),
                    train_sha256=hashlib.sha256(train_file.read_bytes()).hexdigest(),
                    unchanged_split_sha256=hashes, image_ids=selected)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'image_ids'}, indent=2))


if __name__ == '__main__':
    main()

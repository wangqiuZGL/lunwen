"""Audit trusted RefCOCO cache, save deterministic pairs and optional previews."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.pair_supervision import PairIndex, PairBatchSampler


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--train-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pair-fraction', type=float, default=.5)
    parser.add_argument('--max-iou', type=float, default=.5)
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--preview-count', type=int, default=50)
    args = parser.parse_args()
    # The project's annotation caches are Python objects: load only trusted files.
    records = torch.load(args.train_cache, map_location='cpu', weights_only=False)
    index = PairIndex(records, args.max_iou)
    sampler = PairBatchSampler(index, args.batch_size, args.seed, args.pair_fraction)
    args.output.mkdir(parents=True, exist_ok=True)
    audit = index.audit()
    audit['cache_sha256'] = hashlib.sha256(args.train_cache.read_bytes()).hexdigest()
    (args.output / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    (args.output / 'pair_schedule_epoch_000.json').write_text(json.dumps(sampler.manifest()), encoding='utf-8')
    candidates = []
    with (args.output / 'pair_index.jsonl').open('w', encoding='utf-8') as stream:
        for i, partners in enumerate(index.partners):
            for j in partners:
                if j <= i:
                    continue
                item = dict(indices=[i, j], image_id=str(records[i][0]),
                            boxes_xywh=[list(map(float, records[k][2])) for k in (i, j)],
                            phrases=[records[k][3] for k in (i, j)])
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                if len(candidates) < max(0, args.preview_count):
                    candidates.append(item)
    if args.image_root:
        from PIL import Image, ImageDraw
        preview = args.output / 'previews'
        preview.mkdir(exist_ok=True)
        for number, item in enumerate(candidates):
            with Image.open(args.image_root / item['image_id']) as source:
                image = source.convert('RGB')
            draw = ImageDraw.Draw(image)
            for label, box, color in zip(('A', 'B'), item['boxes_xywh'], ('red', 'blue')):
                x, y, w, h = box
                draw.rectangle((x, y, x+w, y+h), outline=color, width=3)
                draw.text((x, y), label, fill=color)
            image.save(preview / ('pair_%03d.jpg' % number))
            (preview / ('pair_%03d.json' % number)).write_text(
                json.dumps(item, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))
    if not audit['candidate_pairs']:
        raise SystemExit('No pairs available; do not launch paired training.')


if __name__ == '__main__':
    main()

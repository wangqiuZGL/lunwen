"""Overfit a fixed tiny batch before spending the full experimental budget.

These probe weights are discarded; only formal runs are accuracy experiments.
"""
import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_oneref as train
from datasets import build_dataset
from utils.pair_data import PairedDataset, paired_collate
from utils.pair_supervision import PairIndex, query_competition_loss
from utils.loss_utils import one_ref_loss


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    assert args.pair_loss_type == 'query_competition' and args.pair_loss_weight > 0
    assert not args.frozen_backbone
    # Keep all other settings, but use a deterministic input for the fit check.
    args.aug_crop = args.aug_scale = args.aug_translate = False
    dataset = build_dataset('train', args)
    index = PairIndex(dataset.images, max_iou=.1)
    selected, used_images = [], set()
    for i, partners in enumerate(index.partners):
        image = dataset.images[i][0]
        if image in used_images or len(dataset.images[i][3].split()) < 2:
            continue
        j = next((j for j in partners if len(dataset.images[j][3].split()) >= 2), None)
        if j is not None:
            selected.append((i, j))
            used_images.add(image)
        if len(selected) == 4:
            break
    if len(selected) < 4:
        raise RuntimeError('Need four distinct-image candidate pairs for the probe')
    wrapped = PairedDataset(dataset)
    image, text, boxes, masks, meta = paired_collate([wrapped[p] for p in selected])
    device = torch.device(args.device)
    image, boxes, masks = image.to(device), boxes.to(device), masks.to(device)
    model = train.create_model(args.model + '_grounding', sys_args=args, pretrained=False,
                               drop_path_rate=args.drop_path, vocab_size=args.vocab_size,
                               checkpoint_activations=args.checkpoint_activations)
    train.beit3_utils.load_model_and_may_interpolate(args.finetune, model, args.model_key, args.model_prefix)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    def forward():
        with torch.cuda.amp.autocast(enabled=args.amp):
            out = model(image.tensors, image.mask, text, training=True)
        pair_loss, stats = query_competition_loss(out[2], masks, image.mask, meta['pairs'],
                                                 temperature=args.pair_temperature)
        assert stats['pair_valid_comparisons'] > 0
        losses = one_ref_loss(args, out[0], boxes, masks, out[1], out[2], out[3])
        loss = sum(v for k, v in losses.items() if k.startswith('loss_')) + args.pair_loss_weight * pair_loss
        return loss, pair_loss, stats

    model.eval()
    with torch.no_grad():
        _, before, initial = forward()
        before = float(before)
    for step in range(args.probe_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, pair_loss, stats = forward()
        assert torch.isfinite(loss), 'Nonfinite probe loss'
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step % 20 == 0:
            print('PROBE', step, 'pair_loss', float(pair_loss), 'pair_accuracy', stats['pair_query_accuracy'], flush=True)
    model.eval()
    with torch.no_grad():
        _, after, final = forward()
        after = float(after)
    result = dict(initial_loss=before, final_loss=after,
                  initial_accuracy=initial['pair_query_accuracy'], final_accuracy=final['pair_query_accuracy'],
                  steps=args.probe_steps, pairs=[dict(indices=p,
                    phrases=[dataset.images[i][3] for i in p],
                    image=dataset.images[p[0]][0]) for p in selected],
                  probe_only=True, weights_discarded=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'probe_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('PROBE_RESULT', result, flush=True)
    if not (after < before * .8 and final['pair_query_accuracy'] >= .85):
        raise RuntimeError('Tiny-pair learning gate failed; do not launch the formal runs')
    print('TINY_PAIR_OVERFIT_PASSED; PROBE_WEIGHTS_DISCARDED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[train.get_args_parser()])
    parser.add_argument('--probe_steps', type=int, default=150)
    main(parser.parse_args())

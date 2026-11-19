"""Bounded real-CUDA train/validate/save/load check, including the OSS path."""
import argparse
import itertools
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_oneref as train


class Limited:
    def __init__(self, loader, count):
        self.loader, self.count = loader, min(count, len(loader))

    def __len__(self):
        return self.count

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.count)


original_step, original_val, original_create = train.train_one_epoch, train.validate, train.create_model
observed = {}
gradients = set()


def create(*a, **kw):
    model = original_create(*a, **kw)
    for name in ['bbox_embed.layers.2.weight', 'language_head.weight']:
        def hook(g, name=name):
            if torch.isfinite(g).all() and g.float().abs().sum() > 0:
                gradients.add(name)
        dict(model.named_parameters())[name].register_hook(hook)
    return model


def step(args, model, loader, *a, **kw):
    stats = original_step(args, model, Limited(loader, 4), *a, **kw)
    assert stats.get('pair_valid_comparisons_total', 0) > 0
    assert stats.get('loss_pair', 0) > 0
    assert 'pair_query_accuracy_valid_mean' in stats
    observed.update(stats)
    return stats


def validate(args, model, loader, *a, **kw):
    return original_val(args, model, Limited(loader, 2), *a, **kw)


if __name__ == '__main__':
    train.create_model, train.train_one_epoch, train.validate = create, step, validate
    args = argparse.ArgumentParser(parents=[train.get_args_parser()]).parse_args()
    assert args.pair_loss_type == 'query_competition'
    Path(args.output_dir).mkdir(parents=True, exist_ok=False)
    train.main(args)
    assert len(gradients) == 2, gradients
    checkpoint = torch.load(Path(args.output_dir) / 'checkpoint.pth', map_location='cpu')
    assert checkpoint.get('scaler')
    model = original_create(args.model + '_grounding', sys_args=args, pretrained=False,
        drop_path_rate=args.drop_path, vocab_size=args.vocab_size,
        checkpoint_activations=args.checkpoint_activations)
    model.load_state_dict(checkpoint['model'], strict=True)
    print('QUERY_COMPETITION_CUDA_SMOKE_PASSED', observed['pair_valid_comparisons_total'],
          'peak_GiB', torch.cuda.max_memory_allocated() / 1024**3, flush=True)

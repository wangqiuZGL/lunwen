"""Exercise real fused GPU training, then verify checkpoint reload."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_oneref as train

original_create = train.create_model
original_validate = train.validate
checked = False
seen = set()


def create(*args, **kwargs):
    model = original_create(*args, **kwargs)
    for name in ('bbox_embed.layers.2.weight', 'ptp_aux_head.weight'):
        parameter = dict(model.named_parameters())[name]
        def hook(gradient, name=name):
            if torch.isfinite(gradient).all() and gradient.abs().sum() > 0 and name not in seen:
                print('GRADIENT_OK', name, float(gradient.float().norm()))
                seen.add(name)
        parameter.register_hook(hook)
    return model


@torch.no_grad()
def validate(args, model, loader, device):
    global checked
    if not checked:
        model.eval()
        image, text = next(iter(loader))[:2]
        image = image.to(device)
        builder = model.beit3.encoder.syntax_bias_builder
        assert builder is not None
        strength = builder.syntax_bias_lambda
        builder.syntax_bias_lambda = 0
        zero = model(image.tensors, image.mask, text)[0]
        model.beit3.encoder.syntax_bias_builder = None
        plain = model(image.tensors, image.mask, text)[0]
        torch.testing.assert_close(zero, plain, atol=1e-6, rtol=1e-5)
        model.beit3.encoder.syntax_bias_builder = builder
        builder.syntax_bias_lambda = strength
        fused = model(image.tensors, image.mask, text)[0]
        delta = float((fused - plain).abs().max())
        assert delta > 0, 'Syntax has no effect on real model output'
        print('ZERO_BIAS_EQUIVALENT; SYNTAX_OUTPUT_DELTA', delta)
        checked = True
    return original_validate(args, model, loader, device)


train.create_model = create
train.validate = validate
parser = argparse.ArgumentParser(parents=[train.get_args_parser()])
args = parser.parse_args()
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
train.main(args)
assert len(seen) == 2
checkpoint = torch.load(Path(args.output_dir) / 'checkpoint.pth', map_location='cpu')
model = original_create(args.model + '_grounding', sys_args=args, pretrained=False,
                        drop_path_rate=args.drop_path, vocab_size=args.vocab_size,
                        checkpoint_activations=args.checkpoint_activations)
model.load_state_dict(checkpoint['model'], strict=True)
assert checkpoint.get('scaler')
print('CHECKPOINT_RELOAD_OK; FUSED_SMOKE_PASSED')
print('CUDA_PEAK_GIB', torch.cuda.max_memory_allocated() / 1024**3)

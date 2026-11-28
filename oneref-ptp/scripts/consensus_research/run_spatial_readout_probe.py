"""Bounded exploratory three-arm frozen-feature probe; no test split access."""
import argparse
import copy
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def corners(b):
    return torch.cat((b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2), -1)


def iou_giou(b, t, evaluation=False):
    p, q = corners(b), corners(t)
    if evaluation:
        p = p.clamp(0, 1)
    inter = (torch.minimum(p[:, 2:], q[:, 2:])-torch.maximum(p[:, :2], q[:, :2])).clamp(min=0).prod(-1)
    union = p[:, 2:].sub(p[:, :2]).prod(-1)+q[:, 2:].sub(q[:, :2]).prod(-1)-inter
    iou = inter/union.clamp(min=1e-8)
    enclosing = (torch.maximum(p[:, 2:], q[:, 2:])-torch.minimum(p[:, :2], q[:, :2])).clamp(min=0).prod(-1)
    return iou, iou-(enclosing-union)/enclosing.clamp(min=1e-8)


class Readout(nn.Module):
    def __init__(self, original, kind):
        super().__init__()
        self.base = copy.deepcopy(original).float().requires_grad_(True)
        self.kind = kind
        if kind == 'mlp':
            self.extra = nn.Sequential(nn.Linear(768, 320), nn.ReLU(), nn.Linear(320, 4))
        elif kind == 'spatial':
            self.project = nn.Conv2d(771, 128, 1)
            self.local = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, groups=128), nn.ReLU(), nn.Conv2d(128, 128, 1), nn.ReLU())
            self.extra = nn.Sequential(nn.Linear(128*16, 64), nn.ReLU(), nn.Linear(64, 4))
        if kind != 'original':
            nn.init.zeros_(self.extra[-1].weight)
            nn.init.zeros_(self.extra[-1].bias)

    def forward(self, pooled, grid=None, valid=None):
        logits = self.base(pooled)
        if self.kind == 'mlp':
            logits = logits+self.extra(pooled)
        elif self.kind == 'spatial':
            b, n, c = grid.shape
            s = int(n**.5)
            x = grid.float().transpose(1, 2).reshape(b, c, s, s)
            v = valid.float().reshape(b, 1, s, s)
            axis = torch.linspace(-1, 1, s, device=x.device)
            yy, xx = torch.meshgrid(axis, axis, indexing='ij')
            coords = torch.stack((xx, yy))[None].expand(b, -1, -1, -1)
            x = self.project(torch.cat((x*v, coords, v), 1))
            x = self.local(x)*v
            x = F.adaptive_avg_pool2d(x, 4)/F.adaptive_avg_pool2d(v, 4).clamp(min=1e-6)
            logits = logits+self.extra(x.flatten(1))
        return logits.sigmoid()


def main():
    p = argparse.ArgumentParser()
    for k in ('project', 'config', 'checkpoint', 'output'):
        p.add_argument('--'+k, required=True)
    p.add_argument('--train-samples', type=int, default=2048)
    p.add_argument('--val-samples', type=int, default=1024)
    p.add_argument('--steps', type=int, default=384)
    opt = p.parse_args()
    out = Path(opt.output)
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, opt.project)
    spec = importlib.util.spec_from_file_location('probe_eval', Path(opt.project)/'eval.py')
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    args = ev.get_args_parser().parse_args(['--task', 'grounding'])
    for k, v in json.loads(Path(opt.config).read_text()).items():
        if hasattr(args, k):
            setattr(args, k, v)
    args.eval_model = opt.checkpoint
    args.eval_set = 'val'
    args.output_dir = str(out)
    args.batch_size = 8
    args.num_workers = 2
    args.distributed = False
    args.world_size = 1
    args.enable_syntax_bias = False
    args.enable_ptp_aux = False
    (out/'config.json').write_text(json.dumps(vars(args), indent=2))

    def probe(args, model, unused_loader, device):
        started = time.monotonic()
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.set_num_threads(4)
        state = torch.load(opt.checkpoint, map_location='cpu')
        model.load_state_dict(state['model'], strict=True)
        del state
        model.eval().requires_grad_(False)
        from datasets import build_dataset, make_transforms
        from utils.misc import collate_fn
        from torch.utils.data import DataLoader
        train = build_dataset('train', args)
        val = build_dataset('val', args)
        # Use fixed evaluation transforms for both cached sets, avoiding augmentation confounds.
        train.transform = make_transforms(args, 'val')
        train_ids = {str(r[0]) for r in train.images}
        val_ids = {str(r[0]) for r in val.images}
        assert not train_ids.intersection(val_ids), 'Image leakage'
        rng = random.Random(42)
        manifest = {}
        for name, ds, n in [('train', train, opt.train_samples), ('val', val, opt.val_samples)]:
            idx = rng.sample(range(len(ds.images)), min(n, len(ds.images)))
            ds.images = [ds.images[i] for i in idx]
            manifest[name] = [{'source_index': i, 'image': str(r[0]), 'phrase': str(r[3])} for i, r in zip(idx, ds.images)]
        (out/'sample_manifest.json').write_text(json.dumps(manifest))
        captures = {}
        handles = []
        for name in ('vision_head', 'language_head', 'visu_proj'):
            def hook(m, ins, result, key=name):
                captures[key] = result
            handles.append(getattr(model, name).register_forward_hook(hook))
        max_error = 0.
        max_cache_error = 0.
        cache = {}
        try:
            for name, ds in [('train', train), ('val', val)]:
                chunks = {k: [] for k in ('pooled', 'grid', 'valid', 'target', 'pred')}
                loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn)
                for step, batch in enumerate(loader):
                    if time.monotonic()-started > 1200:
                        raise TimeoutError('Cache exceeded 20-minute budget')
                    images, text, target = batch[:3]
                    images = images.to(device)
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        pred = model(images.tensors, images.mask, text)[0]
                        grid = captures['visu_proj'][:, 1:]
                        score = (F.normalize(captures['vision_head'], dim=-1)[:, 1:]*F.normalize(captures['language_head'], dim=-1)[:, 0, None]).sum(-1)
                        pooled = (score.softmax(-1)[:, :, None]*grid).sum(1)
                        reconstructed = model.bbox_embed(pooled).sigmoid()
                    max_error = max(max_error, float((pred-reconstructed).abs().max()))
                    if max_error > .002:
                        raise RuntimeError('AMP prediction reconstruction failed')
                    with torch.no_grad():
                        cached_pred = model.bbox_embed(pooled.float()).sigmoid()
                    max_cache_error = max(max_cache_error, float((pred.float()-cached_pred).abs().max()))
                    valid = F.adaptive_avg_pool2d((~images.mask.bool()).float()[:, None], (24, 24)).flatten(1)
                    for key, value in dict(pooled=pooled.float(), grid=grid.half(), valid=valid, target=target.float(), pred=pred.float()).items():
                        chunks[key].append(value.detach().cpu())
                    captures.clear()
                    if step == 3:
                        elapsed = time.monotonic()-started
                        estimate = elapsed/32*(len(train)+len(val))
                        print(json.dumps(dict(stage='timing_32', seconds=elapsed, estimated_cache_seconds=estimate)), flush=True)
                        if name == 'train' and estimate > 1200:
                            raise TimeoutError('32-sample timing projects beyond cache budget; reduce sample count explicitly')
                    if step % 32 == 0:
                        print(name, step*8, 'cache seconds', time.monotonic()-started, flush=True)
                cache[name] = {k: torch.cat(v) for k, v in chunks.items()}
                torch.save(cache[name], out/(name+'_cache.pt'))
        finally:
            for handle in handles:
                handle.remove()
        original = copy.deepcopy(model.bbox_embed).float().cpu()
        del model
        torch.cuda.empty_cache()
        train, val = cache['train'], cache['val']

        def evaluate(head):
            preds = []
            with torch.no_grad():
                for i in range(0, len(val['target']), 32):
                    preds.append(head(*(val[k][i:i+32].to(device) for k in ('pooled', 'grid', 'valid'))).cpu())
            pred = torch.cat(preds)
            ious, _ = iou_giou(pred, val['target'], evaluation=True)
            return dict(acc50=float((ious >= .5).float().mean()*100), acc75=float((ious >= .75).float().mean()*100), miou=float(ious.mean()*100)), pred, ious

        results = {}
        # Fixed updates and ordering, no validation-based checkpoint selection.
        gen = torch.Generator().manual_seed(42)
        orders = [torch.randperm(len(train['target']), generator=gen) for _ in range((opt.steps*32)//len(train['target'])+2)]
        order = torch.cat(orders)[:opt.steps*32].reshape(opt.steps, 32)
        for kind in ('original', 'mlp', 'spatial'):
            torch.manual_seed(42)
            head = Readout(original, kind).to(device)
            initial, initial_pred, _ = evaluate(head)
            ref = original(val['pooled'][:32]).sigmoid().detach()
            assert float((initial_pred[:32]-ref).abs().max()) < 1e-5, 'Initial readout equivalence failed'
            params = [{'params': list(head.base.parameters()), 'lr': 3e-5}]
            extra = [v for k, v in head.named_parameters() if not k.startswith('base.')]
            if extra:
                params.append({'params': extra, 'lr': 1e-4})
            optim = torch.optim.AdamW(params, weight_decay=1e-4)
            t0 = time.monotonic()
            for step, idx in enumerate(order):
                if time.monotonic()-t0 > 600 or time.monotonic()-started > 3600:
                    raise TimeoutError('Head or total budget exceeded; incomplete arms cannot be compared')
                b = head(*(train[k][idx].to(device) for k in ('pooled', 'grid', 'valid')))
                target = train['target'][idx].to(device)
                _, giou = iou_giou(b, target)
                loss = 5*F.l1_loss(b, target, reduction='none').sum(-1).mean()+2*(1-giou).mean()
                assert torch.isfinite(loss), 'Nonfinite loss'
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                optim.step()
                if step % 64 == 0:
                    print(kind, step, float(loss), flush=True)
            metrics, pred, ious = evaluate(head)
            results[kind] = dict(initial=initial, final=metrics, steps=opt.steps, seconds=time.monotonic()-t0, parameters=sum(p.numel() for p in head.parameters()), extra_parameters=sum(p.numel() for p in extra))
            torch.save(dict(state=head.cpu().state_dict(), pred=pred, iou=ious), out/(kind+'.pt'))
            print(json.dumps({kind: results[kind]}), flush=True)
        # Cluster bootstrap uses images, not individual correlated expressions.
        groups = {}
        for i, row in enumerate(manifest['val']):
            groups.setdefault(row['image'], []).append(i)
        comparisons = {}
        values = {k: torch.load(out/(k+'.pt'))['iou'].numpy() for k in results}
        rng = np.random.RandomState(42)
        for control in ('original', 'mlp'):
            delta = (values['spatial'] >= .5).astype(float)-(values[control] >= .5).astype(float)
            sums = np.array([delta[ix].sum() for ix in groups.values()])
            ns = np.array([len(ix) for ix in groups.values()])
            draw = rng.randint(len(ns), size=(2000, len(ns)))
            bootstrap = sums[draw].sum(1)/ns[draw].sum(1)*100
            comparisons[control] = dict(delta_pp=float(delta.mean()*100), image_bootstrap_95ci_pp=np.percentile(bootstrap, [2.5, 97.5]).tolist(), corrected=int((delta > 0).sum()), regressed=int((delta < 0).sum()))
        report = dict(status='completed', scope='Exploratory fixed-seed cached-feature screen; checkpoint previously selected using full val. Not independent test evidence.', samples={k: len(v['target']) for k, v in cache.items()}, amp_reconstruction_max_error=max_error, cached_float32_prediction_max_error=max_cache_error, results=results, spatial_vs=comparisons, seconds=time.monotonic()-started)
        (out/'results.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return results['original']['final']['acc50']/100

    ev.evaluate = probe
    ev.main(args)


if __name__ == '__main__':
    main()

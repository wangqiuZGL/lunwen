"""Reuse screen01 caches: diagnose old states, then three prespecified repairs."""
import argparse, importlib.util, json, sys, time, shutil
from pathlib import Path
import torch
from layerwise_fusion import LayerwiseHead
from run_spatial_readout_probe import iou_giou


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/hy-tmp/layerwise-20261127')
    p.add_argument('--output', default='diagnostic02')
    p.add_argument('--seconds', type=int, default=600)
    p.add_argument('--diagnose-only', action='store_true')
    a = p.parse_args(); root = Path(a.root); out = root/a.output
    out.mkdir(exist_ok=False); start = time.monotonic(); torch.set_num_threads(4)
    def check_time():
        if time.monotonic()-start > a.seconds:
            raise TimeoutError('Diagnostic compute budget exhausted')
    sys.path.insert(0, str(root/'code'))
    spec = importlib.util.spec_from_file_location('diag_eval', root/'code/eval.py')
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    args = ev.get_args_parser().parse_args(['--task', 'grounding'])
    for k, v in json.loads((root/'probe_config.json').read_text()).items():
        if hasattr(args, k): setattr(args, k, v)
    args.eval_model = str(root/'best_checkpoint.pth'); args.output_dir = str(out)
    args.eval_set = 'val'; args.batch_size = 16; args.num_workers = 2
    args.distributed = False; args.world_size = 1
    args.enable_syntax_bias = False; args.enable_ptp_aux = False
    protocol = dict(steps=256, batch=32, seeds=[42,43], seconds=a.seconds,
        arms={'head_low_lr': dict(kind='layer12', head_lr=3e-6),
              'frozen_shared': dict(kind='shared', gate_lr=.001, strength_lr=.01),
              'frozen_separate': dict(kind='separate', gate_lr=.001, strength_lr=.01)},
        selection='Final step256 only; intermediate evaluations diagnostic, no best-val selection',
        scope='Same exploratory512/256 cache; not independent testing or full fine-tuning',
        rationale='Check smaller head updates and frozen-head residual strength starvation; no new alignment module')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    shutil.copy2(__file__, out/'diagnose_layerwise.py')
    shutil.copy2(root/'layerwise_fusion.py', out/'layerwise_fusion.py')
    def run(args, model, unused, device):
        from utils.loss_utils import one_ref_loss
        state = torch.load(args.eval_model, map_location='cpu')
        model.load_state_dict(state['model'], strict=True); del state
        model.eval().requires_grad_(False)
        data = {s: {k:v.to(device) for k,v in torch.load(root/'screen01'/f'{s}_cache.pt', map_location='cpu').items()} for s in ('train','val')}
        def loss_fn(head, d, ids):
            pred, scores, _ = head(d['features'][ids].float())
            parts = one_ref_loss(args, pred, d['target'][ids], d['mask'][ids], None, visu_sim=scores)
            return sum(v for k,v in parts.items() if k.startswith('loss_')), parts, pred
        def evaluate(head, split):
            d=data[split]; preds=[]; totals={}; n=len(d['target'])
            with torch.no_grad():
                for i in range(0,n,32):
                    check_time(); ids=slice(i,i+32); loss,parts,pred=loss_fn(head,d,ids)
                    preds.append(pred); b=len(pred)
                    for k,v in dict(parts,total=loss).items(): totals[k]=totals.get(k,0.)+float(v)*b/n
            pred=torch.cat(preds); iou,_=iou_giou(pred,d['target'],evaluation=True)
            return dict(acc50=float((iou>=.5).float().mean()*100),miou=float(iou.mean()*100),loss=totals),pred.cpu(),iou.cpu()
        base=LayerwiseHead(model,'layer12').to(device)
        baseline={}; base_iou={}
        for split in data:
            m,pred,iou=evaluate(base,split); baseline[split]=m; base_iou[split]=iou
            original_iou,_=iou_giou(data[split]['pred'],data[split]['target'],evaluation=True)
            baseline[split]['fp16_cache_changed_hits']=int(((iou>=.5)!=(original_iou.cpu()>=.5)).sum())
            baseline[split]['fp16_prediction_max_error']=float((pred-data[split]['pred'].cpu()).abs().max())
        del base
        old={}
        manifest=json.loads((root/'manifest.json').read_text())
        def paired(iou):
            before=base_iou['val']>=.5; after=iou>=.5
            corrected=(after & ~before).nonzero().flatten().tolist()
            regressed=(before & ~after).nonzero().flatten().tolist()
            return dict(corrected=corrected, regressed=regressed,
                net_samples=len(corrected)-len(regressed))
        for seed in (42,43):
            for kind in ('layer12','separate'):
                check_time(); head=LayerwiseHead(model,kind).to(device)
                saved=torch.load(root/'screen01'/f'{kind}_{seed}.pt',map_location='cpu')
                head.load_state_dict(saved['state'],strict=True)
                train,_,_=evaluate(head,'train'); val,pred,iou=evaluate(head,'val')
                if float((pred-saved['pred']).abs().max())>1e-6: raise RuntimeError('Saved state/prediction mismatch')
                old[f'{kind}_{seed}']=dict(train=train,val=val,paired=paired(iou),strength=head.strength.detach().tolist())
                del head,saved
        diagnosis=dict(baseline=baseline,old=old)
        (out/'diagnosis.json').write_text(json.dumps(diagnosis,indent=2)); print('DIAGNOSIS',json.dumps(diagnosis),flush=True)
        if a.diagnose_only:
            return baseline['val']['acc50']/100
        # All proposed runs start from the same original model, never the old adapted head.
        results={}
        for seed in protocol['seeds']:
            g=torch.Generator().manual_seed(seed); n=len(data['train']['target'])
            order=torch.cat([torch.randperm(n,generator=g) for _ in range(256*32//n+1)])[:256*32].reshape(256,32).to(device)
            for arm,cfg in protocol['arms'].items():
                check_time(); torch.manual_seed(seed); head=LayerwiseHead(model,cfg['kind']).to(device)
                if arm.startswith('frozen'):
                    for name in ('vision_head','language_head','visu_proj','bbox_embed'):getattr(head,name).requires_grad_(False)
                    groups=[dict(params=[head.strength],lr=cfg['strength_lr']),dict(params=list(head.gate.parameters())+list(head.gate_norm.parameters()),lr=cfg['gate_lr'])]
                else: groups=[dict(params=[v for v in head.parameters() if v.requires_grad],lr=cfg['head_lr'])]
                params=[v for v in head.parameters() if v.requires_grad]
                opt=torch.optim.AdamW(groups,weight_decay=1e-4)
                init,_,_=evaluate(head,'val')
                if abs(init['acc50']-baseline['val']['acc50'])>1e-6:raise RuntimeError('Initial anchor mismatch')
                trace=[]; grads=[]
                for step,ids in enumerate(order,1):
                    check_time(); opt.zero_grad(); loss,parts,_=loss_fn(head,data['train'],ids)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    loss.backward()
                    if step in (1,2,32,128,256):
                        grads.append(dict(step=step,strength_grad=head.strength.grad.detach().tolist() if head.strength.grad is not None else None,
                            gate_grad_norm=sum(float(v.grad.square().sum()) for v in head.gate.parameters() if v.grad is not None)**.5,
                            head_grad_norm=sum(float(v.grad.square().sum()) for name,v in head.named_parameters() if v.grad is not None and name.split('.')[0] in ('vision_head','language_head','visu_proj','bbox_embed'))**.5))
                    torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
                    if step in (32,128,256):
                        tr,_,_=evaluate(head,'train'); va,pred,iou=evaluate(head,'val')
                        row=dict(step=step,train=tr,val=va,strength=head.strength.detach().tolist());trace.append(row)
                        print(arm,seed,json.dumps(row),flush=True)
                if arm.startswith('frozen'):
                    for name in ('vision_head','language_head','visu_proj','bbox_embed'):
                        if any(not torch.equal(v.float(),getattr(model,name).state_dict()[k].float()) for k,v in getattr(head,name).state_dict().items()):raise RuntimeError('Frozen head changed')
                key=f'{arm}_{seed}';results[key]=dict(trace=trace,grads=grads,paired=paired(iou),trainable_parameters=sum(v.numel() for v in params))
                torch.save(dict(state=head.cpu().state_dict(),pred=pred,iou=iou),out/f'{key}.pt')
                report=dict(status='in_progress',protocol=protocol,diagnosis=diagnosis,results=results,seconds=time.monotonic()-start)
                (out/'results.json').write_text(json.dumps(report,indent=2));del head,opt
        report.update(status='completed',seconds=time.monotonic()-start)
        (out/'results.json').write_text(json.dumps(report,indent=2));print('COMPLETED',flush=True)
        return baseline['val']['acc50']/100
    ev.evaluate=run;ev.main(args)


if __name__=='__main__':main()

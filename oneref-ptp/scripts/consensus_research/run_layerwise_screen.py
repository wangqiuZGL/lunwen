"""Bounded 512/256 multi-layer screen; native REC supervision, no new labels."""
import argparse,copy,importlib.util,json,random,sys,time,hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
from layerwise_fusion import LayerwiseHead
from run_spatial_readout_probe import iou_giou


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/hy-tmp/layerwise-20261127')
    p.add_argument('--output',default='screen01');p.add_argument('--steps',type=int,default=128)
    p.add_argument('--seconds',type=int,default=1800);p.add_argument('--batch',type=int,default=32)
    p.add_argument('--sanity',action='store_true')
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43]);a=p.parse_args()
    root=Path(a.root);out=root/a.output;out.mkdir(exist_ok=False);start=time.monotonic();torch.set_num_threads(4)
    sys.path.insert(0,str(root/'code'));spec=importlib.util.spec_from_file_location('layer_eval',root/'code/eval.py')
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    args=ev.get_args_parser().parse_args(['--task','grounding'])
    for k,v in json.loads((root/'probe_config.json').read_text()).items():
        if hasattr(args,k):setattr(args,k,v)
    args.eval_model=str(root/'best_checkpoint.pth');args.output_dir=str(out);args.eval_set='val';args.batch_size=16;args.num_workers=2
    args.distributed=False;args.world_size=1;args.enable_syntax_bias=False;args.enable_ptp_aux=False
    protocol=dict(vars(a),layers=[6,9,12],arms=['layer6','layer9','layer12','equal','shared','shared_wide','separate'],
        head_lr=3e-5,gate_lr=1e-4,selection='Fixed final update, all arms reported',
        supervision='Native one_ref_loss, including original box-mask focal/dice; frozen backbone and deterministic views',
        scope='Small exploratory cache screen, not full fine-tuning. Original checkpoint was selected on val; no independent-test claim.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'config.json').write_text(json.dumps(vars(args),indent=2))
    import shutil
    for name in ('run_layerwise_screen.py','layerwise_fusion.py','prepare_layerwise_server.py','run_spatial_readout_probe.py'):shutil.copy2(root/name,out/name)
    def run(args,model,unused,device):
        state=torch.load(args.eval_model,map_location='cpu');model.load_state_dict(state['model'],strict=True);del state
        model.eval().requires_grad_(False)
        from datasets import build_dataset,make_transforms
        from utils.misc import collate_fn,mdetr_interpolate
        from utils.loss_utils import one_ref_loss
        from torch.utils.data import DataLoader
        manifest=json.loads((root/'manifest.json').read_text())
        if a.sanity:manifest={k:v[:32 if k=='train' else 16] for k,v in manifest.items()}
        (out/'manifest.json').write_text(json.dumps(manifest))
        assert not {r['image'] for r in manifest['train']} & {r['image'] for r in manifest['val']}
        captures={};handles=[];cache={};max_error=0.
        for index in (5,8,11):
            def hook(module,inputs,result,k=index):captures[k]=result[0].detach()
            handles.append(model.beit3.encoder.layers[index].register_forward_hook(hook))
        base=LayerwiseHead(model,'layer12').to(device)
        try:
            for kind in ('train','val'):
                ds=build_dataset(kind,args)
                if kind=='train':ds.transform=make_transforms(args,'val')
                rows=ds.images;ds.images=[rows[r['source_index']] for r in manifest[kind]]
                assert all(str(row[0])==m['image'] and str(row[3])==m['phrase'] for row,m in zip(ds.images,manifest[kind]))
                chunks={k:[] for k in ('features','target','mask','pred')}
                for step,batch in enumerate(DataLoader(ds,batch_size=16,shuffle=False,num_workers=2,collate_fn=collate_fn)):
                    if time.monotonic()-start>a.seconds:raise TimeoutError('Cache time budget exhausted')
                    images,text,target,obj_mask=batch[:4];images=images.to(device)
                    with torch.no_grad():
                        pred=model(images.tensors,images.mask,text)[0]
                        features=torch.stack([model.beit3.encoder.layer_norm(captures[k]) for k in (5,8,11)],1)
                        reconstructed,reference_scores,_=base(features)
                        error=float((pred-reconstructed).abs().max());max_error=max(error,max_error)
                        if error>1e-5:raise RuntimeError('Original layer12 reconstruction failed '+str(error))
                        half=features.half();cached=base(half.float())[0]
                        if float((cached-pred).abs().max())>.002:raise RuntimeError('FP16 feature cache error too large')
                        patch_mask=mdetr_interpolate(obj_mask.float(),(24,24),mode='nearest')>.5
                        if step==0:
                            full=one_ref_loss(args,reconstructed,target.to(device),obj_mask.to(device),None,visu_sim=reference_scores)
                            reduced=one_ref_loss(args,reconstructed,target.to(device),patch_mask.to(device),None,visu_sim=reference_scores)
                            if any(float((full[n]-reduced[n]).abs())>1e-6 for n in full):raise RuntimeError('Cached mask loss mismatch')
                    for n,v in dict(features=half,target=target,mask=patch_mask,pred=pred).items():chunks[n].append(v.detach().cpu())
                    captures.clear()
                    if step%8==0:print(json.dumps(dict(stage='cache',split=kind,samples=min((step+1)*16,len(ds)),seconds=time.monotonic()-start)),flush=True)
                cache[kind]={n:torch.cat(v) for n,v in chunks.items()};torch.save(cache[kind],out/(kind+'_cache.pt'))
        finally:
            for h in handles:h.remove()
        del base
        # All tiny cached sets stay on GPU, avoiding CPU/GPU transfer every step.
        train={k:v.to(device) for k,v in cache['train'].items()};val={k:v.to(device) for k,v in cache['val'].items()}
        results={}
        def evaluate(head):
            predictions=[];gates=[]
            with torch.no_grad():
                for i in range(0,len(val['target']),a.batch):
                    pred,_,weights=head(val['features'][i:i+a.batch].float());predictions.append(pred.cpu())
                    if weights is not None:gates.append(weights.cpu())
            pred=torch.cat(predictions);ious,_=iou_giou(pred,cache['val']['target'],evaluation=True)
            return dict(acc50=float((ious>=.5).float().mean()*100),acc75=float((ious>=.75).float().mean()*100),miou=float(ious.mean()*100)),pred,ious,torch.cat(gates) if gates else None
        for seed in a.seeds:
            gen=torch.Generator().manual_seed(seed);n=a.steps*a.batch
            order=torch.cat([torch.randperm(len(train['target']),generator=gen) for _ in range(n//len(train['target'])+1)])[:n].reshape(a.steps,a.batch).to(device)
            for arm in protocol['arms']:
                torch.manual_seed(seed);head=LayerwiseHead(model,arm).to(device)
                params=[(n,v) for n,v in head.named_parameters() if v.requires_grad]
                common=[v for n,v in params if not(n.startswith('gate') or n=='strength')]
                extra=[v for n,v in params if n.startswith('gate') or n=='strength']
                optimizer=torch.optim.AdamW([dict(params=common,lr=3e-5),dict(params=extra,lr=1e-4)],weight_decay=1e-4)
                if arm not in ('layer6','layer9'):
                    with torch.no_grad():
                        ref=LayerwiseHead(model,'layer12').to(device);x=val['features'][:8].float()
                        err=float((ref(x)[0]-head(x)[0]).abs().max());del ref
                    if err>1e-5:raise RuntimeError('Anchored initialization failed')
                t0=time.monotonic();torch.cuda.reset_peak_memory_stats()
                for step,ids in enumerate(order):
                    if time.monotonic()-start>a.seconds:raise TimeoutError('Training budget exhausted; incomplete arms incomparable')
                    pred,scores,_=head(train['features'][ids].float())
                    losses=one_ref_loss(args,pred,train['target'][ids],train['mask'][ids],None,visu_sim=scores)
                    loss=sum(v for k,v in losses.items() if k.startswith('loss_'))
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([v for _,v in params],1.);optimizer.step()
                    if step%32==0:print(json.dumps(dict(stage='train',arm=arm,seed=seed,step=step+1,loss=float(loss),seconds=time.monotonic()-t0)),flush=True)
                metrics,pred,ious,weights=evaluate(head);key=arm+'_'+str(seed)
                results[key]=dict(metrics,steps=a.steps,seconds=time.monotonic()-t0,parameters=sum(v.numel() for _,v in params),extra_parameters=sum(v.numel() for v in extra),
                    strength=head.strength.detach().cpu().tolist(),gate_mean=weights.mean(0).tolist() if weights is not None else None,peak_memory_mb=torch.cuda.max_memory_allocated()/1024**2)
                torch.save(dict(state=head.cpu().state_dict(),pred=pred,iou=ious,gates=weights),out/(key+'.pt'))
                (out/'results.json').write_text(json.dumps(dict(status='in_progress',protocol=protocol,max_reconstruction_error=max_error,results=results),indent=2))
                print(key,results[key],flush=True);del head,optimizer
        original_iou,_=iou_giou(cache['val']['pred'],cache['val']['target'],evaluation=True)
        report=dict(status='completed',protocol=protocol,max_reconstruction_error=max_error,original_acc50=float((original_iou>=.5).float().mean()*100),results=results,seconds=time.monotonic()-start)
        (out/'results.json').write_text(json.dumps(report,indent=2));print('COMPLETED',flush=True)
        return results['layer12_'+str(a.seeds[0])]['acc50']/100
    ev.evaluate=run;ev.main(args)


if __name__=='__main__':main()

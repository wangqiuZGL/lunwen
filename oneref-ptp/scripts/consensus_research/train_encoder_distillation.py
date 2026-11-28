"""Bounded last-two-block adaptation on new official-training images."""
import argparse,importlib.util,io,json,random,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch
from run_spatial_readout_probe import iou_giou

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/hy-tmp/layerwise-20261127');p.add_argument('--output',default='encoder_distill17')
    p.add_argument('--steps',type=int,default=128);p.add_argument('--seconds',type=int,default=900)
    p.add_argument('--mode',choices=['last2','full_replay'],default='last2');p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43]);p.add_argument('--arms',nargs='+',choices=['native','all_teacher','anchored_quality'],default=['native','all_teacher','anchored_quality'])
    a=p.parse_args();root=Path(a.root);out=root/a.output;out.mkdir(exist_ok=False)
    start=time.monotonic();torch.set_num_threads(4);sys.path.insert(0,str(root/'code'))
    spec=importlib.util.spec_from_file_location('last_eval',root/'code/eval.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    args=ev.get_args_parser().parse_args(['--task','grounding'])
    for k,v in json.loads((root/'probe_config.json').read_text()).items():
        if hasattr(args,k):setattr(args,k,v)
    args.eval_model=str(root/'baseline_model_only.pth');args.output_dir=str(out);args.eval_set='val';args.batch_size=16;args.num_workers=2
    args.distributed=False;args.world_size=1;args.enable_syntax_bias=False;args.enable_ptp_aux=False
    protocol=dict(seeds=a.seeds,arms=a.arms,steps=a.steps,batch=16,encoder_lr=3e-6,head_lr=1e-5,weight_decay=1e-4,
        train=512,val=2048,kd_weight=2.,teacher_margin=.01,min_teacher_iou=.5,precision=a.precision,training_mode=a.mode,
        selection='Fixed final update, no validation-based early stopping',scope='Encoder blocks10/11 plus grounding heads;512official-train;2048development val; single-forward student; no test access or novelty claim')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    import shutil;shutil.copy2(__file__,out/'train_encoder_distillation.py')
    def check():
        if time.monotonic()-start>a.seconds:raise TimeoutError('Last-block training budget exhausted')
    def run(args,model,unused,device):
        from datasets import build_dataset
        from utils.misc import collate_fn
        from utils.loss_utils import one_ref_loss
        from torch.utils.data import DataLoader
        initial=torch.load(args.eval_model,map_location='cpu')['model'];model.load_state_dict(initial,strict=True)
        manifest=json.loads((root/'unseen04/manifest.json').read_text())
        manifest=dict(train=manifest['calibration'],val=json.loads((root/'confirm05/manifest.json').read_text()))
        with zipfile.ZipFile(root/'splits.zip') as z:full=torch.load(io.BytesIO(z.read('ref_data_shuffled/single_dataset/unc+/unc+_train.pth')))
        if a.mode=='full_replay':
            seen_images={r[0] for r in torch.load(root/'assets/splits/unc+/unc+_train.pth')}
            rng=random.Random(20261201)
            selected=rng.sample([i for i,r in enumerate(full) if r[0] in seen_images],4096)+rng.sample([i for i,r in enumerate(full) if r[0] not in seen_images],4096)
            rng.shuffle(selected)
            manifest=dict(train=[dict(source_index=i,image=str(full[i][0]),phrase=str(full[i][3]),baseline_seen_image=full[i][0] in seen_images) for i in selected],val=json.loads((root/'confirm05/manifest.json').read_text()))
            image_dir=root/'assets/image_data/other/images/mscoco/images/train2014'
            with zipfile.ZipFile(root/'images.zip') as z:
                members={Path(n).name:n for n in z.namelist() if n.endswith('.jpg')}
                for name in sorted({r['image'] for r in manifest['train']}):
                    if not (image_dir/name).exists():(image_dir/name).write_bytes(z.read(members[name]))
        (out/'manifest.json').write_text(json.dumps(manifest))
        from datasets import make_transforms
        train=build_dataset('train',args);train.transform=make_transforms(args,'val');train.images=[full[r['source_index']] for r in manifest['train']]
        val=build_dataset('val',args);val.images=[val.images[r['source_index']] for r in manifest['val']]
        assert not {r[0] for r in train.images}&{r[0] for r in val.images}
        loader=DataLoader(val,batch_size=16,shuffle=False,num_workers=2,collate_fn=collate_fn)
        def evaluate():
            model.eval();pred=[];target=[]
            with torch.no_grad():
                for batch in loader:
                    check();images,text,gt=batch[:3];images=images.to(device)
                    p=model(images.tensors,images.mask,text)[0];pred.append(p.cpu());target.append(gt)
            pred=torch.cat(pred);gt=torch.cat(target);iou,_=iou_giou(pred,gt,evaluation=True)
            return dict(acc50=float((iou>=.5).float().mean()*100),acc75=float((iou>=.75).float().mean()*100),miou=float(iou.mean()*100)),pred,iou
        baseline,bp,bi=evaluate();old=torch.load(root/'confirm05/predictions.pt',map_location='cpu')
        reference=old['pred']['original']
        if float((bp-reference).abs().max())>1e-5:raise RuntimeError('Baseline replay mismatch')
        del old
        prefixes=('beit3.encoder.layers.10.','beit3.encoder.layers.11.','beit3.encoder.layer_norm.','vision_head.','language_head.','visu_proj.','bbox_embed.')
        if a.mode=='full_replay':prefixes=('beit3.','vision_head.','language_head.','visu_proj.','bbox_embed.')
        teacher=torch.load(root/'verify_calibration12/predictions.pt',map_location='cpu')
        assert manifest['train']==json.loads((root/'verify_calibration12/manifest.json').read_text())
        teacher_pred=teacher['pred']['verified_consensus_0.02'].to(device)
        base_teacher=teacher['pred']['original'].to(device);teacher_gt=teacher['target'].to(device)
        gate=((iou_giou(teacher_pred,teacher_gt,evaluation=True)[0]-iou_giou(base_teacher,teacher_gt,evaluation=True)[0])>.01)&(iou_giou(teacher_pred,teacher_gt,evaluation=True)[0]>=.5)
        protocol['accepted_teacher_samples']=int(gate.sum())
        results={}
        for seed in protocol['seeds']:
            for arm in protocol['arms']:
                check();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                model.load_state_dict(initial,strict=True);model.eval().requires_grad_(False)
                if a.mode=='full_replay':model.train()
                for name,v in model.named_parameters():
                    if name.startswith(prefixes):v.requires_grad_(True)
                params=[v for v in model.parameters() if v.requires_grad]
                opt=torch.optim.AdamW([dict(params=[v for n,v in model.named_parameters() if v.requires_grad and n.startswith('beit3.')],lr=3e-6),dict(params=[v for n,v in model.named_parameters() if v.requires_grad and not n.startswith('beit3.')],lr=1e-5)],weight_decay=1e-4);scaler=torch.cuda.amp.GradScaler(enabled=False)
                gen=torch.Generator().manual_seed(seed)
                from torch.utils.data import Subset
                fixed_order=torch.cat([torch.randperm(len(train),generator=gen) for _ in range(a.steps*16//len(train)+1)])[:a.steps*16].tolist()
                trainloader=DataLoader(Subset(train,fixed_order),batch_size=16,shuffle=False,drop_last=True,num_workers=2,collate_fn=collate_fn)
                iterator=iter(trainloader);trace=[];conflict=None;t0=time.monotonic();torch.cuda.reset_peak_memory_stats()
                for step in range(1,a.steps+1):
                    check()
                    try:batch=next(iterator)
                    except StopIteration:iterator=iter(trainloader);batch=next(iterator)
                    images,text,gt,mask=batch[:4];images=images.to(device);gt=gt.to(device);mask=mask.to(device)
                    opt.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=a.precision=='bf16',dtype=torch.bfloat16):
                        output=model(images.tensors,images.mask,text,training=True,enable_ref_mim=False,enable_ref_mlm=False,global_step=step)
                    pred,contrast,scores,seg=output[:4]
                    # Keep geometric loss arithmetic FP32 under mixed-precision backbone forward.
                    parts=one_ref_loss(args,pred.float(),gt,mask,contrast,visu_sim=scores.float(),seg_mask=seg)
                    box=parts['loss_bbox']+parts['loss_giou'];aux=sum(v for k,v in parts.items() if k.startswith('loss_') and k not in ('loss_bbox','loss_giou'))
                    ids=torch.tensor(fixed_order[(step-1)*16:step*16],device=device)
                    if not torch.equal(gt,teacher_gt[ids]):raise RuntimeError('Teacher/student coordinates or order mismatch')
                    target_teacher=torch.where(gate[ids,None],teacher_pred[ids],base_teacher[ids]) if arm=='anchored_quality' else teacher_pred[ids]
                    kd=(pred.float()-target_teacher).abs().sum(-1).mean()+(1-iou_giou(pred.float(),target_teacher)[1]).mean()
                    loss=box+aux+(2*kd if arm!='native' else 0)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    if False:
                        gb=torch.autograd.grad(box,params,retain_graph=True,allow_unused=True)
                        ga=torch.autograd.grad(aux,params,retain_graph=True,allow_unused=True)
                        dot=sum(float((b*g).sum()) for b,g in zip(gb,ga) if b is not None and g is not None)
                        nb=sum(float(b.square().sum()) for b in gb if b is not None)**.5;na=sum(float(g.square().sum()) for g in ga if g is not None)**.5
                        conflict=dict(cosine=dot/max(nb*na,1e-30),box_norm=nb,aux_norm=na)
                        del gb,ga
                    scaler.scale(loss).backward();scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(params,1.)
                    if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
                    scaler.step(opt);scaler.update()
                    if step in (1,32,64,a.steps):
                        row=dict(step=step,loss=float(loss),box=float(box),aux=float(aux),kd=float(kd),grad_norm=float(norm),seconds=time.monotonic()-t0)
                        trace.append(row);print(arm,seed,json.dumps(row),flush=True)
                m,pred,iou=evaluate();key=f'{arm}_{seed}';hit=iou>=.5;before=bi>=.5
                results[key]=dict(final=m,corrected=int((hit&~before).sum()),regressed=int((~hit&before).sum()),gradient_conflict_first_batch=conflict,
                    trace=trace,steps=a.steps,seconds=time.monotonic()-t0,trainable_parameters=sum(v.numel() for v in params),peak_memory_mb=torch.cuda.max_memory_allocated()/1024**2)
                delta={k:v.detach().cpu() for k,v in model.state_dict().items() if k.startswith(prefixes)}
                torch.save(dict(state=delta,pred=pred,iou=iou),out/f'{key}.pt');print('RESULT',key,json.dumps(results[key]),flush=True)
                (out/'results.json').write_text(json.dumps(dict(status='in_progress',baseline=baseline,protocol=protocol,results=results,seconds=time.monotonic()-start),indent=2))
                del opt,scaler,delta,iterator,trainloader;torch.cuda.empty_cache()
        (out/'results.json').write_text(json.dumps(dict(status='completed',baseline=baseline,protocol=protocol,results=results,seconds=time.monotonic()-start),indent=2));print('COMPLETED',flush=True)
        return baseline['acc50']/100
    ev.evaluate=run;ev.main(args)

if __name__=='__main__':main()

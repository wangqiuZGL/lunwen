"""Resumable matched10 frozen +20 full-encoder epochs, single-view students."""
import argparse,importlib.util,io,json,math,os,random,shutil,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch
from run_spatial_readout_probe import iou_giou

CORE=('beit3.','vision_head.','language_head.','visu_proj.','bbox_embed.')
HEAD=CORE[1:]

def atomic_save(value,path):
    temporary=path.with_suffix(path.suffix+'.tmp');torch.save(value,temporary);os.replace(temporary,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/hy-tmp/layerwise-20261127');p.add_argument('--output',required=True)
    p.add_argument('--arm',choices=['native','anchored_quality'],required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    root=Path(a.root);out=root/a.output;out.mkdir(exist_ok=a.resume);torch.set_num_threads(4);start=time.monotonic()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    sys.path.insert(0,str(root/'code'));spec=importlib.util.spec_from_file_location('long_eval',root/'code/eval.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    args=ev.get_args_parser().parse_args(['--task','grounding'])
    for k,v in json.loads((root/'probe_config.json').read_text()).items():
        if hasattr(args,k):setattr(args,k,v)
    args.eval_model=str(root/'baseline_model_only.pth');args.output_dir=str(out);args.eval_set='val';args.batch_size=16;args.num_workers=2
    args.distributed=False;args.world_size=1;args.enable_syntax_bias=False;args.enable_ptp_aux=False
    protocol=dict(arm=a.arm,seed=a.seed,train=4096,val=128 if a.smoke else 2048,batch=16,freeze_epochs=1 if a.smoke else 10,finetune_epochs=1 if a.smoke else 20,
        freeze_head_lr=3e-5,finetune_head_lr=1e-5,finetune_encoder_lr=3e-6,scheduler='Stage-wise cosine to0.1initial LR',weight_decay=1e-4,kd_weight=2.,
        initialization='Same task-trained original checkpoint; continued training, not official pretrained-from-scratch reproduction',
        transforms='Deterministic384original-view resize and pad, matched to teacher; no random crop/flip/translation',
        precision='BF16 autocast train, FP32 losses/evaluation; no FP16 GradScaler',
        selection='Report final30epoch and development-best separately; no test selection. Resume only at completed epoch boundary, restore optimizer and RNG.',
        smoke=a.smoke)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));shutil.copy2(__file__,out/'train_long_consensus.py')
    def run(args,model,unused,device):
        from datasets import build_dataset,make_transforms
        from utils.misc import collate_fn
        from utils.loss_utils import one_ref_loss
        from torch.utils.data import DataLoader,Subset
        if not torch.cuda.is_bf16_supported():raise RuntimeError('Validated BF16 CUDA required')
        initial=torch.load(args.eval_model,map_location='cpu')['model'];model.load_state_dict(initial,strict=True);del initial
        manifest=dict(train=json.loads((root/'teacher_train18/manifest.json').read_text()),val=json.loads((root/'confirm05/manifest.json').read_text()))
        if a.smoke:manifest['val']=manifest['val'][:128]
        assert len(manifest['train'])==4096
        assert not {r['image'] for r in manifest['train']}&{r['image'] for r in manifest['val']}
        (out/'manifest.json').write_text(json.dumps(manifest))
        with zipfile.ZipFile(root/'splits.zip') as z:rows=torch.load(io.BytesIO(z.read('ref_data_shuffled/single_dataset/unc+/unc+_train.pth')))
        train=build_dataset('train',args);train.images=[rows[r['source_index']] for r in manifest['train']];train.transform=make_transforms(args,'val')
        val=build_dataset('val',args);val.images=[val.images[r['source_index']] for r in manifest['val']]
        assert all(str(row[0])==m['image'] and str(row[3])==m['phrase'] for row,m in zip(train.images,manifest['train']))
        teacher=torch.load(root/'teacher_train18/predictions.pt',map_location='cpu');tp=teacher['pred']['verified_consensus_0.02'].to(device);bp=teacher['pred']['original'].to(device);gt=teacher['target'].to(device)
        assert len(tp)==len(train)
        gate=((iou_giou(tp,gt,evaluation=True)[0]-iou_giou(bp,gt,evaluation=True)[0])>.01)&(iou_giou(tp,gt,evaluation=True)[0]>=.5)
        target_teacher=torch.where(gate[:,None],tp,bp).detach();protocol['teacher_accepted']=int(gate.sum());(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        vloader=DataLoader(val,batch_size=16,shuffle=False,num_workers=2,collate_fn=collate_fn)
        def evaluate():
            model.eval();pred=[];target=[]
            with torch.no_grad():
                for batch in vloader:
                    images,text,y=batch[:3];images=images.to(device);pred.append(model(images.tensors,images.mask,text)[0].cpu());target.append(y)
            pred=torch.cat(pred);target=torch.cat(target);iou=iou_giou(pred,target,evaluation=True)[0]
            return dict(acc50=float((iou>=.5).float().mean()*100),acc75=float((iou>=.75).float().mean()*100),miou=float(iou.mean()*100)),pred,iou
        baseline,pred,_=evaluate();reference=torch.load(root/'confirm05/predictions.pt',map_location='cpu')['pred']['original'][:len(pred)]
        if float((pred-reference).abs().max())>1e-5:raise RuntimeError('Original model replay mismatch')
        def model_state():return {k:v.detach().cpu() for k,v in model.state_dict().items() if k.startswith(CORE)}
        completed=0;history=[];best=-1.;resume_state=None
        if a.resume:
            resume_state=torch.load(out/'last_training.pth',map_location='cpu')
            if resume_state['arm']!=a.arm or resume_state['seed']!=a.seed:raise RuntimeError('Resume identity mismatch')
            expected={k for k in model.state_dict() if k.startswith(CORE)}
            if set(resume_state['model'])!=expected:raise RuntimeError('Resume core state incomplete')
            model.load_state_dict(resume_state['model'],strict=False);completed=resume_state['completed_epochs'];history=resume_state['history'];best=resume_state['best']
        total=protocol['freeze_epochs']+protocol['finetune_epochs'];optimizer=None;previous_stage=None
        if completed>=total:
            print('ALREADY_COMPLETED',flush=True)
            return history[-1]['val']['acc50']/100
        for epoch in range(completed,total):
            stage='freeze' if epoch<protocol['freeze_epochs'] else 'finetune'
            local_epoch=epoch if stage=='freeze' else epoch-protocol['freeze_epochs'];stage_epochs=protocol['freeze_epochs'] if stage=='freeze' else protocol['finetune_epochs']
            if stage!=previous_stage:
                model.requires_grad_(False)
                for name,v in model.named_parameters():
                    if name.startswith(HEAD) or (stage=='finetune' and name.startswith('beit3.')):v.requires_grad_(True)
                groups=[dict(params=[v for n,v in model.named_parameters() if v.requires_grad and n.startswith(HEAD)],lr=protocol['freeze_head_lr'] if stage=='freeze' else protocol['finetune_head_lr'])]
                if stage=='finetune':groups.append(dict(params=[v for n,v in model.named_parameters() if v.requires_grad and n.startswith('beit3.')],lr=protocol['finetune_encoder_lr']))
                optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
                if resume_state is not None and resume_state['stage']==stage:
                    optimizer.load_state_dict(resume_state['optimizer'])
                if resume_state is not None:
                    random.setstate(resume_state['random']);np.random.set_state(resume_state['numpy']);torch.set_rng_state(resume_state['torch']);torch.cuda.set_rng_state_all(resume_state['cuda']);resume_state=None
                previous_stage=stage
            model.train()
            if stage=='freeze':model.beit3.eval()
            gen=torch.Generator().manual_seed(a.seed+epoch*100003)
            order=torch.randperm(len(train),generator=gen).tolist()
            if a.smoke:order=order[:32]
            loader=DataLoader(Subset(train,order),batch_size=16,shuffle=False,num_workers=2,drop_last=True,collate_fn=collate_fn)
            sums={};t0=time.monotonic();torch.cuda.reset_peak_memory_stats()
            for step,batch in enumerate(loader):
                frac=(local_epoch+step/len(loader))/stage_epochs;factor=.1+.9*(1+math.cos(math.pi*frac))/2
                optimizer.param_groups[0]['lr']=(protocol['freeze_head_lr'] if stage=='freeze' else protocol['finetune_head_lr'])*factor
                if stage=='finetune':optimizer.param_groups[1]['lr']=protocol['finetune_encoder_lr']*factor
                images,text,y,mask=batch[:4];images=images.to(device);y=y.to(device);mask=mask.to(device);ids=torch.tensor(order[step*16:(step+1)*16],device=device)
                if not torch.equal(y,gt[ids]):raise RuntimeError('Teacher/GT order or coordinate mismatch')
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):outputs=model(images.tensors,images.mask,text,training=True,enable_ref_mim=False,enable_ref_mlm=False,global_step=epoch*256+step)
                box,contrast,scores,seg=outputs[:4];parts=one_ref_loss(args,box.float(),y,mask,contrast,visu_sim=scores.float(),seg_mask=seg)
                native=sum(v for k,v in parts.items() if k.startswith('loss_'));kd=(box.float()-target_teacher[ids]).abs().sum(-1).mean()+(1-iou_giou(box.float(),target_teacher[ids])[1]).mean()
                loss=native+(2*kd if a.arm=='anchored_quality' else 0)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.)
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
                optimizer.step()
                for k,v in dict(native=native,kd=kd,loss=loss).items():sums[k]=sums.get(k,0.)+float(v)/len(loader)
                if step%64==0:print(json.dumps(dict(stage=stage,epoch=epoch+1,step=step+1,total_steps=len(loader),loss=float(loss),seconds=time.monotonic()-t0)),flush=True)
            train_seconds=time.monotonic()-t0;metrics,pred,iou=evaluate()
            row=dict(epoch=epoch+1,stage=stage,train=sums,val=metrics,train_seconds=train_seconds,epoch_seconds=time.monotonic()-t0,peak_memory_mb=torch.cuda.max_memory_allocated()/1024**2);history.append(row)
            print('EPOCH_RESULT',json.dumps(row),flush=True)
            if metrics['acc50']>best:
                best=metrics['acc50']
                if not a.smoke:atomic_save(dict(state=model_state(),epoch=epoch+1,pred=pred,iou=iou),out/'best_model.pt')
            report=dict(status='in_progress',protocol=protocol,baseline=baseline,completed_epochs=epoch+1,history=history,best_development_acc50=best,seconds=time.monotonic()-start)
            (out/'results.json').write_text(json.dumps(report,indent=2))
            if not a.smoke:
                if shutil.disk_usage(root).free<4*1024**3:raise RuntimeError('Insufficient disk for atomic resume checkpoint')
                atomic_save(dict(arm=a.arm,seed=a.seed,stage=stage,completed_epochs=epoch+1,model=model_state(),optimizer=optimizer.state_dict(),history=history,best=best,random=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),out/'last_training.pth')
        if not a.smoke:atomic_save(dict(state=model_state(),epoch=total,pred=pred,iou=iou),out/'final_model.pt')
        report['status']='completed';report['seconds']=time.monotonic()-start;(out/'results.json').write_text(json.dumps(report,indent=2));print('COMPLETED',flush=True)
        return metrics['acc50']/100
    ev.evaluate=run;ev.main(args)

if __name__=='__main__':main()

"""Fixed inference-only reflected view; no GT-dependent acceptance."""
import argparse,importlib.util,io,json,random,re,sys,time,zipfile
from pathlib import Path
import torch
from PIL import Image
from run_spatial_readout_probe import iou_giou,corners

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/hy-tmp/layerwise-20261127');p.add_argument('--output',default='flip10')
    p.add_argument('--confirmation',action='store_true');p.add_argument('--full',action='store_true');p.add_argument('--split',default='val',choices=['val','testA','testB'])
    p.add_argument('--calibration',action='store_true');p.add_argument('--verify',action='store_true');p.add_argument('--train-manifest')
    p.add_argument('--integration-smoke',action='store_true')
    p.add_argument('--seconds',type=int,default=600);a=p.parse_args();root=Path(a.root);out=root/a.output;out.mkdir(exist_ok=False)
    start=time.monotonic();torch.set_num_threads(4);sys.path.insert(0,str(root/'code'))
    if a.split!='val':
        dest=root/f'assets/splits/unc+/unc+_{a.split}.pth'
        if not dest.exists():
            with zipfile.ZipFile(root/'splits.zip') as z:dest.write_bytes(z.read(f'ref_data_shuffled/single_dataset/unc+/unc+_{a.split}.pth'))
    spec=importlib.util.spec_from_file_location('flip_eval',root/'code/eval.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    args=ev.get_args_parser().parse_args(['--task','grounding'])
    for k,v in json.loads((root/'probe_config.json').read_text()).items():
        if hasattr(args,k):setattr(args,k,v)
    args.eval_model=str(root/'baseline_model_only.pth');args.output_dir=str(out);args.eval_set=a.split;args.batch_size=16;args.num_workers=2
    args.distributed=False;args.world_size=1;args.enable_syntax_bias=False;args.enable_ptp_aux=False
    import shutil;shutil.copy2(__file__,out/'flip_consistency_screen.py')
    def run(args,model,unused,device):
        from datasets import build_dataset
        from utils.misc import collate_fn
        from torch.utils.data import DataLoader
        model.load_state_dict(torch.load(args.eval_model,map_location='cpu')['model'],strict=True);model.eval().requires_grad_(False)
        ds=build_dataset(a.split,args);rows=ds.images
        if a.calibration or a.train_manifest:
            ds=build_dataset('train',args)
            from datasets import make_transforms
            ds.transform=make_transforms(args,'val')
            with zipfile.ZipFile(root/'splits.zip') as z:rows=torch.load(io.BytesIO(z.read('ref_data_shuffled/single_dataset/unc+/unc+_train.pth')))
            manifest=json.loads(Path(a.train_manifest).read_text()) if a.train_manifest else json.loads((root/'unseen04/manifest.json').read_text())['calibration']
        elif a.full:manifest=[dict(source_index=i,image=str(r[0]),phrase=str(r[3])) for i,r in enumerate(rows)]
        elif a.confirmation:manifest=json.loads((root/'confirm05/manifest.json').read_text())
        else:manifest=json.loads((root/'manifest.json').read_text())['val']
        ds.images=[rows[r['source_index']] for r in manifest];(out/'manifest.json').write_text(json.dumps(manifest))
        protocol=dict(split='official_train_teacher' if a.train_manifest else 'official_train_calibration' if a.calibration else a.split,full=a.full,confirmation=a.confirmation,samples=len(manifest),verify=a.verify,
            rules=['original','flip','mean','consensus_mean','valid_projection','consensus_projection'],consensus_iou=.5,
            direction_policy='No change on expressions containing left/right/compass-direction/clock tokens; all methods fall back to original',
            scope='Fixed inference rules with extra forward cost; no novelty claim; validation reused, final test only after config freeze')
        if a.verify:
            protocol.update(verification_margins=[0.,.02,.05],frozen_primary='verified_consensus_0.02',
                selection='Primary margin0.02 fixed before full validation; train-calibration512 only. Other margins reported as diagnostics, not selected on final evaluation.')
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        image_dir=root/'assets/image_data/other/images/mscoco/images/train2014'
        with zipfile.ZipFile(root/'images.zip') as z:
            members={Path(n).name:n for n in z.namelist() if n.endswith('.jpg')}
            for name in sorted({r['image'] for r in manifest}):
                if not (image_dir/name).exists():(image_dir/name).write_bytes(z.read(members[name]))
        bounds=[]
        for r in manifest:
            with Image.open(image_dir/r['image']) as im:w,h=im.size
            nw,nh=round(w*384/max(w,h)),round(h*384/max(w,h));left,top=round((384-nw)/2-.1),round((384-nh)/2-.1)
            bounds.append([left/384,top/384,(left+nw)/384,(top+nh)/384])
        base=[];flipped=[];target=[];scoreviews=[];valids=[];captured={};handles=[];integration_pred=None
        if a.verify:
            for name in ('vision_head','language_head'):
                def capture(m,ins,result,key=name):captured[key]=result.detach()
                handles.append(getattr(model,name).register_forward_hook(capture))
        def scores():
            from torch.nn import functional as F
            return (F.normalize(captured['vision_head'],dim=-1)[:,1:]*F.normalize(captured['language_head'],dim=-1)[:,0,None]).sum(-1)
        for step,batch in enumerate(DataLoader(ds,batch_size=16,num_workers=2,shuffle=False,collate_fn=collate_fn)):
            if time.monotonic()-start>a.seconds:raise TimeoutError('Flip diagnostic time exhausted')
            images,text,gt=batch[:3];images=images.to(device)
            with torch.no_grad():
                b=model(images.tensors,images.mask,text)[0]
                if a.verify:s0=scores()
                f=model(images.tensors.flip(-1),images.mask.flip(-1),text)[0];f[:,0]=1-f[:,0]
                if a.verify:
                    s1=scores().reshape(-1,24,24).flip(-1).flatten(1)
                    scoreviews.append(torch.stack((s0,s1),1).cpu())
                    from torch.nn import functional as F
                    valids.append(F.adaptive_avg_pool2d((~images.mask.bool()).float()[:,None],(24,24)).flatten(1).cpu()>.5)
                if a.integration_smoke and step==0:
                    from utils.rec_refinement import predict_refined
                    integration_pred=predict_refined(model,images.tensors,images.mask,text,'verified_flip')[0].cpu()
            base.append(b.cpu());flipped.append(f.cpu());target.append(gt)
            if step%64==0:print('EVAL',(step+1)*16,time.monotonic()-start,flush=True)
        b=torch.cat(base);f=torch.cat(flipped);gt=torch.cat(target);bounds=torch.tensor(bounds)
        skip=torch.tensor([bool(re.search(r'left|right|north|south|east|west|clock',r['phrase'].lower())) for r in manifest])
        f=torch.where(skip[:,None],b,f)
        def clip(p,window):
            c=corners(p);c=torch.maximum(torch.minimum(c,window[:,[2,3,2,3]]),window[:,[0,1,0,1]])
            return torch.cat(((c[:,:2]+c[:,2:])/2,c[:,2:]-c[:,:2]),-1)
        frame=torch.tensor([[0.,0.,1.,1.]]).expand(len(b),-1)
        agreement=iou_giou(clip(b,frame),clip(f,frame),evaluation=True)[0]
        mean=(b+f)/2;consensus=torch.where((agreement>=.5)[:,None],mean,b)
        pred=dict(original=b,flip=f,mean=mean,consensus_mean=consensus,valid_projection=clip(b,bounds),consensus_projection=clip(consensus,bounds))
        extra={}
        if a.verify:
            for h in handles:h.remove()
            sv=torch.cat(scoreviews);valid=torch.cat(valids)
            axis=(torch.arange(24)+.5)/24;yy,xx=torch.meshgrid(axis,axis,indexing='ij');xy=torch.stack((xx,yy),-1).reshape(1,576,2)
            support=[]
            for proposal in (b,f):
                c=corners(clip(proposal,frame));inside=((xy>=c[:,None,:2])&(xy<=c[:,None,2:])).all(-1)&valid
                empty=~inside.any(-1)
                if empty.any():
                    distance=(xy-proposal[:,None,:2]).square().sum(-1).masked_fill(~valid,torch.inf)
                    inside[empty]=False;inside[empty,distance[empty].argmin(-1)]=True
                support.append((sv*inside[:,None]).sum(-1)/inside.sum(-1)[:,None].clamp_min(1))
            advantage=support[1]-support[0]
            # Each proposal is compared within the same fixed score map in each view.
            # This avoids comparing uncalibrated absolute scores between views.
            for margin in (0.,.02,.05):
                accept=(advantage>margin).all(-1)&~skip
                choice=torch.where(accept[:,None],f,b)
                candidate=torch.where((agreement>=.5)[:,None],mean,choice)
                pred[f'verified_{margin}']=clip(choice,bounds)
                pred[f'verified_consensus_{margin}']=clip(candidate,bounds)
                extra[f'verified_{margin}_accepts']=int(accept.sum())
            extra['score_rule']='Area-normalized mean native patch cosine inside candidate, using only valid-image patch centers; candidate comparison within each of two aligned maps; both must prefer flipped candidate.'
        ious={k:iou_giou(v,gt,evaluation=True)[0] for k,v in pred.items()};before=ious['original']>=.5;results={}
        if integration_pred is not None:
            error=float((integration_pred-pred['verified_consensus_0.02'][:len(integration_pred)]).abs().max())
            if error>1e-6:raise RuntimeError('Deployed inference module parity failed '+str(error))
            extra['integration_prediction_max_error']=error
        for k,iou in ious.items():
            hit=iou>=.5;results[k]=dict(acc50=float(hit.float().mean()*100),acc75=float((iou>=.75).float().mean()*100),miou=float(iou.mean()*100),
                corrected=int((hit&~before).sum()),regressed=int((~hit&before).sum()))
        torch.save(dict(pred=pred,iou=ious,target=gt,agreement=agreement,skip=skip,score_advantage=advantage if a.verify else None),out/'predictions.pt')
        report=dict(status='completed',protocol=protocol,skipped=int(skip.sum()),accepted=int((agreement>=.5).sum()),extra=extra,results=results,seconds=time.monotonic()-start)
        (out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
        return results['original']['acc50']/100
    ev.evaluate=run;ev.main(args)

if __name__=='__main__':main()

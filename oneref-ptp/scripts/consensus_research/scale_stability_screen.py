"""Prespecified scale/letterbox stability and equal-weight ensemble diagnostic."""
import argparse,importlib.util,json,sys,time,zipfile,random
from pathlib import Path
import torch
from PIL import Image
from run_spatial_readout_probe import iou_giou


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/hy-tmp/layerwise-20261127')
    p.add_argument('--output',default='scale03');p.add_argument('--confirmation',type=int,default=0)
    p.add_argument('--seconds',type=int,default=600);a=p.parse_args();root=Path(a.root);out=root/a.output;out.mkdir(exist_ok=False)
    start=time.monotonic();torch.set_num_threads(4);sys.path.insert(0,str(root/'code'))
    spec=importlib.util.spec_from_file_location('scale_eval',root/'code/eval.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    args=ev.get_args_parser().parse_args(['--task','grounding'])
    for k,v in json.loads((root/'probe_config.json').read_text()).items():
        if hasattr(args,k):setattr(args,k,v)
    args.eval_model=str(root/'best_checkpoint.pth');args.output_dir=str(out);args.eval_set='val';args.batch_size=16;args.num_workers=2
    args.distributed=False;args.world_size=1;args.enable_syntax_bias=False;args.enable_ptp_aux=False
    protocol=dict(scales=[384,352,320],arms=['original','scale352','scale320','mean384_352','mean3'],
        rule='Equal arithmetic mean in common original384 coordinates; fixed rules, no GT-dependent selection',
        confirmation=a.confirmation,scope='Exploratory scaling diagnostic, extra inference compute, no novelty claim')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    import shutil;shutil.copy2(__file__,out/'scale_stability_screen.py')
    def run(args,model,unused,device):
        from datasets import build_dataset
        import datasets.transforms as T
        from utils.misc import collate_fn
        from torch.utils.data import DataLoader
        model.load_state_dict(torch.load(args.eval_model,map_location='cpu')['model'],strict=True);model.eval().requires_grad_(False)
        old=json.loads((root/'manifest.json').read_text());manifest=old['val']
        source=build_dataset('val',args);rows=source.images
        if a.confirmation:
            excluded={r['image'] for r in manifest}
            candidates=[i for i,r in enumerate(rows) if str(r[0]) not in excluded]
            ids=random.Random(20261128).sample(candidates,min(a.confirmation,len(candidates)))
            manifest=[dict(source_index=i,image=str(rows[i][0]),phrase=str(rows[i][3])) for i in ids]
        (out/'manifest.json').write_text(json.dumps(manifest))
        image_dir=root/'assets/image_data/other/images/mscoco/images/train2014'
        needed={m['image'] for m in manifest if not (image_dir/m['image']).exists()}
        if needed:
            with zipfile.ZipFile(root/'images.zip') as z:
                members={Path(n).name:n for n in z.namelist() if n.endswith('.jpg')}
                for name in sorted(needed):(image_dir/name).write_bytes(z.read(members[name]))
        sizes=[Image.open(image_dir/m['image']).size for m in manifest]
        allpred={};targets={}
        def affine(scale):
            offsets=[];original=[]
            for w,h in sizes:
                wh=[round(v*scale/max(w,h)) for v in (w,h)]
                orig=[round(v*384/max(w,h)) for v in (w,h)]
                offsets.append([round((384-v)/2-.1) for v in wh]);original.append([round((384-v)/2-.1) for v in orig])
            return torch.tensor(offsets),torch.tensor(original)
        for scale in protocol['scales']:
            source.images=[rows[m['source_index']] for m in manifest]
            source.transform=T.Compose([T.RandomResize([scale]),T.ToTensor(),T.NormalizeAndPad(size=384)])
            preds=[];gt=[]
            for step,batch in enumerate(DataLoader(source,batch_size=16,shuffle=False,num_workers=2,collate_fn=collate_fn)):
                if time.monotonic()-start>a.seconds:raise TimeoutError('Scale diagnostic budget exhausted')
                images,text,target=batch[:3];images=images.to(device)
                with torch.no_grad():pred=model(images.tensors,images.mask,text)[0]
                preds.append(pred.cpu());gt.append(target)
            pred=torch.cat(preds);target=torch.cat(gt);offset,original=affine(scale)
            for box in (pred,target):
                box[:,:2]=((box[:,:2]*384-offset)*(384/scale)+original)/384
                box[:,2:]*=384/scale
            allpred[str(scale)]=pred;targets[str(scale)]=target
            if scale!=384 and float((target-targets['384']).abs().max())>2e-6:raise RuntimeError('Box coordinate affine round-trip failed')
            print('SCALE_DONE',scale,'seconds',time.monotonic()-start,flush=True)
        if not a.confirmation:
            cached=torch.load(root/'screen01/val_cache.pt',map_location='cpu')
            err=float((allpred['384']-cached['pred']).abs().max())
            if err>1e-5:raise RuntimeError('Baseline replay mismatch '+str(err))
        else:err=None
        predictions=dict(original=allpred['384'],scale352=allpred['352'],scale320=allpred['320'],
            mean384_352=(allpred['384']+allpred['352'])/2,mean3=sum(allpred.values())/3)
        ious={k:iou_giou(v,targets['384'],evaluation=True)[0] for k,v in predictions.items()}
        results={}
        base=ious['original']>=.5
        for key,iou in ious.items():
            hit=iou>=.5;results[key]=dict(acc50=float(hit.float().mean()*100),acc75=float((iou>=.75).float().mean()*100),miou=float(iou.mean()*100),
                corrected=int((hit & ~base).sum()),regressed=int((base & ~hit).sum()),
                corrected_indices=(hit & ~base).nonzero().flatten().tolist(),regressed_indices=(base & ~hit).nonzero().flatten().tolist())
        torch.save(dict(pred=predictions,iou=ious,target=targets['384']),out/'predictions.pt')
        report=dict(status='completed',protocol=protocol,results=results,baseline_replay_error=err,samples=len(manifest),seconds=time.monotonic()-start)
        (out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
        return results['original']['acc50']/100
    ev.evaluate=run;ev.main(args)


if __name__=='__main__':main()

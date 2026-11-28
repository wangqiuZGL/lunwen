"""Frozen last-block checkpoints, enlarged/full validation or one final test."""
import argparse,importlib.util,io,json,sys,time,zipfile,random
from pathlib import Path
import torch
from run_spatial_readout_probe import iou_giou

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/hy-tmp/layerwise-20261127');p.add_argument('--output',default='pair_full20')
    p.add_argument('--experiment',default='paired_distill19');p.add_argument('--split',default='val',choices=['val','testA','testB'])
    p.add_argument('--full',action='store_true');p.add_argument('--seconds',type=int,default=1800);a=p.parse_args();root=Path(a.root);out=root/a.output;out.mkdir(exist_ok=False)
    start=time.monotonic();torch.set_num_threads(4);sys.path.insert(0,str(root/'code'))
    if a.split!='val':
        dest=root/f'assets/splits/unc+/unc+_{a.split}.pth'
        if not dest.exists():
            with zipfile.ZipFile(root/'splits.zip') as z:dest.write_bytes(z.read(f'ref_data_shuffled/single_dataset/unc+/unc+_{a.split}.pth'))
    spec=importlib.util.spec_from_file_location('last_confirm_eval',root/'code/eval.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    args=ev.get_args_parser().parse_args(['--task','grounding'])
    for k,v in json.loads((root/'probe_config.json').read_text()).items():
        if hasattr(args,k):setattr(args,k,v)
    args.eval_model=str(root/'baseline_model_only.pth');args.output_dir=str(out);args.eval_set=a.split;args.batch_size=16;args.num_workers=2
    args.distributed=False;args.world_size=1;args.enable_syntax_bias=False;args.enable_ptp_aux=False
    import shutil;shutil.copy2(__file__,out/'confirm_lastblocks.py')
    def check():
        if time.monotonic()-start>a.seconds:raise TimeoutError('Confirmation budget exhausted')
    def run(args,model,unused,device):
        from datasets import build_dataset
        from utils.misc import collate_fn
        from torch.utils.data import DataLoader
        initial=torch.load(args.eval_model,map_location='cpu')['model'];model.load_state_dict(initial,strict=True);model.eval().requires_grad_(False)
        ds=build_dataset(a.split,args);rows=ds.images
        if a.full:
            manifest=[dict(source_index=i,image=str(r[0]),phrase=str(r[3])) for i,r in enumerate(rows)]
        else:
            assert a.split=='val'
            manifest=json.loads((root/'confirm05/manifest.json').read_text());ds.images=[rows[r['source_index']] for r in manifest]
        (out/'manifest.json').write_text(json.dumps(manifest))
        protocol=dict(experiment=a.experiment,split=a.split,full=a.full,samples=len(manifest),seeds=[42,43],arms=['native','anchored_quality'],
            selection='Both fixed final1024step checkpoints, no best-seed or best-step selection',
            scope='Official val previously used for baseline checkpoint choice and exploratory analyses; final test only after method frozen' if a.split=='val' else 'Frozen configuration final test; never used for training or threshold selection')
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        image_dir=root/'assets/image_data/other/images/mscoco/images/train2014'
        with zipfile.ZipFile(root/'images.zip') as z:
            members={Path(n).name:n for n in z.namelist() if n.endswith('.jpg')}
            for name in sorted({m['image'] for m in manifest}):
                if not (image_dir/name).exists():(image_dir/name).write_bytes(z.read(members[name]))
        loader=DataLoader(ds,batch_size=16,shuffle=False,num_workers=2,collate_fn=collate_fn)
        predictions={};targets=None;results={}
        for key in ['original']+[f'{arm}_{seed}' for seed in (42,43) for arm in protocol['arms']]:
            check();model.load_state_dict(initial,strict=True)
            if key!='original':
                state=torch.load(root/a.experiment/f'{key}.pt',map_location='cpu')['state']
                expected={k for k in initial if k.startswith(('beit3.encoder.layers.10.','beit3.encoder.layers.11.','beit3.encoder.layer_norm.','vision_head.','language_head.','visu_proj.','bbox_embed.'))}
                if set(state)!=expected:raise RuntimeError('Incomplete or unexpected adaptation state')
                missing=model.load_state_dict(state,strict=False)
                if missing.unexpected_keys:raise RuntimeError('Unexpected adaptation keys')
                del state
            model.eval();pred=[];gt=[]
            with torch.no_grad():
                for step,batch in enumerate(loader):
                    check();images,text,target=batch[:3];images=images.to(device)
                    pred.append(model(images.tensors,images.mask,text)[0].cpu());gt.append(target)
                    if step%128==0:print('EVAL',key,(step+1)*16,time.monotonic()-start,flush=True)
            pred=torch.cat(pred);current=torch.cat(gt)
            if targets is None:targets=current
            elif not torch.equal(targets,current):raise RuntimeError('Target order changed')
            predictions[key]=pred;iou,_=iou_giou(pred,targets,evaluation=True)
            results[key]=dict(acc50=float((iou>=.5).float().mean()*100),acc75=float((iou>=.75).float().mean()*100),miou=float(iou.mean()*100))
            print('RESULT',key,json.dumps(results[key]),flush=True)
            (out/'results.json').write_text(json.dumps(dict(status='in_progress',protocol=protocol,results=results,seconds=time.monotonic()-start),indent=2))
        ious={k:iou_giou(v,targets,evaluation=True)[0] for k,v in predictions.items()};base=ious['original']>=.5
        for k,iou in ious.items():
            hit=iou>=.5;results[k].update(corrected=int((hit&~base).sum()),regressed=int((~hit&base).sum()),delta_pp=results[k]['acc50']-results['original']['acc50'])
        torch.save(dict(pred=predictions,iou=ious,target=targets),out/'predictions.pt')
        (out/'results.json').write_text(json.dumps(dict(status='completed',protocol=protocol,results=results,seconds=time.monotonic()-start),indent=2));print('COMPLETED',flush=True)
        return results['original']['acc50']/100
    ev.evaluate=run;ev.main(args)

if __name__=='__main__':main()

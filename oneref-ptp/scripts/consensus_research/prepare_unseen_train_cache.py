"""Cache new official training images outside the baseline20% image subset."""
import argparse,importlib.util,io,json,sys,time,zipfile,random
from pathlib import Path
import torch
from run_spatial_readout_probe import iou_giou
from layerwise_fusion import LayerwiseHead

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/hy-tmp/layerwise-20261127');a=p.parse_args()
    root=Path(a.root);out=root/'unseen04';out.mkdir(exist_ok=False);start=time.monotonic();torch.set_num_threads(4)
    sys.path.insert(0,str(root/'code'));spec=importlib.util.spec_from_file_location('unseen_eval',root/'code/eval.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    args=ev.get_args_parser().parse_args(['--task','grounding'])
    for k,v in json.loads((root/'probe_config.json').read_text()).items():
        if hasattr(args,k):setattr(args,k,v)
    args.eval_model=str(root/'best_checkpoint.pth');args.output_dir=str(out);args.eval_set='val';args.batch_size=16;args.num_workers=2
    args.distributed=False;args.world_size=1;args.enable_syntax_bias=False;args.enable_ptp_aux=False
    import shutil;shutil.copy2(__file__,out/'prepare_unseen_train_cache.py')
    def run(args,model,unused,device):
        from datasets import build_dataset,make_transforms
        from utils.misc import collate_fn,mdetr_interpolate
        from torch.utils.data import DataLoader
        model.load_state_dict(torch.load(args.eval_model,map_location='cpu')['model'],strict=True);model.eval().requires_grad_(False)
        with zipfile.ZipFile(root/'splits.zip') as z:full=torch.load(io.BytesIO(z.read('ref_data_shuffled/single_dataset/unc+/unc+_train.pth')))
        seen=torch.load(root/'assets/splits/unc+/unc+_train.pth');seen_images={r[0] for r in seen}
        rng=random.Random(20261129);newimages=sorted({r[0] for r in full}-seen_images);rng.shuffle(newimages)
        cal_images=set(newimages[:len(newimages)//4]);train_images=set(newimages[len(newimages)//4:])
        choices={}
        for split,n,allowed in [('train',1536,train_images),('calibration',512,cal_images)]:
            eligible=[i for i,r in enumerate(full) if r[0] in allowed];choices[split]=rng.sample(eligible,n)
        manifest={k:[dict(source_index=i,image=str(full[i][0]),phrase=str(full[i][3])) for i in ids] for k,ids in choices.items()}
        old_manifest=json.loads((root/'manifest.json').read_text());manifest['val']=old_manifest['val']
        sets={k:{r['image'] for r in rows} for k,rows in manifest.items()}
        assert not sets['train']&sets['calibration'] and not sets['val']&(sets['train']|sets['calibration'])
        assert not seen_images&(sets['train']|sets['calibration'])
        (out/'manifest.json').write_text(json.dumps(manifest))
        image_dir=root/'assets/image_data/other/images/mscoco/images/train2014'
        with zipfile.ZipFile(root/'images.zip') as z:
            members={Path(n).name:n for n in z.namelist() if n.endswith('.jpg')}
            for name in sorted(sets['train']|sets['calibration']):
                if not (image_dir/name).exists():(image_dir/name).write_bytes(z.read(members[name]))
        print('DATA_READY',len(sets['train']),len(sets['calibration']),flush=True)
        captured={}
        def hook(m,ins,result):captured['x']=model.beit3.encoder.layer_norm(result[0].detach())[:,:578]
        handle=model.beit3.encoder.layers[11].register_forward_hook(hook)
        base=LayerwiseHead(model,'layer12').to(device);max_error=0.;summary={}
        try:
            for split in ('train','calibration'):
                ds=build_dataset('train',args);ds.transform=make_transforms(args,'val');ds.images=[full[i] for i in choices[split]]
                chunks={k:[] for k in ('features','target','mask','pred')}
                for step,batch in enumerate(DataLoader(ds,batch_size=16,shuffle=False,num_workers=2,collate_fn=collate_fn)):
                    if time.monotonic()-start>600:raise TimeoutError('Unseen cache budget exhausted')
                    images,text,target,obj_mask=batch[:4];images=images.to(device)
                    with torch.no_grad():
                        pred=model(images.tensors,images.mask,text)[0];features=captured.pop('x').half()
                        replay=base(features.float()[:,None].expand(-1,3,-1,-1))[0]
                        error=float((replay-pred).abs().max());max_error=max(max_error,error)
                        if error>.002:raise RuntimeError('Cache replay failed')
                        mask=mdetr_interpolate(obj_mask.float(),(24,24),mode='nearest')>.5
                    for k,v in dict(features=features,target=target,mask=mask,pred=pred).items():chunks[k].append(v.detach().cpu())
                    if step%32==0:print('CACHE',split,(step+1)*16,time.monotonic()-start,flush=True)
                cache={k:torch.cat(v) for k,v in chunks.items()};torch.save(cache,out/f'{split}_cache.pt')
                iou,_=iou_giou(cache['pred'],cache['target'],evaluation=True)
                summary[split]=dict(samples=len(iou),acc50=float((iou>=.5).float().mean()*100),miou=float(iou.mean()*100))
                del cache,chunks
        finally:handle.remove()
        old=torch.load(root/'screen01/val_cache.pt',map_location='cpu');old['features']=old['features'][:,2,:578].contiguous();torch.save(old,out/'val_cache.pt')
        report=dict(status='completed',baseline=summary,max_replay_error=max_error,seconds=time.monotonic()-start,
            scope='1536 new official-training descriptions +512 image-disjoint quality-calibration descriptions; all outside baseline20% training images. Existing exploratory256 val reused. Extra data is a confound explicitly controlled by plain head adaptation.')
        (out/'cache_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
        torch.save({name:getattr(model,name).float().cpu().state_dict() for name in ('vision_head','language_head','visu_proj','bbox_embed')},out/'original_heads.pt')
        return summary['calibration']['acc50']/100
    ev.evaluate=run;ev.main(args)

if __name__=='__main__':main()

"""Frozen-config enlarged validation for both candidate seeds and matched controls."""
import argparse,importlib.util,json,sys,time,zipfile,random
from pathlib import Path
import torch
from torch.nn import functional as F
from train_unseen_refinement import Correction,roi_samples,original_heads
from layerwise_fusion import LayerwiseHead
from run_spatial_readout_probe import iou_giou

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/hy-tmp/layerwise-20261127');p.add_argument('--samples',type=int,default=2048)
    p.add_argument('--output',default='confirm05');p.add_argument('--seconds',type=int,default=600);a=p.parse_args();root=Path(a.root);out=root/a.output;out.mkdir(exist_ok=False)
    start=time.monotonic();torch.set_num_threads(4);sys.path.insert(0,str(root/'code'))
    spec=importlib.util.spec_from_file_location('confirm_eval',root/'code/eval.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    args=ev.get_args_parser().parse_args(['--task','grounding'])
    for k,v in json.loads((root/'probe_config.json').read_text()).items():
        if hasattr(args,k):setattr(args,k,v)
    args.eval_model=str(root/'best_checkpoint.pth');args.output_dir=str(out);args.eval_set='val';args.batch_size=16;args.num_workers=2
    args.distributed=False;args.world_size=1;args.enable_syntax_bias=False;args.enable_ptp_aux=False
    import shutil;shutil.copy2(__file__,out/'confirm_refinement.py');shutil.copy2(root/'train_unseen_refinement.py',out/'train_unseen_refinement.py')
    def run(args,model,unused,device):
        from datasets import build_dataset
        from utils.misc import collate_fn
        from torch.utils.data import DataLoader
        model.load_state_dict(torch.load(args.eval_model,map_location='cpu')['model'],strict=True);model.eval().requires_grad_(False)
        ds=build_dataset('val',args);rows=ds.images
        explored=json.loads((root/'manifest.json').read_text())['val'];exclude={r['image'] for r in explored}
        eligible=[i for i,r in enumerate(rows) if str(r[0]) not in exclude]
        ids=random.Random(20261130).sample(eligible,min(a.samples,len(eligible)))
        ds.images=[rows[i] for i in ids];manifest=[dict(source_index=i,image=str(rows[i][0]),phrase=str(rows[i][3])) for i in ids]
        assert not exclude&{r['image'] for r in manifest}
        protocol=dict(samples=len(manifest),seeds=[42,43],selection='Frozen final256step candidate and controls, no hyperparameter adjustment after enlarged evaluation',
            scope='Disjoint images from current256 exploratory subset; official val previously used for baseline checkpoint selection and possibly older experiments, not an independent test',
            training='Candidate and controls trained on1536 previously-unused official-training descriptions; native_calibration additionally sees512 calibration labels')
        (out/'manifest.json').write_text(json.dumps(manifest));(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        image_dir=root/'assets/image_data/other/images/mscoco/images/train2014';wanted={r['image'] for r in manifest}
        with zipfile.ZipFile(root/'images.zip') as z:
            members={Path(n).name:n for n in z.namelist() if n.endswith('.jpg')}
            for name in sorted(wanted):
                if not (image_dir/name).exists():(image_dir/name).write_bytes(z.read(members[name]))
        print('ASSETS_READY',len(wanted),time.monotonic()-start,flush=True)
        heads={};orig=original_heads(torch.load(root/'unseen04/original_heads.pt',map_location='cpu'))
        for seed in (42,43):
            for kind in ('native','native_calibration','global','roi'):
                head=LayerwiseHead(orig,'layer12') if kind.startswith('native') else Correction(kind)
                head.load_state_dict(torch.load(root/'refinement04'/f'{kind}_{seed}.pt',map_location='cpu')['state'],strict=True)
                heads[f'{kind}_{seed}']=head.to(device).eval().requires_grad_(False)
        chunks={k:[] for k in ['original',*heads]};target=[];captured={}
        def hook(m,ins,result):captured['x']=model.beit3.encoder.layer_norm(result[0].detach())[:,:578]
        handle=model.beit3.encoder.layers[11].register_forward_hook(hook)
        try:
            for step,batch in enumerate(DataLoader(ds,batch_size=16,shuffle=False,num_workers=2,collate_fn=collate_fn)):
                if time.monotonic()-start>a.seconds:raise TimeoutError('Confirmation budget exhausted')
                images,text,gt=batch[:3];images=images.to(device)
                with torch.no_grad():
                    base=model(images.tensors,images.mask,text)[0];x=captured.pop('x')
                    vision=F.normalize(model.vision_head(x[:,:577]),dim=-1)[:,1:];lang=F.normalize(model.language_head(x[:,577]),dim=-1)
                    score=(vision*lang[:,None]).sum(-1);pooled=(score.softmax(-1)[:,:,None]*model.visu_proj(x[:,:577])[:,1:]).sum(1)
                    roi=roi_samples(x,base);chunks['original'].append(base.cpu())
                    for key,h in heads.items():
                        if key.startswith('native'):pred=h(x[:,None].expand(-1,3,-1,-1))[0]
                        else:pred=h(roi if key.startswith('roi') else pooled,base)[0]
                        chunks[key].append(pred.cpu())
                target.append(gt)
                if step%32==0:print('EVAL',(step+1)*16,time.monotonic()-start,flush=True)
        finally:handle.remove()
        target=torch.cat(target);pred={k:torch.cat(v) for k,v in chunks.items()};ious={k:iou_giou(v,target,evaluation=True)[0] for k,v in pred.items()}
        base=ious['original']>=.5;results={}
        for k,iou in ious.items():
            hit=iou>=.5;results[k]=dict(acc50=float(hit.float().mean()*100),acc75=float((iou>=.75).float().mean()*100),miou=float(iou.mean()*100),
                corrected=int((hit&~base).sum()),regressed=int((~hit&base).sum()),delta_pp=float(hit.float().mean()*100-base.float().mean()*100))
        torch.save(dict(pred=pred,iou=ious,target=target),out/'predictions.pt')
        report=dict(status='completed',protocol=protocol,results=results,seconds=time.monotonic()-start)
        (out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
        return results['original']['acc50']/100
    ev.evaluate=run;ev.main(args)

if __name__=='__main__':main()

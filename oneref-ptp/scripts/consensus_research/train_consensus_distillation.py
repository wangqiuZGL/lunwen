"""Single-forward student: native vs unfiltered vs improvement-gated teacher KD."""
import argparse,json,sys,time,random,shutil
from pathlib import Path
from types import SimpleNamespace
import torch
from train_unseen_refinement import original_heads
from layerwise_fusion import LayerwiseHead
from run_spatial_readout_probe import iou_giou

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='/hy-tmp/layerwise-20261127');p.add_argument('--output',default='distill15');p.add_argument('--anchored',action='store_true');a=p.parse_args()
 root=Path(a.root);out=root/a.output;out.mkdir(exist_ok=False);start=time.monotonic();torch.set_num_threads(4)
 sys.path.insert(0,str(root/'code'));from utils.loss_utils import one_ref_loss
 args=SimpleNamespace(**json.loads((root/'probe_config.json').read_text()))
 manifest=json.loads((root/'unseen04/manifest.json').read_text())
 assert manifest['calibration']==json.loads((root/'verify_calibration12/manifest.json').read_text())
 assert not {r['image'] for r in manifest['calibration']}&{r['image'] for r in manifest['val']}
 teacher=torch.load(root/'verify_calibration12/predictions.pt',map_location='cpu')
 caches={s:{k:v.cuda() for k,v in torch.load(root/'unseen04'/f'{s}_cache.pt',map_location='cpu').items()} for s in ('calibration','val')}
 train=caches['calibration'];val=caches['val'];tp=teacher['pred']['verified_consensus_0.02'].cuda()
 assert float((teacher['pred']['original'].cuda()-train['pred']).abs().max())<1e-5
 assert torch.equal(teacher['target'].cuda(),train['target'])
 tiou=iou_giou(tp,train['target'],evaluation=True)[0];biou=iou_giou(train['pred'],train['target'],evaluation=True)[0]
 gate=((tiou-biou)>.01)&(tiou>=.5)
 protocol=dict(train=512,val=256,seeds=[42,43],arms=['native','all_teacher','quality_gated'],steps=256,batch=32,lr=1e-5,kd_weight=2.,margin=.01,min_teacher_iou=.5,
  teacher='Frozen verified_consensus_0.02, original+flip two-view inference on official training only',
  student='Frozen backbone cached final features, train original grounding heads only; single original-view inference',
  quality_rule='Train-GT relative IoU gain>0.01 and teacherIoU>=0.5; no gate/GT at student inference',
  selection='Fixed final256step, all arms and both seeds reported; exploratory val only, no testA/testB tuning',
  scope='Small mechanism pilot, not full end-to-end training, no novelty or improvement claim before results',accepted=int(gate.sum()))
 if a.anchored:protocol.update(arms=['anchored_quality'],quality_rule='Improvement gate selects verified teacher; rejected samples distill original frozen prediction. Original native/all_teacher controls from distill15 use identical batches, init and optimizer.')
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'manifest.json').write_text(json.dumps(dict(train=manifest['calibration'],val=manifest['val'])))
 shutil.copy2(__file__,out/'train_consensus_distillation.py')
 original=original_heads(torch.load(root/'unseen04/original_heads.pt',map_location='cpu'))
 def evaluate(h):
  predictions=[]
  with torch.no_grad():
   for i in range(0,len(val['target']),32):predictions.append(h(val['features'][i:i+32].float()[:,None].expand(-1,3,-1,-1))[0])
  pred=torch.cat(predictions);iou=iou_giou(pred,val['target'],evaluation=True)[0]
  return dict(acc50=float((iou>=.5).float().mean()*100),acc75=float((iou>=.75).float().mean()*100),miou=float(iou.mean()*100)),pred,iou
 baseline_head=LayerwiseHead(original,'layer12').cuda();baseline,_,_=evaluate(baseline_head);del baseline_head
 results={}
 for seed in protocol['seeds']:
  generator=torch.Generator().manual_seed(seed);order=torch.cat([torch.randperm(512,generator=generator) for _ in range(17)])[:8192].reshape(256,32).cuda()
  for arm in protocol['arms']:
   torch.manual_seed(seed);head=LayerwiseHead(original,'layer12').cuda();opt=torch.optim.AdamW([v for v in head.parameters() if v.requires_grad],lr=1e-5,weight_decay=1e-4);trace=[]
   for step,ids in enumerate(order,1):
    if time.monotonic()-start>600:raise TimeoutError('Pilot600second budget exhausted')
    pred,scores,_=head(train['features'][ids].float()[:,None].expand(-1,3,-1,-1))
    parts=one_ref_loss(args,pred,train['target'][ids],train['mask'][ids],None,visu_sim=scores)
    native=sum(v for k,v in parts.items() if k.startswith('loss_'))
    reference=torch.where(gate[ids,None],tp[ids],train['pred'][ids]) if arm=='anchored_quality' else tp[ids]
    kd=(pred-reference).abs().sum(-1)+(1-iou_giou(pred,reference)[1])
    if arm=='quality_gated':kd=kd*gate[ids].float()
    loss=native+(2*kd.mean() if arm!='native' else 0)
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
    opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_([v for v in head.parameters() if v.requires_grad],1.)
    if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
    opt.step()
    if step in (1,128,256):
     row=dict(step=step,native=float(native),kd=float(kd.mean()),loss=float(loss));trace.append(row);print(arm,seed,json.dumps(row),flush=True)
   metrics,pred,iou=evaluate(head);key=f'{arm}_{seed}';results[key]=dict(final=metrics,steps=256,trace=trace)
   torch.save(dict(state=head.cpu().state_dict(),pred=pred.cpu(),iou=iou.cpu()),out/f'{key}.pt');print('RESULT',key,json.dumps(metrics),flush=True)
   (out/'results.json').write_text(json.dumps(dict(status='in_progress',protocol=protocol,baseline=baseline,results=results),indent=2));del head,opt
 (out/'results.json').write_text(json.dumps(dict(status='completed',protocol=protocol,baseline=baseline,results=results,seconds=time.monotonic()-start),indent=2));print('COMPLETED',flush=True)

if __name__=='__main__':main()

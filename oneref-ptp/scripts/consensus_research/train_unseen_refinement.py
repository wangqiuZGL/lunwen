"""New-data controls and bounded ROI corrections; quality labels only in training."""
import argparse,copy,json,sys,time,math
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from layerwise_fusion import LayerwiseHead
from run_spatial_readout_probe import iou_giou

class MLP(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList([nn.Linear(768,768),nn.Linear(768,768),nn.Linear(768,4)])
    def forward(self,x):
        for i,l in enumerate(self.layers):x=F.relu(l(x)) if i<2 else l(x)
        return x

def original_heads(state):
    model=SimpleNamespace(vision_head=nn.Linear(768,768,bias=False),language_head=nn.Linear(768,768,bias=False),visu_proj=nn.Linear(768,768),bbox_embed=MLP())
    for k,v in state.items():getattr(model,k).load_state_dict(v,strict=True)
    return model

class Correction(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind;self.project=nn.Linear(768,48)
        width=436 if kind=='roi' else 52;hidden=128 if kind=='roi' else 1024
        self.hidden=nn.Linear(width,hidden);self.final=nn.Linear(hidden,4)
        nn.init.zeros_(self.final.weight);nn.init.zeros_(self.final.bias)
    def forward(self,x,base):
        x=F.relu(self.project(x));x=x.flatten(1)
        h=F.relu(self.hidden(torch.cat((x,base),-1)))
        delta=.5*self.final(h).tanh()
        pred=(torch.logit(base.clamp(1e-6,1-1e-6))+delta).sigmoid()
        return pred,h

def roi_samples(features,box):
    axis=torch.linspace(-.5,.5,3,device=features.device)
    yy,xx=torch.meshgrid(axis,axis,indexing='ij');offset=torch.stack((xx,yy),-1)
    coords=box[:,None,None,:2]+offset[None]*box[:,None,None,2:]*1.25
    grid=features[:,1:577].transpose(1,2).reshape(-1,768,24,24).float()
    return F.grid_sample(grid,coords*2-1,align_corners=False,padding_mode='zeros').flatten(2).transpose(1,2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/hy-tmp/layerwise-20261127');p.add_argument('--seconds',type=int,default=600)
    a=p.parse_args();root=Path(a.root);source=root/'unseen04';out=root/'refinement04';out.mkdir(exist_ok=False);start=time.monotonic();torch.set_num_threads(4)
    sys.path.insert(0,str(root/'code'));from utils.loss_utils import one_ref_loss
    args=SimpleNamespace(**json.loads((root/'probe_config.json').read_text()))
    protocol=dict(seeds=[42,43],steps=256,batch=32,native_head_lr=3e-5,correction_lr=3e-4,quality_steps=128,quality_lr=1e-3,
        correction_bound='0.5*tanh(delta) in original box-logit coordinates',quality_positive_margin=.01,quality_threshold=.5,
        train=1536,quality_calibration=512,val=256,selection='Fixed final256step; quality threshold fixed0.5, no best-val selection',
        controls='All methods receive the same1536 new official-training descriptions. Quality heads use512 image-disjoint training descriptions for128additional updates. Native_calibration also receives128extra native-loss updates on those512 to control extra supervision; compute and trainable capacity still differ.',
        supervision='Native REC loss for head adaptation; native L1+GIoU for frozen-head box corrections (mask terms constant). No test labels.',
        scope='Exploratory reused256val; new training data is not an architectural contribution; no novelty claim')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    import shutil;shutil.copy2(__file__,out/'train_unseen_refinement.py');shutil.copy2(source/'manifest.json',out/'manifest.json')
    def check():
        if time.monotonic()-start>a.seconds:raise TimeoutError('Refinement budget exhausted')
    caches={k:{n:v.cuda() for n,v in torch.load(source/f'{k}_cache.pt',map_location='cpu').items()} for k in ('train','calibration','val')}
    original=original_heads(torch.load(source/'original_heads.pt',map_location='cpu'));basehead=LayerwiseHead(original,'layer12').cuda().requires_grad_(False)
    prepared={}
    with torch.no_grad():
        for split,c in caches.items():
            chunks=[];rois=[]
            for i in range(0,len(c['target']),32):
                check();x=c['features'][i:i+32].float();box=c['pred'][i:i+32]
                vision=F.normalize(basehead.vision_head(x[:,:577]),dim=-1)[:,1:]
                lang=F.normalize(basehead.language_head(x[:,577]),dim=-1)
                scores=(vision*lang[:,None]).sum(-1)
                pooled=(scores.softmax(-1)[:,:,None]*basehead.visu_proj(x[:,:577])[:,1:]).sum(1)
                chunks.append(pooled);rois.append(roi_samples(x,box))
                replay=basehead.bbox_embed(pooled).sigmoid()
                if float((replay-box).abs().max())>.002:raise RuntimeError('Baseline head replay failed')
            prepared[split]=dict(global_x=torch.cat(chunks),roi_x=torch.cat(rois))
    del basehead
    def metrics(pred,split='val'):
        iou,_=iou_giou(pred,caches[split]['target'],evaluation=True)
        base,_=iou_giou(caches[split]['pred'],caches[split]['target'],evaluation=True)
        hit=iou>=.5;before=base>=.5
        return dict(acc50=float(hit.float().mean()*100),acc75=float((iou>=.75).float().mean()*100),miou=float(iou.mean()*100),
            corrected=int((hit&~before).sum()),regressed=int((~hit&before).sum())),iou
    baseline={s:metrics(c['pred'],s)[0] for s,c in caches.items()};results={}
    def evaluate(head,kind,split):
        c=caches[split];pred=[];hidden=[]
        with torch.no_grad():
            for i in range(0,len(c['target']),32):
                check()
                if kind=='native':p,_,_=head(c['features'][i:i+32].float()[:,None].expand(-1,3,-1,-1));h=None
                else:p,h=head(prepared[split][kind+'_x'][i:i+32],c['pred'][i:i+32])
                pred.append(p)
                if h is not None:hidden.append(h)
        return torch.cat(pred),torch.cat(hidden) if hidden else None
    for seed in protocol['seeds']:
        gen=torch.Generator().manual_seed(seed);n=len(caches['train']['target'])
        order=torch.cat([torch.randperm(n,generator=gen) for _ in range(256*32//n+1)])[:256*32].reshape(256,32).cuda()
        for kind in ('native','global','roi'):
            check();torch.manual_seed(seed)
            head=LayerwiseHead(original,'layer12').cuda() if kind=='native' else Correction(kind).cuda()
            initial,_=evaluate(head,kind,'val')
            if float((initial-caches['val']['pred']).abs().max())>.002:raise RuntimeError('Initial anchor failed')
            opt=torch.optim.AdamW([v for v in head.parameters() if v.requires_grad],lr=3e-5 if kind=='native' else 3e-4,weight_decay=1e-4)
            for step,ids in enumerate(order,1):
                check();c=caches['train'];opt.zero_grad()
                if kind=='native':
                    pred,scores,_=head(c['features'][ids].float()[:,None].expand(-1,3,-1,-1))
                    parts=one_ref_loss(args,pred,c['target'][ids],c['mask'][ids],None,visu_sim=scores)
                    loss=sum(v for k,v in parts.items() if k.startswith('loss_'))
                else:
                    pred,_=head(prepared['train'][kind+'_x'][ids],c['pred'][ids])
                    loss=(pred-c['target'][ids]).abs().sum(-1).mean()+(1-iou_giou(pred,c['target'][ids])[1]).mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_([v for v in head.parameters() if v.requires_grad],1.);opt.step()
            pred,h=evaluate(head,kind,'val');m,iou=metrics(pred);key=f'{kind}_{seed}'
            tr,_=evaluate(head,kind,'train');results[key]=dict(final=m,train=metrics(tr,'train')[0],parameters=sum(v.numel() for v in head.parameters() if v.requires_grad),steps=256)
            torch.save(dict(state=head.cpu().state_dict(),pred=pred.cpu(),iou=iou.cpu()),out/f'{key}.pt');head.cuda()
            print(key,json.dumps(results[key]),flush=True)
            if kind=='native':
                for extra_step in range(128):
                    check();ids=torch.randint(len(caches['calibration']['target']),(32,),generator=gen).cuda();c=caches['calibration']
                    pred2,scores,_=head(c['features'][ids].float()[:,None].expand(-1,3,-1,-1))
                    parts=one_ref_loss(args,pred2,c['target'][ids],c['mask'][ids],None,visu_sim=scores)
                    loss2=sum(v for k,v in parts.items() if k.startswith('loss_'))
                    if not torch.isfinite(loss2):raise RuntimeError('Nonfinite additional control loss')
                    opt.zero_grad();loss2.backward();torch.nn.utils.clip_grad_norm_([v for v in head.parameters() if v.requires_grad],1.);opt.step()
                p2,_=evaluate(head,kind,'val');m2,i2=metrics(p2);key2=f'native_calibration_{seed}'
                results[key2]=dict(final=m2,steps=384,note='256new-train updates plus128extra calibration-label updates')
                torch.save(dict(state=head.cpu().state_dict(),pred=p2.cpu(),iou=i2.cpu()),out/f'{key2}.pt');head.cuda()
                print(key2,json.dumps(results[key2]),flush=True)
            if kind!='native':
                head.requires_grad_(False);cp,ch=evaluate(head,kind,'calibration')
                ciou,_=iou_giou(cp,caches['calibration']['target'],evaluation=True)
                biou,_=iou_giou(caches['calibration']['pred'],caches['calibration']['target'],evaluation=True)
                labels=(ciou-biou>.01).float()
                qx=torch.cat((ch,cp,caches['calibration']['pred']),-1).detach()
                vx=torch.cat((h,pred,caches['val']['pred']),-1).detach()
                torch.manual_seed(seed+1000);quality=nn.Sequential(nn.Linear(qx.shape[-1],64),nn.ReLU(),nn.Linear(64,1)).cuda()
                qopt=torch.optim.AdamW(quality.parameters(),lr=.001,weight_decay=1e-4)
                qgen=torch.Generator().manual_seed(seed+1000)
                for step in range(128):
                    check();ids=torch.randint(len(qx),(32,),generator=qgen).cuda()
                    qloss=F.binary_cross_entropy_with_logits(quality(qx[ids]).squeeze(-1),labels[ids])
                    qopt.zero_grad();qloss.backward();qopt.step()
                with torch.no_grad():confidence=quality(vx).sigmoid().squeeze(-1);accept=confidence>.5;gated=torch.where(accept[:,None],pred,caches['val']['pred'])
                gm,giou=metrics(gated);key=f'{kind}_quality_{seed}';true=(iou-iou_giou(caches['val']['pred'],caches['val']['target'],evaluation=True)[0]>.01)
                pos=confidence[true];neg=confidence[~true]
                auc=float(((pos[:,None]>neg[None]).float()+.5*(pos[:,None]==neg[None]).float()).mean()) if len(pos) and len(neg) else None
                results[key]=dict(final=gm,accepted=int(accept.sum()),calibration_positive_rate=float(labels.mean()),quality_auc=auc,
                    oracle_correctable=int(((iou>=.5)&(iou_giou(caches['val']['pred'],caches['val']['target'],evaluation=True)[0]<.5)).sum()),oracle_note='Post-hoc diagnostic upper bound, never inference selection')
                torch.save(dict(quality=quality.cpu().state_dict(),candidate=f'{kind}_{seed}',pred=gated.cpu(),iou=giou.cpu(),confidence=confidence.cpu()),out/f'{key}.pt')
                print(key,json.dumps(results[key]),flush=True);del quality,qopt
            (out/'results.json').write_text(json.dumps(dict(status='in_progress',protocol=protocol,baseline=baseline,results=results,seconds=time.monotonic()-start),indent=2));del head,opt
    (out/'results.json').write_text(json.dumps(dict(status='completed',protocol=protocol,baseline=baseline,results=results,seconds=time.monotonic()-start),indent=2));print('COMPLETED',flush=True)

if __name__=='__main__':main()

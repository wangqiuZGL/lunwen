"""Inference-only clipping to image content bounds, using image dimensions only."""
from pathlib import Path
import json
import torch
from PIL import Image
from run_spatial_readout_probe import iou_giou,corners

root=Path('/hy-tmp/layerwise-20261127');out=root/'valid_projection08';out.mkdir(exist_ok=False)
reports={}
for folder in ('scale03','confirm05'):
    manifest=json.loads((root/folder/'manifest.json').read_text())
    cache=torch.load(root/folder/'predictions.pt',map_location='cpu');p=cache['pred']['original'];t=cache['target']
    bounds=[]
    for r in manifest:
        with Image.open(root/'assets/image_data/other/images/mscoco/images/train2014'/r['image']) as im:w,h=im.size
        ratio=384/max(w,h);nw,nh=round(w*ratio),round(h*ratio);left,top=round((384-nw)/2-.1),round((384-nh)/2-.1)
        bounds.append([left/384,top/384,(left+nw)/384,(top+nh)/384])
    bounds=torch.tensor(bounds);pc=corners(p);clipped=torch.maximum(torch.minimum(pc,bounds[:,[2,3,2,3]]),bounds[:,[0,1,0,1]])
    result=torch.cat(((clipped[:,:2]+clipped[:,2:])/2,clipped[:,2:]-clipped[:,:2]),-1)
    before,_=iou_giou(p,t,evaluation=True);after,_=iou_giou(result,t,evaluation=True)
    b=before>=.5;a=after>=.5;gt=corners(t)
    report=dict(samples=len(p),original_acc50=float(b.float().mean()*100),projected_acc50=float(a.float().mean()*100),
        original_miou=float(before.mean()*100),projected_miou=float(after.mean()*100),corrected=int((a&~b).sum()),regressed=int((~a&b).sum()),
        touched=int((pc-clipped).abs().amax(-1).gt(1e-6).sum()),true_boxes_outside_image=int(((gt[:,:2]<bounds[:,:2]-1e-6).any(-1)|(gt[:,2:]>bounds[:,2:]+1e-6).any(-1)).sum()),
        iou_decreased=int((after<before-1e-6).sum()),corrected_indices=(a&~b).nonzero().flatten().tolist())
    reports[folder]=report;torch.save(dict(pred=result,iou=after,original_iou=before,bounds=bounds),out/f'{folder}.pt')
(out/'results.json').write_text(json.dumps(dict(rule='Clip predicted box to resized-image content rectangle determined only by input dimensions; no GT-dependent inference',results=reports),indent=2))
print(json.dumps(reports))

"""Render all exploratory baseline errors; geometric proxies are not object labels."""
from pathlib import Path
import json,math,textwrap
import torch
from PIL import Image,ImageDraw,ImageFont
from run_spatial_readout_probe import iou_giou,corners

root=Path('/hy-tmp/layerwise-20261127');out=root/'failure_audit';out.mkdir(exist_ok=True)
torch.set_num_threads(4)
c=torch.load(root/'screen01/val_cache.pt',map_location='cpu')
manifest=json.loads((root/'manifest.json').read_text())['val']
p=c['pred'];t=c['target'];iou,_=iou_giou(p,t,evaluation=True);pc=corners(p);tc=corners(t)
failed=(iou<.5).nonzero().flatten().tolist()
try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
except OSError:font=ImageFont.load_default()
rows=[]
for i in failed:
    center_inside=bool(((p[i,:2]>=tc[i,:2]) & (p[i,:2]<=tc[i,2:])).all())
    rows.append(dict(index=i,**manifest[i],iou=float(iou[i]),pred=p[i].tolist(),target=t[i].tolist(),
        pred_center_inside_gt=center_inside,area_ratio=float(p[i,2:].prod()/t[i,2:].prod()),
        normalized_center_error=float(((p[i,:2]-t[i,:2])/t[i,2:].clamp_min(.01)).norm()),
        note='Geometric diagnostic only; center overlap does not establish correct object identity'))
for page in range(math.ceil(len(rows)/16)):
    sheet=Image.new('RGB',(384*4,456*4),'white')
    for k,r in enumerate(rows[page*16:(page+1)*16]):
        img=Image.open(root/'assets/image_data/other/images/mscoco/images/train2014'/r['image']).convert('RGB')
        ratio=384/max(img.size);w,h=[round(v*ratio) for v in img.size]
        left=round((384-w)/2-.1);top=round((384-h)/2-.1)
        canvas=Image.new('RGB',(384,456),(124,116,104));canvas.paste(img.resize((w,h),Image.Resampling.BILINEAR),(left,top))
        draw=ImageDraw.Draw(canvas)
        for b,col in ((tc[r['index']],'lime'),(pc[r['index']],'red')):
            draw.rectangle([float(v*384) for v in b],outline=col,width=3)
        draw.rectangle((0,384,384,456),fill='white')
        title=f"#{r['index']} IoU={r['iou']:.3f}  green=GT red=pred"
        draw.text((4,387),title,fill='black',font=font)
        for j,line in enumerate(textwrap.wrap(r['phrase'],width=43)[:3]):draw.text((4,405+j*15),line,fill='black',font=font)
        sheet.paste(canvas,((k%4)*384,(k//4)*456))
    sheet.save(out/f'errors_{page+1}.jpg',quality=90)
summary=dict(total=len(p),correct=int((iou>=.5).sum()),failed=len(rows),
    zero_or_tiny_overlap=sum(r['iou']<.1 for r in rows),near_threshold=sum(.4<=r['iou']<.5 for r in rows),
    predicted_center_in_gt=sum(r['pred_center_inside_gt'] for r in rows),
    interpretation='No automatic object-choice vs localization ground truth. Review rendered images and describe ambiguous cases.')
(out/'cases.json').write_text(json.dumps(rows,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))

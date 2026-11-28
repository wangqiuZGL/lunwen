"""Paired image-cluster bootstrap: inference/selected checkpoint comparison."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch

def cluster_bootstrap(delta,images,draws=5000):
    names=sorted(set(images));lookup={s:i for i,s in enumerate(names)}
    sums=np.zeros(len(names));counts=np.zeros(len(names))
    for value,name in zip(delta,images):i=lookup[name];sums[i]+=float(value);counts[i]+=1
    rng=np.random.default_rng(20261202);values=[]
    for _ in range(0,draws,100):
        indices=rng.integers(0,len(names),size=(min(100,draws-len(values)),len(names)))
        values.extend((sums[indices].sum(1)/counts[indices].sum(1)*100).tolist())
    return dict(delta_pp=float(np.mean(delta)*100),ci95_pp=np.percentile(values,[2.5,97.5]).tolist(),clusters=len(names),draws=draws)

def main():
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--candidate',default='consensus_projection');a=p.parse_args();root=Path(a.folder)
    cache=torch.load(root/'predictions.pt',map_location='cpu');manifest=json.loads((root/'manifest.json').read_text())
    base=cache['iou']['original'].numpy();other=cache['iou'][a.candidate].numpy();b=base>=.5;c=other>=.5
    corrected=int((c&~b).sum());regressed=int((~c&b).sum());discordant=corrected+regressed
    # Exact two-sided McNemar/binomial test, paired by expression; bootstrap clusters by image.
    pvalue=min(1.,2*sum(math.comb(discordant,k) for k in range(min(corrected,regressed)+1))/(1<<discordant)) if discordant else 1.
    images=[r['image'] for r in manifest]
    result=dict(candidate=a.candidate,samples=len(base),corrected=corrected,regressed=regressed,expression_exact_mcnemar_p=pvalue,
        acc50=cluster_bootstrap(c.astype(float)-b.astype(float),images),miou=cluster_bootstrap(other-base,images),
        scope='Fixed-model paired image-cluster resampling; no training-seed uncertainty or multiple-search correction. Val estimates remain exploratory after method search.')
    (root/'paired_statistics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()

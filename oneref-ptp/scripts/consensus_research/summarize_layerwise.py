import json
from pathlib import Path
import numpy as np
import torch

root=Path('/hy-tmp/layerwise-20261127');out=root/'screen01'
r=json.loads((out/'results.json').read_text());arms=r['protocol']['arms'];seeds=r['protocol']['seeds']
if r['status']!='completed' or len(r['results'])!=len(arms)*len(seeds):raise RuntimeError('Incomplete experiment')
saved={k:torch.load(out/(k+'.pt'),map_location='cpu') for k in r['results']}
r['means']={arm:{m:float(np.mean([r['results'][arm+'_'+str(s)][m] for s in seeds])) for m in ('acc50','acc75','miou')} for arm in arms}
r['layer_complementarity']={}
for seed in seeds:
    h={arm:saved[arm+'_'+str(seed)]['iou']>=.5 for arm in ('layer6','layer9','layer12')}
    oracle=h['layer6']|h['layer9']|h['layer12']
    r['layer_complementarity'][str(seed)]=dict(layer12_hits=int(h['layer12'].sum()),oracle_hits=int(oracle.sum()),
        extra_oracle_hits=int((oracle&~h['layer12']).sum()),scope='Post-hoc correct-answer union; unreachable oracle upper bound, not a deployable gain.')
r['separate_comparisons']={}
for control in [a for a in arms if a!='separate']:
    counts=[];deltas=[]
    for seed in seeds:
        a=saved[control+'_'+str(seed)]['iou']>=.5;b=saved['separate_'+str(seed)]['iou']>=.5
        deltas.append(float((b.float()-a.float()).mean()*100));counts.append(dict(seed=seed,corrected=int((~a&b).sum()),regressed=int((a&~b).sum())))
    r['separate_comparisons'][control]=dict(seed_delta_pp=deltas,mean_delta_pp=float(np.mean(deltas)),counts=counts)
passed=all(all(d>0 for d in v['seed_delta_pp']) and v['mean_delta_pp']>=.3 for v in r['separate_comparisons'].values())
passed=passed and r['means']['separate']['acc50']>r['original_acc50']
r['decision']=dict(passes_spend_gate=passed,scope='256 exploratory validation samples and adaptation-order seeds only; no statistical or independent-test claim.',
    action='Expand confirmation only' if passed else 'No automatic scaling; inspect strongest control and stop this configuration')
(out/'summary.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:r[k] for k in ('status','seconds','original_acc50','means','layer_complementarity','separate_comparisons','decision')},indent=2))

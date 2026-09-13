"""All-seed, all-view comparison for a fixed new-scene validation."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--backup',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.backup;out=a.out;out.mkdir(parents=True,exist_ok=True)
j=json.loads((r/'fern_runs/compare/results.json').read_text());assert j['state']=='complete' and len(j['runs'])==6
rows=j['runs'];summary={}
for mode in ['off','ordinary']:
 rr=sorted([x for x in rows if x['mode']==mode],key=lambda x:x['seed']);assert [x['seed'] for x in rr]==[0,1,2]
 summary[mode]={q:{k:{'mean':float(np.mean([x['metrics'][q][k] for x in rr])),'std':float(np.std([x['metrics'][q][k] for x in rr],ddof=1))} for k in ['psnr','ssim','lpips_vgg_standard_m11']} for q in ['1.0','0.85']}
 summary[mode].update(mean_training_seconds=float(np.mean([x['training_seconds'] for x in rr])),max_allocated_mb=max(x['memory']['peak_allocated_mb'] for x in rr),max_reserved_mb=max(x['memory']['peak_reserved_mb'] for x in rr))
paired=[];viewpairs=[];exposure=[]
for seed in range(3):
 off=next(x for x in rows if x['seed']==seed and x['mode']=='off');ordinary=next(x for x in rows if x['seed']==seed and x['mode']=='ordinary')
 paired.append(dict(seed=seed,**{k:ordinary['metrics']['1.0'][k]-off['metrics']['1.0'][k] for k in ['psnr','ssim','lpips_vgg_standard_m11']}))
 logs=[json.loads(s) for s in (r/f'fern_runs/compare/ordinary_seed{seed}/rank_stats.jsonl').read_text().splitlines()];assert len(logs)==9000
 exposure.append(dict(seed=seed,selected=sum(x['selected'] for x in logs),active=sum(x['active'] for x in logs),zero_pair_steps=sum(x['selected']==0 for x in logs),mean_pool_agreement=float(np.mean([x['agreed']/x['pool_valid'] for x in logs]))))
 metas={m:json.loads((r/f'fern_runs/compare/{m}_seed{seed}/evaluation/metrics.json').read_text()) for m in ['off','ordinary']}
 assert len(metas['off']['per_view'])==len(metas['ordinary']['per_view'])==3
 for b,c in zip(metas['off']['per_view'],metas['ordinary']['per_view']):
  assert b['image']==c['image'];viewpairs.append(dict(seed=seed,image=b['image'],lpips_delta=c['lpips_vgg_standard_m11']-b['lpips_vgg_standard_m11']))
base=summary['off']['1.0']['lpips_vgg_standard_m11']['mean'];cand=summary['ordinary']['1.0']['lpips_vgg_standard_m11']['mean'];improvement=100*(1-cand/base)
result=dict(state='complete',scene='fern',summary=summary,paired=paired,per_view=viewpairs,exposure=exposure,relative_lpips_improvement_percent=improvement,seed_wins=sum(x['lpips_vgg_standard_m11']<0 for x in paired),view_wins=sum(x['lpips_delta']<0 for x in viewpairs),runs=rows)
(out/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['runs','per_view']},indent=2))

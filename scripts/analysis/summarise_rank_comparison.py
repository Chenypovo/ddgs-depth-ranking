"""Summarise every registered run; no selection or omitted negative results."""
import argparse,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
j=json.loads(args.results.read_text());assert j['state']=='complete' and len(j['runs'])==9
rows=j['runs'];out=args.out;out.mkdir(parents=True,exist_ok=True)
summary={}
for mode in ['off','ordinary','consensus']:
 rr=sorted([r for r in rows if r['mode']==mode],key=lambda r:r['seed']);assert [r['seed'] for r in rr]==[0,1,2]
 stat={}
 for q in ['1.0','0.85']:
  stat[q]={k:dict(mean=float(np.mean([r['metrics'][q][k] for r in rr])),std=float(np.std([r['metrics'][q][k] for r in rr],ddof=1))) for k in ['psnr','ssim','lpips_vgg_standard_m11']}
 stat['mean_training_seconds']=float(np.mean([r['training_seconds'] for r in rr]));stat['max_peak_allocated_mb']=max(r['memory']['peak_allocated_mb'] for r in rr)
 summary[mode]=stat
pairwise={}
for other in ['off','ordinary']:
 pairs=[]
 for s in range(3):
  a=next(r for r in rows if r['mode']==other and r['seed']==s);c=next(r for r in rows if r['mode']=='consensus' and r['seed']==s)
  pairs.append(dict(seed=s,lpips_delta=c['metrics']['1.0']['lpips_vgg_standard_m11']-a['metrics']['1.0']['lpips_vgg_standard_m11'],psnr_delta=c['metrics']['1.0']['psnr']-a['metrics']['1.0']['psnr'],ssim_delta=c['metrics']['1.0']['ssim']-a['metrics']['1.0']['ssim']))
 base=summary[other]['1.0']['lpips_vgg_standard_m11']['mean'];candidate=summary['consensus']['1.0']['lpips_vgg_standard_m11']['mean']
 pairwise[other]=dict(pairs=pairs,lpips_relative_improvement_percent=100*(1-candidate/base),wins=sum(r['lpips_delta']<0 for r in pairs))
result=dict(state='complete',summary=summary,consensus_vs=pairwise,runs=rows,evidence='Development comparison only; generic depth ranking and confidence supervision are existing research, candidate novelty unconfirmed')
(out/'summary.json').write_text(json.dumps(result,indent=2))
lines=['# Ordinal supervision development comparison','','All three seeds; complete10000step models; lower standard VGG LPIPS is better.','']
for q in ['1.0','0.85']:
 lines.extend([f'## Inference opacity multiplier {q}','','|Mode|PSNR|SSIM|LPIPS|Train seconds|Peak allocated MiB|','|---|---:|---:|---:|---:|---:|'])
 for mode,stat in summary.items():
  m=stat[q];lines.append(f"|{mode}|{m['psnr']['mean']:.5f}|{m['ssim']['mean']:.6f}|{m['lpips_vgg_standard_m11']['mean']:.6f}|{stat['mean_training_seconds']:.2f}|{stat['max_peak_allocated_mb']:.1f}|")
 lines.append('')
for other,s in pairwise.items():lines.append(f"Consensus vs {other}: relative LPIPS improvement {s['lpips_relative_improvement_percent']:.3f}%, wins {s['wins']}/3.")
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

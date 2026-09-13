"""Summarise all planned correction runs without selecting favourable seeds."""
import argparse
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
data=json.loads(a.results.read_text());assert data['state']=='complete' and data['stage']=='compare'
runs=data['runs'];assert len(runs)==9
arms=[('upstream','upstream'),('camera_z','upstream'),('upstream','refresh')]
labels=['baseline','depth_only','density_only'];metrics=['psnr','ssim','lpips_vgg_standard_m11','lpips_vgg_repo_01']
summary={}
for label,(d,c) in zip(labels,arms):
 rows=sorted([r for r in runs if (r['depth'],r['density'])==(d,c)],key=lambda r:r['seed'])
 assert [r['seed'] for r in rows]==[0,1,2]
 summary[label]={'mean':{m:statistics.mean(r['metrics'][m] for r in rows) for m in metrics},
                 'std':{m:statistics.stdev(r['metrics'][m] for r in rows) for m in metrics},
                 'training_seconds_mean':statistics.mean(r['training_seconds'] for r in rows),
                 'runs':rows}
for label in labels[1:]:
 summary[label]['paired_deltas']=[{m:r['metrics'][m]-b['metrics'][m] for m in metrics}
   for r,b in zip(summary[label]['runs'],summary['baseline']['runs'])]
 summary[label]['primary_relative_change_percent']=100*(summary[label]['mean']['lpips_vgg_standard_m11']/summary['baseline']['mean']['lpips_vgg_standard_m11']-1)
result={'evidence':'Single development scene, three paired seeds, engineering corrections only','arms':summary}
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# D²GS correction comparison','','All planned seeds reported. Lower LPIPS is better. No novelty claim.','',
       '| Arm | PSNR | SSIM | LPIPS standard | Mean train seconds |','|---|---:|---:|---:|---:|']
for label in labels:
 r=summary[label];m=r['mean'];lines.append(f"| {label} | {m['psnr']:.5f} | {m['ssim']:.6f} | {m['lpips_vgg_standard_m11']:.6f} | {r['training_seconds_mean']:.2f} |")
lines+=['','Paired deltas and all metrics are recorded in summary.json. Historical GPU timings are not directly comparable across machines.']
(a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))

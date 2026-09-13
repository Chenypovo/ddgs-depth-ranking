"""Bounded, paired engineering correction experiment; no hyperparameter search.

Run diagnose first and review its telemetry before launching compare.
An isolated DDGS_corrections source directory is required. Original runs are kept.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
 p=argparse.ArgumentParser()
 p.add_argument('--stage',choices=['diagnose','compare'],required=True)
 p.add_argument('--deadline',required=True,help='Absolute ISO time with timezone')
 p.add_argument('--dry-run',action='store_true')
 args=p.parse_args()
 dt=datetime.fromisoformat(args.deadline)
 if dt.tzinfo is None:raise ValueError('Timezone required')
 deadline=min(dt.timestamp(),time.time()+3600)
 if deadline<=time.time():raise ValueError('Deadline already passed')
 root=Path(__file__).resolve().parent
 source=root/'DDGS_corrections'
 out=root/'correction_runs'/args.stage
 modes=[('upstream','upstream'),('camera_z','upstream'),('upstream','refresh')]
 seeds=[0] if args.stage=='diagnose' else [0,1,2]
 steps=1100 if args.stage=='diagnose' else 10000
 plan=[dict(depth=d,density=c,seed=s,steps=steps) for s in seeds for d,c in modes]
 if args.dry_run:
  print(json.dumps(dict(stage=args.stage,plan=plan,max_seconds=max(0,int(deadline-time.time())),source=str(source)),indent=2));return
 assert source.is_dir() and source.resolve()!=(root/'DDGS').resolve()
 assert not out.exists(),'Existing cohort cannot be overwritten or silently resumed'
 py=root/'.venv/bin/python'
 subprocess.run([str(py),'-c','import torch; assert torch.cuda.is_available(), "GPU required"'],check=True)
 split=json.loads((root/'dataset/nerf_llff_data/trex/3_views/split.json').read_text())
 assert split['state']=='complete' and len(split['train_images'])==3 and len(split['test_images'])==7
 if args.stage=='compare':
  diagnostic=json.loads((root/'correction_runs/diagnose/results.json').read_text())
  assert diagnostic['state']=='complete','Diagnostics must complete first'
 out.mkdir(parents=True)
 (out/'plan.json').write_text(json.dumps(plan,indent=2))
 files=['train.py','gaussian_renderer/__init__.py','scene/gaussian_model.py','utils/dropout_controls.py','utils/general_utils.py']
 hashes={f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in files}
 hashes['evaluator']=hashlib.sha256((root/'evaluate_corrections.py').read_bytes()).hexdigest()
 (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
 rows=[]
 def status(state,**extra):
  tmp=out/'status.tmp';tmp.write_text(json.dumps(dict(state=state,time=datetime.now().astimezone().isoformat(),completed=rows,**extra),indent=2));tmp.replace(out/'status.json')
 def run(cmd,log,env):
  remaining=int(deadline-time.time())
  if remaining<=0:raise TimeoutError('Cohort deadline reached')
  with log.open('x') as f:
   subprocess.run(['timeout','--signal=TERM','--kill-after=15',str(remaining),*map(str,cmd)],cwd=source,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 try:
  for item in plan:
   label=f"depth_{item['depth']}_density_{item['density']}_seed{item['seed']}"
   model=out/label;model.mkdir()
   env=os.environ.copy();env.update(DDGS_SOURCE=str(source),DDGS_DEPTH_MODE=item['depth'],DDGS_DENSITY_MODE=item['density'],DDGS_DROPOUT='independent',DDGS_SEED=str(item['seed']),DDGS_DIAGNOSTICS=str(model/'diagnostics.jsonl'))
   cmd=[py,'train.py','-s',root/'dataset/nerf_llff_data/trex','-m',model,
        '--depth_weight','0.15','--density_weight','0.85','--drop_min','0.05','--drop_max','0.5',
        '--mask_param','5','--lambda_far','0.5','--eval','-r','8','--n_views','3',
        '--iterations',steps,'--test_iterations',steps,'--save_iterations',steps]
   (model/'identity.json').write_text(json.dumps(dict(**item,command=list(map(str,cmd)),environment={k:v for k,v in env.items() if k.startswith('DDGS_')}),indent=2))
   status('training',current=item);start=time.monotonic();run(cmd,model/'train.log',env)
   elapsed=time.monotonic()-start
   telemetry=[json.loads(s) for s in (model/'diagnostics.jsonl').read_text().splitlines()]
   assert len(telemetry)==steps
   summary={k:sum(bool(x[k]) for x in telemetry) for k in ['density_length_fallback','density_topology_stale','density_refreshed_before_render']}
   depth_rows=[x for x in telemetry if 'depth_band_disagreement_z' in x]
   summary['sampled_z_band_disagreement_mean']=sum(x['depth_band_disagreement_z'] for x in depth_rows)/len(depth_rows)
   if item['density']=='refresh':assert summary['density_length_fallback']==summary['density_topology_stale']==0
   row=dict(**item,model=str(model),training_seconds=elapsed,diagnostics=summary)
   if args.stage=='compare':
    status('evaluating',current=item)
    run([py,root/'evaluate_corrections.py','--model',model],model/'evaluation.log',env)
    row['metrics']=json.loads((model/'evaluation/metrics.json').read_text())['mean']
   rows.append(row)
   (out/'partial.json').write_text(json.dumps(rows,indent=2))
  (out/'results.json').write_text(json.dumps(dict(state='complete',stage=args.stage,runs=rows,primary='lpips_vgg_standard_m11',evidence='Engineering correction comparison, not research novelty'),indent=2))
  status('complete')
 except Exception as exc:
  status('failed',error=str(exc));raise


if __name__=='__main__':main()

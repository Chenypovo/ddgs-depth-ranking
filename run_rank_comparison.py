"""Bounded, no-retry ordinal supervision experiment with all three arms/seeds."""
import argparse,hashlib,json,os,subprocess,time
from datetime import datetime
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','compare'],required=True);p.add_argument('--max-seconds',type=int,default=2700);a=p.parse_args()
 root=Path(__file__).resolve().parent;source=root/'DDGS_rank';py=root/'.venv/bin/python'
 out=root/'rank_runs'/a.stage;assert not out.exists(),'Do not overwrite or selectively resume'
 assert json.loads((root/'rank_integration.json').read_text())['state']=='passed'
 if a.stage=='compare':assert json.loads((root/'rank_runs/pilot/results.json').read_text())['state']=='complete'
 assert hashlib.sha256((root/'dataset/nerf_llff_data/trex/3_views/dense/fused.ply').read_bytes()).hexdigest()=='1b1b9eb79101c0a3e56d07009be5ae1d0c1b7c0c9d99a320d7e0a2d43f4cfd4c'
 split=json.loads((root/'dataset/nerf_llff_data/trex/3_views/split.json').read_text());assert len(split['train_images'])==3 and len(split['test_images'])==7
 plan=[dict(seed=0,mode='consensus',steps=1200)] if a.stage=='pilot' else [dict(seed=s,mode=m,steps=10000) for s in range(3) for m in ['off','ordinary','consensus']]
 out.mkdir(parents=True);deadline=time.time()+min(a.max_seconds,2700);rows=[]
 hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [source/'train.py',source/'utils/rank_supervision.py',source/'gaussian_renderer/__init__.py',source/'scene/gaussian_model.py',root/'evaluate_rank.py',root/'run_rank_comparison.py',*[root/f'depth_reliability/view{i}.npz' for i in range(3)]]}
 (out/'plan.json').write_text(json.dumps(dict(plan=plan,source_hashes=hashes,deadline=datetime.fromtimestamp(deadline).astimezone().isoformat(),primary='lpips_vgg_standard_m11',development_scene=True),indent=2))
 def status(state,**kw):
  tmp=out/'status.tmp';tmp.write_text(json.dumps(dict(state=state,time=datetime.now().astimezone().isoformat(),completed=rows,**kw),indent=2));tmp.replace(out/'status.json')
 def run(cmd,log,env):
  seconds=int(deadline-time.time());assert seconds>0,'Budget exhausted'
  with log.open('x') as handle:subprocess.run(['timeout','--signal=TERM','--kill-after=15',str(seconds),*map(str,cmd)],cwd=source,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
 try:
  for item in plan:
   model=out/f"{item['mode']}_seed{item['seed']}";model.mkdir()
   env=os.environ.copy()
   for k in list(env):
    if k.startswith('DDGS_'):del env[k]
   env.update(DDGS_SOURCE=str(source),DDGS_SEED=str(item['seed']),DDGS_DROPOUT='independent',DDGS_DEPTH_MODE='upstream',DDGS_DENSITY_MODE='upstream',DDGS_RANK_MODE=item['mode'],DDGS_RANK_CACHE=str(root/'depth_reliability'))
   cmd=[py,'train.py','-s',root/'dataset/nerf_llff_data/trex','-m',model,'--depth_weight','0.15','--density_weight','0.85','--drop_min','0.05','--drop_max','0.5','--mask_param','5','--lambda_far','0.5','--eval','-r','8','--n_views','3','--iterations',item['steps'],'--test_iterations',item['steps'] if a.stage=='compare' else 0,'--save_iterations',item['steps']]
   (model/'identity.json').write_text(json.dumps(dict(**item,command=list(map(str,cmd)),environment={k:v for k,v in env.items() if k.startswith('DDGS_')}),indent=2))
   status('training',current=item);start=time.monotonic();run(cmd,model/'train.log',env)
   row=dict(**item,model=str(model),training_seconds=time.monotonic()-start,memory=json.loads((model/'memory.json').read_text()))
   if item['mode']!='off':
    rr=[json.loads(x) for x in (model/'rank_stats.jsonl').read_text().splitlines()];assert len(rr)==item['steps']-1000
    row['rank']=dict(logged_steps=len(rr),zero_pairs=sum(r['selected']==0 for r in rr),invalid_selected_pairs=sum(r['selected']-r['active'] for r in rr),mean_pairs=sum(r['selected'] for r in rr)/len(rr),mean_raw_loss=sum(r['loss'] for r in rr)/len(rr))
    assert row['rank']['zero_pairs']==0
   if a.stage=='compare':
    row['metrics']={}
    for q,folder in [(1.,'evaluation'),(.85,'evaluation_q085')]:
     status('evaluating',current=item,keep_probability=q)
     run([py,root/'evaluate_rank.py','--model',model,'--keep-probability',q],model/f'eval_{q}.log',env)
     row['metrics'][str(q)]=json.loads((model/folder/'metrics.json').read_text())['mean']
   rows.append(row);(out/'partial.json').write_text(json.dumps(rows,indent=2))
  if a.stage=='compare':
   for s in range(3):
    bb=[json.loads(x) for x in (out/f'ordinary_seed{s}/rank_stats.jsonl').read_text().splitlines()]
    cc=[json.loads(x) for x in (out/f'consensus_seed{s}/rank_stats.jsonl').read_text().splitlines()]
    assert [(r['iteration'],r['camera'],r['selected']) for r in bb]==[(r['iteration'],r['camera'],r['selected']) for r in cc],f'Count/camera mismatch seed{s}'
  (out/'results.json').write_text(json.dumps(dict(state='complete',stage=a.stage,runs=rows),indent=2));status('complete')
 except Exception as exc:
  status('failed',error=str(exc));raise

if __name__=='__main__':main()

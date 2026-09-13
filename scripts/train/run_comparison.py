from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
from pathlib import Path
import subprocess
import time

root=Path(__file__).resolve().parents[2]
py=str(root/'.venv/bin/python')
train_deadline=datetime(2026,9,14,0,10,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
eval_deadline=datetime(2026,9,14,1,10,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
def status(state,**kw):
    d={'state':state,'at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),**kw}
    temp=root/'comparison_status.tmp';temp.write_text(json.dumps(d,indent=2));temp.replace(root/'comparison_status.json')
    print(json.dumps(d),flush=True)
def run(cmd,log,env,deadline):
    remaining=int(deadline-time.time())
    if remaining<=0: raise TimeoutError('Hard deadline reached')
    with log.open('w') as f:
        result=subprocess.run(['timeout','--signal=TERM','--kill-after=30',str(remaining),*cmd],
            cwd=root/'DDGS',env=env,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'{log}: return code {result.returncode}')
rows=[]
baseline0=root/'runs/trex_baseline_seed0/evaluation/metrics.json'
base=json.loads(baseline0.read_text())
rows.append({'variant':'independent','seed':0,'metrics':base['mean'],'training_seconds':142.92663598060608,'model':str(baseline0.parents[1])})
try:
    assert json.loads((root/'sampling_checks.json').read_text())['state']=='integration_pass'
    assert 'passed' in (root/'candidate_smoke.log').read_text()
    for variant,seed in [('spatial_stratified',0),('independent',1),('spatial_stratified',1),('independent',2),('spatial_stratified',2)]:
        model=root/f'runs/trex_{variant}_seed{seed}'
        assert not model.exists(), f'Existing run cannot be overwritten: {model}'
        env=os.environ.copy();env.update(DDGS_DROPOUT=variant,DDGS_SEED=str(seed))
        status('training',variant=variant,seed=seed,completed=rows)
        model.mkdir(parents=True)
        tb=Path('/root/tf-logs')/f'ddgs_trex_{variant}_seed{seed}'
        if not tb.exists(): tb.symlink_to(model,target_is_directory=True)
        command=[py,'train.py','-s',str(root/'dataset/nerf_llff_data/trex'),'-m',str(model),
            '--depth_weight','0.15','--density_weight','0.85','--drop_min','0.05','--drop_max','0.5',
            '--mask_param','5','--lambda_far','0.5','--eval','-r','8','--n_views','3',
            '--iterations','10000','--test_iterations','5000','10000','--save_iterations','5000','10000',
            '--checkpoint_iterations','5000','10000']
        (model/'run_identity.json').write_text(json.dumps({'variant':variant,'seed':seed,'command':command},indent=2))
        t=time.time();run(command,model/'train.log',env,train_deadline);training_seconds=time.time()-t
        status('evaluating',variant=variant,seed=seed,training_seconds=training_seconds,completed=rows)
        run([py,str(root/'scripts/evaluate/evaluate_baseline.py'),'--model',str(model)],model/'evaluation.log',env,eval_deadline)
        metrics=json.loads((model/'evaluation/metrics.json').read_text())
        rows.append({'variant':variant,'seed':seed,'metrics':metrics['mean'],'training_seconds':training_seconds,'model':str(model)})
        (root/'comparison_partial.json').write_text(json.dumps(rows,indent=2))
    means={v:{k:sum(r['metrics'][k] for r in rows if r['variant']==v)/3 for k in rows[0]['metrics']} for v in ['independent','spatial_stratified']}
    delta={k:means['spatial_stratified'][k]-means['independent'][k] for k in means['independent']}
    result={'state':'complete','runs':rows,'means':means,'candidate_minus_baseline':delta,
            'primary':'lpips_vgg_standard_m11','lower_is_better':True,
            'evidence':'Single scene, three paired seeds; exploratory mechanism comparison, novelty unverified'}
    (root/'comparison.json').write_text(json.dumps(result,indent=2))
    status('complete_review_required',result=str(root/'comparison.json'))
except Exception as exc:
    status('failed',error=str(exc),completed=rows)
    raise

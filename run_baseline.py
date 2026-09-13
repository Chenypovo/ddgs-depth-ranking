"""One bounded pilot, then a fixed full baseline; no candidate or retries hidden here."""
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
os.chdir(root/'DDGS')
py=str(root/'.venv/bin/python')
train_deadline=datetime(2026,9,14,0,10,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
eval_deadline=datetime(2026,9,14,1,10,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
def status(state,**extra):
    content={'state':state,'at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),**extra}
    temp=root/'status.tmp';temp.write_text(json.dumps(content,indent=2));temp.replace(root/'status.json')
    print(json.dumps(content),flush=True)
def run(command,log,deadline):
    remaining=int(deadline-time.time())
    if remaining<=0: raise TimeoutError('Hard deadline reached')
    with (root/log).open('w') as f:
        # GNU timeout owns the process group and kills it at the absolute budget bound.
        result=subprocess.run(['timeout','--signal=TERM','--kill-after=30',str(remaining),*command],stdout=f,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'{log} exited {result.returncode}')
status('waiting_preprocessing')
try:
    splitpath=root/'dataset/nerf_llff_data/trex/3_views/split.json'
    preparation_deadline=min(train_deadline-1800,time.time()+7200)
    while True:
        if time.time()>preparation_deadline: raise TimeoutError('Preparation budget exhausted')
        if splitpath.exists() and json.loads(splitpath.read_text()).get('state')=='complete': break
        time.sleep(15)
    split=json.loads(splitpath.read_text())
    masks=root/'DDGS/preprocessed_masks_5/trex/r8'
    for name in split['train_images']: assert (masks/(Path(name).stem+'.pt')).is_file()
    assert (root/'SETUP_COMPLETE').exists()
    assert (root/'dataset/nerf_llff_data/trex/3_views/dense/fused.ply').stat().st_size>0
    args=['train.py','-s',str(root/'dataset/nerf_llff_data/trex'),
          '--depth_weight','0.15','--density_weight','0.85','--drop_min','0.05','--drop_max','0.5',
          '--mask_param','5','--lambda_far','0.5','--eval','-r','8','--n_views','3']
    pilot=root/'runs/trex_integration_100'
    baseline=root/'runs/trex_baseline_seed0'
    assert not pilot.exists() and not baseline.exists(), 'Do not overwrite prior runs'
    status('integration_100_steps')
    t=time.time()
    run([py,*args,'-m',str(pilot),'--iterations','100','--test_iterations','100','--save_iterations','100','--detect_anomaly'],
        'pilot.log',min(train_deadline,time.time()+600))
    pilot_seconds=time.time()-t
    assert (pilot/'point_cloud/iteration_100/point_cloud.ply').is_file()
    status('training',iterations=10000,pilot_seconds=pilot_seconds,model=str(baseline))
    baseline.mkdir(parents=True)
    tb=Path('/root/tf-logs/ddgs_trex_baseline_seed0');tb.parent.mkdir(exist_ok=True)
    if not tb.exists(): tb.symlink_to(baseline,target_is_directory=True)
    start=time.time()
    run([py,*args,'-m',str(baseline),'--iterations','10000','--test_iterations','5000','10000',
         '--save_iterations','5000','10000','--checkpoint_iterations','5000','10000'], 'baseline.log',train_deadline)
    training_seconds=time.time()-start
    status('evaluating',model=str(baseline),training_seconds=training_seconds)
    run([py,str(root/'evaluate_baseline.py'),'--model',str(baseline)],'evaluation.log',eval_deadline)
    status('baseline_complete_review_required',model=str(baseline),training_seconds=training_seconds,
           metrics=str(baseline/'evaluation/metrics.json'),video=str(baseline/'evaluation/camera_path_10s.mp4'),
           candidate_started=False)
except Exception as exc:
    status('failed',error=str(exc))
    raise

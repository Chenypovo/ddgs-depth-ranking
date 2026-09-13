"""Audit the completed archive, paired exposure, and per-view primary changes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
r=a.root;out=a.out;out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((r/'rank_archive_manifest.json').read_text())
for name,record in manifest.items():
    path=r/name
    assert path.stat().st_size==record['bytes'],name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],name
plan=json.loads((r/'rank_runs/compare/plan.json').read_text())
for name,digest in plan['source_hashes'].items():
    assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest,name
rows=json.loads((r/'rank_runs/compare/results.json').read_text())['runs'];assert len(rows)==9
exposure=[];per_view=[]
for seed in range(3):
    logs={m:[json.loads(line) for line in (r/f'rank_runs/compare/{m}_seed{seed}/rank_stats.jsonl').read_text().splitlines()] for m in ['ordinary','consensus']}
    b,c=logs['ordinary'],logs['consensus'];assert len(b)==len(c)==9000
    assert [(x['iteration'],x['camera'],x['selected']) for x in b]==[(x['iteration'],x['camera'],x['selected']) for x in c]
    for m,ll in logs.items():
        selected=sum(x['selected'] for x in ll);active=sum(x['active'] for x in ll)
        exposure.append(dict(seed=seed,mode=m,selected=selected,active=active,invalid=selected-active,
            mean_pool_agreement=float(np.mean([x['agreed']/x['pool_valid'] for x in ll]))))
    metrics={m:json.loads((r/f'rank_runs/compare/{m}_seed{seed}/evaluation/metrics.json').read_text())['per_view'] for m in ['off','ordinary','consensus']}
    for b,c,o in zip(metrics['ordinary'],metrics['consensus'],metrics['off']):
        assert b['image']==c['image']==o['image']
        key='lpips_vgg_standard_m11'
        per_view.append(dict(seed=seed,image=b['image'],ordinary_minus_off=b[key]-o[key],consensus_minus_off=c[key]-o[key],consensus_minus_ordinary=c[key]-b[key]))
result=dict(state='passed',verified_files=len(manifest),source_hashes_match=True,exposure=exposure,per_view=per_view,
    wins={k:sum(x[k]<0 for x in per_view) for k in ['ordinary_minus_off','consensus_minus_off','consensus_minus_ordinary']},
    notes='21 image-model comparisons share views and seeds; they are not 21 independent training repeats.')
(out/'artifact_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='per_view'},indent=2))

"""Actual training cameras against frozen final models; CPU diagnostic only."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from audit_baseline_mechanism import read_xyz

root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('controls',(root/'vendor/DDGS' if (root/'vendor/DDGS').exists() else root/'DDGS')/'utils/dropout_controls.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
poses=json.loads((root/'reports/corrections/train_camera_poses.json').read_text())['rows']
torch.set_num_threads(2)
rows=[]
for p in sorted((root/'reports/comparison/runs').glob('trex_*seed*/point_cloud/iteration_10000/point_cloud.ply')):
 xyz=torch.from_numpy(read_xyz(p))
 for pose in poses:
  w=torch.eye(4);w[:3,:3]=torch.tensor(pose['R']);w[:3,3]=torch.tensor(pose['T'])
  cam=SimpleNamespace(world_view_transform=w.T,camera_center=torch.linalg.inv(w)[:3,3])
  d={k:m.camera_depths(xyz,cam,k) for k in ('upstream','camera_z','euclidean')}
  def band(z):
   s=z.sort().values
   return (z>s[int(len(z)*.33)]).int()+(z>s[int(len(z)*.67)]).int()
  ba=band(d['upstream']);bz=band(d['camera_z']);be=band(d['euclidean'])
  def score(z):return 1-(z-z.min())/(z.max()-z.min()+1e-6)
  rows.append(dict(run=p.parents[2].name,camera=pose['name'],points=len(xyz),
      z_band_disagreement=float((ba!=bz).float().mean()),
      euclidean_band_disagreement=float((ba!=be).float().mean()),
      normalised_z_score_mae=float((score(d['upstream'])-score(d['camera_z'])).abs().mean())))
r={'evidence':'Real training poses and six saved final models; not initial geometry or per-step telemetry',
   'rows':rows,'z_band_disagreement_mean':float(np.mean([x['z_band_disagreement'] for x in rows])),
   'z_band_disagreement_min':min(x['z_band_disagreement'] for x in rows),
   'z_band_disagreement_max':max(x['z_band_disagreement'] for x in rows)}
(root/'reports/corrections/real_camera_depths.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))

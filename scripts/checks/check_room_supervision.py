"""Real room initial-scene integration; no metric-based model selection."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
r=Path(__file__).resolve().parents[2];sys.path.insert(0,str(r/'DDGS_room'))
from scene import Scene,GaussianModel
from gaussian_renderer import render
from utils.rank_supervision import RankSupervisor
from utils.general_utils import safe_state
safe_state(False)
folder=r/'room_integration';folder.mkdir(exist_ok=False)
data=r/'dataset/nerf_llff_data/room';split=json.loads((data/'3_views/split.json').read_text());assert split['state']=='complete'
g=GaussianModel(3);ds=SimpleNamespace(sh_degree=3,source_path=str(data),model_path=str(folder),images='images',eval=True,n_views=3,resolution=8,white_background=False,data_device='cuda')
scene=Scene(ds,g,shuffle=False);cams=scene.getTrainCameras();assert [c.image_name for c in cams]==[Path(n).stem for n in split['train_images']];assert len(scene.getTestCameras())==6
assert g.get_xyz.shape[0]==split['dense_points']
pipe=SimpleNamespace(debug=False,compute_cov3D_python=False,convert_SHs_python=False)
sup=RankSupervisor(r/'room_priors','ordinary',0,folder/'rank_stats.jsonl');stats=[]
for cam in cams:
 assert tuple(sup.priors[cam.image_name].shape[1:])==tuple(cam.original_image.shape[1:])
 for p in [g._xyz,g._scaling,g._opacity,g._rotation,g._features_dc,g._features_rest]:p.grad=None
 state=torch.cuda.get_rng_state().clone();loss=sup.loss(cam,g,pipe,render,3000);assert torch.equal(state,torch.cuda.get_rng_state());assert torch.isfinite(loss) and loss.item()>0;loss.backward()
 gradients={n:float(p.grad.abs().max()) for n,p in [('xyz',g._xyz),('scale',g._scaling),('opacity',g._opacity)]};assert all(np.isfinite(x) and x>0 for x in gradients.values())
 stats.append(dict(image=cam.image_name,shape=list(cam.original_image.shape),loss=float(loss.detach()),gradients=gradients))
result=dict(state='passed',rows=stats,initial_points=split['dense_points'],train_images=split['train_images'],evaluation_views=6,global_rng_unchanged=True)
(r/'room_integration.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

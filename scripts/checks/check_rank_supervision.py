"""Real-cache, real-renderer integration checks before any training."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'DDGS_rank'))
from scene import Scene,GaussianModel
from gaussian_renderer import render
from utils.rank_supervision import select_pairs,render_inverse_depth,ordinal_loss

torch.manual_seed(17);torch.cuda.manual_seed_all(17)
priors=torch.from_numpy(np.load(root/'depth_reliability/view0.npz')['normalised']).float().cuda()
before=torch.cuda.get_rng_state().clone()
a,b,s,st=select_pairs(priors,0,1101,'ordinary')
aa,bb,ss,tt=select_pairs(priors,0,1101,'consensus')
assert st==tt and a.numel()==aa.numel()>0
assert torch.equal(before,torch.cuda.get_rng_state()),'Global RNG changed'
d=priors.flatten(1)[:,aa]-priors.flatten(1)[:,bb]
assert ((d.sign()==ss[None]).all(0)).all()
assert (priors.flatten(1)[0,a]-priors.flatten(1)[0,b]).abs().min()>=.05
# Loss sign and gradient on a tiny nontrivial map.
x=torch.tensor([[1.,2.],[3.,4.]],device='cuda',requires_grad=True)
valid=torch.ones_like(x,dtype=torch.bool)
wrong,_=ordinal_loss(x,valid,torch.tensor([0],device='cuda'),torch.tensor([3],device='cuda'),torch.ones(1,device='cuda'))
right,_=ordinal_loss(x,valid,torch.tensor([3],device='cuda'),torch.tensor([0],device='cuda'),torch.ones(1,device='cuda'))
assert wrong>0 and right==0
wrong.backward();assert x.grad[0,0]<0 and x.grad[1,1]>0
g=GaussianModel(3)
ds=SimpleNamespace(sh_degree=3,source_path=str(root/'dataset/nerf_llff_data/trex'),model_path=str(root/'correction_runs/compare/depth_upstream_density_upstream_seed0'),images='images',eval=True,n_views=3,resolution=8,white_background=False,data_device='cuda')
scene=Scene(ds,g,load_iteration=10000,shuffle=False);cam=scene.getTrainCameras()[0]
pipe=SimpleNamespace(debug=False,compute_cov3D_python=False,convert_SHs_python=False);bg=torch.zeros(3,device='cuda')
with torch.no_grad():
 baseline_before=render(cam,g,pipe,bg)['render'].clone()
inverse,valid,pack=render_inverse_depth(cam,g,pipe,render)
assert inverse.shape==(378,504) and torch.isfinite(inverse).all()
with torch.no_grad():
 xyz=g.get_xyz;z=(torch.cat([xyz,torch.ones_like(xyz[:,:1])],1)@cam.world_view_transform)[:,2].clamp_min(0);scale=z.max()
 num=render(cam,g,pipe,bg,override_color=(z/scale)[:,None].expand(-1,3).contiguous())['render'][0]*scale
 alpha=render(cam,g,pipe,bg,override_color=torch.ones_like(xyz))['render'][0]
 separate=alpha/num.clamp_min(1e-6)
 assert torch.allclose(inverse[valid],separate[valid],rtol=1e-5,atol=1e-6),'Packed depth mismatch'
assert set(np.load(root/'depth_reliability/view0.npz').files)
loss,extra=ordinal_loss(inverse,valid,aa,bb,ss)
assert torch.isfinite(loss) and loss>0
loss.backward()
grads={}
for key in ['_xyz','_scaling','_opacity']:
 grad=getattr(g,key).grad
 assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0,key
 grads[key]=float(grad.abs().max())
with torch.no_grad():assert torch.equal(baseline_before,render(cam,g,pipe,bg)['render'])
result=dict(state='passed',pairs=st,active=extra,loss=float(loss),max_gradients=grads,baseline_unchanged=True,packed_matches_separate=True,global_rng_unchanged=True)
(root/'rank_integration.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

"""CPU regression checks using real helper and topology mutation methods."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch

root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('controls',(root/'vendor/DDGS' if (root/'vendor/DDGS').exists() else root/'DDGS')/'utils/dropout_controls.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.manual_seed(901)
xyz=torch.randn(128,3,requires_grad=True)
M=torch.eye(4);M[:3,:3]=torch.tensor([[0.,0.,-1.],[0.,1.,0.],[1.,0.,0.]])
M[3,:3]=torch.tensor([1.,2.,3.])
cam=SimpleNamespace(world_view_transform=M,camera_center=torch.linalg.inv(M)[3,:3])
rng=torch.random.get_rng_state().clone()
h=torch.cat([xyz,torch.ones(128,1)],1)
a=m.camera_depths(xyz,cam,'upstream')
assert torch.equal(a,(h@M.T)[:,2])
b=m.camera_depths(xyz,cam,'camera_z');flat=M.contiguous().flatten()
assert torch.allclose(b,xyz[:,0]*flat[2]+xyz[:,1]*flat[6]+xyz[:,2]*flat[10]+flat[14])
b.sum().backward();assert torch.isfinite(xyz.grad).all()
assert torch.equal(rng,torch.random.get_rng_state())
for bad in ['invalid','']:
 try: m.camera_depths(xyz,cam,bad)
 except ValueError: pass
 else: raise AssertionError('Invalid mode accepted')

# Execute the actual class definition without importing CUDA dependencies.
tree=ast.parse(((root/'vendor/DDGS' if (root/'vendor/DDGS').exists() else root/'DDGS')/'scene/gaussian_model.py').read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GaussianModel')
ns={'torch':torch,'BasicPointCloud':object};exec(compile(ast.Module(body=[cls],type_ignores=[]),'gaussian_model.py','exec'),ns)
g=ns['GaussianModel'].__new__(ns['GaussianModel'])
attrs={'xyz':'_xyz','f_dc':'_features_dc','f_rest':'_features_rest','opacity':'_opacity','scaling':'_scaling','rotation':'_rotation'}
for attr in attrs.values(): setattr(g,attr,torch.arange(12.).reshape(4,3).clone())
g.density_score=torch.tensor([.1,.2,.3,.4]);g._density_topology_version=0
g.xyz_gradient_accum=torch.zeros(4,1);g.denom=torch.zeros(4,1);g.max_radii2D=torch.zeros(4)
g._prune_optimizer=lambda valid:{k:getattr(g,v)[valid] for k,v in attrs.items()}
g.prune_points(torch.tensor([False,True,False,False]))
assert g._ddgs_topology_version==1
assert not m.ensure_density(g,'upstream')
assert torch.equal(g.density_score,torch.tensor([.1,.2,.3,.4]))
calls=[]
def refresh():
 calls.append(1);g.density_score=torch.arange(len(g.get_xyz),dtype=torch.float32)
 g._density_topology_version=g._ddgs_topology_version
g.update_density_score=refresh
assert m.ensure_density(g,'refresh')
assert not m.ensure_density(g,'refresh')
g.cat_tensors_to_optimizer=lambda d:{k:torch.cat([getattr(g,attrs[k]),v]) for k,v in d.items()}
zeros=torch.zeros
# Only substitute device allocation; execute unchanged densification_postfix body.
def cpu_zeros(*a,**kw):
 if kw.get('device')=='cuda':kw['device']='cpu'
 return zeros(*a,**kw)
with patch.object(torch,'zeros',cpu_zeros):
 g.densification_postfix(*[torch.ones(1,3) for _ in range(6)])
assert g._ddgs_topology_version==2
assert m.ensure_density(g,'refresh') and len(calls)==2
assert len(g.density_score)==len(g.get_xyz)==4
r={'state':'cpu_checks_pass','upstream_depth_exact':True,'renderer_depth_matches':True,
   'backward_finite':True,'rng_unchanged':True,'actual_prune_append_invalidate_cache':True,
   'refresh_once_per_dirty_cache':True,'cuda_integration':'not yet tested'}
(root/'reports/corrections/cpu_checks.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))

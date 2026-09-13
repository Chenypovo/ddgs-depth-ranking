"""CUDA integration check only; this is not a reconstruction experiment."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent/'DDGS'))
from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera
from utils.graphics_utils import BasicPointCloud

torch.manual_seed(0)
rng = np.random.default_rng(0)
xyz = rng.uniform(-.5,.5,(256,3)).astype(np.float32)
xyz[:,2] += 2
pc = GaussianModel(3)
pc.create_from_pcd(BasicPointCloud(xyz,rng.uniform(0,1,(256,3)).astype(np.float32),np.zeros_like(xyz)),1.0)
pc.update_density_score()
camera = Camera(0,np.eye(3),np.zeros(3),1.,1.,torch.zeros(3,64,64),None,'smoke',0,np.array([.1,5.]))
pipe = SimpleNamespace(debug=False,compute_cov3D_python=False,convert_SHs_python=False)
out = render(camera,pc,pipe,torch.zeros(3,device='cuda'),is_train=True,iteration=1)
loss = out['render'].square().mean()
assert torch.isfinite(loss) and loss.item() > 0
loss.backward()
assert pc._xyz.grad is not None and torch.isfinite(pc._xyz.grad).all()
assert pc._xyz.grad.abs().sum() > 0
print('CUDA rasterizer forward/backward passed',out['render'].shape,loss.item(),flush=True)

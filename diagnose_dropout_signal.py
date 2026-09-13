"""Does fixed-checkpoint dropout sensitivity identify wrong pixels?

Uniform 15% masks are a fixed diagnostic probe, NOT the historical DDGS
depth/density schedule. No GT enters a mask or the Monte Carlo mean.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'DDGS_corrections'))
from scene import Scene,GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch.modules.lpips import LPIPS

class OpacityProbe:
    def __init__(self,g,mask):self.g,self.mask=g,mask
    def __getattr__(self,name):return getattr(self.g,name)
    @property
    def get_opacity(self):return self.g.get_opacity*self.mask[:,None]

out=root/'failure_analysis'
assert not (out/'dropout_signal.json').exists()
metric=LPIPS('vgg').cuda().eval()
pipe=SimpleNamespace(debug=False,compute_cov3D_python=False,convert_SHs_python=False)
bg=torch.zeros(3,device='cuda')
rows=[]
with torch.inference_mode():
 for seed in range(3):
  model=root/'correction_runs/compare'/f'depth_upstream_density_upstream_seed{seed}'
  dataset=SimpleNamespace(sh_degree=3,source_path=str(root/'dataset/nerf_llff_data/trex'),model_path=str(model),images='images',eval=True,n_views=3,resolution=8,white_background=False,data_device='cuda')
  g=GaussianModel(3);scene=Scene(dataset,g,load_iteration=10000,shuffle=False)
  for vi,cam in enumerate(scene.getTestCameras()):
   gen=torch.Generator(device='cuda').manual_seed(10000+seed*100+vi)
   orig=render(cam,g,pipe,bg)['render'];gt=cam.original_image
   samples=[]
   for k in range(8):
    mask=(torch.rand(len(g.get_xyz),generator=gen,device='cuda')>=.15).float()
    samples.append(render(cam,OpacityProbe(g,mask),pipe,bg)['render'])
   stack=torch.stack(samples);mean=stack.mean(0);unc=stack.std(0).mean(0)
   err=(orig-gt).abs().mean(0)
   hi=unc>=torch.quantile(unc,.8);bad=err>=torch.quantile(err,.8)
   # Rank correlation also within 3x3 GT gradient buckets reduces edge confounding.
   gray=gt.mean(0);dy=torch.zeros_like(gray);dx=torch.zeros_like(gray)
   dx[:,1:]=(gray[:,1:]-gray[:,:-1]).abs();dy[1:]=(gray[1:]-gray[:-1]).abs();edge=dx+dy
   corr=[]
   for a,b in [(0,.5),(.5,.8),(.8,1)]:
    sel=(edge>=torch.quantile(edge,a))&(edge<=torch.quantile(edge,b))
    corr.append(float(spearmanr(unc[sel].cpu().numpy(),err[sel].cpu().numpy()).statistic))
   row=dict(seed=seed,image=cam.image_name,
    rank_correlation=float(spearmanr(unc.flatten().cpu().numpy(),err.flatten().cpu().numpy()).statistic),
    edge_bucket_correlations=corr,top20_error_recall=((hi&bad).sum()/bad.sum()).item(),
    high_uncertainty_mae=err[hi].mean().item(),remaining_mae=err[~hi].mean().item(),
    baseline_lpips=metric(2*orig[None]-1,2*gt[None]-1).item(),
    mc8_lpips=metric(2*mean[None]-1,2*gt[None]-1).item(),
    mc8_psnr=psnr(mean[None],gt[None]).item(),mc8_ssim=ssim(mean[None],gt[None]).item(),
    mean_dropout_change=(mean-orig).abs().mean().item())
   assert all(np.isfinite(v) for k,v in row.items() if isinstance(v,float))
   rows.append(row)
   if seed==0:
    for label,img in [('mc8',mean),('uncertainty',(unc/torch.quantile(unc,.99).clamp_min(1e-6)).clamp(0,1)[None].expand(3,-1,-1))]:
     Image.fromarray((img.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(out/f'development_{vi}_{label}.png')
 summary={k:float(np.mean([r[k] for r in rows])) for k in ['rank_correlation','top20_error_recall','high_uncertainty_mae','remaining_mae','baseline_lpips','mc8_lpips','mc8_psnr','mc8_ssim','mean_dropout_change']}
 result=dict(state='complete',rows=rows,summary=summary,probe='8 independent uniform opacity masks, p=.15, fixed seeds, all 21 development image/model pairs; diagnostic only, not trained candidate, no threshold sweep')
 (out/'dropout_signal.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))

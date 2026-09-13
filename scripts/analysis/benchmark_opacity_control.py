"""Matched end-to-end render timing; synchronised, rotated method order."""
import json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'DDGS_corrections'))
from scene import Scene,GaussianModel
from gaussian_renderer import render
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'methods'))
from opacity_expectation import OpacityExpectation

class MaskProxy(OpacityExpectation):
    def __init__(self,g,mask):self.gaussians=g;self.mask=mask
    @property
    def get_opacity(self):return self.gaussians.get_opacity*self.mask[:,None]

out=root/'failure_analysis/benchmark.json'
assert not out.exists()
pipe=SimpleNamespace(debug=False,compute_cov3D_python=False,convert_SHs_python=False)
bg=torch.zeros(3,device='cuda');rows=[]
with torch.inference_mode():
 for seed in range(3):
  model=root/'correction_runs/compare'/f'depth_upstream_density_upstream_seed{seed}'
  ds=SimpleNamespace(sh_degree=3,source_path=str(root/'dataset/nerf_llff_data/trex'),model_path=str(model),images='images',eval=True,n_views=3,resolution=8,white_background=False,data_device='cuda')
  g=GaussianModel(3);scene=Scene(ds,g,load_iteration=10000,shuffle=False)
  for vi,cam in enumerate(scene.getTestCameras()):
   def run(mode):
    if mode=='baseline':return render(cam,g,pipe,bg)['render']
    if mode=='opacity085':return render(cam,OpacityExpectation(g,.85),pipe,bg)['render']
    mean=torch.zeros_like(cam.original_image)
    for _ in range(8):
     mask=(torch.rand(len(g.get_xyz),device='cuda')>=.15).float()
     mean+=render(cam,MaskProxy(g,mask),pipe,bg)['render']/8
    return mean
   modes=['baseline','opacity085','mc8'];shift=(seed+vi)%3;modes=modes[shift:]+modes[:shift]
   for mode in modes:
    for _ in range(2):run(mode)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();mem=torch.cuda.memory_allocated()
    start=time.perf_counter()
    for _ in range(5):run(mode)
    torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000/5
    rows.append(dict(seed=seed,image=cam.image_name,mode=mode,ms=elapsed,extra_peak_mb=(torch.cuda.max_memory_allocated()-mem)/2**20))
 summary={m:dict(median_ms=float(np.median([r['ms'] for r in rows if r['mode']==m])),mean_ms=float(np.mean([r['ms'] for r in rows if r['mode']==m])),max_extra_peak_mb=max(r['extra_peak_mb'] for r in rows if r['mode']==m)) for m in ['baseline','opacity085','mc8']}
 out.write_text(json.dumps(dict(state='complete',rows=rows,summary=summary,notes='End-to-end synchronised wall time includes wrapper, random mask generation and MC mean; no metric/video IO; 5 repeats per 21 camera/model pairs, rotated method order. GPU name is virtualised.'),indent=2))
 print(json.dumps(summary,indent=2))

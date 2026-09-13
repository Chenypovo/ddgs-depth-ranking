"""Fixed final-checkpoint evaluation and RGB camera-path preview."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp
import imageio.v2 as imageio

root = Path(__file__).resolve().parents[2]
sys.path.insert(0,os.environ.get('DDGS_SOURCE', str(root/'DDGS')))
from scene import Scene, GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch.modules.lpips import LPIPS

p = argparse.ArgumentParser()
p.add_argument('--model',type=Path,required=True)
p.add_argument('--iteration',type=int,default=10000)
args = p.parse_args()
model_path = args.model.resolve()
outdir = model_path/'evaluation'
outdir.mkdir(exist_ok=True)
dataset = SimpleNamespace(sh_degree=3, source_path=str(root/'dataset/nerf_llff_data/trex'),
    model_path=str(model_path), images='images',eval=True,n_views=3,resolution=8,
    white_background=False,data_device='cuda')
gaussians = GaussianModel(3)
scene = Scene(dataset,gaussians,load_iteration=args.iteration,shuffle=False)
pipe = SimpleNamespace(debug=False,compute_cov3D_python=False,convert_SHs_python=False)
background = torch.zeros(3,device='cuda')
metric = LPIPS('vgg').cuda().eval()
split = json.loads((root/'dataset/nerf_llff_data/trex/3_views/split.json').read_text())
assert [c.image_name for c in scene.getTestCameras()] == [Path(x).stem for x in split['test_images']]
rows, panels = [], []
def to_image(t):
    return Image.fromarray((t.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8))
start=time.time()
with torch.inference_mode():
    for i,cam in enumerate(scene.getTestCameras()):
        prediction = render(cam,gaussians,pipe,background)['render']
        truth = cam.original_image
        x,y=prediction[None],truth[None]
        row={'image':cam.image_name,'psnr':psnr(x,y).item(),'ssim':ssim(x,y).item(),
             'lpips_vgg_repo_01':metric(x,y).item(),
             'lpips_vgg_standard_m11':metric(2*x-1,2*y-1).item()}
        assert all(np.isfinite(v) for k,v in row.items() if k!='image')
        rows.append(row)
        pred_im,gt_im=to_image(prediction),to_image(truth)
        pred_im.save(outdir/f'{cam.image_name}_prediction.png')
        gt_im.save(outdir/f'{cam.image_name}_gt.png')
        panel=Image.new('RGB',(pred_im.width*2,pred_im.height+28),'white')
        panel.paste(gt_im,(0,28)); panel.paste(pred_im,(pred_im.width,28))
        ImageDraw.Draw(panel).text((5,5),f'{i}: Ground truth | Prediction',fill='black')
        panels.append(panel)
        print(row,flush=True)
    summary={'state':'complete','iteration':args.iteration,'views':len(rows),'per_view':rows,
             'mean':{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='image'},
             'metric_notes':'PSNR/SSIM repository functions on unquantised RGB. Both LPIPS variants use repository VGG v0.1: repo_01 preserves upstream [0,1] input; standard_m11 rescales to [-1,1]. Do not conflate these definitions.',
             'gaussians':len(gaussians.get_xyz),'evaluation_seconds':time.time()-start,
             'evidence':'one-scene one-seed adapted official baseline; not paper-wide reproduction'}
    (outdir/'metrics.json').write_text(json.dumps(summary,indent=2))
    contact=Image.new('RGB',(panels[0].width,sum(x.height for x in panels)),'white')
    y0=0
    for panel in panels:
        contact.paste(panel,(0,y0));y0+=panel.height
    contact.save(outdir/'contact.png')
    # Known camera poses only: interpolate a path inside the captured view range.
    cameras=sorted(scene.getTrainCameras()+scene.getTestCameras(),key=lambda c:c.image_name)
    transforms=[]
    for cam in cameras:
        w2c=np.eye(4);w2c[:3,:3]=cam.R.T;w2c[:3,3]=cam.T
        transforms.append(np.linalg.inv(w2c))
    transforms=np.stack(transforms)
    knots=np.arange(len(cameras),dtype=float)
    rotations=Slerp(knots,Rotation.from_matrix(transforms[:,:3,:3]))
    ref=cameras[0]
    with imageio.get_writer(outdir/'camera_path_10s.mp4',fps=24,codec='libx264',quality=8,macro_block_size=2) as writer:
        for t in np.linspace(0,len(cameras)-1,240):
            c2w=np.eye(4);c2w[:3,:3]=rotations(t).as_matrix()
            c2w[:3,3]=[np.interp(t,knots,transforms[:,k,3]) for k in range(3)]
            w2c=np.linalg.inv(c2w)
            cam=Camera(0,w2c[:3,:3].T,w2c[:3,3],ref.FoVx,ref.FoVy,
                torch.zeros_like(ref.original_image),None,'path',0,ref.bounds)
            prediction=render(cam,gaussians,pipe,background)['render']
            writer.append_data(np.asarray(to_image(prediction)))
    (outdir/'VIDEO_COMPLETE').write_text('240 RGB frames, 24 fps; novel-view camera interpolation, not future prediction\n')
print(json.dumps(summary['mean'],indent=2),flush=True)

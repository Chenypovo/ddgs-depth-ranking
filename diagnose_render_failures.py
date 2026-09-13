"""Frozen-checkpoint diagnostic: appearance ablation, train/test gap, edge errors.

No checkpoint changes. SH0 is an intervention, not a trained candidate.
trex is explicitly a development scene after repeated previous inspection.
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'DDGS_corrections'))
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch.modules.lpips import LPIPS

out = root / 'failure_analysis'
out.mkdir(exist_ok=True)
assert not (out/'diagnosis.json').exists(), 'Do not overwrite completed diagnostics'
pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
metric = LPIPS('vgg').cuda().eval()
bg = torch.zeros(3, device='cuda')
rows = []
start = time.time()

def pil(x):
    return Image.fromarray((x.clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8'))

with torch.inference_mode():
    for seed in range(3):
        model = root/'correction_runs/compare'/f'depth_upstream_density_upstream_seed{seed}'
        dataset = SimpleNamespace(sh_degree=3, source_path=str(root/'dataset/nerf_llff_data/trex'),
            model_path=str(model), images='images', eval=True,n_views=3,resolution=8,
            white_background=False,data_device='cuda')
        g = GaussianModel(3)
        scene = Scene(dataset,g,load_iteration=10000,shuffle=False)
        centres = torch.stack([c.camera_center for c in scene.getTrainCameras()])
        for split, cams in [('train',scene.getTrainCameras()),('development',scene.getTestCameras())]:
            for vi,cam in enumerate(cams):
                gt = cam.original_image
                gray = gt.mean(0)[None,None]
                k = torch.tensor([[-1.,0,1],[-2,0,2],[-1,0,1]],device='cuda')[None,None]
                grad = (F.conv2d(gray,k,padding=1).square()+F.conv2d(gray,k.transpose(2,3),padding=1).square()).sqrt()[0,0]
                edge = grad >= torch.quantile(grad,0.8)
                flat = grad <= torch.quantile(grad,0.5)
                preds=[]
                for degree in [3,0]:
                    g.active_sh_degree=degree
                    pred=render(cam,g,pipe,bg)['render']
                    err=(pred-gt).abs().mean(0)
                    row=dict(seed=seed,split=split,image=cam.image_name,degree=degree,
                        psnr=psnr(pred[None],gt[None]).item(),ssim=ssim(pred[None],gt[None]).item(),
                        lpips=metric(2*pred[None]-1,2*gt[None]-1).item(),mae=err.mean().item(),
                        edge_mae=err[edge].mean().item(),flat_mae=err[flat].mean().item(),
                        edge_error_share=err[edge].sum().item()/err.sum().item(),
                        nearest_train_distance=torch.linalg.norm(centres-cam.camera_center,dim=1).min().item())
                    rows.append(row);preds.append(pred)
                    if seed==0:
                        pil(pred).save(out/f'{split}_{vi}_sh{degree}.png')
                if seed==0:
                    pil(gt).save(out/f'{split}_{vi}_gt.png')
                    heat=(preds[0]-gt).abs().mean(0).clamp(0,0.25)/0.25
                    heat_rgb=torch.stack([heat,torch.zeros_like(heat),1-heat])
                    panels=[pil(gt),pil(preds[0]),pil(preds[1]),pil(heat_rgb)]
                    sheet=Image.new('RGB',(gt.shape[2]*4,gt.shape[1]+25),'white')
                    draw=ImageDraw.Draw(sheet)
                    for j,(im,label) in enumerate(zip(panels,['Ground truth','Baseline SH3','SH0 intervention','Absolute error: red = high'])):
                        sheet.paste(im,(j*im.width,25));draw.text((j*im.width+5,5),label,fill='black')
                    sheet.save(out/f'{split}_{vi}_panel.png')
        g.active_sh_degree=3
        old=json.loads((model/'evaluation/metrics.json').read_text())['mean']
        selected=[r for r in rows if r['seed']==seed and r['split']=='development' and r['degree']==3]
        assert abs(np.mean([r['lpips'] for r in selected])-old['lpips_vgg_standard_m11'])<1e-5, 'Baseline rendering/evaluator drift'
    summaries=[]
    for split in ['train','development']:
        for degree in [3,0]:
            rr=[r for r in rows if r['split']==split and r['degree']==degree]
            summaries.append(dict(split=split,degree=degree,**{k:float(np.mean([r[k] for r in rr])) for k in ['psnr','ssim','lpips','mae','edge_mae','flat_mae','edge_error_share']}))
    result=dict(state='complete',seconds=time.time()-start,rows=rows,summary=summaries,
        notes='SH0 is a fixed-checkpoint intervention, not retraining; cannot uniquely establish causal source. Edge top20%, flat bottom50% GT Sobel. Development scene only. Exact baseline metrics checked against saved unquantised evaluation.')
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(summaries,indent=2),flush=True)

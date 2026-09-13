"""Training-only depth-prior audit; sparse triangulation is a proxy, not GT.

Three predeclared transformations: original518, horizontal flip518, original392.
Reference: 3-view tracks with COLMAP reprojection error <=1 original pixel.
No final-test views, no training updates and no hyperparameter search.
"""
import hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pycolmap
from scipy.stats import spearmanr
import torch
from PIL import Image

root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'depth-anything-v2'))
from depth_anything_v2.dpt import DepthAnythingV2
sys.path.insert(0,str(root/'DDGS_corrections'))
from scene import Scene,GaussianModel
from gaussian_renderer import render

def sample(a,xy):
    return cv2.remap(a.astype('float32'),xy[:,0].astype('float32')[None],xy[:,1].astype('float32')[None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0]

def robust_normalise(a):
    return (a-np.median(a))/(np.quantile(a,.75)-np.quantile(a,.25)+1e-6)

def corr(a,b):
    if len(a)<3:return None
    value=float(spearmanr(a,b).statistic)
    return value if np.isfinite(value) else None

start=time.time()
out=root/'depth_reliability';out.mkdir(exist_ok=False)
data=root/'dataset/nerf_llff_data/trex'
split=json.loads((data/'3_views/split.json').read_text())
ref=pycolmap.Reconstruction(data/'3_views/triangulated')
assert {i.name for i in ref.images.values()}==set(split['train_images'])
ckpt=root/'depth-anything-v2/checkpoints/depth_anything_v2_vits.pth'
net=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384])
net.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True));net=net.cuda().eval()
g=GaussianModel(3)
model=root/'correction_runs/compare/depth_upstream_density_upstream_seed0'
ds=SimpleNamespace(sh_degree=3,source_path=str(data),model_path=str(model),images='images',eval=True,n_views=3,resolution=8,white_background=False,data_device='cuda')
scene=Scene(ds,g,load_iteration=10000,shuffle=False)
cams={c.image_name:c for c in scene.getTrainCameras()}
pipe=SimpleNamespace(debug=False,compute_cov3D_python=False,convert_SHs_python=False)
rows=[]
with torch.inference_mode():
 for ii,name in enumerate(split['train_images']):
    bgr=cv2.imread(str(data/'images'/name));h,w=bgr.shape[:2];size=(round(w/8),round(h/8));stem=Path(name).stem
    raw=net.infer_image(bgr,input_size=518)
    flipped=net.infer_image(np.ascontiguousarray(bgr[:,::-1]),input_size=518)[:,::-1].copy()
    small=net.infer_image(bgr,input_size=392)
    priors=np.stack([cv2.resize(a,size,interpolation=cv2.INTER_LINEAR) for a in [raw,flipped,small]])
    assert np.isfinite(priors).all()
    normalised=np.stack([robust_normalise(a) for a in priors])
    disagreement=normalised.std(0)
    im=next(i for i in ref.images.values() if i.name==name)
    transform=im.cam_from_world().matrix()
    xy=[];z=[];ids=[];reproj=[]
    for obs in im.points2D:
      if not obs.has_point3D():continue
      p=ref.points3D[obs.point3D_id]
      if p.track.length()<3 or p.error>1.0:continue
      camxyz=transform[:,:3]@p.xyz+transform[:,3]
      if camxyz[2]<=0:continue
      xy.append((obs.xy+.5)*np.array(size)/np.array([w,h])-.5);z.append(camxyz[2]);ids.append(int(obs.point3D_id));reproj.append(float(p.error))
    xy=np.asarray(xy);z=np.asarray(z);ids=np.asarray(ids)
    assert len(z)>100
    values=np.stack([sample(a,xy) for a in priors]);unc=sample(disagreement,xy)
    # Independent point-ID halves for affine fit and diagnostic error reporting.
    ordered=np.argsort(ids);fit=ordered[::2];check=ordered[1::2]
    affine=np.linalg.lstsq(np.stack([values[0,fit],np.ones(len(fit))],axis=1),1/z[fit],rcond=None)[0]
    predinv=affine[0]*values[0]+affine[1]
    inv_err=np.abs(predinv-1/z)/(1/z)
    check_q=np.quantile(unc[check],.5);stable=unc[check]<=check_q
    # All eligible spatial pairs; no pairing tuned from outcomes.
    aa,bb=np.triu_indices(len(z),1)
    distance=np.linalg.norm(xy[aa]-xy[bb],axis=1)
    refgap=np.abs(z[aa]-z[bb])/np.minimum(z[aa],z[bb])
    valid=(distance>=8)&(distance<=64)&(refgap>=.10)
    aa,bb=aa[valid],bb[valid]
    truth=np.sign(z[bb]-z[aa]) # larger prior means nearer
    signs=np.sign(values[:,aa]-values[:,bb])
    correct=signs[0]==truth
    agreement=np.all(signs==signs[0:1],axis=0)&(signs[0]!=0)
    pair_unc=np.maximum(unc[aa],unc[bb]);keep=pair_unc<=np.quantile(unc,.5)
    # Render conditional expected camera-z for visual diagnostics only.
    cam=cams[stem];xyz=g.get_xyz
    hom=torch.cat([xyz,torch.ones_like(xyz[:,:1])],dim=1)
    depths=(hom@cam.world_view_transform)[:,2].clamp_min(0)
    scale=depths.max().clamp_min(1e-6)
    colors=(depths/scale)[:,None].expand(-1,3).contiguous()
    depthsum=render(cam,g,pipe,torch.zeros(3,device='cuda'),override_color=colors)['render'][0]*scale
    alpha=render(cam,g,pipe,torch.zeros(3,device='cuda'),override_color=torch.ones_like(colors))['render'][0]
    rendered=(depthsum/alpha.clamp_min(1e-6)).cpu().numpy();alpha=alpha.cpu().numpy()
    expected_z=sample(rendered,xy)
    row=dict(image=name,reference_points=len(z),mean_reprojection_error=float(np.mean(reproj)),
        prior_inverse_depth_spearman=corr(values[0],1/z),affine_inverse_depth_scale=float(affine[0]),
        independent_check_points=len(check),check_median_relative_inverse_depth_error=float(np.median(inv_err[check])),
        stable_half_check_median_error=float(np.median(inv_err[check][stable])),unstable_half_check_median_error=float(np.median(inv_err[check][~stable])),
        instability_error_spearman=corr(unc[check],inv_err[check]),
        local_pairs=len(aa),local_rank_accuracy=float(correct.mean()),
        augmentation_agreement_coverage=float(agreement.mean()),augmentation_agreement_rank_accuracy=float(correct[agreement].mean()) if agreement.any() else None,
        stable_half_pair_coverage=float(keep.mean()),stable_half_pair_rank_accuracy=float(correct[keep].mean()) if keep.any() else None,
        rendered_depth_spearman=corr(expected_z,z),rendered_depth_median_relative_error=float(np.median(np.abs(expected_z-z)/z)),
        transforms=['original518','horizontal_flip518_unflipped','original392'])
    rows.append(row)
    np.savez_compressed(out/f'view{ii}.npz',rgb=cv2.cvtColor(cv2.resize(bgr,size),cv2.COLOR_BGR2RGB),priors=priors,normalised=normalised,disagreement=disagreement,rendered_z=rendered,alpha=alpha,xy=xy,z=z,ids=ids,prior_samples=values,check=check,affine=affine,reference_relative_inverse_depth_error=inv_err,pair_a=aa,pair_b=bb,pair_correct=correct,pair_agreement=agreement)
    print(json.dumps(row),flush=True)
 result=dict(state='complete',rows=rows,seconds=time.time()-start,
     prior_checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
     reconstruction='3_views/triangulated: training-only 2199 total points; use track length3 and reprojection error<=1',
     reference_caveat='Sparse triangulated geometry is a noisy reference, not true depth. Same point tracks recur across views; pairs are correlated. Dense/occluded regions are not certified.',
     protocol='fixed3augmentations; local pair distance8..64 output pixels, reference depth separation>=10%; no unseen views or threshold sweep; no training updates')
 (out/'results.json').write_text(json.dumps(result,indent=2))

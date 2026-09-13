"""Frozen three-training-image priors and DAFE masks, identical trex settings."""
import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
r=Path(__file__).resolve().parents[2];sys.path.insert(0,str(r/'depth-anything-v2'))
from depth_anything_v2.dpt import DepthAnythingV2
scene=r/'dataset/nerf_llff_data/room';split=json.loads((scene/'3_views/split.json').read_text());assert split['state']=='complete'
out=r/'room_priors';out.mkdir(exist_ok=False)
masks=r/'DDGS_room/preprocessed_masks_5/room/r8';masks.mkdir(parents=True,exist_ok=False)
ckpt=r/'depth-anything-v2/checkpoints/depth_anything_v2_vits.pth'
net=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384]);net.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True));net=net.cuda().eval();rows=[]
with torch.inference_mode():
 for i,name in enumerate(split['train_images']):
  im=cv2.imread(str(scene/'images'/name));h,w=im.shape[:2];size=(round(w/8),round(h/8))
  raw=net.infer_image(im,input_size=518)
  flip=net.infer_image(np.ascontiguousarray(im[:,::-1]),input_size=518)[:,::-1].copy()
  small=net.infer_image(im,input_size=392)
  priors=np.stack([cv2.resize(x,size,interpolation=cv2.INTER_LINEAR) for x in [raw,flip,small]])
  normalised=np.stack([(x-np.median(x))/(np.quantile(x,.75)-np.quantile(x,.25)+1e-6) for x in priors]);assert np.isfinite(normalised).all()
  rgb=cv2.cvtColor(cv2.resize(im,size),cv2.COLOR_BGR2RGB)
  np.savez_compressed(out/f'view{i}.npz',normalised=normalised,priors=priors,rgb=rgb)
  Image.fromarray(rgb).save(out/f'view{i}_train.png')
  mask=(priors[0]<=np.quantile(priors[0],.05)).astype(np.float32);assert .04<=mask.mean()<=.06
  torch.save(torch.from_numpy(mask),masks/f'{Path(name).stem}.pt')
  rows.append(dict(image=name,size=size,far_fraction=float(mask.mean()),transforms=['original518','horizontal_flip518_unflipped','original392']))
 result=dict(state='complete',rows=rows,prior_checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),protocol='same trex frozen priors, normalisation, and farthest5percent masks; training images only')
 (out/'results.json').write_text(json.dumps(result,indent=2));(masks/'manifest.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)

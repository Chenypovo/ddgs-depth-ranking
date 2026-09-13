"""Reconstructed DAFE preprocessing, explicitly not author-provided masks."""
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root/'depth-anything-v2'))
from depth_anything_v2.dpt import DepthAnythingV2

scene = root/'dataset/nerf_llff_data/trex'
split = json.loads((scene/'3_views/split.json').read_text())
checkpoint = root/'depth-anything-v2/checkpoints/depth_anything_v2_vits.pth'
model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
model = model.cuda().eval()
output = root/'DDGS/preprocessed_masks_5/trex/r8'
output.mkdir(parents=True, exist_ok=True)
manifest = {'method': 'DepthAnything V2 Small frozen relative inverse depth; lowest 5 percent pixels at training resolution',
            'interpretation': 'Paper section 4.3 top 5 percent farthest, not equation 4 absolute maximum threshold',
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'images': {}}
for name in split['train_images']:
    bgr = cv2.imread(str(scene/'images'/name))
    h, w = bgr.shape[:2]
    size = (round(w/8), round(h/8))
    with torch.inference_mode():
        inverse_depth = model.infer_image(bgr, input_size=518)
    inverse_depth = cv2.resize(inverse_depth, size, interpolation=cv2.INTER_LINEAR)
    assert np.isfinite(inverse_depth).all()
    # Relative DepthAnything predicts larger values for nearer surfaces.
    threshold = float(np.quantile(inverse_depth, .05))
    mask = (inverse_depth <= threshold).astype(np.float32)
    assert .04 <= mask.mean() <= .06
    stem = Path(name).stem
    torch.save(torch.from_numpy(mask), output/f'{stem}.pt')
    rgb = cv2.cvtColor(cv2.resize(bgr, size), cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    overlay[mask.astype(bool)] = (.45*overlay[mask.astype(bool)] + .55*np.array([255,40,40])).astype(np.uint8)
    Image.fromarray(np.concatenate([rgb, overlay], axis=1)).save(output/f'{stem}_preview.jpg')
    manifest['images'][name] = {'size': size, 'far_fraction': float(mask.mean()), 'inverse_depth_threshold': threshold}
    print(name,manifest['images'][name],flush=True)
(output/'manifest.json').write_text(json.dumps(manifest,indent=2))

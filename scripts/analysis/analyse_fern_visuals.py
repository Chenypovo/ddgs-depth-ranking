"""All views and seeds; full-image contact sheets without favourable crop selection."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--backup',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.backup/'fern_runs/compare';a.out.mkdir(parents=True,exist_ok=True);rows=[]
for seed in range(3):
 for mode in ['off','ordinary']:
  for path in sorted((r/f'{mode}_seed{seed}/evaluation').glob('*_gt.png')):
   truth=np.array(Image.open(path),dtype=np.float32)/255;pred=np.array(Image.open(path.with_name(path.name.replace('_gt.png','_prediction.png'))),dtype=np.float32)/255
   gray=truth.mean(2);pad=np.pad(gray,1,mode='constant');gx=np.zeros_like(gray);gy=np.zeros_like(gray);k=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
   for i in range(3):
    for j in range(3):gx+=k[i,j]*pad[i:i+gray.shape[0],j:j+gray.shape[1]];gy+=k.T[i,j]*pad[i:i+gray.shape[0],j:j+gray.shape[1]]
   edge=np.sqrt(gx*gx+gy*gy);mask=edge>=np.quantile(edge,.8);err=np.abs(pred-truth).mean(2)
   rows.append(dict(seed=seed,mode=mode,image=path.name,mae=float(err.mean()),edge_mae=float(err[mask].mean())))
assert len(rows)==18
(a.out/'edge_slices.json').write_text(json.dumps(dict(rows=rows,means={m:{k:float(np.mean([x[k] for x in rows if x['mode']==m])) for k in ['mae','edge_mae']} for m in ['off','ordinary']},note='Secondary PNG diagnostic; 3views x3seeds; same constant-padding Sobel definition as the latest trex diagnostic; not a geometry metric.'),indent=2))
paths=sorted((r/'off_seed0/evaluation').glob('*_gt.png'));w,h=Image.open(paths[0]).size
canvas=Image.new('RGB',(w*3,(h+26)*3),'white');d=ImageDraw.Draw(canvas)
for i,path in enumerate(paths):
 for col,(mode,title) in enumerate([('gt','Ground truth'),('off','Original'),('ordinary','Depth ranking')]):
  f=path if mode=='gt' else r/f'{mode}_seed0/evaluation'/path.name.replace('_gt.png','_prediction.png')
  canvas.paste(Image.open(f),(col*w,i*(h+26)+26));d.text((col*w+5,i*(h+26)+5),f'View {i} | {title}',fill='black')
canvas.save(a.out/'all_views_seed0.png')

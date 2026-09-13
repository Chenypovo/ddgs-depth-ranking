"""Fixed seed0 crops plus all-seed/all-view PNG edge diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
rows=[]
for seed in range(3):
 for mode in ['off','ordinary','consensus']:
  folder=a.runs/f'{mode}_seed{seed}'/'evaluation'
  for gtpath in sorted(folder.glob('*_gt.png')):
   name=gtpath.name[:-7];truth=np.array(Image.open(gtpath),dtype=np.float32)/255;pred=np.array(Image.open(folder/f'{name}_prediction.png'),dtype=np.float32)/255
   gray=truth.mean(2);pad=np.pad(gray,1,mode='constant');gx=np.zeros_like(gray);gy=np.zeros_like(gray)
   k=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
   for i in range(3):
    for j in range(3):gx+=k[i,j]*pad[i:i+gray.shape[0],j:j+gray.shape[1]];gy+=k.T[i,j]*pad[i:i+gray.shape[0],j:j+gray.shape[1]]
   edge=np.sqrt(gx*gx+gy*gy);mask=edge>=np.quantile(edge,.8);err=np.abs(pred-truth).mean(2)
   rows.append(dict(seed=seed,mode=mode,image=name,edge_mae=float(err[mask].mean()),mae=float(err.mean())))
assert len(rows)==63
(a.out/'edge_slices.json').write_text(json.dumps(dict(rows=rows,means={m:{k:float(np.mean([r[k] for r in rows if r['mode']==m])) for k in ['mae','edge_mae']} for m in ['off','ordinary','consensus']},notes='PNG-quantised RGB diagnostic, GT top20% Sobel. All7views x3seeds. Same slice as initial failure report, secondary only.'),indent=2))
names=sorted(r['image'] for r in rows if r['seed']==0 and r['mode']=='off')
regions=[(1,(15,5,195,185)),(4,(0,180,270,378))]
fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained')
for ri,(vi,box) in enumerate(regions):
 name=names[vi]
 paths=[a.runs/'off_seed0/evaluation'/f'{name}_gt.png']+[a.runs/f'{m}_seed0/evaluation'/f'{name}_prediction.png' for m in ['off','ordinary','consensus']]
 for col,(path,title) in enumerate(zip(paths,['Ground truth','Original','Ordinary ranking','Consensus ranking'])):
  arr=np.asarray(Image.open(path));x0,y0,x1,y1=box;axes[ri,col].imshow(arr[y0:y1,x0:x1],interpolation='nearest');axes[ri,col].axis('off')
  if ri==0:axes[ri,col].set_title(title)
fig.suptitle('Trex | fixed seed 0 | same crops chosen before this training comparison',fontsize=12)
fig.savefig(a.out/'local_comparison.png',dpi=150);plt.close(fig)

"""Assemble reproducible scientific comparisons, without retouching pixels."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
root=Path(__file__).resolve().parents[2]/'reports/failure_analysis'
data=root/'failure_analysis'
diag=json.loads((data/'diagnosis.json').read_text())
mc=json.loads((data/'dropout_signal.json').read_text())
base={s:{k:np.mean([r[k] for r in diag['rows'] if r['split']=='development' and r['degree']==3 and r['seed']==s]) for k in ['psnr','ssim','lpips']} for s in range(3)}
new={s:json.loads((data/f'opacity085_depth_upstream_density_upstream_seed{s}'/'metrics.json').read_text()) for s in range(3)}
mapkey={'psnr':'psnr','ssim':'ssim','lpips':'lpips_vgg_standard_m11'}
means={name:{k:float(np.mean([vals(s,k) for s in range(3)])) for k in mapkey} for name,vals in [('baseline',lambda s,k:base[s][k]),('opacity085',lambda s,k:new[s]['mean'][mapkey[k]])]}
rows=[]
for s in range(3):
 rows.append(dict(seed=s,baseline=base[s],opacity085={k:new[s]['mean'][v] for k,v in mapkey.items()}))
summary=dict(mean=means,seeds=rows,lpips_relative_reduction_percent=100*(1-means['opacity085']['lpips']/means['baseline']['lpips']))
(root/'summary.json').write_text(json.dumps(summary,indent=2))
# Fixed illustrative regions from initial baseline inspection, not selected by gain.
regions=[(1,'Head and ceiling: spurious contours',(15,5,195,185)),(4,'Railings: blurred and duplicated lines',(0,180,270,378))]
fig,axes=plt.subplots(2,3,figsize=(12,7.5),layout='constrained')
for row,(v,label,box) in enumerate(regions):
 name=[r['image'] for r in diag['rows'] if r['seed']==0 and r['split']=='development' and r['degree']==3][v]
 paths=[data/f'development_{v}_gt.png',data/f'development_{v}_sh3.png',data/'opacity085_depth_upstream_density_upstream_seed0'/f'{name}_prediction.png']
 for col,(p,title) in enumerate(zip(paths,['Ground truth','Original reconstruction','Opacity x 0.85'])):
  arr=np.asarray(Image.open(p));x0,y0,x1,y1=box
  axes[row,col].imshow(arr[y0:y1,x0:x1],interpolation='nearest');axes[row,col].axis('off')
  axes[row,col].set_title(title if row==0 else '',fontsize=12)
 axes[row,0].text(0,-.06,label,transform=axes[row,0].transAxes,fontsize=10,va='top')
fig.suptitle('Trex development scene | same seed 0 checkpoint | improvement is modest; artifacts remain',fontsize=12)
fig.savefig(root/'local_comparison.png',dpi=160);plt.close(fig)
# All-view error/uncertainty illustration for fixed view 1.
v=1
imgs=[Image.open(data/f'development_{v}_gt.png'),Image.open(data/f'development_{v}_sh3.png')]
gt,pred=[np.asarray(x)/255 for x in imgs];err=np.abs(gt-pred).mean(-1)
unc=np.asarray(Image.open(data/f'development_{v}_uncertainty.png'))[:,:,0]/255
fig,ax=plt.subplots(1,4,figsize=(16,4),layout='constrained')
for a,x,t in zip(ax,[gt,pred,err,unc],['Ground truth','Baseline','Absolute RGB error','Dropout sensitivity']):
 a.imshow(x,**({'cmap':'magma','vmin':0,'vmax':.25 if t=='Absolute RGB error' else 1} if x.ndim==2 else {}));a.set_title(t);a.axis('off')
fig.savefig(root/'failure_map.png',dpi=140);plt.close(fig)
print(json.dumps(summary,indent=2))

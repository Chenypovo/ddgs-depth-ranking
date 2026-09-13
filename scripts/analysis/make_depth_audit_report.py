from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]/'reports/depth_reliability'
data=root/'depth_reliability'
j=json.loads((data/'results.json').read_text())
fig,axes=plt.subplots(3,4,figsize=(15,9),layout='constrained')
for i in range(3):
 d=np.load(data/f'view{i}.npz')
 images=[d['rgb'],d['priors'][0],1/np.maximum(d['rendered_z'],1e-6),d['disagreement']]
 names=['Training image','DepthAnything: brighter = nearer','GS expected depth: brighter = nearer','Change under flip / resize']
 for a,img,title in zip(axes[i],images,names):
  if img.ndim==3:a.imshow(img)
  else:a.imshow(img,cmap='magma',vmin=np.quantile(img,.02),vmax=np.quantile(img,.98))
  a.axis('off')
  if i==0:a.set_title(title,fontsize=10)
 axes[i,0].text(0,-.04,f'Training view {i+1}',transform=axes[i,0].transAxes)
fig.suptitle('Training-only audit | depth maps use separate display scales; do not compare colours as metric distances',fontsize=12)
fig.savefig(root/'depth_overview.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for i,a in enumerate(axes):
 d=np.load(data/f'view{i}.npz');ch=d['check'];xy=d['xy'][ch];err=d['reference_relative_inverse_depth_error'][ch]
 a.imshow(d['rgb']);sc=a.scatter(xy[:,0],xy[:,1],c=err,s=5,cmap='turbo',vmin=0,vmax=.15);a.axis('off');a.set_title(f'View {i+1}: checked sparse references',fontsize=11)
fig.colorbar(sc,ax=axes,label='Relative inverse-depth error against triangulation',shrink=.7)
fig.savefig(root/'sparse_reference_errors.png',dpi=150);plt.close(fig)
summary={'seconds':j['seconds'],'views':[]}
for r in j['rows']:
 summary['views'].append({k:r[k] for k in ['local_rank_accuracy','augmentation_agreement_coverage','augmentation_agreement_rank_accuracy','stable_half_check_median_error','unstable_half_check_median_error','instability_error_spearman']})
(root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

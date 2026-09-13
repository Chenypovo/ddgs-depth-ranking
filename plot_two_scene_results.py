"""Plot all three paired seeds per scene, without best-run selection."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--fern',type=Path,required=True);p.add_argument('--trex',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
for ax,name,path in zip(axes,['Trex (development)','Fern (fixed settings)'],[a.trex,a.fern]):
 j=json.loads(path.read_text());rr=j['runs'];values=[]
 for seed in range(3):
  pair=[next(x for x in rr if x['seed']==seed and x['mode']==m)['metrics']['1.0']['lpips_vgg_standard_m11'] for m in ['off','ordinary']]
  values.append(pair);ax.plot([0,1],pair,'o-',alpha=.7,label=f'Seed {seed}',linewidth=1.5)
 values=np.array(values);ax.plot([0,1],values.mean(0),'kD--',linewidth=2,label='Mean');ax.set_xticks([0,1],['Original','Depth ranking']);ax.set_ylabel('VGG LPIPS (lower is better)');ax.set_title(name);ax.grid(axis='y',alpha=.2);ax.set_xlim(-.2,1.2)
axes[1].legend(fontsize=8);fig.suptitle('D²GS | 3 input views | 10,000 steps | all paired seeds',fontsize=12)
fig.savefig(a.out,dpi=160);plt.close(fig)

"""Matched-count ordinary/augmentation-consensus ordinal supervision.

Teacher outputs are frozen training-only caches. No global RNG is consumed.
"""
import json
from pathlib import Path
import numpy as np
import torch


def select_pairs(priors, seed, iteration, mode, maximum=4096, pool=16384):
    if mode not in ('ordinary','consensus'):
        raise ValueError(mode)
    _,h,w=priors.shape
    gen=torch.Generator(device=priors.device).manual_seed(1000003*seed+iteration+7919)
    xy=torch.stack([torch.randint(w,(pool,),device=priors.device,generator=gen),torch.randint(h,(pool,),device=priors.device,generator=gen)],1)
    delta=torch.randint(-64,65,(pool,2),device=priors.device,generator=gen)
    other=xy+delta
    distance2=(delta*delta).sum(1)
    valid=(distance2>=64)&(distance2<=4096)&(other[:,0]>=0)&(other[:,0]<w)&(other[:,1]>=0)&(other[:,1]<h)
    xy,other=xy[valid],other[valid]
    a=xy[:,1]*w+xy[:,0];b=other[:,1]*w+other[:,0]
    diff=priors.flatten(1)[:,a]-priors.flatten(1)[:,b]
    reliable=(diff[0].abs()>=.05)&torch.isfinite(diff).all(0)
    a,b,diff=a[reliable],b[reliable],diff[:,reliable]
    sign=diff.sign()
    agree=(sign==sign[:1]).all(0)&(sign[0]!=0)
    indices=torch.nonzero(agree,as_tuple=False).flatten()
    count=min(maximum,indices.numel())
    chosen=torch.arange(count,device=priors.device) if mode=='ordinary' else indices[:count]
    return a[chosen],b[chosen],sign[0,chosen],dict(pool_valid=a.numel(),agreed=indices.numel(),selected=count)


def render_inverse_depth(camera,gaussians,pipe,renderer):
    xyz=gaussians.get_xyz
    homogeneous=torch.cat([xyz,torch.ones_like(xyz[:,:1])],1)
    z=(homogeneous@camera.world_view_transform)[:,2].clamp_min(0)
    scale=z.detach().max().clamp_min(1e-6)
    # Packed channels: depth numerator / scale, alpha, unused zero.
    colours=torch.stack([z/scale,torch.ones_like(z),torch.zeros_like(z)],1)
    package=renderer(camera,gaussians,pipe,torch.zeros(3,device=xyz.device),override_color=colours)
    depth_sum=package['render'][0]*scale
    alpha=package['render'][1]
    inverse=alpha/depth_sum.clamp_min(1e-6)
    valid=(alpha.detach()>1e-4)&(depth_sum.detach()>1e-6)&torch.isfinite(inverse.detach())
    safe=torch.where(valid,inverse,torch.zeros_like(inverse))
    return safe,valid,package


def ordinal_loss(inverse,valid,a,b,sign):
    if a.numel()==0:return inverse.sum()*0,dict(active=0,violated=0,iqr=0.)
    values=inverse.detach()[valid]
    if values.numel()<2:return inverse.sum()*0,dict(active=0,violated=0,iqr=0.)
    quartiles=torch.quantile(values,torch.tensor([.25,.75],device=values.device))
    iqr=(quartiles[1]-quartiles[0]).clamp_min(1e-6)
    flat=inverse.flatten();good=valid.flatten()[a]&valid.flatten()[b]
    residual=.05-sign*(flat[a]-flat[b])/iqr
    loss=(residual.relu()*good).sum()/a.numel()
    return loss,dict(active=int(good.sum()),violated=int(((residual>0)&good).sum()),iqr=float(iqr))


class RankSupervisor:
    def __init__(self,cache,mode,seed,log):
        self.mode,self.seed=mode,seed
        self.priors={}
        cache=Path(cache)
        rows=json.loads((cache/'results.json').read_text())['rows']
        for i,row in enumerate(rows):
            array=np.load(cache/f'view{i}.npz')['normalised']
            assert array.shape==(3,378,504) and np.isfinite(array).all()
            self.priors[Path(row['image']).stem]=torch.from_numpy(array).float().cuda()
        self.file=Path(log).open('x',buffering=1)

    def loss(self,cam,g,pipe,renderer,iteration):
        weight=.01*min(1.,max(0.,(iteration-1000)/2000))
        if weight==0:return g.get_xyz.new_zeros(())
        a,b,sign,stats=select_pairs(self.priors[cam.image_name],self.seed,iteration,self.mode)
        inverse,valid,_=render_inverse_depth(cam,g,pipe,renderer)
        loss,extra=ordinal_loss(inverse,valid,a,b,sign)
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite rank loss')
        row=dict(iteration=iteration,camera=cam.image_name,mode=self.mode,weight=weight,loss=float(loss.detach()),**stats,**extra)
        self.file.write(json.dumps(row)+'\n')
        return weight*loss

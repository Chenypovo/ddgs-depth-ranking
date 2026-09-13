"""Check marginal probabilities, local count variance, and spatial ordering."""
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'DDGS'))
from utils.spatial_dropout import morton_order,stratified_keep_mask
torch.manual_seed(712)
xyz=torch.rand(64,3,device='cuda')
order=morton_order(xyz)
assert torch.equal(order.sort().values,torch.arange(64,device='cuda'))
keep=torch.linspace(.5,.99,64,device='cuda')
sample=torch.stack([stratified_keep_mask(keep,order) for _ in range(4000)])
error=(sample.mean(0)-keep).abs().max().item()
assert error<.035,error
uniform=torch.full((64,),.8,device='cuda')
strat=torch.stack([stratified_keep_mask(uniform,order)[order].reshape(-1,8).sum(-1) for _ in range(2000)])
ind=(torch.rand(2000,8,8,device='cuda')<.8).float().sum(-1)
sv=strat.var(dim=0).mean().item();iv=ind.var(dim=0).mean().item()
assert sv<iv*.4,(sv,iv)
result={'state':'integration_pass','maximum_marginal_error':error,'stratified_local_count_variance':sv,
        'independent_local_count_variance':iv,'note':'Sampling property only, not a reconstruction improvement'}
print(json.dumps(result,indent=2))

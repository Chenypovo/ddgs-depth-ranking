"""Create an isolated source snapshot with one opt-in loss insertion."""
import hashlib,json,shutil
from pathlib import Path
root=Path(__file__).resolve().parent
src=root/'DDGS_corrections';dst=root/'DDGS_rank'
assert not dst.exists()
shutil.copytree(src,dst)
target=dst/'train.py';original=target.read_text()
anchor='    gaussians.training_setup(opt)\n'
assert original.count(anchor)==1
s=original.replace(anchor,anchor+'''    rank_supervisor = None
    rank_mode = os.environ.get('DDGS_RANK_MODE', 'off')
    if rank_mode != 'off':
        from utils.rank_supervision import RankSupervisor
        rank_supervisor = RankSupervisor(os.environ['DDGS_RANK_CACHE'], rank_mode,
            int(os.environ.get('DDGS_SEED', '0')), os.path.join(dataset.model_path, 'rank_stats.jsonl'))
''')
anchor='        loss.backward()\n'
assert s.count(anchor)==1
s=s.replace(anchor,'''        if rank_supervisor is not None:
            loss = loss + rank_supervisor.loss(viewpoint_cam, gaussians, pipe, render, iteration)
'''+anchor)
anchor='    print("\\nTraining complete.")'
assert s.count(anchor)==1
s=s.replace(anchor,'''    import json
    with open(os.path.join(args.model_path, 'memory.json'), 'w') as f:
        json.dump({'peak_allocated_mb': torch.cuda.max_memory_allocated()/2**20,
                   'peak_reserved_mb': torch.cuda.max_memory_reserved()/2**20}, f)
'''+anchor)
target.write_text(s)
shutil.copy2(root/'rank_supervision.py',dst/'utils/rank_supervision.py')
(root/'rank_source_identity.json').write_text(json.dumps({'source_train_sha256':hashlib.sha256(original.encode()).hexdigest(),'rank_train_sha256':hashlib.sha256(s.encode()).hexdigest(),'insertion':'opt-in ordinal loss before backward; densification screen-space statistics remain from original RGB pass; defaults unchanged'},indent=2))
print('Isolated DDGS_rank source prepared')

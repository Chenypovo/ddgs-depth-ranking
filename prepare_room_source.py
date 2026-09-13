"""Isolate room validation; preserve frozen trex method and evaluator math."""
import hashlib,json,shutil
from pathlib import Path
r=Path(__file__).resolve().parent;src=r/'DDGS_rank';dst=r/'DDGS_room';assert not dst.exists()
shutil.copytree(src,dst)
p=dst/'utils/rank_supervision.py';old=p.read_text();new=old.replace('array.shape==(3,378,504)','array.ndim==3 and array.shape[0]==3 and min(array.shape[1:])>=128');assert new!=old;p.write_text(new)
# Only the scene-dependent shape guard is generalised. Loss/sampling are unchanged.
(r/'room_source_identity.json').write_text(json.dumps({'parent':'DDGS_rank','train_sha256':hashlib.sha256((dst/'train.py').read_bytes()).hexdigest(),'old_rank_sha256':hashlib.sha256(old.encode()).hexdigest(),'new_rank_sha256':hashlib.sha256(new.encode()).hexdigest(),'change':'shape assertion only; all train/loss/sampling hyperparameters frozen'},indent=2))
print('Room source prepared')

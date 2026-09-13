"""Archive completed fern comparison and pre-training room failure provenance."""
import hashlib,json,tarfile
from pathlib import Path
r=Path(__file__).resolve().parents[2];assert json.loads((r/'fern_runs/compare/results.json').read_text())['state']=='complete'
archive=r/'fern_completed.tar.gz';assert not archive.exists()
data_files=sorted(p for p in (r/'dataset/nerf_llff_data/fern/images').glob('*') if p.is_file())
data_files+=sorted(p for p in (r/'dataset/nerf_llff_data/fern/sparse/0').glob('*') if p.is_file())
data_files += [r/'dataset/nerf_llff_data/fern/poses_bounds.npy']
identity={str(p.relative_to(r)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in data_files}
(r/'fern_data_identity.json').write_text(json.dumps(identity,indent=2))
names=['fern_data_identity.json','downloads/source.txt','scripts/archive/package_fern_results.py','fern_runs','DDGS_fern','fern_priors','fern_integration.json','fern_integration','fern_source_identity.json','scripts/setup/prepare_fern_source.py','scripts/data/prepare_fern_priors.py','scripts/checks/check_fern_supervision.py','scripts/evaluate/evaluate_fern.py','scripts/train/run_fern_comparison.py','methods/opacity_expectation.py','scripts/data/prepare_scene.py','FERN_VALIDATION_PROTOCOL.md','fern_prepare.log','fern_runner.log','dataset/nerf_llff_data/fern/3_views/split.json','dataset/nerf_llff_data/fern/3_views/dense/fused.ply','dataset/nerf_llff_data/fern/3_views/created','dataset/nerf_llff_data/fern/3_views/triangulated','room_prepare.log','room_fusion_repair.log','scripts/data/repair_room_fusion.py','ROOM_VALIDATION_PROTOCOL.md','dataset/nerf_llff_data/room/3_views/split.json','dataset/nerf_llff_data/room/3_views/dense/fused_default5_failed.ply','dataset/nerf_llff_data/room/3_views/dense/fused_min3.ply']
files=[]
for n in names:
 p=r/n;assert p.exists(),p;files.extend([x for x in p.rglob('*') if x.is_file()] if p.is_dir() else [p])
files=sorted(set(p for p in files if '.git' not in p.parts and '__pycache__' not in p.parts))
manifest={}
for p in files:
 assert not p.is_symlink();manifest[str(p.relative_to(r))]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
mp=r/'fern_archive_manifest.json';mp.write_text(json.dumps(manifest,indent=2))
with tarfile.open(archive,'w:gz') as t:
 for p in [*files,mp]:t.add(p,arcname=str(p.relative_to(r)),recursive=False)
receipt=dict(archive=archive.name,bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files))
(r/'fern_archive_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)

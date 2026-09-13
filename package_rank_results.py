"""Archive a completed cohort, with a per-file manifest and archive checksum."""
import hashlib
import json
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parent
results = root / 'rank_runs/compare/results.json'
assert json.loads(results.read_text())['state'] == 'complete'
archive = root / 'rank_completed.tar.gz'
assert not archive.exists(), 'Preserve previous exports'
entries = [root / p for p in [
    'rank_runs/compare', 'rank_runs/pilot', 'DDGS_rank',
    'rank_source_identity.json', 'rank_integration.json',
    'rank_supervision.py', 'prepare_rank_source.py', 'check_rank_supervision.py',
    'evaluate_rank.py', 'run_rank_comparison.py', 'opacity_expectation.py',
    'RANK_COMPARISON_PROTOCOL.md', 'depth_reliability',
    'dataset/nerf_llff_data/trex/3_views/split.json',
    'dataset/nerf_llff_data/trex/3_views/dense/fused.ply',
]]
files = []
for entry in entries:
    assert entry.exists(), entry
    files.extend([p for p in entry.rglob('*') if p.is_file()] if entry.is_dir() else [entry])
files = sorted(set(files))
files = [p for p in files if '.git' not in p.parts and '__pycache__' not in p.parts]
manifest = {}
for path in files:
    assert not path.is_symlink(), path
    manifest[str(path.relative_to(root))] = dict(
        bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
manifest_path = root / 'rank_archive_manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2))
with tarfile.open(archive, 'w:gz') as tar:
    for path in [*files, manifest_path]:
        tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
receipt = dict(archive=archive.name, bytes=archive.stat().st_size,
               sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
               files=len(files), state='complete')
(root / 'rank_archive_receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)

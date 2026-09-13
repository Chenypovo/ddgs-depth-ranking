"""Materialise the pinned upstream code and verify the experiment-control patch.

This downloads source code only. It does not install packages or launch training.
--local-upstream uses committed git objects for an offline reconstruction check.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--local-upstream', type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parents[2]
    identity = json.loads((here / 'patches/source_identity.json').read_text())
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = root / 'DDGS'
    controls = root / 'DDGS_corrections'
    if source.exists() or controls.exists():
        raise FileExistsError('Refusing to overwrite existing source snapshots')
    if args.local_upstream:
        with tempfile.TemporaryFile() as archive:
            subprocess.run(['git', '-C', str(args.local_upstream.resolve()), 'archive', identity['commit']], stdout=archive, check=True)
            archive.seek(0)
            source.mkdir()
            with tarfile.open(fileobj=archive) as tar:
                for item in tar.getmembers():
                    path = Path(item.name)
                    if path.is_absolute() or '..' in path.parts or not (item.isfile() or item.isdir()):
                        raise ValueError(f'Unsupported archive entry: {item.name}')
                tar.extractall(source)
    else:
        subprocess.run(['git', 'clone', '--no-checkout', identity['upstream'], str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', identity['commit']], check=True)
    patch = here / 'patches/ddgs-experiment-controls.patch'
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=source, check=True)
    subprocess.run(['git', 'apply', str(patch)], cwd=source, check=True)
    for name, expected in identity['patched_files'].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source hash mismatch: {name}')
    if hashlib.sha256((source / 'train.py').read_bytes()).hexdigest() != identity['train_sha256']:
        raise ValueError('Unexpected upstream training source')
    shutil.copytree(source, controls, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    receipt = dict(state='verified', upstream=identity['upstream'], commit=identity['commit'],
                   checked_files=len(identity['patched_files']) + 1,
                   scope='Source reconstruction only; no fresh GPU validation')
    (root / 'source-bootstrap.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

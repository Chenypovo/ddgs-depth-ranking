"""Extract one scene from an existing LLFF archive, without downloading data."""
import argparse
from pathlib import Path
import stat
import zipfile

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene',choices=['trex','fern','room'],required=True)
p.add_argument('--archive',type=Path,default=Path('downloads/nerf_llff_data.zip'))
p.add_argument('--output',type=Path,default=Path('dataset'))
a=p.parse_args();root=a.output.resolve();target=root/'nerf_llff_data'/a.scene
if target.exists():raise FileExistsError(target)
with zipfile.ZipFile(a.archive) as z:
    members=[m for m in z.infolist() if m.filename.startswith(f'nerf_llff_data/{a.scene}/')]
    if not members or sum(m.file_size for m in members)>12*1024**3:raise ValueError('Unexpected scene archive')
    for m in members:
        path=(root/m.filename).resolve()
        if root not in path.parents or stat.S_ISLNK(m.external_attr>>16):raise ValueError(m.filename)
    z.extractall(root,members)
print(target)

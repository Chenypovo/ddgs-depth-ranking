"""Resume the identified COLMAP 4.2 output-format failure without recomputing depth."""
import hashlib
import json
from pathlib import Path
import time
import pycolmap as pc
from plyfile import PlyData

work=Path(__file__).resolve().parent/'dataset/nerf_llff_data/trex/3_views'
manifest=json.loads((work/'split.json').read_text())
assert {p.name for p in (work/'images').iterdir()} == set(manifest['train_images'])
assert not set(manifest['train_images']) & set(manifest['test_images'])
for name in manifest['train_images']:
    assert (work/'dense/stereo/depth_maps'/f'{name}.geometric.bin').is_file()
pc.stereo_fusion(work/'dense/fused.ply',work/'dense',output_type='ply',
                options=pc.StereoFusionOptions(num_threads=12,cache_size=4))
ply=work/'dense/fused.ply'
vertices=PlyData.read(ply)['vertex']
assert len(vertices)>100
manifest.update(state='complete',finished=time.time(),
    sparse_points=pc.Reconstruction(work/'triangulated').num_points3D(),dense_points=len(vertices),
    ply_sha256=hashlib.sha256(ply.read_bytes()).hexdigest(),
    compatibility_note='Explicit output_type=ply required by pycolmap 4.2; resumed fusion only after first output format failure')
(work/'split.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2),flush=True)

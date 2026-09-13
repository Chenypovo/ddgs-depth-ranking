"""Scoped LLFF preprocessing; never use held-out pixels to build the point cloud."""
import argparse
import hashlib
import json
import pathlib
import shutil
import time

import numpy as np
import pycolmap as pc

p = argparse.ArgumentParser()
p.add_argument('--scene', type=pathlib.Path, required=True)
args = p.parse_args()
scene = args.scene.resolve()
work = scene / '3_views'
work.mkdir(exist_ok=True)
source = pc.Reconstruction(scene / 'sparse/0')
images = sorted(source.images.values(), key=lambda x: x.name)
pool = [im for i, im in enumerate(images) if i % 8 != 0]
selected = [pool[round(i)] for i in np.linspace(0, len(pool)-1, 3)]
train_names = [im.name for im in selected]
test_names = [im.name for i, im in enumerate(images) if i % 8 == 0]
assert not set(train_names) & set(test_names)
manifest = {'train_images': train_names, 'test_images': test_names,
            'all_images': len(images), 'protocol': 'every eighth held out; 3 evenly spaced remaining views',
            'pycolmap': pc.__version__, 'source_camera_poses': 'LLFF provided poses',
            'point_cloud_source': 'three training images ONLY; no full-view points copied',
            'state': 'preprocessing', 'started': time.time()}
(work/'split.json').write_text(json.dumps(manifest, indent=2))
for folder in ['images', 'created', 'triangulated']:
    (work/folder).mkdir(exist_ok=True)
for im in selected:
    dest = work/'images'/im.name
    if not dest.exists():
        shutil.copy2(scene/'images'/im.name, dest)
assert {x.name for x in (work/'images').iterdir()} == set(train_names)
db_path = work/'database.db'
camera_ids = {im.camera_id for im in selected}
assert len(camera_ids) == 1, 'This scoped adapter expects the LLFF single camera'
camera = source.cameras[next(iter(camera_ids))]
reader = pc.ImageReaderOptions(camera_model=camera.model_name,
    camera_params=','.join(str(v) for v in camera.params))
extraction = pc.FeatureExtractionOptions(max_image_size=4032, num_threads=12, use_gpu=False)
extraction.sift.max_num_features = 32768
extraction.sift.estimate_affine_shape = True
extraction.sift.domain_size_pooling = True
pc.extract_features(db_path, work/'images', image_names=train_names,
    camera_mode=pc.CameraMode.SINGLE, reader_options=reader,
    extraction_options=extraction, device=pc.Device.cpu)
matching = pc.FeatureMatchingOptions(guided_matching=True, max_num_matches=32768, num_threads=12)
pc.match_exhaustive(db_path, matching_options=matching, device=pc.Device.cuda)
with pc.Database.open(db_path) as db:
    db_images = db.read_all_images()
assert {im.name for im in db_images} == set(train_names)
by_name = {im.name: im for im in selected}
db_camera_ids = {im.camera_id for im in db_images}
assert len(db_camera_ids) == 1
db_camera_id = next(iter(db_camera_ids))
created = work/'created'
(created/'cameras.txt').write_text(f'{db_camera_id} {camera.model_name} {camera.width} {camera.height} ' +
                                  ' '.join(str(v) for v in camera.params) + '\n')
with (created/'images.txt').open('w') as f:
    for im in sorted(db_images, key=lambda x: x.image_id):
        pose = by_name[im.name].cam_from_world()
        q = pose.rotation.quat  # xyzw -> COLMAP text wxyz
        values = [q[3], q[0], q[1], q[2], *pose.translation]
        f.write(f'{im.image_id} ' + ' '.join(str(v) for v in values) + f' {db_camera_id} {im.name}\n\n')
(created/'points3D.txt').write_text('')
reference = pc.Reconstruction(created)
assert reference.num_points3D() == 0
assert reference.num_reg_images() == 3
options = pc.IncrementalPipelineOptions(num_threads=12, random_seed=0,
    ba_local_max_num_iterations=40, ba_local_max_refinements=3, ba_global_max_num_iterations=100)
triangulated = pc.triangulate_points(reference, db_path, work/'images', work/'triangulated', options=options)
assert triangulated.num_points3D() > 100, 'Insufficient triangulated training-only points'
pc.undistort_images(work/'dense', work/'triangulated', work/'images', num_threads=12)
pc.patch_match_stereo(work/'dense', options=pc.PatchMatchOptions(gpu_index='0', cache_size=4, num_threads=8))
pc.stereo_fusion(work/'dense/fused.ply', work/'dense', output_type='ply', options=pc.StereoFusionOptions(num_threads=12, cache_size=4))
from plyfile import PlyData
ply = work/'dense/fused.ply'
vertices = PlyData.read(ply)['vertex']
assert len(vertices) > 100, 'Insufficient dense training-only points'
manifest.update(state='complete', finished=time.time(), sparse_points=triangulated.num_points3D(),
                dense_points=len(vertices), ply_sha256=hashlib.sha256(ply.read_bytes()).hexdigest())
(work/'split.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2), flush=True)

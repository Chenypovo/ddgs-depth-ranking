"""One recorded preprocessing repair before any room training/metrics."""
import hashlib,json,shutil,time
from pathlib import Path
import pycolmap as pc
from plyfile import PlyData
r=Path(__file__).resolve().parents[2];p=r/'dataset/nerf_llff_data/room/3_views';s=json.loads((p/'split.json').read_text());assert s['state']=='preprocessing'
old=p/'dense/fused.ply';assert len(PlyData.read(old)['vertex'])==14
shutil.copy2(old,p/'dense/fused_default5_failed.ply')
output=p/'dense/fused_min3.ply';assert not output.exists()
opts=pc.StereoFusionOptions(num_threads=12,cache_size=4,min_num_pixels=3)
pc.stereo_fusion(output,p/'dense',output_type='ply',options=opts)
count=len(PlyData.read(output)['vertex']);print('repaired_count',count,flush=True)
assert count>100,'Single repair failed; do not tune further'
shutil.copy2(output,old);ref=pc.Reconstruction(p/'triangulated')
s.update(state='complete',finished=time.time(),sparse_points=ref.num_points3D(),dense_points=count,ply_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),preprocessing_deviation={'failed_default_fusion_points':14,'min_num_pixels_original':5,'min_num_pixels_used':3,'reason':'insufficient points under default fusion before any training or validation metrics','same_initialisation_for_both_arms':True})
(p/'split.json').write_text(json.dumps(s,indent=2));print(json.dumps(s),flush=True)

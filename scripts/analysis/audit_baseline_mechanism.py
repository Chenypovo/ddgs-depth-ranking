"""CPU-only audit of frozen sources and saved PLYs; does not change training.

Synthetic checks establish implementation behaviour, not an image-quality gain.
Final-checkpoint neighbourhood measurements are posthoc, not training telemetry.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial import cKDTree
import torch

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / 'vendor/DDGS' if (ROOT / 'vendor/DDGS').exists() else ROOT / 'DDGS'
OUT = ROOT / 'reports/audit'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def upstream(path):
    return subprocess.check_output(['git', '-C', str(VENDOR), 'show', f'HEAD:{path}'], text=True)


def camera_check():
    graphics = module(VENDOR / 'utils/graphics_utils.py', 'audit_graphics')
    angle = np.pi / 3
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    matrix = torch.tensor(graphics.getWorld2View2(rotation.T, np.array([1., -2., 4.]))).T
    rng = np.random.default_rng(123)
    camera = np.column_stack([rng.uniform(-3, 3, (1000, 2)), rng.uniform(1, 10, 1000), np.ones(1000)])
    points = torch.tensor(camera, dtype=torch.float32) @ torch.linalg.inv(matrix)
    expected = torch.tensor(camera[:, 2], dtype=torch.float32)
    official = (points @ matrix.T)[:, 2]
    corrected = (points @ matrix)[:, 2]
    # Literal flattened-memory indexing from CUDA auxiliary.h transformPoint4x3.
    flat = matrix.contiguous().flatten()
    renderer = points[:, 0]*flat[2] + points[:, 1]*flat[6] + points[:, 2]*flat[10] + flat[14]
    assert torch.allclose(corrected, expected, atol=3e-6)
    assert torch.allclose(corrected, renderer, atol=3e-6)
    assert not torch.allclose(official, renderer)
    def bands(z):
        ordered = z.sort().values
        return (z > ordered[int(len(z)*.33)]).int() + (z > ordered[int(len(z)*.67)]).int()
    source = upstream('gaussian_renderer/__init__.py')
    assert 'torch.matmul(gaussian_positions_homo, viewpoint_camera.world_view_transform.T)' in source
    return {'upstream_contains_extra_transpose': True,
            'synthetic_points': len(points),
            'official_depth_mae': float((official-expected).abs().mean()),
            'corrected_depth_max_error': float((corrected-expected).abs().max()),
            'synthetic_band_disagreement': float((bands(official)!=bands(corrected)).float().mean()),
            'meaning': 'Proves convention mismatch on a known rotated camera. Not the trex error rate.'}


def density_check():
    source = upstream('scene/gaussian_model.py')
    prune = source.split('    def prune_points(')[1].split('\n    def ')[0]
    append = source.split('    def densification_postfix(')[1].split('\n    def ')[0]
    assert 'density_score' not in prune and 'density_score' not in append
    renderer = upstream('gaussian_renderer/__init__.py')
    assert 'if pc.density_score.numel() >= opacity.shape[0]:' in renderer
    assert 'density_norm = torch.ones_like(depth_score) * 0.5' in renderer
    # Distinct values denote cached scores attached to distinct point IDs.
    ids = np.array([0, 1, 2, 3])
    cache = np.array([.1, .2, .3, .4])
    valid = np.array([True, False, True, True])
    retained = ids[valid]
    used = cache[:len(retained)]
    expected = cache[valid]
    assert not np.array_equal(used, expected)
    return {'upstream_prune_does_not_remap_density': True,
            'upstream_append_does_not_extend_density': True,
            'prune_example_retained_ids': retained.tolist(),
            'prune_example_used_scores': used.tolist(),
            'prune_example_expected_scores': expected.tolist(),
            'growth_branch': 'When current N exceeds cache N, all density_norm values become 0.5.',
            'meaning': 'Source-confirmed lifecycle issue; frequency and quality effect in recorded runs unknown.'}


def mask_reduction_check():
    losses = module(VENDOR/'utils/loss_utils.py', 'audit_losses')
    mask = torch.zeros(10, 10)
    mask.flatten()[:5] = 1
    error = torch.ones(3, 10, 10)
    full_mean = losses.l1_loss(error*mask, torch.zeros_like(error))
    masked_rgb_mean = (error*mask).sum()/(3*mask.sum())
    assert abs(float(full_mean/masked_rgb_mean)-.05) < 1e-6
    return {'mask_area_fraction': .05, 'full_image_mean': float(full_mean),
            'masked_pixel_rgb_mean': float(masked_rgb_mean),
            'meaning': 'Same channel averaging in both expressions: full-image reduction is 0.05 times masked reduction. Paper RGB-norm convention may introduce an additional channel factor. This is not evidence that changing the loss improves quality.'}


def read_xyz(path):
    # Saved Gaussian PLY schema: one vertex element with scalar float32 fields.
    with path.open('rb') as f:
        assert f.readline().strip() == b'ply'
        assert f.readline().strip() == b'format binary_little_endian 1.0'
        count, fields = None, []
        while True:
            line = f.readline().decode('ascii').strip()
            if line == 'end_header':
                break
            if line.startswith('element vertex '):
                count = int(line.split()[-1])
            elif line.startswith('property '):
                _, kind, name = line.split()
                assert kind == 'float'
                fields.append((name, '<f4'))
            else:
                raise ValueError(f'Unsupported PLY header: {line}')
        data = np.fromfile(f, dtype=np.dtype(fields), count=count)
    assert len(data) == count
    return np.column_stack([data[name] for name in ('x', 'y', 'z')])


def grouping_check():
    spatial = module(VENDOR / 'utils/spatial_dropout.py', 'audit_spatial')
    records = []
    for path in sorted((ROOT/'reports/comparison/runs').glob('*/point_cloud/iteration_10000/point_cloud.ply')):
        xyz = read_xyz(path)
        order = spatial.morton_order(torch.from_numpy(xyz)).numpy()
        assert np.array_equal(np.sort(order), np.arange(len(xyz)))
        # Full groups only: tail of at most seven points excluded explicitly.
        groups = order[:len(order)//8*8].reshape(-1, 8)
        rng = np.random.default_rng(713)
        chosen = rng.choice(len(groups), min(512, len(groups)), replace=False)
        sample = groups[chosen]
        distances, nearest = cKDTree(xyz).query(xyz[sample].reshape(-1, 3), k=8)
        neighbours = nearest[:, 1:].reshape(-1, 8, 7)
        group_distance = np.linalg.norm(xyz[sample][:, :, None]-xyz[sample][:, None, :], axis=-1)
        farthest = group_distance.max(-1).reshape(-1)
        nearest_radius = np.maximum(distances[:, -1], 1e-12)
        overlaps = []
        for ids, neighbours_for_group in zip(sample, neighbours):
            for own_id, nn in zip(ids, neighbours_for_group):
                overlaps.append(len((set(ids.tolist())-{int(own_id)}) & set(nn.tolist()))/7)
        ratio = farthest / nearest_radius
        records.append({'run': path.parents[2].name, 'points': len(xyz),
                        'sampled_groups': len(sample), 'excluded_tail_points': len(xyz)%8,
                        'same_group_recall_of_seven_nearest_mean': float(np.mean(overlaps)),
                        'group_farthest_over_seventh_neighbour_median': float(np.median(ratio)),
                        'group_farthest_over_seventh_neighbour_p95': float(np.quantile(ratio, .95)),
                        'meaning': 'Fresh Morton groups at final checkpoint. Geometric proxy, not semantic correctness or historical cache state.'})
    return records


def main():
    torch.set_num_threads(2)
    result = {'evidence': 'CPU numerical checks, source audit, and saved-checkpoint geometry only',
              'upstream_commit': subprocess.check_output(['git', '-C', str(VENDOR), 'rev-parse', 'HEAD'], text=True).strip(),
              'camera_convention': camera_check(), 'density_lifecycle': density_check(),
              'mask_reduction': mask_reduction_check(),
              'final_checkpoint_grouping': grouping_check()}
    result['audited_source_sha256'] = {name: hashlib.sha256((VENDOR/name).read_bytes()).hexdigest()
        for name in ['scene/cameras.py', 'scene/gaussian_model.py', 'gaussian_renderer/__init__.py', 'utils/spatial_dropout.py']}
    frozen = json.loads((ROOT/'reports/comparison/comparison_source_hashes.json').read_text())
    for name in ['gaussian_renderer/__init__.py', 'utils/spatial_dropout.py']:
        assert result['audited_source_sha256'][name] == frozen['DDGS/'+name]
    result['renderer_and_candidate_match_recorded_run_hashes'] = True
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'mechanism_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

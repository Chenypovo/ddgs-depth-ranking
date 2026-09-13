"""Post-hoc diagnostic of saved test renders; does not train or select a model."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import sobel

root = Path(__file__).resolve().parent / 'reports/comparison/runs'
runs = {
    ('independent', 0): root / 'trex_baseline_seed0/evaluation',
    ('independent', 1): root / 'trex_independent_seed1/evaluation',
    ('independent', 2): root / 'trex_independent_seed2/evaluation',
    ('spatial_stratified', 0): root / 'trex_spatial_stratified_seed0/evaluation',
    ('spatial_stratified', 1): root / 'trex_spatial_stratified_seed1/evaluation',
    ('spatial_stratified', 2): root / 'trex_spatial_stratified_seed2/evaluation',
}

rows = []
for (variant, seed), directory in runs.items():
    for gt_path in sorted(directory.glob('*_gt.png')):
        name = gt_path.name.removesuffix('_gt.png')
        pred_path = directory / f'{name}_prediction.png'
        gt = np.asarray(Image.open(gt_path).convert('RGB'), dtype=np.float32) / 255
        pred = np.asarray(Image.open(pred_path).convert('RGB'), dtype=np.float32) / 255
        grey = gt @ np.array([.299, .587, .114], dtype=np.float32)
        edge_strength = np.hypot(sobel(grey, axis=0), sobel(grey, axis=1))
        threshold = np.quantile(edge_strength, .8)
        edge = edge_strength >= threshold
        error = np.abs(pred - gt).mean(axis=2)
        rows.append({
            'variant': variant,
            'seed': seed,
            'image': name,
            'edge_mae': float(error[edge].mean()),
            'nonedge_mae': float(error[~edge].mean()),
            'overall_mae': float(error.mean()),
            'edge_threshold': float(threshold),
            'edge_fraction': float(edge.mean()),
        })

means = {}
for variant in ('independent', 'spatial_stratified'):
    chosen = [r for r in rows if r['variant'] == variant]
    means[variant] = {
        metric: float(np.mean([r[metric] for r in chosen]))
        for metric in ('edge_mae', 'nonedge_mae', 'overall_mae')
    }

paired = {}
for metric in ('edge_mae', 'nonedge_mae', 'overall_mae'):
    differences = []
    for seed in (0, 1, 2):
        for image in sorted({r['image'] for r in rows}):
            base = next(r[metric] for r in rows if r['variant']=='independent' and r['seed']==seed and r['image']==image)
            candidate = next(r[metric] for r in rows if r['variant']=='spatial_stratified' and r['seed']==seed and r['image']==image)
            differences.append(candidate - base)
    paired[metric] = {
        'candidate_minus_baseline_mean': float(np.mean(differences)),
        'candidate_better_pairs': int(np.sum(np.array(differences) < 0)),
        'pairs': len(differences),
    }

result = {
    'status': 'posthoc_diagnostic_only',
    'definition': 'Sobel magnitude on GT luminance; top 20% pixels per view are edge slice; RGB PNG mean absolute error.',
    'warning': 'This slice was chosen after the headline result and is not a new primary metric or proof of thin-structure quality.',
    'means': means,
    'paired': paired,
    'rows': rows,
}
out = Path(__file__).resolve().parent / 'reports/comparison/failure_slices.json'
out.write_text(json.dumps(result, indent=2))
print(json.dumps({'means': means, 'paired': paired}, indent=2))

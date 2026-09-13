# Code map

Run commands from the repository root. The reorganisation changes script locations, not model settings, data splits or recorded results.

| Purpose | Entry point |
|---|---|
| Recover pinned upstream source | [bootstrap_sources.py](../scripts/setup/bootstrap_sources.py) |
| Integrate the opt-in ranking branch | [prepare_rank_source.py](../scripts/setup/prepare_rank_source.py) |
| Core ranking loss | [rank_supervision.py](../methods/rank_supervision.py) |
| Prepare scene geometry | [prepare_scene.py](../scripts/data/prepare_scene.py) |
| Generate frozen fern depth priors | [prepare_fern_priors.py](../scripts/data/prepare_fern_priors.py) |
| Check fern integration | [check_fern_supervision.py](../scripts/checks/check_fern_supervision.py) |
| Run the fixed fern comparison | [run_fern_comparison.py](../scripts/train/run_fern_comparison.py) |
| Evaluate fern checkpoints | [evaluate_fern.py](../scripts/evaluate/evaluate_fern.py) |
| Plot every paired seed | [plot_two_scene_results.py](../scripts/analysis/plot_two_scene_results.py) |

Read [REPRODUCING.md](REPRODUCING.md) before starting a run. The `archive` folder contains historical packaging helpers; their old run directories remain prerequisites. Large source snapshots, data and model artifacts are intentionally ignored by Git.

[Old-to-new filename map](script_locations.json)

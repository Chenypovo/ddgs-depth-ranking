# Reproducing the comparisons

Run from the repository root on a CUDA Linux host. These commands allocate GPU work when indicated; review the configuration and budget first. The source packaging was verified offline; this is not a claim that every command was rerun from a fresh machine after packaging.

## 1. Recover the recorded D²GS source

```bash
python3 scripts/setup/bootstrap_sources.py
bash scripts/setup/setup_remote.sh
python3 scripts/setup/prepare_rank_source.py
python3 scripts/setup/prepare_fern_source.py
```

The bootstrap creates `DDGS` and `DDGS_corrections`, using the same patched source as the recorded controls. The subsequent scripts create isolated snapshots and refuse to overwrite them. CUDA architecture 8.9 and the tested AutoDL base environment are assumed by the installation helper; adapt installation for another GPU before running it. Some packages are not fully locked; inspect the helper rather than assuming bitwise environment reproducibility.

## 2. Obtain the data and frozen teacher

`scripts/data/download_data.sh` downloads the LLFF archive and extracts trex. To additionally extract fern from that archive:

```bash
python3 scripts/data/extract_llff_scene.py --scene fern
```

Clone the official [Depth Anything V2 repository](https://github.com/DepthAnything/Depth-Anything-V2) into `depth-anything-v2/` and obtain the Small/vits checkpoint using the upstream instructions. Place it at `depth-anything-v2/checkpoints/depth_anything_v2_vits.pth`. Verify its SHA256 against `patches/depth_teacher_identity.json` before creating priors. The original teacher source commit is unknown, so fresh-source numerical equivalence is not guaranteed.

No full-view point cloud may be used as fallback initialisation. Only the three selected training images are used for geometric preprocessing.

## 3. Prepare fern and check integration

```bash
.venv/bin/python scripts/data/prepare_scene.py --scene dataset/nerf_llff_data/fern
.venv/bin/python scripts/data/prepare_fern_priors.py
.venv/bin/python scripts/checks/check_fern_supervision.py
.venv/bin/python scripts/train/run_fern_comparison.py --stage pilot --max-seconds 300
```

Check that the split reports `state: complete`, three train and three held-out views, and a valid dense point cloud. The recorded fern run had 4,507 dense points; future preprocessing need not be bitwise identical. Do not bypass a failed point-count gate or select another scene after looking at quality metrics. The 1,200-step pilot checks integration, not final quality.

## 4. Run the fixed comparison

```bash
.venv/bin/python scripts/train/run_fern_comparison.py --stage compare --max-seconds 2700
```

Six runs: off/ordinary × seeds 0/1/2, each with 10,000 steps. Read `fern_runs/compare/status.json`. The runner refuses existing output directories and has a bounded runtime; it does not shut down a cloud instance. Final-checkpoint evaluation includes original opacity and a fixed 0.85 multiplier for every arm. Do not select the best seed or switch metric definitions after seeing results.

See [the frozen fern protocol](FERN_VALIDATION_PROTOCOL.md). Trex's historical three-arm run also requires the earlier baseline checkpoints and depth audit caches; `scripts/train/run_rank_comparison.py` is not a standalone first command.

## Historical scripts

Several diagnostic and archival runners deliberately retain experiment-specific paths, prerequisites or historical deadlines. In particular, `scripts/train/run_baseline.py` and `scripts/train/run_comparison.py` are records of the original bounded session, not general launchers. Inspect these scripts before use. `package_*` scripts refer to local archive layout and are not needed to train or evaluate.

## Results and provenance

Published `results/*/metrics.json` files preserve numerical scores but normalise machine paths. Full checkpoints, per-frame videos and archives remain local. Reports retain some historical artifact names; these are evidence descriptions, not promises that every large file is included in Git. Do not compare this adapted baseline directly with the authors' full benchmark table.

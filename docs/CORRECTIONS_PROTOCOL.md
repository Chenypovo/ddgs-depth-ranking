# Isolated engineering corrections — 2026-09-13

Status: implemented and staged, CPU checks pass; GPU integration and training pending availability. These changes are implementation corrections, not a claimed research innovation. No new quality scores exist.

## Source isolation

Original remote source and results remain under `${PROJECT_ROOT}/DDGS` and existing `runs`. Corrected source is `${PROJECT_ROOT}/DDGS_corrections`, copied from the 55 MB existing source before overlaying three changed files. Package staging directory: `${PROJECT_ROOT}/correction_stage_20260913`.

`evaluate_baseline.py` is unchanged, verified against the previous run SHA256. A separate `evaluate_corrections.py` accepts `DDGS_SOURCE` but keeps all metric computations, data and camera-path generation unchanged. Locally the working vendor source contains opt-in switches; defaults retain upstream behaviour. Historical audit SHA256 checks intentionally detect this later source change.

## Factors

| Arm | DDGS_DEPTH_MODE | DDGS_DENSITY_MODE | Dropout sampling |
|---|---|---|---|
| baseline replay | upstream | upstream | independent |
| depth correction only | camera_z | upstream | independent |
| density correction only | upstream | refresh | independent |

The euclidean option is present for diagnostics of the paper/code definition difference; it is not part of this quality comparison. The failed spatial-stratified candidate is not enabled. No mask/loss, data, seed, iteration or checkpoint-rule tuning.

Density correction tracks topology versions through actual prune/append operations. It recomputes density before its next use if topology or count changed, while retaining scheduled upstream refreshes. Added kNN computation and telemetry are included in wall time. Position drift between refreshes is retained, as in the upstream schedule; this correction targets topology validity only.

## Completed checks

- CPU: upstream depth exactly matches prior expression; corrected z agrees with renderer indexing; finite backward; no new random-number consumption.
- CPU: actual `GaussianModel.prune_points` and `densification_postfix` bodies invalidate topology version; refresh occurs once per dirty cache; upstream mode does not recompute it.
- Local and remote Python syntax compilation; runner dry-run enumeration.
- Original evaluator hash retained. No CUDA forward/backward validation yet.
- Real LLFF three-view camera extrinsics retrieved from the authorised instance and saved locally. Against all six final PLYs, near/mid/far band disagreement is 7.559% on average across 18 camera/model pairs (2.707–13.730%). This is final-checkpoint geometry, not initial geometry or per-step behaviour. It establishes a real-scene signal difference, not quality improvement.

## Runtime sequence

1. `run_corrections.py --stage diagnose --deadline <absolute ISO timestamp with timezone>` runs three 1100-step diagnostic arms, seed 0. This crosses multiple densification/cache cycles. Inspect sampled depth-band differences, cache fallback/staleness counts and refresh costs. Corrected density must have zero fallback/staleness after refresh. These short runs are integration diagnostics, not baseline scores.
2. Only after inspecting diagnostics, launch `--stage compare` with a valid remaining deadline. Nine fresh runs: three arms × seeds 0/1/2, each complete 10000 iterations. Fresh baseline replay controls for instrumentation and source deployment. Final evaluation remains seven test views, standard VGG LPIPS primary, PSNR/SSIM secondary. Report all runs, paired differences, and time; no cherry-picking or silent retries.
3. Original split, masks, initial points, resolution 504×378 and trex official hyperparameters remain fixed. Trex is a development scene; no held-out cross-scene research claim.

The runner refuses an existing cohort directory and refuses execution without CUDA. It logs source hashes, commands, factor environment, telemetry, partial completion and failures. GNU timeout bounds each subprocess by the remaining absolute deadline and caps a runner invocation at one hour. The overall session must also be bounded by the same platform shutdown time; do not extend it between stages. The old absolute timestamp must be updated only when a fresh run is actually authorised/scheduled, never used as a guessed current deadline.

## Resource status

At approximately 18:43, starting instance `<instance>` with GPU was rejected by AutoDL because the host had 0 free GPUs. No GPU training began. CPU-only boot (¥0.10/hour) was used to read poses and stage source. Normal GPU price shown was ¥1.58/hour. The platform shutdown was moved earlier to 19:40 as a safeguard; early shutdown was requested after staging at about 18:50 and subsequently confirmed as 已关机 in AutoDL.

Expected diagnostic/full comparison budget: initially allow 30–60 minutes, capped at one hour for the whole session, with throughput rechecked after density refresh diagnostics. Nine runs at prior 143-second timing alone take about 22 minutes; refresh overhead and evaluation are additional. Original dataset/model downloads are reused. Staging adds about 55 MB; final model/evaluation artifacts roughly 0.3–0.6 GB, variable with point counts. No other instance was started or cloned. Training is blocked on GPU availability, not code preparation.

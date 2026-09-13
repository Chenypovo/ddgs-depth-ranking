# Local stratified dropout pilot comparison

Frozen 2026-09-13 before any candidate training. Baseline seed 0 already completed and inspected; no candidate hyperparameter search will use test results.

Observed baseline issue: the dinosaur and building are recognisable, while some thin railings/bone boundaries and background regions show ghosting or blur. This observation does not establish dropout as the cause. Hypothesis: independent dropout has unnecessarily high variation in how many nearby points remain at each training step.

Candidate: sort Gaussian centres in approximate local order using 10-bit-per-axis Morton codes, form consecutive groups of 8, and use a shared uniform random phase plus evenly spaced offsets within each group. This retains each point's original marginal keep probability, while reducing local count variation for similar probabilities. Refresh ordering when point count changes or after 500 iterations. Partial final group uses its true size. No extra learned parameters or additional images; grouping adds sorting/caching cost. It does not guarantee semantic structures or stable counts for arbitrary varying probabilities. This is an experimental variance-reduction mechanism; novelty is unverified.

Original path remains `DDGS_DROPOUT=independent`; candidate is `spatial_stratified`. `DDGS_SEED` explicitly sets seed, default 0 exactly as before. All other training, preprocessing and evaluation settings are unchanged from PROTOCOL.md.

Fixed runs: baseline seeds 0 (already complete), 1, 2; candidate seeds 0, 1, 2. Each is 10000 iterations. No selected best checkpoint, no seed replacement and no parameter retuning after results. Report every seed, mean and paired differences. Predeclared primary is standard VGG LPIPS [-1,1], lower is better. PSNR/SSIM and total training time are secondary. Treat under 1% average LPIPS reduction, mixed seed directions, or notable secondary regression as inconclusive for a meaningful improvement; do not claim validated improvement from one scene even if all seeds improve. Report efficiency separately, including grouping cost inside the candidate training wall time.

Expected extra compute from observed 143-second baseline: five additional runs plus evaluation, roughly 15–25 minutes allowing candidate overhead; under the existing session authorisation and deadlines. Data and masks reused. Stop rather than silently retry after an unidentified failure. Final reports include a comparison JSON/table and seed-0 visual evidence for both variants. No broader benchmark or novel-method claim.

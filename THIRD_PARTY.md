# Sources and attribution

- **D²GS:** https://github.com/Insta360-Research-Team/DDGS — fixed revision `9adb7355092e8bd2b302e16b194c693fa51004d8`. The patch modifies upstream source; original file notices must remain in place. The repository contains code derived from the Graphdeco Gaussian Splatting implementation with research/non-commercial restrictions. Consult the upstream file notices and rasterizer licence before reuse.
- **SparseNeRF:** https://sparsenerf.github.io/ — conceptual reference for local depth ranking from inaccurate depth priors. Our code adapts that idea to Gaussian optimisation; it does not implement the full SparseNeRF method.
- **Depth Anything V2:** https://github.com/DepthAnything/Depth-Anything-V2 — frozen Small/vits depth teacher. Checkpoint SHA256 is in `patches/depth_teacher_identity.json`. The teacher source commit was not captured in the original local artifacts; it is not presented as pinned. Obtain code and weights under their upstream terms.
- **LLFF:** data and supplied camera poses. The download helper records the source and retains the MatchNeRF-hosted backup URL as a fallback. Raw data and model weights are excluded from Git.
- **COLMAP/PyCOLMAP and LPIPS:** used for training-only geometric preprocessing and perceptual evaluation respectively. LPIPS results distinguish the standard [-1,1] input convention from the upstream repository's [0,1] convention.

No blanket permissive licence is applied to upstream-derived material. This repository does not change the terms of any third-party dependency. Independent experiment scripts and modifications are distinguished from upstream code through the patch and source identity records.

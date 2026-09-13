#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:/root/miniconda3/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4
export PIP_DISABLE_PIP_VERSION_CHECK=1
source /etc/network_turbo >/dev/null 2>&1 || true
trap 'printf "Setup exited with status %s at %s\n" "$?" "$(date -Is)"' EXIT
date -Is
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install 'torch==2.8.0' 'numpy<2' ninja plyfile tqdm matplotlib 'torchmetrics==1.2.0' 'opencv-python<4.12' scipy scikit-learn 'pykeops==2.3' gdown tensorboard
# pycolmap's CUDA toolkit meta-package selects a runtime newer than base Torch.
# Reuse the already installed CUDA 12 runtime, and verify the actual import below.
.venv/bin/python -m pip install --no-deps 'pycolmap-cuda12==4.2.0'
.venv/bin/python -m pip install --no-build-isolation ./DDGS/submodules/diff-gaussian-rasterization ./DDGS/submodules/simple-knn
.venv/bin/python - <<'PY'
import torch, pycolmap, diff_gaussian_rasterization, simple_knn._C
print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'GPU', torch.cuda.get_device_name())
print('pycolmap', pycolmap.__version__, 'CUDA support', pycolmap.has_cuda)
assert pycolmap.has_cuda, 'Dense COLMAP requires CUDA support'
from simple_knn._C import distCUDA2
x=torch.rand((64,3),device='cuda')
d=distCUDA2(x)
assert torch.isfinite(d).all()
print('KNN CUDA smoke passed', d.shape)
PY
.venv/bin/python -m pip freeze > environment-freeze.txt
date -Is > SETUP_COMPLETE

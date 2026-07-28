#!/usr/bin/env bash
# One-time setup on the vast.ai instance (vastai/pytorch cuda-12.6 image).
#   bash setup_server.sh
set -euo pipefail
cd "$(dirname "$0")"

apt-get update -y >/dev/null 2>&1 || true
apt-get install -y --no-install-recommends tmux htop rsync unzip build-essential libglib2.0-0 libgl1-mesa-glx ninja-build >/dev/null 2>&1 || true

python -m pip install -U pip wheel setuptools
python -m pip install -r requirements.txt

# RTX 30xx = 8.6 (Ampere), RTX 40xx = 8.9 (Ada Lovelace)
export TORCH_CUDA_ARCH_LIST="8.6;8.9"
export MAX_JOBS=4
python -m pip install --no-cache-dir gsplat==1.4.0

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available!"
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
PY

echo "Building/JIT-compiling gsplat CUDA kernels (first time takes minutes)..."
python - <<'PY'
import time, torch
t0 = time.time()
from gsplat import rasterization
d = torch.device("cuda")
n = 2000
means = torch.randn(n, 3, device=d)
quats = torch.nn.functional.normalize(torch.randn(n, 4, device=d), dim=-1)
scales = torch.rand(n, 3, device=d) * 0.02
opac = torch.rand(n, device=d)
colors = torch.rand(n, 3, device=d)
K = torch.tensor([[300., 0., 160.], [0., 300., 120.], [0., 0., 1.]], device=d)[None]
vm = torch.eye(4, device=d)[None]; vm[0, 2, 3] = 4.0
img, alpha, info = rasterization(means, quats, scales, opac, colors, vm, K, 320, 240)
assert img.shape == (1, 240, 320, 3) and float(img.mean()) > 0
print(f"gsplat smoke test OK ({time.time()-t0:.1f}s incl. JIT)")
PY

python -m pip freeze > "pip_freeze_$(date +%Y%m%d).txt"
echo "SETUP DONE"

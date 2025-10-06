#!/usr/bin/env bash
set -euo pipefail

# Assumes the conda env from environment.yml is already active.

use_gpu=false
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "[bootstrap] GPU detected -> installing Torch cu121 + matching PyG"
  use_gpu=true
else
  echo "[bootstrap] No GPU -> installing Torch CPU + matching PyG"
fi

if $use_gpu; then
  pip install "torch==2.2.*+cu121" --index-url https://download.pytorch.org/whl/cu121
  pip install torch_geometric \
    pyg_lib==0.4.0 torch_scatter==2.1.2 torch_sparse==0.6.18 \
    torch_cluster==1.6.3 torch_spline_conv==1.2.2 \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
else
  pip install "torch==2.2.*+cpu" --index-url https://download.pytorch.org/whl/cpu
  pip install torch_geometric \
    pyg_lib==0.4.0 torch_scatter==2.1.2 torch_sparse==0.6.18 \
    torch_cluster==1.6.3 torch_spline_conv==1.2.2 \
    -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
fi

python - <<'PY'
import torch, torch_geometric as tg
print("Torch:", torch.__version__, "| CUDA avail?", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
print("PyG:", tg.__version__)
PY

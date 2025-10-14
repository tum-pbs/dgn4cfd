#!/usr/bin/env bash
set -euo pipefail

# Versions you want to standardize on
TORCH_VER="2.2.2"
CUDA_TAG="cu121"        # change if you standardize on a different CUDA
VIS_VER="0.17.2"
AUDIO_VER="2.2.2"

PYG_PYTORCH_TAG="${TORCH_VER}+${CUDA_TAG}"   # for find-links URL
PYG_CPU_TAG="${TORCH_VER}+cpu"

have_nvidia() {
  command -v nvidia-smi >/dev/null 2>&1
}

echo "==> Detecting GPU..."
if have_nvidia; then
  echo "NVIDIA GPU detected. Installing CUDA ${CUDA_TAG} wheels for Torch ${TORCH_VER}."
  # Install Torch w/ CUDA
  pip install \
    torch==${TORCH_VER}+${CUDA_TAG} \
    torchvision==${VIS_VER}+${CUDA_TAG} \
    torchaudio==${AUDIO_VER}+${CUDA_TAG} \
    --index-url https://download.pytorch.org/whl/${CUDA_TAG}

  # Core PyG + compiled extensions (match the torch tag in the URL!)
  pip install \
    torch-geometric \
    pyg_lib==0.4.0 \
    torch_scatter==2.1.2 \
    torch_sparse==0.6.18 \
    torch_cluster==1.6.3 \
    torch_spline_conv==1.2.2 \
    --find-links https://data.pyg.org/whl/torch-${PYG_PYTORCH_TAG}.html
else
  echo "No NVIDIA GPU detected (or drivers not available). Installing CPU wheels for Torch ${TORCH_VER}."
  # CPU Torch (no index-url)
  pip install \
    torch==${TORCH_VER} \
    torchvision==${VIS_VER} \
    torchaudio==${AUDIO_VER}

  # CPU PyG wheels
  pip install \
    torch-geometric \
    pyg_lib==0.4.0 \
    torch_scatter==2.1.2 \
    torch_sparse==0.6.18 \
    torch_cluster==1.6.3 \
    torch_spline_conv==1.2.2 \
    --find-links https://data.pyg.org/whl/torch-${PYG_CPU_TAG}.html
fi

echo "==> Verifying install..."
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
from importlib import import_module as I
mods = ["torch_geometric","torch_scatter","torch_sparse","torch_cluster","torch_spline_conv"]
for m in mods:
    mod = I(m)
    print(m, getattr(mod, "__version__", "OK"))
PY

echo "==> Done."

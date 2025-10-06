#!/usr/bin/env bash
# One-shot setup + preprocess + smoke-train for UniFoil → DGN4CFD

set -euo pipefail

# ---------- USER OVERRIDES ----------
DATA_ROOT="${DATA_ROOT:-/mnt/d/dgn4unifoil/raw}"                # abs path with your .cgns
SPLIT_CSV="${SPLIT_CSV:-dgn4cfd_ext/configs/splits_example.csv}" # case list with abs cgns_path, Re, Mach, AoA
CONFIG_YAML="${CONFIG_YAML:-dgn4cfd_ext/configs/unifoil.yaml}"    # model/dataset config
ENV_NAME="${ENV_NAME:-dgn4cfd-unifoil}"                           # conda env name
FORCE_BOOTSTRAP="${FORCE_BOOTSTRAP:-false}"                       # set true to reinstall Torch/PyG
# -----------------------------------

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

say(){ echo -e "\033[1;32m[run_unifoil]\033[0m $*"; }
warn(){ echo -e "\033[1;33m[run_unifoil]\033[0m $*"; }
die(){ echo -e "\033[1;31m[run_unifoil]\033[0m $*"; exit 1; }

[[ -d dgn4cfd_ext ]] || die "dgn4cfd_ext/ not found at repo root."
[[ -f "$SPLIT_CSV" ]] || die "Split CSV missing at $SPLIT_CSV."
[[ -d "$DATA_ROOT" ]] || warn "DATA_ROOT $DATA_ROOT not found yet (ensure .cgns paths in CSV are valid)."

# conda helper
if ! command -v conda >/dev/null 2>&1; then
  die "conda not found. Install Miniconda/Anaconda and re-run."
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# create env if needed
if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  say "Conda env ${ENV_NAME} exists."
else
  say "Creating conda env ${ENV_NAME} from environment.yml…"
  conda env create -n "${ENV_NAME}" -f environment.yml
fi

# activate
conda activate "${ENV_NAME}"

# install Torch+PyG if first time or forced
if $FORCE_BOOTSTRAP || ! python -c 'import torch,sys; print(torch.__version__)' >/dev/null 2>&1; then
  say "Bootstrapping Torch + PyG…"
  ./tools/bootstrap_torch_pyg.sh
else
  say "Torch already present. Skip bootstrap (set FORCE_BOOTSTRAP=true to reinstall)."
fi

# preprocess to cached graphs
say "Preprocessing CGNS → cached graphs…"
python dgn4cfd_ext/scripts/preprocess_unifoil.py \
  --data-root "$DATA_ROOT" \
  --split-csv "$SPLIT_CSV" \
  --out-root data/unifoil \
  --fields_at cells

# sanity tests (non-fatal if missing)
if [[ -d tests ]]; then
  say "Running sanity tests…"
  pytest tests/test_unifoil_build.py -q || warn "Sanity test failed (non-fatal)."
fi

# IMPORTANT: ensure train_unifoil.py calls your repo's trainer (see README_UNIFOIL.md)
say "Starting smoke train…"
python dgn4cfd_ext/scripts/train_unifoil.py --config "$CONFIG_YAML"

say "DONE. Cached graphs: data/unifoil/proc ; logs/checkpoints under runs/ (depending on trainer)."


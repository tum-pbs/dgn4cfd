#!/usr/bin/env bash
# tools/convert_cgns_to_hdf5.sh
# Convert CGNS ADF files to CGNS HDF5 in batch.
#
# Usage:
#   bash tools/convert_cgns_to_hdf5.sh /abs/path/to/dir
#   bash tools/convert_cgns_to_hdf5.sh /abs/path/to/*.cgns
#   # no args -> tries ../raw/*.cgns OR ./raw/*.cgns relative to this repo

set -euo pipefail

say()  { echo -e "\033[1;32m[convert]\033[0m $*"; }
warn() { echo -e "\033[1;33m[convert]\033[0m $*"; }
die()  { echo -e "\033[1;31m[convert]\033[0m $*"; exit 1; }

# ---------- resolve inputs ----------
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if (( $# == 0 )); then
  # common layouts: raw lives next to repo or inside repo
  CAND1="$ROOT/../raw/*.cgns"
  CAND2="$ROOT/raw/*.cgns"
  shopt -s nullglob
  files=( $CAND1 )
  if (( ${#files[@]} == 0 )); then files=( $CAND2 ); fi
  shopt -u nullglob
else
  if [[ -d "$1" ]]; then
    shopt -s nullglob
    files=( "$1"/*.cgns )
    shopt -u nullglob
  else
    # treat argument as a glob or specific file
    shopt -s nullglob
    files=( $1 )
    shopt -u nullglob
  fi
fi

(( ${#files[@]} > 0 )) || die "No .cgns files found for the given path/glob."

say "Found ${#files[@]} file(s) to convert."

# ---------- ensure converter tool exists ----------
choose_converter() {
  if command -v cgnsconvert >/dev/null 2>&1; then
    echo "cgnsconvert"
    return 0
  fi
  if command -v adf2hdf >/dev/null 2>&1; then
    echo "adf2hdf"
    return 0
  fi

  warn "CGNS converter not found. Attempting to install 'cgns-convert' (Ubuntu/WSL)…"
  if command -v sudo >/dev/null 2>&1; then
    sudo add-apt-repository -y universe || true
    sudo apt-get update -y
    # Ubuntu 24.04 package name:
    if ! sudo apt-get install -y cgns-convert; then
      warn "Package 'cgns-convert' not available; trying legacy names…"
      sudo apt-get install -y cgns-tools || sudo apt-get install -y cgns-utils || true
    fi
  else
    die "No 'sudo' available to install converter. Please install 'cgns-convert' manually."
  fi

  if command -v cgnsconvert >/dev/null 2>&1; then
    echo "cgnsconvert"
  elif command -v adf2hdf >/dev/null 2>&1; then
    echo "adf2hdf"
  else
    die "Could not install a CGNS converter (cgnsconvert/adf2hdf)."
  fi
}

CONV=$(choose_converter)
say "Using converter: $CONV"

# ---------- ensure 'file' and python/h5py for verification ----------
if ! command -v file >/dev/null 2>&1; then
  warn "'file' command not found; skipping file-signature check."
fi

PYBIN="${PYBIN:-python}"
if ! $PYBIN - <<'PY' >/dev/null 2>&1; then
import sys
try:
    import h5py  # noqa
    ok=1
except Exception:
    ok=0
sys.exit(0 if ok else 1)
PY
then
  warn "h5py not found in current Python environment; attempting to install into current env…"
  $PYBIN -m pip install -U h5py >/dev/null 2>&1 || warn "Could not install h5py; will skip h5py-open verification."
fi

verify_h5() {
  local out="$1"
  local ok=1
  if command -v file >/dev/null 2>&1; then
    if ! file "$out" | grep -qi "Hierarchical Data Format"; then
      warn "file(1) did not report HDF5 signature for: $out"
    fi
  fi
  $PYBIN - <<PY "$out" >/dev/null 2>&1 || ok=0
import sys
try:
    import h5py
    with h5py.File(sys.argv[1],"r"): pass
    print("HDF5 open OK")
except Exception as e:
    print("HDF5 open FAILED:", e)
    raise SystemExit(1)
PY
  return $ok
}

# ---------- convert loop ----------
fail=0
for f in "${files[@]}"; do
  [[ -f "$f" ]] || { warn "Skipping non-file: $f"; continue; }
  out="${f%.cgns}_h5.cgns"
  say "Converting: $(basename "$f")  →  $(basename "$out")"

  if [[ "$CONV" == "cgnsconvert" ]]; then
    # '-h' = write HDF5
    if ! cgnsconvert -h "$f" "$out"; then
      warn "cgnsconvert failed for $f"; fail=1; continue
    fi
  else
    # adf2hdf writes HDF5 directly
    if ! adf2hdf "$f" "$out"; then
      warn "adf2hdf failed for $f"; fail=1; continue
    fi
  fi

  if verify_h5 "$out"; then
    say "Verified HDF5: $out"
  else
    warn "Verification failed (not HDF5 or cannot open): $out"
    fail=1
  fi
done

if (( fail )); then
  die "One or more files failed conversion/verification."
else
  say "All files converted and verified successfully."
fi

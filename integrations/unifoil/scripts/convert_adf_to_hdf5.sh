#!/usr/bin/env bash
set -euo pipefail

IN="${1:-integrations/unifoil/data/raw/cgns}"
OUT="${2:-integrations/unifoil/data/raw/cgns_h5}"

if ! command -v cgnsconvert >/dev/null 2>&1; then
  echo "ERROR: cgnsconvert not found. Install via: conda install -c conda-forge cgns"
  exit 1
fi

mkdir -p "$OUT"

# HDF5 magic: 89 48 44 46 0d 0a 1a 0a
is_hdf5 () {
  dd if="$1" bs=8 count=1 2>/dev/null | od -An -tx1 | tr -d ' \n' | grep -qi '^894844460d0a1a0a$'
}

echo "IN : $IN"
echo "OUT: $OUT"

find "$IN" -type f -name '*.cgns' -print0 | while IFS= read -r -d '' src; do
  rel="${src#$IN/}"
  dst="$OUT/$rel"
  mkdir -p "$(dirname "$dst")"

  if is_hdf5 "$src"; then
    if [[ -f "$dst" ]]; then
      echo "[skip:hdf5-exists] $rel"
    else
      cp -p "$src" "$dst" && echo "[copy:hdf5] $rel"
    fi
  else
    echo "[conv:adf->hdf5] $rel"
    if cgnsconvert -h "$src" "$dst" >/dev/null 2>&1; then
      if is_hdf5 "$dst"; then
        echo "  -> OK"
      else
        echo "  -> WARN: output missing HDF5 signature"
      fi
    else
      echo "  -> ERROR: cgnsconvert failed"
    fi
  fi
done

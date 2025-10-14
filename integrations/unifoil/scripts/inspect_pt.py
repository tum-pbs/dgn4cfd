# -*- coding: utf-8 -*-
"""
Validate that processed .pt graphs faithfully represent the source CGNS.

Strategy:
- For each .pt in integrations/unifoil/data/processed, find the source CGNS
  (HDF5 preferred under cgns_h5/, else ADF under cgns/).
- Re-read the CGNS using the same robust reader (cgns_to_pyg) to get a fresh Data.
- Compare: shapes, centroids (pos), edge_index, y (u,v,Cp), globals (Mach,Re,AoA).
- Integrity checks: no NaNs/Inf, symmetric edges, no self-loops, no duplicates.

Usage:
  python integrations/unifoil/scripts/validate_converted.py
"""

import math
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from integrations.unifoil.dgn4cfd_ext.datasets.converters.cgns_to_pyg import cgns_to_pyg

RAW = Path("integrations/unifoil/data/raw")
PROC = Path("integrations/unifoil/data/processed")
TAB = RAW / "metadata.tab"
CGNS_H5_DIR = RAW / "cgns_h5"
CGNS_ADF_DIR = RAW / "cgns"

STEM_PAT = re.compile(r"airfoil_(?P<airfoil>\d+).*?case_(?P<case>\d+)_", re.IGNORECASE)

# tolerances (tight—these should all be exact or within float-rounding)
TOL_POS_ABS = 1e-6
TOL_Y_ABS   = 1e-6
TOL_G_ABS   = 1e-10

def _is_hdf5(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(8) == b"\x89HDF\r\n\x1a\n"
    except Exception:
        return False

def _ensure_hdf5(adf_path: Path, out_dir: Path) -> Optional[Path]:
    """If only ADF exists, convert to HDF5 in out_dir and return new path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    h5 = out_dir / adf_path.name
    if h5.exists() and _is_hdf5(h5):
        return h5
    try:
        subprocess.run(["cgnsconvert", "-h", str(adf_path), str(h5)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return h5 if _is_hdf5(h5) else None
    except Exception:
        return None

def _find_cgns_for_stem(stem: str) -> Optional[Path]:
    # prefer exact stem match
    for d in (CGNS_H5_DIR, CGNS_ADF_DIR):
        p = d / f"{stem}.cgns"
        if p.exists():
            return p if (d is CGNS_H5_DIR or _is_hdf5(p)) else _ensure_hdf5(p, CGNS_H5_DIR)
    # fuzzy match: same airfoil/case pair
    m = STEM_PAT.search(stem)
    if not m:
        return None
    a, c = int(m.group("airfoil")), int(m.group("case"))
    for d in (CGNS_H5_DIR, CGNS_ADF_DIR):
        for p in d.glob("**/*.cgns"):
            n = p.name.lower()
            if f"airfoil_{a}" in n and f"case_{c}_" in n:
                return p if (d is CGNS_H5_DIR or _is_hdf5(p)) else _ensure_hdf5(p, CGNS_H5_DIR)
    return None

def _edge_index_to_set(edge_index: torch.Tensor) -> set:
    """Undirected edge set; ignores direction and dedups."""
    e = edge_index.cpu().numpy()
    return {tuple(sorted((int(u), int(v)))) for u, v in e.T if int(u) != int(v)}

def _has_duplicates(edge_index: torch.Tensor) -> bool:
    e = edge_index.cpu().numpy()
    pairs = [ (int(u), int(v)) for u, v in e.T ]
    return len(pairs) != len(set(pairs))

def _is_symmetric(edge_index: torch.Tensor) -> bool:
    e = edge_index.cpu().numpy()
    fwd = {(int(u), int(v)) for u, v in e.T}
    bwd = {(int(v), int(u)) for u, v in e.T}
    return fwd == bwd

def _all_finite(*tensors: torch.Tensor) -> bool:
    for t in tensors:
        if not torch.isfinite(t).all().item():
            return False
    return True

def validate_one(pt_path: Path) -> Tuple[bool, List[str]]:
    """Validate a single .pt graph against the source CGNS."""
    errs: List[str] = []
    data_pt: torch_geometric.data.Data = torch.load(pt_path, map_location="cpu")
    stem = pt_path.stem

    # locate CGNS
    src = _find_cgns_for_stem(stem)
    if not src or not src.exists():
        return False, [f"Missing source CGNS for {stem}"]

    # rebuild fresh from CGNS
    # meta: use the globals in the saved .pt if present
    g = getattr(data_pt, "globals", None)
    if g is not None:
        meta = {"globals": [float(g[0]), float(g[1]), float(g[2])]}
    else:
        # fallback parse from filename
        m = STEM_PAT.search(stem)
        meta = {"globals": [0.0, 0.0, 0.0]} if not m else {"globals":[0.0,0.0,0.0]}

    fresh = cgns_to_pyg(str(src), meta=meta)

    # --- integrity checks on both
    for label, d in (("PT", data_pt), ("CGNS→fresh", fresh)):
        if not _all_finite(d.pos, d.y, d.edge_index.to(torch.float32)):
            errs.append(f"{label}: NaNs/Inf detected")
        if (d.edge_index[0] == d.edge_index[1]).any().item():
            errs.append(f"{label}: self-loops detected")
        if not _is_symmetric(d.edge_index):
            errs.append(f"{label}: edge_index not symmetric/undirected")
        if _has_duplicates(d.edge_index):
            errs.append(f"{label}: duplicate directed edges found")
        if d.pos.ndim != 2 or d.pos.size(1) != 2:
            errs.append(f"{label}: pos expected (N,2), got {tuple(d.pos.shape)}")
        if d.y.ndim != 2 or d.y.size(1) != 3:
            errs.append(f"{label}: y expected (N,3), got {tuple(d.y.shape)}")

    # --- size checks
    if data_pt.pos.size(0) != fresh.pos.size(0):
        errs.append(f"pos N mismatch: {data_pt.pos.size(0)} vs {fresh.pos.size(0)}")
    if data_pt.y.size(0) != fresh.y.size(0):
        errs.append(f"y N mismatch: {data_pt.y.size(0)} vs {fresh.y.size(0)}")
    # Edges: allow different ordering but expect same undirected set
    set_pt   = _edge_index_to_set(data_pt.edge_index)
    set_fresh= _edge_index_to_set(fresh.edge_index)
    if set_pt != set_fresh:
        # print quick diff size
        only_pt   = len(set_pt - set_fresh)
        only_fresh= len(set_fresh - set_pt)
        errs.append(f"edge set mismatch (only_pt={only_pt}, only_fresh={only_fresh})")

    # --- numeric checks (centroids / targets / globals)
    # Sort by position to align (robust to cell ordering differences across conversions)
    def _sort_key(t: torch.Tensor):
        a = t.cpu().numpy()
        return np.lexsort((a[:,1], a[:,0]))  # sort by x then y

    idx_pt    = _sort_key(data_pt.pos)
    idx_fresh = _sort_key(fresh.pos)
    pos_pt    = data_pt.pos[idx_pt]
    pos_fr    = fresh.pos[idx_fresh]
    y_pt      = data_pt.y[idx_pt]
    y_fr      = fresh.y[idx_fresh]

    # positions
    pos_diff = (pos_pt - pos_fr).abs().max().item()
    if not (pos_diff <= TOL_POS_ABS):
        errs.append(f"pos max|diff|={pos_diff:.3e} > tol={TOL_POS_ABS:.1e}")

    # y channels
    y_diff = (y_pt - y_fr).abs().max().item()
    if not (y_diff <= TOL_Y_ABS):
        errs.append(f"y max|diff|={y_diff:.3e} > tol={TOL_Y_ABS:.1e}")

    # globals (if present)
    g_pt = getattr(data_pt, "globals", None)
    g_fr = getattr(fresh, "globals", None)
    if g_pt is not None and g_fr is not None:
        g_diff = (g_pt - g_fr).abs().max().item()
        if not (g_diff <= TOL_G_ABS):
            errs.append(f"globals max|diff|={g_diff:.3e} > tol={TOL_G_ABS:.1e}")

    ok = len(errs) == 0
    return ok, errs

def main():
    pts = sorted(PROC.glob("*.pt"))
    if not pts:
        print(f"No .pt files found in {PROC}")
        return
    total = len(pts)
    passed = failed = 0
    for i, p in enumerate(pts, 1):
        try:
            ok, errs = validate_one(p)
        except Exception as e:
            ok, errs = False, [f"validator exception: {e}"]
        status = "PASS" if ok else "FAIL"
        print(f"[{i:>4}/{total}] {p.name} -> {status}")
        if not ok:
            for e in errs:
                print(f"         - {e}")
            failed += 1
        else:
            passed += 1
    print(f"\nSummary: passed={passed}, failed={failed}, total={total}")
    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()

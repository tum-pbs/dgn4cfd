# -*- coding: utf-8 -*-
"""
Batch: metadata.tab (airfoil,case,Mach,AoA,Re) + CGNS(HDF5) → .pt graphs
- Reads HDF5-CGNS from integrations/unifoil/data/raw/cgns_h5
- Matches files like: airfoil_<airfoil>_..._case_<case>_... .cgns
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

from integrations.unifoil.dgn4cfd_ext.datasets.converters.cgns_to_pyg import cgns_to_pyg

RAW = Path("integrations/unifoil/data/raw")
PROC = Path("integrations/unifoil/data/processed")
TAB = RAW / "metadata.tab"     # header: airfoil,case,Mach,AoA,Re
CGNS_DIR = RAW / "cgns_h5"     # HDF5 CGNS input dir

def read_meta_csv(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                rows.append({
                    "airfoil": int(str(r["airfoil"]).strip()),
                    "case":    int(str(r["case"]).strip()),
                    "Mach":    float(str(r["Mach"]).strip()),
                    "AoA":     float(str(r["AoA"]).strip()),
                    "Re":      float(str(r["Re"]).strip()),
                })
            except Exception as e:
                print(f"[warn] skip row {r}: {e}")
    if not rows:
        raise ValueError(f"No valid rows parsed from {path}")
    return rows

def infer_airfoil_case_index(cgns_dir: Path) -> Dict[Tuple[int, int], Path]:
    idx: Dict[Tuple[int, int], Path] = {}
    if not cgns_dir.exists():
        print(f"[warn] CGNS dir doesn't exist: {cgns_dir}")
        return idx
    pat = re.compile(r"airfoil_(?P<airfoil>\d+).*?case_(?P<case>\d+)_", re.IGNORECASE)
    for p in cgns_dir.glob("**/*.cgns"):
        m = pat.search(p.name)
        if m:
            idx[(int(m.group("airfoil")), int(m.group("case")))] = p
    return idx

def fallback_guess_pair(cgns_dir: Path, airfoil: int, case_id: int) -> Optional[Path]:
    candidates = [
        cgns_dir / f"airfoil_{airfoil}_case_{case_id}.cgns",
        cgns_dir / f"airfoil_{airfoil}_case_{str(case_id).zfill(4)}.cgns",
    ]
    for p in candidates:
        if p.exists(): return p
    tok_air, tok_case = f"airfoil_{airfoil}", f"case_{case_id}"
    for p in cgns_dir.glob("**/*.cgns"):
        name = p.name.lower()
        if tok_air in name and tok_case in name:
            return p
    return None

def main():
    PROC.mkdir(parents=True, exist_ok=True)
    rows = read_meta_csv(TAB)
    index = infer_airfoil_case_index(CGNS_DIR)

    total = len(rows)
    saved = skipped = errored = 0

    for i, r in enumerate(rows, 1):
        airfoil, case_id = r["airfoil"], r["case"]
        mach, Re, aoa   = r["Mach"], r["Re"], r["AoA"]

        resolved = index.get((airfoil, case_id)) or fallback_guess_pair(CGNS_DIR, airfoil, case_id)
        if not resolved or not resolved.exists():
            print(f"[skip] no CGNS for airfoil={airfoil}, case={case_id} in {CGNS_DIR}")
            skipped += 1
            continue

        meta = {"globals": [mach, Re, aoa]}
        try:
            data = cgns_to_pyg(str(resolved), meta=meta)
        except Exception as e:
            print(f"[error] {resolved.name}: {e}")
            errored += 1
            continue

        out = PROC / f"{resolved.stem}.pt"
        torch.save(data, out)
        saved += 1

        if i % 10 == 0 or i == total:
            print(f"[{i:>4}/{total}] saved: {out.name}  (saved={saved}, skipped={skipped}, errors={errored})")

    print(f"done. saved={saved}, skipped={skipped}, errors={errored}, total_rows={total}")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
CGNS(HDF5) → PyTorch Geometric Data

Supports:
 - Structured surface zones (build QUAD connectivity from I×J)
 - Unstructured TRI/QUAD/NGON (content-based discovery anywhere under Zone)
 - Robust coordinate reading (CoordinateX/Y/Z or packed 2D arrays; handles ' data' node)

Graph is cell-centered:
 - data.pos        : (Nc,2) centroids (Z ignored if present)
 - data.edge_index : (2,E) shared-edge adjacency (undirected)
 - data.edge_attr  : (E,1) dummy ones
 - data.y          : (Nc,3) [u, v, Cp]
 - data.globals    : (3,)  [Mach, Re, AoA]
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch_geometric.data import Data

# ---- Field & coordinate name candidates (extend if your files differ) ----
NAME_U        = ["VelocityX", "u", "U", "Ux"]
NAME_V        = ["VelocityY", "v", "V", "Uy"]
NAME_CP       = ["CoefPressure", "Cp", "cp", "PressureCoeff", "CpMean"]
NAME_COORD_X  = ["CoordinateX", "X", "CoordX", "Coordinate1", "Coord1"]
NAME_COORD_Y  = ["CoordinateY", "Y", "CoordY", "Coordinate2", "Coord2"]
NAME_COORD_Z  = ["CoordinateZ", "Z", "CoordZ", "Coordinate3", "Coord3"]

# ---------------- HDF5 helpers ----------------
def _is_hdf5(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(8) == b"\x89HDF\r\n\x1a\n"

def _is_group(x) -> bool:
    return hasattr(x, "keys") and hasattr(x, "__getitem__")

def _label(x) -> str:
    try:
        v = x.attrs.get("label", b"")
        return v.decode(errors="ignore") if isinstance(v, (bytes, bytearray)) else (str(v) if v is not None else "")
    except Exception:
        return ""

def _children(g):
    return [(k, g[k]) for k in g.keys()] if _is_group(g) else []

def _find_first_by_label(g, label: str):
    for k, v in _children(g):
        if _label(v) == label:
            return k, v
    return None, None

def _walk(g):
    stack = [g]
    while stack:
        cur = stack.pop()
        if _is_group(cur):
            yield cur
            for _, v in _children(cur):
                if _is_group(v):
                    stack.append(v)

def _get_dataarray(group, name: str):
    """Return numpy array from a CGNS DataArray_t node:
       - If group[name] is a group, try 'Data' or ' data' (note the space)
       - Else, try reading dataset directly
    """
    if not _is_group(group) or name not in group:
        return None
    node = group[name]
    try:
        if _is_group(node):
            if "Data" in node:   return np.asarray(node["Data"][...])
            if " data" in node:  return np.asarray(node[" data"][...])  # some writers use ' data'
            # last resort: first child dataset we can read
            for kk, vv in _children(node):
                try:
                    return np.asarray(vv[...])
                except Exception:
                    pass
            return None
        else:
            return np.asarray(node[...])
    except Exception:
        return None

def _first_existing(group, names):
    """Return the first DataArray present in group from a list of names."""
    for nm in names:
        arr = _get_dataarray(group, nm)
        if arr is not None:
            return arr
    return None

def _gridlocation_str(fs) -> str:
    """Robustly read FlowSolution/GridLocation as a lowercase string."""
    val = _get_dataarray(fs, "GridLocation")
    if val is None:
        return ""
    try:
        if isinstance(val, (bytes, bytearray)):
            return val.decode(errors="ignore").lower().strip()
        arr = np.asarray(val)
        if arr.dtype.kind in ("S", "U"):
            try:
                return arr.astype("S").tobytes().decode(errors="ignore").lower().strip()
            except Exception:
                return "".join(map(str, arr.flatten().tolist())).lower().strip()
        try:
            b = bytes(arr.astype("uint8").ravel().tolist())
            return b.decode(errors="ignore").lower().strip()
        except Exception:
            return str(arr).lower().strip()
    except Exception:
        return ""

# ---------------- geometry helpers ----------------
def _centroids(points: np.ndarray, conn: Union[np.ndarray, List[List[int]]]) -> np.ndarray:
    C = []
    it = conn if isinstance(conn, list) else [e.tolist() for e in conn]
    for elem in it:
        idx = np.asarray(elem, dtype=int)
        C.append(points[idx, :2].mean(axis=0))
    return np.asarray(C, dtype=np.float32)

def _edge_adjacency(conn: Union[np.ndarray, List[List[int]]]) -> np.ndarray:
    edge_to_cells = {}
    it = conn if isinstance(conn, list) else [e.tolist() for e in conn]
    for cid, elem in enumerate(it):
        k = len(elem)
        for i in range(k):
            a, b = int(elem[i]), int(elem[(i + 1) % k])
            e = (a, b) if a < b else (b, a)
            lst = edge_to_cells.setdefault(e, [])
            if len(lst) < 2:
                lst.append(cid)
    pairs = []
    for inc in edge_to_cells.values():
        if len(inc) == 2:
            i, j = inc
            pairs.append((i, j)); pairs.append((j, i))
    return (np.asarray(pairs, dtype=np.int64).T if pairs else np.empty((2, 0), dtype=np.int64))

# ---------------- coordinate extraction ----------------
def _extract_xyz_from_pairs(pairs: List[Tuple[str, np.ndarray]], n_vertices: Optional[int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Accepts list of (name_lower, array). Returns (X,Y,Z) 1D float32 or None.
       Supports separate 1D axes and packed 2D matrices (N×2/3 or 2/3×N).
    """
    if n_vertices is not None:
        filtered, matrix = [], []
        for n, a in pairs:
            a = np.asarray(a)
            if a.ndim == 1 and a.size == n_vertices:
                filtered.append((n, a))
            elif a.ndim == 2 and (n_vertices in a.shape) and (2 in a.shape or 3 in a.shape):
                matrix.append((n, a))
        pairs = filtered or pairs
        for _, A in matrix:
            A = np.asarray(A)
            if A.ndim == 2 and (A.shape[0] in (2,3) or A.shape[1] in (2,3)):
                if A.shape[0] in (2,3) and A.shape[1] == n_vertices:
                    A = A.T  # (N,D)
                elif A.shape[1] in (2,3) and A.shape[0] == n_vertices:
                    pass
                else:
                    continue
                X = A[:,0].astype(np.float32).reshape(-1)
                Y = A[:,1].astype(np.float32).reshape(-1)
                Z = A[:,2].astype(np.float32).reshape(-1) if A.shape[1] == 3 else None
                return X, Y, Z

    X_KEYS = ("coordinatex", "x", "coordx", "coordinate1", "coord1")
    Y_KEYS = ("coordinatey", "y", "coordy", "coordinate2", "coord2")
    Z_KEYS = ("coordinatez", "z", "coordz", "coordinate3", "coord3")

    name_to_arr = {n: np.asarray(a) for (n, a) in pairs if np.asarray(a).ndim == 1}
    x = next((name_to_arr[n] for n in X_KEYS if n in name_to_arr), None)
    y = next((name_to_arr[n] for n in Y_KEYS if n in name_to_arr), None)
    z = next((name_to_arr[n] for n in Z_KEYS if n in name_to_arr), None)

    if (x is not None and y is not None):
        return x.astype(np.float32).reshape(-1), y.astype(np.float32).reshape(-1), (z.astype(np.float32).reshape(-1) if z is not None else None)

    one_d = [np.asarray(a) for (_, a) in pairs if np.asarray(a).ndim == 1]
    if n_vertices is not None:
        one_d_nv = [a for a in one_d if a.size == n_vertices]
        if one_d_nv:
            one_d = one_d_nv
    if len(one_d) >= 2:
        return one_d[0].astype(np.float32).reshape(-1), one_d[1].astype(np.float32).reshape(-1), (one_d[2].astype(np.float32).reshape(-1) if len(one_d) >= 3 else None)

    return None, None, None

# ---------------- unstructured elements (recursive, content-based) ----------------
def _gather_unstructured(zone):
    """Find any Elements-like group:
       - NGON if ElementConnectivity + ElementStartOffset/offsets present
       - else TRI/QUAD inferred from connectivity arity
    """
    tri = quad = None
    ngon_conn = ngon_offs = None
    for grp in _walk(zone):
        econn = _first_existing(grp, ["ElementConnectivity", "connectivity"])
        if econn is None:
            continue
        offs = _first_existing(grp, ["ElementStartOffset", "offsets"])
        if offs is not None:
            ngon_conn = np.asarray(econn, dtype=np.int64).ravel()
            ngon_offs = np.asarray(offs,  dtype=np.int64).ravel()
            break
        e = np.asarray(econn, dtype=np.int64).ravel()
        if e.size % 3 == 0 and tri is None:
            tri = e.reshape(-1, 3)
        elif e.size % 4 == 0 and quad is None:
            quad = e.reshape(-1, 4)
    if ngon_conn is not None and ngon_offs is not None:
        if len(ngon_offs) and ngon_offs[0] != 0:
            ngon_offs = np.concatenate([[0], ngon_offs])
        polys = []
        for i in range(len(ngon_offs) - 1):
            a, b = int(ngon_offs[i]), int(ngon_offs[i + 1])
            verts = ngon_conn[a:b].tolist()
            if len(verts) >= 3:
                polys.append(verts)
        return polys  # ragged list
    if tri is not None:  return tri
    if quad is not None: return quad
    return None

# ---------------- structured surface (build QUADs from I×J) ----------------
def _structured_surface(zone) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if "GridCoordinates" not in zone:
        return None, None
    gc = zone["GridCoordinates"]

    def pick_coord(names):
        for nm in names:
            arr = _get_dataarray(gc, nm)
            if arr is not None:
                return np.asarray(arr)
        return None

    X = pick_coord(NAME_COORD_X)
    Y = pick_coord(NAME_COORD_Y)
    Z = pick_coord(NAME_COORD_Z)  # optional

    if X is None or Y is None or X.ndim != 2 or Y.ndim != 2:
        return None, None

    J, I = X.shape  # (J, I)
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype(np.float32)

    conn = []
    for j in range(J - 1):
        for i in range(I - 1):
            v0 = j * I + i
            v1 = j * I + (i + 1)
            v2 = (j + 1) * I + (i + 1)
            v3 = (j + 1) * I + i
            conn.append([v0, v1, v2, v3])
    conn = np.asarray(conn, dtype=np.int64)
    return pts, conn

# ---------------- FlowSolution (cell-centered with rind) ----------------
def _pick_flowsolution(zone):
    # Prefer label-based; fall back to first group named like FlowSolution
    for grp in _walk(zone):
        if _label(grp) == "FlowSolution_t":
            return grp
    # name fallback
    for k, v in _children(zone):
        if "flow" in k.lower():
            return v
    return None

def _read_cell_center_fields(fs, I, J):
    """Read VelocityX, VelocityY, CoefPressure from a cell-centered FlowSolution
       with rind (1,1,1,1). Arrays are shaped (Jc+2, Ic+2); strip rind → (J-1, I-1).
    """
    gls = _gridlocation_str(fs)
    if gls and ("cell" not in gls):
        return None, None, None

    def pick(names):
        for nm in names:
            arr = _get_dataarray(fs, nm)
            if arr is not None:
                return np.asarray(arr)
        return None

    Ux = pick(NAME_U); Uy = pick(NAME_V); Cp = pick(NAME_CP)
    if Ux is None or Uy is None or Cp is None:
        return None, None, None

    # strip rind: [1:-1, 1:-1] -> (J-1, I-1)
    Ux = Ux[1:-1, 1:-1].astype(np.float32)
    Uy = Uy[1:-1, 1:-1].astype(np.float32)
    Cp = Cp[1:-1, 1:-1].astype(np.float32)

    return Ux.reshape(-1), Uy.reshape(-1), Cp.reshape(-1)

# ---------------- main CGNS→PyG ----------------
def cgns_to_pyg(cgns_path: str, meta: Dict[str, Any]) -> Data:
    if not _is_hdf5(cgns_path):
        raise RuntimeError("File is not HDF5 CGNS (ADF detected). Convert ADF→HDF5 first.")

    import h5py
    with h5py.File(cgns_path, "r") as f:
        # Base
        _, base = _find_first_by_label(f, "CGNSBase_t")
        if base is None:
            base = next(iter(f.values()))  # name-based fallback (e.g., 'BaseSurfaceSol')

        # Zones (prefer a wall-like one; else first)
        zones = [(k, v) for k, v in _children(base) if _label(v) == "Zone_t"]
        if not zones:
            zones = [(k, v) for k, v in _children(base) if "Zone" in k]
        zname, zone = next(((k, v) for k, v in zones if "wall" in k.lower()), zones[0])

        # Try structured surface first
        points, conn = _structured_surface(zone)
        structured = (points is not None) and (conn is not None)

        if not structured:
            # Unstructured fallback
            conn = _gather_unstructured(zone)
            if conn is None:
                raise RuntimeError("Could not find mesh connectivity (structured or unstructured).")
            # Points for unstructured: coordinates from GridCoordinates
            if "GridCoordinates" in zone:
                gc = zone["GridCoordinates"]
                X = _first_existing(gc, NAME_COORD_X)
                Y = _first_existing(gc, NAME_COORD_Y)
                if X is None or Y is None:
                    raise RuntimeError("No CoordinateX/CoordinateY for unstructured zone.")
                points = np.stack([np.asarray(X).reshape(-1), np.asarray(Y).reshape(-1)], axis=1).astype(np.float32)
            else:
                raise RuntimeError("No GridCoordinates in zone.")

        # Build graph geometry
        cent = _centroids(points, conn)
        edge_index = _edge_adjacency(conn)
        edge_attr = np.ones((edge_index.shape[1], 1), dtype=np.float32)

        # Fields
        fs = _pick_flowsolution(zone)
        if fs is None:
            raise RuntimeError("No FlowSolution found in zone.")

        if structured:
            # I,J from coordinates
            gc = zone["GridCoordinates"]
            X = _first_existing(gc, NAME_COORD_X)
            if X is None:
                raise RuntimeError("CoordinateX not found in structured GridCoordinates.")
            J, I = np.asarray(X).shape
            u, v, cp = _read_cell_center_fields(fs, I, J)
        else:
            # Unstructured: expect 1D cell arrays
            def pick1(names):
                for nm in names:
                    arr = _get_dataarray(fs, nm)
                    if arr is not None:
                        return np.asarray(arr).reshape(-1).astype(np.float32)
                return None
            u = pick1(NAME_U); v = pick1(NAME_V); cp = pick1(NAME_CP)

        if u is None or v is None or cp is None:
            raise RuntimeError("Missing one or more fields (u, v, Cp). Extend NAME_* lists if needed.")

        y = np.stack([u, v, cp], axis=-1).astype(np.float32)

        g = np.array(
            [
                float(meta.get("Mach", meta.get("globals", [0.1, 1e6, 0.0])[0])),
                float(meta.get("Re",   meta.get("globals", [0.1, 1e6, 0.0])[1])),
                float(meta.get("AoA",  meta.get("globals", [0.1, 1e6, 0.0])[2])),
            ],
            dtype=np.float32,
        )

        data = Data(
            pos=torch.from_numpy(cent),
            edge_index=torch.from_numpy(edge_index.astype(np.int64)),
            edge_attr=torch.from_numpy(edge_attr),
            y=torch.from_numpy(y),
        )
        data.globals = torch.from_numpy(g)
        return data

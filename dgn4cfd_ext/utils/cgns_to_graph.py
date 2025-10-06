
from typing import Dict, Tuple
import numpy as np
import torch
import meshio

def _cell_centers(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    return points[cells].mean(axis=1)[:, :2]

def _build_edges_from_cells(cells: np.ndarray) -> torch.Tensor:
    edges = set()
    for f in cells:
        k = len(f)
        for i in range(k):
            a, b = int(f[i]), int(f[(i+1)%k])
            if a != b:
                edges.add((min(a,b), max(a,b)))
    undirected = list(edges)
    # make directed
    directed = undirected + [(b,a) for (a,b) in undirected]
    ei = torch.tensor(directed, dtype=torch.long).t().contiguous()  # (2,E)
    return ei

def cgns_to_graph(cgns_path: str, case_row: Dict, *, fields_at='cells') -> Dict[str, torch.Tensor]:
    """Read a UniFoil CGNS file and return a dict of PyG-compatible tensors.

    Args:
        cgns_path: path to .cgns
        case_row: dict with keys like {'Re':..., 'Mach':..., 'AoA':..., 'id':...}
        fields_at: 'cells' or 'points' depending on where UniFoil stored fields.

    Returns keys: pos, edge_index, edge_attr, vc, y, wall_mask, far_mask
    NOTE: This is a template; adjust field names to your CGNS ('VelocityX','VelocityY','Cp').
    """
    m = meshio.read(cgns_path)
    points = m.points[:, :2]
    # Grab the first cell block for simplicity; adjust if multiple blocks exist
    if 'quad' in m.cells_dict:
        cells = m.cells_dict['quad']
    elif 'triangle' in m.cells_dict:
        cells = m.cells_dict['triangle']
    else:
        # fallback: first block
        cells = list(m.cells)[0].data
    if fields_at == 'cells':
        X = _cell_centers(points, cells)
    else:
        X = points

    # helper to fetch field, checking point_data then cell_data
    def get_field(name):
        if name in m.point_data: return m.point_data[name]
        for _, d in m.cell_data_dict.items():
            if name in d: return d[name][0]
        raise KeyError(f'Field {name} not found in CGNS data.')

    # Try common aliases
    try:
        u = get_field('VelocityX')
        v = get_field('VelocityY')
    except KeyError:
        u = get_field('u'); v = get_field('v')
    try:
        Cp = get_field('Cp')
    except KeyError:
        Cp = get_field('PressureCoefficient')

    # If fields are at cells but X is at points, you may need interpolation.
    if fields_at == 'cells' and X.shape[0] != u.shape[0]:
        # align with cell centers
        X = _cell_centers(points, cells)

    pos = torch.tensor(X, dtype=torch.float32)
    # build edges from cell connectivity
    edge_index = _build_edges_from_cells(cells)
    # edge_attr = relative positions
    src, dst = edge_index
    edge_attr = pos[dst] - pos[src]

    # node conditions Vc = [Re, Ma, AoA, one-hot(node-type: inner, wall, far)]
    Re = float(case_row.get('Re', 1.0))
    Ma = float(case_row.get('Mach', 0.1))
    AoA = float(case_row.get('AoA', 0.0))
    N = pos.shape[0]
    Vc_scalar = torch.tensor(np.stack([np.full(N, Re), np.full(N, Ma), np.full(N, AoA)], axis=1), dtype=torch.float32)

    # Placeholder masks; replace with CGNS boundary-family-based masks if available
    wall_mask = torch.zeros(N, dtype=torch.bool)
    far_mask  = torch.zeros(N, dtype=torch.bool)
    inner_mask = ~(wall_mask | far_mask)
    omega = torch.stack([inner_mask.float(), wall_mask.float(), far_mask.float()], dim=1)

    # Targets y = [u, v, Cp]
    y = torch.tensor(np.stack([u, v, Cp], axis=1), dtype=torch.float32)
    if y.shape[0] != N:
        # If targets are at cells, remap or keep at cell graph instead of point graph.
        # For simplicity, force N to match by recomputing pos at cells:
        pos = torch.tensor(_cell_centers(points, cells), dtype=torch.float32)
        N = pos.shape[0]
        # recompute edge_index for a cell-centered graph is non-trivial; you can
        # instead choose to build a dual-graph (cells as nodes). For now, assume
        # fields_at='cells' and X already matches y.
        edge_index = _build_edges_from_cells(cells)
        src, dst = edge_index
        edge_attr = pos[dst] - pos[src]
        Vc_scalar = torch.tensor(np.stack([np.full(N, Re), np.full(N, Ma), np.full(N, AoA)], axis=1), dtype=torch.float32)
        wall_mask = torch.zeros(N, dtype=torch.bool)
        far_mask  = torch.zeros(N, dtype=torch.bool)
        inner_mask = ~(wall_mask | far_mask)
        omega = torch.stack([inner_mask.float(), wall_mask.float(), far_mask.float()], dim=1)

    vc = torch.cat([Vc_scalar, omega], dim=1)
    return dict(pos=pos, edge_index=edge_index, edge_attr=edge_attr, vc=vc, y=y,
                wall_mask=wall_mask, far_mask=far_mask)

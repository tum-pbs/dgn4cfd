
import os, argparse, pandas as pd, torch
from pathlib import Path
from tqdm import tqdm
from dgn4cfd_ext.utils.cgns_to_graph import cgns_to_graph

def main(args):
    data_root = Path(args.data_root)
    out_root  = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    proc_dir = out_root / 'proc'
    proc_dir.mkdir(exist_ok=True)

    # Expect a CSV with columns: split,id,cgns_path,Re,Mach,AoA
    df = pd.read_csv(args.split_csv)
    for split in ['train', 'val', 'test']:
        sub = df[df['split']==split].reset_index(drop=True)
        graphs = []
        for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f'Building {split}'):
            case_row = dict(Re=row['Re'], Mach=row['Mach'], AoA=row['AoA'], id=row['id'])
            g = cgns_to_graph(row['cgns_path'], case_row, fields_at=args.fields_at)
            # Standardize tensor dtypes
            for k in ['pos','edge_index','edge_attr','vc','y']:
                assert k in g, f'Missing key {k}'
            graphs.append(torch_geometric_Data_from_dict(g))
        # collate & save
        from torch_geometric.data import InMemoryDataset, Data
        data_list = graphs
        # Manual collate to avoid custom class dependency
        keys = data_list[0].keys
        slices = {k: [0] for k in keys}
        out = {}
        for k in keys:
            out[k] = torch.cat([d[k] for d in data_list], dim=0 if k!='edge_index' else 1)
        # Build slices (simple; assumes same-sized graphs not required)
        # For proper slices, using InMemoryDataset.collate is better. Let's do that:
        class _TmpDS(InMemoryDataset):
            def __init__(self, data_list):
                super().__init__('.')
                self.data, self.slices = self.collate(data_list)
        tmp = _TmpDS(data_list)
        torch.save((tmp.data, tmp.slices), proc_dir / f'unifoil_{split}.pt')
    print('Done. Files written to', proc_dir)

def torch_geometric_Data_from_dict(gdict):
    from torch_geometric.data import Data
    return Data(
        pos=gdict['pos'],
        edge_index=gdict['edge_index'],
        edge_attr=gdict['edge_attr'],
        vc=gdict['vc'],
        y=gdict['y'],
        wall_mask=gdict['wall_mask'],
        far_mask=gdict['far_mask'],
    )

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True, help='Root path of UniFoil data (not strictly used if cgns_path is absolute).')
    p.add_argument('--split-csv', required=True, help='CSV listing cases with columns: split,id,cgns_path,Re,Mach,AoA')
    p.add_argument('--out-root', required=True, help='Output root for cached .pt files')
    p.add_argument('--fields_at', default='cells', choices=['cells','points'])
    args = p.parse_args()
    main(args)

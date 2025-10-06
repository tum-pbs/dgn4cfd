
import os, torch

def test_cached_graphs_exist():
    proc = os.path.join('data','unifoil','proc')
    assert os.path.isdir(proc), 'Run preprocess_unifoil.py first to create cached graphs.'
    for split in ['train','val','test']:
        f = os.path.join(proc, f'unifoil_{split}.pt')
        assert os.path.isfile(f), f'Missing {f}'

def test_graph_shapes():
    proc = os.path.join('data','unifoil','proc')
    train_file = os.path.join(proc, 'unifoil_train.pt')
    data, slices = torch.load(train_file)
    # spot-check a few keys exist
    for k in ['pos','edge_index','edge_attr','vc','y']:
        assert hasattr(data, k), f'Missing key {k} in cached data.'

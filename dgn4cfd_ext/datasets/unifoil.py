
import os
import torch
from torch_geometric.data import InMemoryDataset
from typing import Optional
from pathlib import Path

class UniFoilDataset(InMemoryDataset):
    """Loads cached PyG graphs produced by preprocess_unifoil.py.

    Expects files at: <root>/proc/unifoil_{split}.pt
    """
    def __init__(self, root: str, split: str = 'train', transform=None, pre_transform=None):
        self.split = split
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'unifoil_{self.split}.pt']

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'proc')

    def process(self):
        # No-op: preprocessing occurs in scripts/preprocess_unifoil.py
        raise RuntimeError('Run scripts/preprocess_unifoil.py to create cached .pt files.')

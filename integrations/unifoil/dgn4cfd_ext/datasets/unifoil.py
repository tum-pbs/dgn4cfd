
import glob
import math
import os
import torch
from torch_geometric.data import Dataset, Data

class UniFoilDataset(Dataset):
    """
    Loads processed PyG graphs saved as .pt from a directory.
    Splits by contiguous index ranges (80/10/10 by default).
    """
    def __init__(self, root: str, split: str = "train", split_cfg=None, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.files = sorted(glob.glob(os.path.join(root, "*.pt")))
        if split_cfg is None:
            split_cfg = {"train": 0.8, "val": 0.1, "test": 0.1}
        n = len(self.files)
        n_train = math.floor(n * split_cfg["train"])
        n_val = math.floor(n * split_cfg["val"])
        idxs = {
            "train": range(0, n_train),
            "val":   range(n_train, n_train + n_val),
            "test":  range(n_train + n_val, n),
        }
        self.idxs = list(idxs[split])

    def len(self):
        return len(self.idxs)

    def get(self, idx):
        path = self.files[self.idxs[idx]]
        data: Data = torch.load(path)
        return data


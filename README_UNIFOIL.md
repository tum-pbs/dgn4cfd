
# UniFoil → DGN4CFD Integration (Scaffold)
This repo adds a **UniFoil dataset adapter** to DGN4CFD. You only need the **dataset files** (CGNS + case table). No need to clone UniFoil code.

## Prereqs
- Linux/WSL2 with Conda (Miniconda/Anaconda).
- (Optional) NVIDIA GPU visible in WSL via `nvidia-smi`.

## Dataset layout
Put a few `.cgns` files somewhere stable (absolute paths recommended), e.g.:


This folder contains a minimal, **drop-in** extension to plug the [UniFoil](https://github.com/rohitroxkp7/UniFoil) airfoil dataset into the [DGN4CFD](https://github.com/tum-pbs/dgn4cfd) training pipeline.

> Copy or symlink the `dgn4cfd_ext/` directory into your DGN4CFD repo root (or keep it adjacent and fix PYTHONPATH) and run:
>
> ```bash
> conda env create -f environment.yml
> conda activate dgn4cfd-unifoil
> python dgn4cfd_ext/scripts/preprocess_unifoil.py --data-root /path/to/unifoil --split-csv dgn4cfd_ext/configs/splits_example.csv --out-root data/unifoil
> python dgn4cfd_ext/scripts/train_unifoil.py --config dgn4cfd_ext/configs/unifoil.yaml
> ```

## Contents
- `dgn4cfd_ext/datasets/unifoil.py` — PyG dataset glue (loads cached `.pt` graphs).
- `dgn4cfd_ext/utils/cgns_to_graph.py` — CGNS→graph converter (using `meshio`), with TODOs to adapt to your CGNS layout.
- `dgn4cfd_ext/scripts/preprocess_unifoil.py` — one-off converter to cache graphs and write train/val/test splits.
- `dgn4cfd_ext/scripts/train_unifoil.py` — thin wrapper that imports DGN4CFD's trainer but points to the UniFoil dataset and config.
- `dgn4cfd_ext/configs/unifoil.yaml` — config example (out_channels=3 for u,v,Cp).
- `dgn4cfd_ext/configs/splits_example.csv` — example split file (edit with your case IDs).
- `tests/test_unifoil_build.py` — tiny shape/sanity tests for your preprocessed graphs.
- `environment.yml` — Conda env file.

## Quick start
1. Ensure you have a working clone of `dgn4cfd` next to this folder.
2. Put UniFoil CGNS + case tables under `/path/to/unifoil`.
3. Edit `configs/splits_example.csv` to list a few cases for a smoke test.
4. Run `preprocess_unifoil.py` to create cached graphs under `data/unifoil/proc`.
5. Run `train_unifoil.py` to do a one-epoch smoke test (batch_size=1).

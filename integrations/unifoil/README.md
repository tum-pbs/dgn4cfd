Quickstart:
1) conda create -n dgn4cfd-unifoil python=3.10 -y && conda activate dgn4cfd-unifoil
2) pip install -r integrations/unifoil/requirements.txt
3) pip install -e .   # installs dgn4unifoil library
4) python integrations/unifoil/scripts/convert_cgns_batch.py
5) python integrations/unifoil/scripts/train_unifoil.py

## Setup (works on GPU or CPU)

```bash
# create the base env
conda env create -f environment.yml
conda activate dgn4cfd-unifoil

# install Torch + PyG for your system (auto-detects GPU)
./integrations/unifoil/scripts/install_torch_pyg.sh

# optional: install your fork as an editable lib
pip install -e .


## Data preparation (ADF CGNS → HDF5 CGNS → PyG graphs)

This repo expects **HDF5-CGNS** files for conversion to PyTorch Geometric graphs.

### 0) Prerequisites
```bash
conda activate dgn4cfd-unifoil
conda install -y -c conda-forge cgns   # provides cgnsconvert

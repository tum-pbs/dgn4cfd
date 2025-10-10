Quickstart:
1) conda create -n dgn4cfd-unifoil python=3.10 -y && conda activate dgn4cfd-unifoil
2) pip install -r integrations/unifoil/requirements.txt
3) pip install -e .   # installs dgn4unifoil library
4) python integrations/unifoil/scripts/convert_cgns_batch.py
5) python integrations/unifoil/scripts/train_unifoil.py


import argparse, yaml, os
from pathlib import Path

def main(args):
    # This script is a thin wrapper that delegates to DGN4CFD's training entrypoint.
    # Modify this to match the exact API of the dgn4cfd repo you use.
    config = yaml.safe_load(open(args.config))
    # Example: set env var so the main repo can find our dataset class
    os.environ['PYTHONPATH'] = os.getcwd() + ':' + os.environ.get('PYTHONPATH','')
    # You will typically do something like:
    # from dgn4cfd.train import main as train_main
    # train_main(config)
    print('Load your dgn4cfd training entrypoint here with config:', config)
    print('NOTE: Replace this print with a call into the original repo.')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    main(args)

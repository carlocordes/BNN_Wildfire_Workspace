"""
Visualization of the results based on a test set and and trained model
"""

# Internal
from src.core.utils import load_config
from src.core.systempaths import MODELS, DATASETS
# External
import argparse
from pathlib import Path

import torch

def main(model : str, dataset : str):
    # Load model
    print(DATASETS / dataset)
    print(MODELS / model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Visualize model predictions against a sample dataset')
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help='Name of model in data/models/',
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of testing dataset at /data/datasets"
    )
    args = parser.parse_args()
    main(model = args.model, dataset = args.dataset)
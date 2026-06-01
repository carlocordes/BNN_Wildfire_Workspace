"""
Evaluates model with respect to expected true positives/
expected true negatives
"""

# Internal
from src.core.utils import load_config
from src.models.vit.vit import STViT
from scripts.dataset import SpatialTemporalDataset, skip_missing_collate_fn

# External
import re
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# --------------- CONFIG --------------- #
cfg_base = Path('files', 'configs')
exp_base = Path('files', 'experiments')

out_path = Path('exports', 'model_evaluation', 'benchmarks.parquet')

# Name configurations
run_names = {
    'SampleExt3' : 't009_1',
    'SampleExt5' : 't009_2',
    'SampleExt7' : 't009_3',
    'SampleExt1' : 't009_4',

    'Lead-5' : 't010_1',
    'Lead0' : 't009_2', 
    'Lead5' : 't010_2',
    'Lead10' : 't010_3',
    'Lead20' : 't010_4',
}

calibration_logit = math.log(600)
# -------------------------------------- #




if __name__ == '__main__':
    df_runs = pd.DataFrame([
        { 
            'name' : name,
            'cfg_path' : cfg_base / f'config_{exp}.yaml',
            'model_path' : exp_base / exp / f'{exp}_model_best.pt',
            'log_path' : exp_base / exp / 'training.log'
        } for name, exp in run_names.items()
    ])



    # Initialize empty arrays to later store into df_runs
    train_histories, val_histories, test_losses, best_val_losses = [], [], [], []
    yearly_burn, yearly_risk = [], []

    for idx, row in df_runs.iterrows():
        pass
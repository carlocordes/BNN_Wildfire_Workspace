"""
Evaluate performance metrics of multiple models accross disciplines with Temperature Scaling calibration
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

# Helper functions
def parse_training_log(log_path: Path) -> Dict[str, Any]:
    """Parses training log file to extract histories."""
    epoch_pattern = re.compile(
        r"Epoch \[\d+/100\] Train Loss: (?P<train_loss>[\d\.]+) \| Val Loss: (?P<val_loss>[\d\.]+)"
    )
    test_pattern = re.compile(r"Final Test Loss: (?P<test_loss>[\d\.]+)")

    results = {"train_losses": [], "val_losses": [], "test_loss": None}

    with open(log_path, 'r') as f:
        for line in f:
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                results["train_losses"].append(float(epoch_match.group("train_loss")))
                results["val_losses"].append(float(epoch_match.group("val_loss")))
                continue
            test_match = test_pattern.search(line)
            if test_match:
                results["test_loss"] = float(test_match.group("test_loss"))
    return results



def evaluate(model, cfg_path, device):
    dataset = SpatialTemporalDataset(cfg_path, split_type="test", benchmark_mode=True)
    set_dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1,
        pin_memory=False, collate_fn=skip_missing_collate_fn
    )

    model.eval()

    # Empty arrays
    all_targets, all_probs = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(set_dataloader):
            print(f'Processing test batch_idx {batch_idx + 1} / {len(set_dataloader)}')
            if batch is None:
                continue

            static_x = batch['static'].to(device)
            single_dynamic_x = batch['single_dynamic'].to(device)
            dynamic_x = batch['dynamic'].to(device)
            targets = batch['target'].to(device)
            first_date = datetime.strptime(batch['meta_first_date'][0], '%Y-%m-%d')

            ## Predict (Raw Logits)
            logits = model(x_static=static_x, x_dynamic=dynamic_x, x_single_dynamic=single_dynamic_x)

            # Calibrate
            cal_logits = logits - calibration_logit

            probs = torch.sigmoid(cal_logits)

            np_target = targets.squeeze(0).detach().cpu().numpy()
            np_prob = probs.squeeze(0).detach().cpu().numpy()

            all_targets.append(np_target)
            all_probs.append(np_prob)

    # Produce yearly aggregated risk values
    probs_stack = np.stack(all_probs)
    targets_stack = np.stack(all_targets)
    yearly_exp_risk = np.sum(probs_stack, axis = 0).flatten()
    yearly_burn= np.max(targets_stack, axis=0).flatten()


    return {
        'aggregated' : {
            'yearly_burn' : yearly_burn,
            'yearly_risk' : yearly_exp_risk
        }
    }
    


def load_model(cfg_model : nn.Module, cfg_data, model_path, device):
    model = STViT(
        num_dynamic_channels=cfg_model['num_dynamic_channels'],
        num_static_channels=cfg_model['num_static_channels'] + cfg_model['num_single_dynamic_channels'],
        num_timestamps_per_sample=cfg_data['temporal_extent']['sample_extent'],
        patch_size=cfg_model['patch_size'],
        embedding_dim=cfg_model['embedding_dim']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)


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
        print(f"\n Evaluating model run: {row['name']}")
        
        ## 1. Parse logfile
        # Parse
        log_data = parse_training_log(row['log_path'])

        # Extract
        train_histories.append(log_data['train_losses'])
        val_histories.append(log_data['val_losses'])
        test_losses.append(log_data['test_loss'])
        best_val_losses.append(min(log_data['val_losses']) if log_data['val_losses'] else None)


        ## 2. Evaluate on validation set
        # Setup Configs and Models
        cfg = load_config(row['cfg_path'])
        device = 'mps'
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg['training']["pos_weight"]])).to(device)
        model = load_model(cfg_model=cfg['model'], cfg_data=cfg['data'], model_path=row['model_path'], device=device)

        # Evaluate
        metrics = evaluate(model=model, cfg_path=row['cfg_path'], device=device)

        # Store
        metrics_hist = metrics['aggregated']
        yearly_burn.append(metrics_hist['yearly_burn'])
        yearly_risk.append(metrics_hist['yearly_risk'])

        ## 3. OPT


    ## Append to df for all models
    # Log contents
    df_runs['train_history'] = train_histories
    df_runs['val_history'] = val_histories
    df_runs['test_loss'] = test_losses
    df_runs['best_val_loss'] = best_val_losses

    # History contents
    df_runs['yearly_exp_risk'] = yearly_risk
    df_runs['yearly_agg_burn'] = yearly_burn


    # Drop columns
    df_runs = df_runs.drop(columns = ['cfg_path', 'model_path', 'log_path'])

    # Write to parquet
    df_runs.to_parquet(path = out_path)
    print(f'Saved to {out_path}')
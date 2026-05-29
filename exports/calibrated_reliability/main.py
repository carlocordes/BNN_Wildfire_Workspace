"""
Evaluate performance metrics of multiple models accross disciplines with Temperature Scaling calibration
"""

# Internal
from src.core.utils import load_config
from src.models.vit.vit import STViT
from scripts.dataset import SpatialTemporalDataset, skip_missing_collate_fn

# External
import numpy as np
import math
from pathlib import Path
import pandas as pd
from datetime import datetime
import scipy.optimize as opt
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryCalibrationError
from sklearn.calibration import calibration_curve


# --------------- CONFIG --------------- #
cfg_base = Path('files', 'configs')
exp_base = Path('files', 'experiments')

out_path = Path('exports', 'model_evaluation', 'benchmarks.parquet')

# Name configurations
run_names = {
    'SampleExt3' : 't009_1',
    #'SampleExt5' : 't009_2',
    #'SampleExt7' : 't009_3',
    'SampleExt1' : 't009_4',

    #'Lead-5' : 't010_1',
    #'Lead0' : 't009_2', 
    #'Lead5' : 't010_2',
    #'Lead10' : 't010_3',
    #'Lead20' : 't010_4',
}

calibration_logit = math.log(600)

# -------------------------------------- #


def load_model(cfg_model, cfg_data, model_path, device):
    model = STViT(
        num_dynamic_channels=cfg_model['num_dynamic_channels'],
        num_static_channels=cfg_model['num_static_channels'] + cfg_model['num_single_dynamic_channels'],
        num_timestamps_per_sample=cfg_data['temporal_extent']['sample_extent'],
        patch_size=cfg_model['patch_size'],
        embedding_dim=cfg_model['embedding_dim']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)



def evaluate(model: nn.Module, cfg_path, device):
    dataset = SpatialTemporalDataset(
        cfg_path,
        split_type="test",
        benchmark_mode=True
    )

    set_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        collate_fn=skip_missing_collate_fn
    )

    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(set_dataloader):
            print(f'Processing batch {batch_idx + 1} / {len(set_dataloader)}')

            if batch is None:
                continue

            static_x = batch['static'].to(device)
            single_dynamic_x = batch['single_dynamic'].to(device)
            dynamic_x = batch['dynamic'].to(device)
            targets = batch['target'].to(device)

            logits = model(
                x_static=static_x,
                x_dynamic=dynamic_x,
                x_single_dynamic=single_dynamic_x
            )

            # calibrated logits
            logits = logits - calibration_logit
            probs = torch.sigmoid(logits)

            np_target = targets.squeeze(0).detach().cpu().numpy()
            np_prob = probs.squeeze(0).detach().cpu().numpy()

            all_targets.append(np_target)
            all_probs.append(np_prob)


    return all_probs, all_targets


def plot_aggregate_captured_fire_ratio(df_runs: pd.DataFrame):
    """
    Plots the Spatially Aggregated Captured-Fire Ratio.
    Collapses the temporal dimension first by calculating the accumulated expected
    risk map vs. the cumulative annual burn map.
    """
    plt.figure(figsize=(10, 7))
    
    for idx, row in df_runs.iterrows():
        model_name = row['name']
        
        # 1. Stack list of timesteps into single arrays: shape (Timesteps, H, W) or (Timesteps, Patches)
        probs_stack = np.stack(row['all_probs'])
        targets_stack = np.stack(row['all_targets'])
        
        # 2. TEMPORAL AGGREGATION
        # Sum risk probabilities over time to find expected fire accumulation per pixel
        spatial_expected_risk = np.sum(probs_stack, axis=0).flatten()
        
        # Max over time for targets: 1 if pixel burned at least once during the year, else 0
        spatial_actual_burns = np.max(targets_stack, axis=0).flatten()
        
        # 3. Sort pixels by cumulative annual risk in descending order
        sort_idx = np.argsort(spatial_expected_risk)[::-1]
        sorted_actual_burns = spatial_actual_burns[sort_idx]
        
        # 4. Compute cumulative spatial fires caught
        cum_fires = np.cumsum(sorted_actual_burns)
        total_fires = cum_fires[-1] if len(cum_fires) > 0 else 0
        
        if total_fires == 0:
            print(f"Warning: No actual fires found in the aggregated map for {model_name}. Skipping.")
            continue
            
        # 5. Scale to percentages
        y_captured_fraction = cum_fires / total_fires
        x_area_fraction = np.linspace(0, 1, len(cum_fires))
        
        # Plot model performance
        plt.plot(x_area_fraction * 100, y_captured_fraction * 100, label=f'{model_name}', linewidth=2)
        
        # Print operational spatial metrics to the console
        print(f"\n--- Spatially Aggregated Metrics [{model_name}] ---")
        for pct in [1, 5, 10]:
            idx_pct = int((pct / 100) * (len(x_area_fraction) - 1))
            print(f"Top {pct}% highest-risk land area accounts for {y_captured_fraction[idx_pct]*100:.2f}% of the year's total burned pixels.")

    # Reference baseline line
    plt.plot([0, 100], [0, 100], color='grey', linestyle='--', label='Random Spatial Baseline')
    
    # Formatting
    plt.title('Spatially Aggregated Captured-Fire Ratio (Annual Risk)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('% of Geographic Land Area Monitored (Sorted by Annual Accumulated Risk)', fontsize=12, labelpad=10)
    plt.ylabel('% of Total Annual Burned Pixels Captured', fontsize=12, labelpad=10)
    
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    
    plt.show()

if __name__ == '__main__':

    all_probs, all_targets = [], []
    df_runs = pd.DataFrame([
        { 
            'name' : name,
            'cfg_path' : cfg_base / f'config_{exp}.yaml',
            'model_path' : exp_base / exp / f'{exp}_model_best.pt',
            'log_path' : exp_base / exp / 'training.log'
        } for name, exp in run_names.items()
    ])



    for idx, row in df_runs.iterrows():
        print(f"\n Evaluating model run: {row['name']}")
        cfg = load_config(row['cfg_path'])



        device = 'mps'
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg['training']["pos_weight"]])).to(device)
        model = load_model(cfg_model=cfg['model'], cfg_data=cfg['data'], model_path=row['model_path'], device=device)


        probs, targets = evaluate(model, row['cfg_path'], device = device)

        all_probs.append(probs)
        all_targets.append(targets)

    df_runs['all_probs'] = all_probs
    df_runs['all_targets'] = all_targets



    plot_aggregate_captured_fire_ratio(df_runs)
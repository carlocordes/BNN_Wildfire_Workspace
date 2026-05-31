# Internal
from src.core.utils import load_config
from src.models.vit.vit import STViT
from scripts.dataset import SpatialTemporalDataset, skip_missing_collate_fn

# External
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
from pathlib import Path
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

# ----- CONFIG ----- #
correction_logit = math.log(600)
out_path = Path('exports', 'seasonal_prediction', 'season_pred.png')
# ------------------ #


def load_trained_model(cfg_model : nn.Module, cfg_data, model_path, device):
    model = STViT(
        num_dynamic_channels=cfg_model['num_dynamic_channels'],
        num_static_channels=cfg_model['num_static_channels'] + cfg_model['num_single_dynamic_channels'],
        num_timestamps_per_sample=cfg_data['temporal_extent']['sample_extent'],
        patch_size=cfg_model['patch_size'],
        embedding_dim=cfg_model['embedding_dim']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)

def predict_ds_from_model(model, dataloader : DataLoader, device):
    model.eval()

    all_probs = []
    all_dates = []

    # Primary sweep with full model
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f'Processing test batch_idx {batch_idx + 1} / {len(dataloader)}')
            if batch is None:
                continue

            static_x = batch['static'].to(device)
            single_dynamic_x = batch['single_dynamic'].to(device)
            dynamic_x = batch['dynamic'].to(device)

            first_date = datetime.strptime(batch['meta_first_date'][0], '%Y-%m-%d')

            ## Predict
            logits = model(x_static=static_x,
                           x_dynamic=dynamic_x,
                           x_single_dynamic=single_dynamic_x)

            probs = torch.sigmoid(logits).squeeze() / 50

            all_probs.append(probs)
            all_dates.append(first_date)
            

    # Stack all results together
    all_preds_tensor = torch.stack(all_probs, dim = 0)
    
    return all_preds_tensor.detach().cpu().numpy(), all_dates

def plot_seasonal_preds(preds: np.ndarray, dates: list):
    """
    Plots the first prediction of each season (quarter) in a 2x2 grid
    using a unified Turbo colormap and a shared horizontal colorbar at the bottom.
    """
    num_samples = len(preds)
    if num_samples < 4:
        raise ValueError(f"Not enough data points ({num_samples}) to split into 4 seasons.")

    # 1. Calculate indices for the start of each quarter
    quarter_size = num_samples // 4
    seasonal_indices = [0, quarter_size, quarter_size * 2, quarter_size * 3]
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']

    # Extract the specific 4 frames and their corresponding dates
    selected_preds = [preds[i] for i in seasonal_indices]
    selected_dates = [dates[i].strftime('%Y-%m-%d') for i in seasonal_indices]

    # 2. Determine global min/max for the unified colorbar
    vmin = min(p.min() for p in selected_preds)
    vmax = max(p.max() for p in selected_preds)

    # 3. Initialize the Seaborn/Matplotlib figure layout (2 rows, 2 columns)
    sns.set_theme(style="white")
    
    # Square-ish dimensions work best for a 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 18))
    
    # CRITICAL: Flatten axes from a 2D matrix [[ax1, ax2], [ax3, ax4]] to a 1D array
    axes = axes.flatten()

    # 4. Plot each season
    for i, ax in enumerate(axes):
        sns.heatmap(
            selected_preds[i],
            ax=ax,
            cmap='gnuplot2',      # Changed back to turbo as requested originally
            vmin=vmin,
            vmax=vmax,
            cbar=False,        # Turn off individual colorbars
            xticklabels=False, # Hide ticks for a cleaner map look
            yticklabels=False
        )
        ax.set_title(f"{seasons[i]}\n({selected_dates[i]})", fontsize=12, fontweight='bold')

    # Adjust layout to make room for the bottom colorbar safely
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15) 

    # 5. Add the single, unified colorbar at the bottom
    # [left, bottom, width, height] relative to the whole figure
    cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.03]) 
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    hcbar = ColorbarBase(
        cbar_ax, 
        cmap=plt.get_cmap('gnuplot2'), 
        norm=norm,
        orientation='horizontal'
    )
    hcbar.set_label('Prediction Probability / Intensity', fontsize=12, labelpad=8)

    # Save and show
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Seasonal prediction plot saved to {out_path}")
    return None


if __name__ == '__main__':
    
    # Load cfg
    exp_name = 't010_1' # 't009_1'

    cfg_file = 'config_' + exp_name + '.yaml'
    cfg_path = Path('files', 'configs', cfg_file)
    cfg = load_config(cfg_path)
    cfg_model = cfg['model']
    cfg_data = cfg['data']

    # Load model path
    model_path = Path('files', 'experiments', exp_name, f'{exp_name}_model_best.pt')

    # Device
    device = 'mps'

    # Select baseline model
    model = load_trained_model(
        cfg_model=cfg_model,
        cfg_data=cfg_data,
        model_path=model_path,
        device = device
    )

    # Data
    dataset = SpatialTemporalDataset(cfg_path, split_type="test", benchmark_mode=True)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1,
        pin_memory=False, collate_fn=skip_missing_collate_fn
    )


    preds, dates = predict_ds_from_model(model = model, dataloader = dataloader, device=device)

    plot_seasonal_preds(preds, dates)

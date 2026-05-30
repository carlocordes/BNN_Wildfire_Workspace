"""
Takes an instance of STViT and measures predictional variability
of a baseline model against ablated versions of the same model
"""

# Internal
from src.core.utils import load_config
from src.models.vit.vit import STViT
from scripts.dataset import SpatialTemporalDataset, skip_missing_collate_fn


# External
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


# -------- CONFIG -------- #
out_path = Path('exports', 'ablation')
# --- Helper functions --- #

class ablation():
    def __init__(self, static_ablations, dynamic_ablations, name : str):
        self.static_ablations = static_ablations
        self.dynamic_ablations = dynamic_ablations
        self.name = name

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


def evaluate_ablated_model(model, dataloader : DataLoader, cfg_path, device, ablation_cfg : ablation):
    model.eval()

    all_probs = []

    # Primary sweep with full model
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f'Processing test batch_idx {batch_idx + 1} / {len(dataloader)}')
            if batch is None:
                continue

            static_x = batch['static'].to(device)
            single_dynamic_x = batch['single_dynamic'].to(device)
            dynamic_x = batch['dynamic'].to(device)
            #targets = batch['target'].to(device)
            #first_date = datetime.strptime(batch['meta_first_date'][0], '%Y-%m-%d')

            ## Predict
            logits = model(x_static=static_x, x_dynamic=dynamic_x, x_single_dynamic=single_dynamic_x,
                           ablate_static_idxs = ablation_cfg.static_ablations,
                           ablate_dynamic_idxs = ablation_cfg.dynamic_ablations # Ablated versions
                           )

            probs = torch.sigmoid(logits).squeeze()

            all_probs.append(probs)
            

    # Stack all results together
    all_preds_tensor = torch.stack(all_probs, dim = 0)
    
    return all_preds_tensor.detach().cpu().numpy()



# --- Plotting functions --- #
def plot_implicit_seasonal_trend(all_predictions, out_path=Path('.')):
    """
    Computes image-level MAE over the 25 implicit timestamps
    and tracks the seasonal trend across a calendar year with unique line colors.
    """
    if "Baseline" not in all_predictions:
        raise KeyError("Could not find 'Baseline' key in all_predictions dictionary!")
        
    baseline_preds = all_predictions["Baseline"]
    num_timestamps = baseline_preds.shape[0]  # Extracts 25
    time_steps = np.arange(num_timestamps)
    
    # Filter out baseline to get the true number of ablation lines
    ablation_keys = [name for name in all_predictions.keys() if name != "Baseline"]
    num_ablations = len(ablation_keys)
    
    # --- FIX RECURRING COLORS ---
    # Use a large discrete colormap like 'tab20' if you have <= 20 lines, 
    # or a continuous map like 'jet' / 'turbo' if you have a ton.
    if num_ablations <= 20:
        colormap = plt.cm.get_cmap('tab20')
    else:
        colormap = plt.cm.get_cmap('jet')
        
    # Generate an array of unique color values spaced evenly between 0 and 1
    unique_colors = [colormap(i) for i in np.linspace(0, 1, num_ablations)]
    
    plt.figure(figsize=(14, 6.5))
    
    # Process each ablation configuration with its designated unique color
    for idx, name in enumerate(ablation_keys):
        ablated_preds = all_predictions[name]
            
        # Compute MAE for each image slice individually (averaging over H and W dimensions)
        image_level_mae = np.mean(np.abs(baseline_preds - ablated_preds), axis=(1, 2))
        
        # Plot with our unique color token mapping
        plt.plot(
            time_steps, 
            image_level_mae, 
            marker='o',          # Re-added small markers so individual points are visible
            markersize=4,
            linestyle='-', 
            linewidth=1.8, 
            color=unique_colors[idx], 
            label=name
        )
        
    # Style the X-axis to mimic an implicit annual cycle (Jan -> Dec)
    plt.xlabel('Implicit Timeline Across One Year (Sample Index)', fontsize=12, labelpad=10)
    plt.ylabel('Image-Level MAE (Deviation from Baseline)', fontsize=12)
    plt.title('Seasonal Ablation Profile: Component Reliance Across Time', fontsize=14, fontweight='bold', pad=15)
    
    # Create clear x-ticks for your 25 frames
    plt.xticks(time_steps)
    plt.xlim(-0.5, num_timestamps - 0.5)
    
    # Add minor decorations to split the year visually if desired
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Position the legend outside or stretched slightly so long names don't overlap lines
    plt.legend(loc='upper left', fontsize=9, frameon=True, shadow=True, bbox_to_anchor=(1.01, 1))
    
    # Visual assist: Shading sections to denote changing seasons if relevant
    plt.axvspan(0, num_timestamps // 4, color='blue', alpha=0.02, label='Winter/Early Year')
    plt.axvspan(num_timestamps * 3 // 4, num_timestamps, color='blue', alpha=0.02)
    
    # Adjusted tight layout to account for the legend box being on the outside right
    plt.tight_layout()
    
    fig_name = 'seasonal_implicit_trend.png'
    out_file = Path(out_path) / fig_name
    out_file.parent.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
    
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Successfully generated unique-colored seasonal chart: '{out_file}'")


def plot_ablation_metrics(all_predictions):
    """
    Computes MAE and MSE for each ablation against the baseline,
    and plots them side-by-side in a single figure.
    """
    # Extract baseline
    baseline_preds = all_predictions["Baseline"]
    
    # Storage for results
    ablation_names = []
    mae_values = []
    mse_values = []
    
    # Compute metrics for each ablation case
    for name, ablated_preds in all_predictions.items():
        if name == "Baseline":
            continue
        
        ablation_names.append(name)
        
        # 1. Mean Absolute Error (Average absolute probability shift)
        mae = np.mean(np.abs(baseline_preds - ablated_preds))
        mae_values.append(mae)
        
        # 2. Mean Squared Error (Penalizes highly drastic confidence shifts)
        mse = np.mean((baseline_preds - ablated_preds) ** 2)
        mse_values.append(mse)
    
    # --- PLOTTING LOGIC ---
    # Create one figure containing 2 subplots side-by-side (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    
    # Horizontal positions for the bars
    y_pos = np.arange(len(ablation_names))
    
    # 1. Left Plot: Mean Absolute Error (MAE)
    axes[0].barh(y_pos, mae_values, color='skyblue', edgecolor='navy', height=0.5)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(ablation_names, fontsize=10)
    axes[0].invert_yaxis()  # Put top ablations at the top
    axes[0].set_xlabel('Mean Absolute Error (MAE)', fontsize=12)
    axes[0].set_title('Average Probability Shift per Pixel', fontsize=13, fontweight='bold')
    axes[0].grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add numerical value tags to the end of each bar
    for i, v in enumerate(mae_values):
        axes[0].text(v + (max(mae_values) * 0.01), i, f'{v:.5f}', va='center', fontsize=9)

    # 2. Right Plot: Mean Squared Error (MSE)
    axes[1].barh(y_pos, mse_values, color='salmon', edgecolor='darkred', height=0.5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([])  # Hide labels on right plot to avoid overlapping
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Mean Squared Error (MSE)', fontsize=12)
    axes[1].set_title('Squared Deviation', fontsize=13, fontweight='bold')
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(mse_values):
        axes[1].text(v + (max(mse_values) * 0.01), i, f'{v:.5f}', va='center', fontsize=9)

    # Clean up and layout adjustment
    plt.tight_layout()
    
    # Save and display
    fig_name = 'ablation_mse_mae.png'
    out_file = out_path / fig_name
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Successfully generated and saved comparison plot to '{out_file}'")



if __name__ == '__main__':
    
    # Load cfg
    exp_name = 't009_4' # 't009_1'

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


    # Build ablation configurations
    ablations = [
        ablation(static_ablations = None, dynamic_ablations = None, name = 'Baseline'),
        ablation(static_ablations=None, dynamic_ablations=[0, 1], name = 'Aspect'),
        ablation(static_ablations=None, dynamic_ablations=[2], name = 'Slope'),
        ablation(static_ablations = [0, 1, 2], dynamic_ablations = None, name = 'Terrain'),
        ablation(static_ablations = [3], dynamic_ablations = None, name = 'Roads'),
        ablation(static_ablations = [4], dynamic_ablations = None, name = 'Burn History'),
        ablation(static_ablations=[5], dynamic_ablations=None, name = 'Precipitation'),
        ablation(static_ablations=None, dynamic_ablations=[0], name = 'NDVI'),
        ablation(static_ablations=None, dynamic_ablations=[1], name = 'NDWI'),
        ablation(static_ablations=None, dynamic_ablations=[2, 3, 4], name = 'Wind'),
        ablation(static_ablations=None, dynamic_ablations=[2, 3], name = 'Wind Direction'),
        ablation(static_ablations=None, dynamic_ablations=[4], name = 'Wind Speed'),
        ablation(static_ablations=None, dynamic_ablations=[5], name = 'LST'),
    ]
    

    all_predictions = {}
    for abltn in ablations:
        
        # Precict with ablationo configuration
        ablation_predictions = evaluate_ablated_model(
            model=model,
            dataloader=dataloader,
            cfg_path=cfg_path,
            device=device,
            ablation_cfg=abltn)
        
        # Store
        all_predictions[abltn.name] = ablation_predictions


    plot_ablation_metrics(all_predictions)
    plot_implicit_seasonal_trend(all_predictions)

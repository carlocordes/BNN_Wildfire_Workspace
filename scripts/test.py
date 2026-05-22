"""
Visualization of the results based on a test set and trained model saved as a GIF
"""

# Internal
from src.core.utils import load_config
from src.core.systempaths import MODELS, DATASETS
from src.models.vit.vit import STViT



# External
import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scripts.dataset import SpatialTemporalDataset, skip_missing_collate_fn
from torch.utils.data import DataLoader, Dataset


def main(model_path: Path, config_path: Path):
    # Load config parameters
    cfg = load_config(config_path)
    cfg_model = cfg["model"]
    cfg_training = cfg['training']
    cfg_data = cfg['data']

    # Force batch size to 1 to ensure we iterate through samples individually
    cfg_training['batch_size'] = 1

    dataset = SpatialTemporalDataset(config_path, split_type="val")

    set_dataloader = DataLoader(
        dataset, 
        batch_size=cfg_training['batch_size'], 
        shuffle=False,         
        num_workers=4,         
        pin_memory=True,       
        collate_fn=skip_missing_collate_fn
    )

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STViT(
        num_dynamic_channels=cfg_model['num_dynamic_channels'],
        num_static_channels=cfg_model['num_static_channels'] + 
                        cfg_model['num_single_dynamic_channels'],
        num_timestamps_per_sample=cfg_data['temporal_extent']['sample_extent'],
        patch_size=cfg_model['patch_size'],
        embedding_dim=cfg_model['embedding_dim']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- 1. Extract and process data batches first ---
    print("Extracting evaluation batches...")
    batches = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(set_dataloader):
            #if batch_idx > 10: 
            #    break

            print(f'Predicting batch {batch_idx+1} / {len(set_dataloader)}')
            
            static_x = batch['static'].to(device)
            dynamic_x = batch['dynamic'].to(device)
            targets = batch['target'].to(device)
            single_dynamic = batch['single_dynamic'].to(device)
            first_date = batch['meta_first_date']

            logits = model(static_x, dynamic_x, single_dynamic)
            prediction = torch.sigmoid(logits).squeeze().cpu().numpy()
            target = targets.squeeze().cpu().numpy()
            
            batches.append((prediction, target, batch_idx, first_date))

    # Ensure we actually have data before proceeding
    if not batches:
        raise ValueError("The dataloader returned no samples.")

    # --- 2. Dynamic Aspect Ratio Setup ---
    # Grab the real spatial shape from the very first processed frame
    sample_prediction = batches[0][0]
    height, width = sample_prediction.shape  # Dynamic height and width

    # Dynamically adjust the figure size so rectangular tensors don't stretch weirdly
    # We give width a bit more padding since there are 2 subplots side by side
    aspect_ratio = width / height
    fig_height = 6
    fig_width = fig_height * aspect_ratio * 2.2  
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    
    # Initialize the plots using the real sample shape instead of a hardcoded 64x64 square
    import numpy as np
    init_data = np.zeros((height, width)) 
    
    # aspect='equal' ensures pixels remain perfectly square even if the whole tensor is rectangular
    im1 = ax1.imshow(init_data, cmap='magma', vmin=0, vmax=1, aspect='equal')
    im2 = ax2.imshow(init_data, cmap='magma', vmin=0, vmax=1, aspect='equal')
    
    ax1.axis('off')
    ax2.axis('off')
    
    title1 = ax1.set_title('Prediction (Batch 0)', fontsize=12)
    title2 = ax2.set_title('Target (Batch 0)', fontsize=12)

    # Universal color bar setup
    cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation='horizontal', pad=0.1, shrink=0.5)
    cbar.set_label('Wildfire Probability / Intensity')

    # --- 3. Animation Update Function ---
    def update(frame_idx):
        prediction, target, batch_idx, date = batches[frame_idx]
        
        im1.set_data(prediction)
        im2.set_data(target)
        
        title1.set_text(f'Prediction: {date} (Batch {batch_idx})')
        title2.set_text(f'Target: {date} (Batch {batch_idx})')
        
        return im1, im2, title1, title2

    print("Generating GIF frames...")
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(batches), 
        interval=100, 
        blit=True
    )

    output_filename = "prediction_results.gif"
    print(f"Saving GIF to {output_filename}...")
    ani.save(output_filename, writer='pillow')
    plt.close(fig) 
    print("Done.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize model predictions and save to GIF')
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    
    main(model_path=Path(args.model), config_path=Path(args.config))
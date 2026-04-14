"""
Visualization of the results based on a test set and trained model saved as a GIF
"""

# Internal
from src.core.utils import load_config
from src.core.systempaths import MODELS, DATASETS
from src.models.vit.vit import STViT
from scripts.train import WildfireDataset
from scripts.train import build_dataloader

# External
import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main(model_path: Path, dataset_path: Path, config_path: Path):
    # Load config parameters
    cfg = load_config(config_path)
    cfg_model = cfg["model"]
    cfg_training = cfg['training']

    # Force batch size to 1 to ensure we iterate through samples individually
    cfg_training['batch_size'] = 1

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STViT(
        num_dynamic_channels=4,
        num_static_channels=3,
        num_timestamps_per_sample=10,
        patch_size=cfg_model['patch_size'],
        embedding_dim=cfg_model['embedding_dim']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load dataset
    dataset_dict = torch.load(dataset_path, weights_only=False)
    dataloader = build_dataloader(dataset_dict, cfg_training)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    frames = []  # List to store artists for each frame

    print("Generating frames...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Limit frames if dataset is huge (optional)
            if batch_idx > 50: 
                break

            static_x = batch['static'].to(device)
            dynamic_x = batch['dynamic'].to(device)
            targets = batch['target'].to(device)

            logits = model(static_x, dynamic_x)
            prediction = torch.sigmoid(logits).squeeze().cpu().numpy()
            target = targets.squeeze().cpu().numpy()

            # Plot both images
            # Note: add_axes=False and using the return of imshow to create a list of artists
            im1 = ax1.imshow(prediction, cmap='magma', animated=True)
            im2 = ax2.imshow(target, cmap='magma', animated=True)
            
            # Add text labels as artists so they update/persist correctly in the GIF
            txt1 = ax1.text(0.5, 1.05, f'Prediction (Batch {batch_idx})', 
                            transform=ax1.transAxes, ha="center", color="black", fontsize=12)
            txt2 = ax2.text(0.5, 1.05, f'Target (Batch {batch_idx})', 
                            transform=ax2.transAxes, ha="center", color="black", fontsize=12)

            frames.append([im1, im2, txt1, txt2])

    # Clean up axes
    ax1.axis('off')
    ax2.axis('off')

    # Create the animation
    # interval=1000 means 1000ms (1 second) per frame
    ani = animation.ArtistAnimation(fig, frames, interval=1000, blit=True, repeat_delay=1000)

    output_filename = "prediction_results.gif"
    print(f"Saving GIF to {output_filename}...")
    ani.save(output_filename, writer='pillow')
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize model predictions and save to GIF')
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    main(model_path=Path(args.model), dataset_path=Path(args.dataset), config_path=Path(args.config))
"""
Visualization of the results based on a test set and and trained model
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
def main(modelname : str, datasetname : str, config_path : str):
    # Load config parameters
    cfg = load_config(config_path)
    cfg_model = cfg["model"]

    # Load model
    model = STViT(
        num_modules=cfg_model["num_modules"],
        patch_size=cfg_model["patch_size"],
        embedding_dim=cfg_model["embedding_dim"]
    )
    model.load_state_dict(torch.load(MODELS / (modelname + '.pt')))
    model.eval()

    #print(model)

    # Load dataset
    dataset_path = DATASETS / f"{datasetname}.pt"
    dataset = torch.load(dataset_path, weights_only=False)


    input_tensor, target_tensor = dataset[30]  # Get the first sample for testing

    # Predict and visualize results
    with torch.no_grad():
        logits = model(input_tensor.unsqueeze(0))  # Add batch dimension
        prediction = torch.sigmoid(logits).squeeze().cpu().numpy()
        target = target_tensor.cpu()

        # 4. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # If your data is 1-band (e.g., [1, H, W]), we squeeze the channel too
    im1 = axes[0].imshow(target.squeeze(), cmap='viridis')
    axes[0].set_title("Target (Ground Truth)")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(prediction.squeeze(), cmap='viridis')
    axes[1].set_title("Model Prediction")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()

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
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    main(modelname = args.model, datasetname = args.dataset, config_path = args.config)

    # Example usage from /code:
    # uv run -m scripts.test --model test_model --dataset test_dataset
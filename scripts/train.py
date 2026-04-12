# Internal
from src.core.systempaths import DATASETS, MODELS
from src.core.utils import load_config
from src.models.vit.vit import STViT

#External
from pathlib import Path
import argparse
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class WildfireDataset(Dataset):
    """
    Dataset wrapper
    """
    def __init__(self, data_dict):
        self.static = data_dict['static']
        self.dynamic = data_dict['dynamic']
        self.target = data_dict['target']

    def __len__(self):
        return len(self.dynamic)
    
    def __getitem__(self, idx):
        return {
            'static' : self.static,
            'dynamic' : self.dynamic[idx],
            'target' : self.target[idx]
        }


# Dataset-info
def get_dataset_parameters(dataset : str):
    print(f"Loaded dataset with {dataset['static'].shape[0]} static "\
          f"and {dataset['dynamic'].shape[1]} with {dataset['dynamic'].shape[2]} timesteps each")
    return {
        'num_static_channels' : dataset['static'].shape[0],
        'num_dynamic_channels' : dataset['dynamic'].shape[1],
        'num_timestamps_per_sample' : dataset['dynamic'].shape[2]
    }


# Dataset
def build_dataloader(data_dict, cfg_training):

    dataset = WildfireDataset(data_dict)
    return DataLoader(
        dataset,
        batch_size=cfg_training["batch_size"],
        shuffle=False # Dont shuffle chronological data
    )


# Model
def build_model(cfg_model, device, ds_info):
    model = STViT(
        num_static_channels = ds_info['num_static_channels'],
        num_dynamic_channels = ds_info['num_dynamic_channels'],
        num_timestamps_per_sample = ds_info['num_timestamps_per_sample'],
        patch_size = cfg_model['patch_size'],
        embedding_dim = cfg_model['embedding_dim']
    )
    return model.to(device)


# Training
def train(model, loss_fn, dataloader, cfg_training, device):
    optimizer = optim.Adam(model.parameters(), lr=cfg_training["learning_rate"])

    model.train()

    for epoch in range(cfg_training['num_epochs']):
        epoch_loss = 0.0

        for batch in dataloader:

            static_x = batch['static'].to(device)
            dynamic_x = batch['dynamic'].to(device)
            targets = batch['target'].to(device)

            # Pass forward
            preds = model(x_static = static_x, x_dynamic = dynamic_x)

            # Loss
            loss = loss_fn(preds, targets)

            optimizer.zero_grad() # zero gradients
            loss.backward() # Calculate losses
            optimizer.step() # Update weights

            epoch_loss += loss.item() # Calculate loss

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{cfg_training['num_epochs']}] - Loss: {avg_loss:.6f}")


# ----------------------------
# Main
# ----------------------------

def main(config_path: Path, dataset_name: str):

    # ---- Load config ----
    cfg = load_config(config_path)
    cfg_data = cfg["data"]
    cfg_model = cfg["model"]
    cfg_training = cfg["training"]
    cfg_data_sets = cfg["data_sets"]

    # ---- Device ----
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # ---- Load Data ----
    dataset_path = DATASETS / f'{dataset_name}.pt'
    print(f'Loading dataset from: {dataset_path}')
    dataset_dict = torch.load(dataset_path, weights_only = False)

    # ---- Build components ----
    dataloader = build_dataloader(dataset_dict, cfg_training)
    ds_info = get_dataset_parameters(dataset = dataset_dict)

    model = build_model(cfg_model = cfg_model, device = device, ds_info = ds_info)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight = torch.Tensor([cfg_training["pos_weight"]])).to(device)

    # ---- Train ----
    train(model, loss_fn, dataloader, cfg_training, device)
    """
    # ---- Save model ----
    model_dir = MODELS / (dataset_name + "_model.pt")
    print(f'Storing model as {model_dir}')
    torch.save(model.state_dict(), f = model_dir)

    """
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--datasetname", type=str, required=True)

    args = parser.parse_args()
    main(args.config, args.datasetname)

    # Example usage from /code:
    # uv run -m scripts.train --config configs/project.yaml --datasetname test
# Internal
from src.core.systempaths import DATASETS, MODELS
from src.core.utils import load_config
from src.models.vit.vit import STViT

#External
from pathlib import Path
import json
import argparse
from omegaconf import OmegaConf
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

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
    logging.info(f"Loaded dataset with {dataset['static'].shape[0]} static "\
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

def load_dataset_dict(path_to_ds: Path):
    if path_to_ds.is_file():
        return torch.load(path_to_ds, weights_only=False)
    
    # Temporary lists to hold the tensor chunks
    chunks = {
        "static": None, # We'll store the first one we find
        "dynamic": [],
        "target": []
    }
    
    # Use .pt or .pth (torch.load doesn't usually use .json)
    for i, file_path in enumerate(sorted(path_to_ds.glob("*.pt"))):
        logging.info(f'Loading and appending dataset: {file_path}')
        file_content = torch.load(file_path, weights_only=False)
        
        # 1. Handle Static (Keep only the first instance)
        if chunks["static"] is None:
            chunks["static"] = file_content.get("static")
        
        # 2. Collect chunks for concatenation
        for key in ["dynamic", "target"]:
            if key in file_content:
                chunks[key].append(file_content[key])

    # 3. Concatenate the lists of tensors back into single tensors
    return {
        "static": chunks["static"],
        "dynamic": torch.cat(chunks["dynamic"], dim=0) if chunks["dynamic"] else None,
        "target": torch.cat(chunks["target"], dim=0) if chunks["target"] else None
    }


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
def train(model, loss_fn, dataloader, cfg_training, device, writer, model_save_path):
    optimizer = optim.Adam(model.parameters(), lr=cfg_training["learning_rate"])
    
    # Track the best loss to save the best model state
    best_loss = float('inf') 

    model.train()
    logging.info("Starting training loop...")

    for epoch in range(cfg_training['num_epochs']):
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            logging.info(f'Processing batch index: {batch_idx}')
            static_x = batch['static'].to(device)
            dynamic_x = batch['dynamic'].to(device)
            targets = batch['target'].to(device)

            preds = model(x_static=static_x, x_dynamic=dynamic_x)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        logging.info(f"Epoch [{epoch+1}/{cfg_training['num_epochs']}] - Loss: {avg_loss:.6f}")
        
        # 1. Log to TensorBoard
        writer.add_scalar('Training/Loss', avg_loss, epoch)

        # 2. Save the model ONLY if it improved (Best Practice)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"--> Validation loss improved. Model saved to {model_save_path}")

    logging.info("Training complete.")

# ----------------------------
# Main
# ----------------------------
def main(config_path: Path, dataset_path: str, experiment_path: Path):
    


    # --- Setup Logging Directory and File ---
    exp_name = experiment_path.stem

    # Configure Python Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(experiment_path / "training.log"), # Save to file
            logging.StreamHandler() # Print to console
        ]
    )
    
    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir=str(experiment_path))

    # ---- Load config ----
    cfg = load_config(config_path)
    cfg_model = cfg["model"]
    cfg_training = cfg["training"]
    logging.info(f'Using config file from: {config_path}')

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logging.info(f"Using {device} device")

    # ---- Load Data ----
    logging.info(f"Loading dataset from: {dataset_path}")
    dataset_dict = load_dataset_dict(dataset_path)

    # ---- Build components ----
    dataloader = build_dataloader(dataset_dict, cfg_training)
    ds_info = get_dataset_parameters(dataset_dict)

    model = build_model(cfg_model, device, ds_info)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg_training["pos_weight"]])).to(device)

    # --- Define Save Path ---
    experiment_path.mkdir(exist_ok=True)
    model_save_path = experiment_path / f"{exp_name}_model_best.pt"

    # ---- Train ----
    train(model, loss_fn, dataloader, cfg_training, device, writer, model_save_path)

    # Close the TensorBoard writer
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--experiment_path", type=Path, required=True)

    args = parser.parse_args()
    main(experiment_path=args.experiment_path,
         config_path=args.config_path,
         dataset_path=args.dataset_path)

    # Example usage:
    # uv run -m scripts.train --config configs/project.yaml --datasetname test.pt
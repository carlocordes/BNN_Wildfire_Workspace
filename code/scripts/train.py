# Internal
from src.models.vit.vit import STViT

#External
from pathlib import Path
import argparse
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Config handling
def load_config(config_path: Path):
    cfg = OmegaConf.load(config_path)

    return {
        "data": cfg["data_sets"],
        "model": cfg["model"],
        "training": cfg["training"]
    }


# Dataset
def build_dataloader(cfg_data, cfg_training, dataset_name):
    dataset_path = Path(cfg_data["path"]) / f"{dataset_name}.pt"

    dataset = torch.load(dataset_path, weights_only=False)

    return DataLoader(
        dataset,
        batch_size=cfg_training["batch_size"],
        shuffle=True
    )


# Model
def build_model(cfg_model, device):
    model = STViT(
        num_modules=cfg_model["num_modules"],
        patch_size=cfg_model["patch_size"],
        embedding_dim=cfg_model["embedding_dim"]
    )
    return model.to(device)


# Training
def train(model, dataloader, cfg_training, device):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg_training["learning_rate"])

    model.train()

    for epoch in range(cfg_training["num_epochs"]):
        epoch_loss = 0.0

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)
            loss = loss_fn(preds, targets.squeeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

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

    # ---- Device ----
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # ---- Build components ----
    dataloader = build_dataloader(cfg_data, cfg_training, dataset_name)
    model = build_model(cfg_model, device)

    # ---- Train ----
    train(model, dataloader, cfg_training, device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--datasetname", type=str, required=True)

    args = parser.parse_args()
    main(args.config, args.datasetname)
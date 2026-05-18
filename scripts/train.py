# Internal
from src.core.utils import load_config
from src.models.vit.vit import STViT

#External
from pathlib import Path
import json
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
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


# Dataloaders
def build_dataloaders(data_dict, cfg_training):
    """
    Loads dataloaders split by 70 / 15 / 15 (Train/Validate/Test)
    """
    dataset = WildfireDataset(data_dict)

    N = len(dataset)
    
    end_train = cfg_training['train_size']
    end_val = 1 - cfg_training['val_size']

    train_end = int(0.70 * N)
    val_end = int(0.85 * N)

    train_dataset = Subset(
        dataset,
        range(0, train_end)
    )

    val_dataset = Subset(
        dataset,
        range(train_end, val_end)
    )

    test_dataset = Subset(
        dataset,
        range(val_end, N)
    )

    logging.info("Dataset split:")
    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Validation samples: {len(val_dataset)}")
    logging.info(F"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg_training["batch_size"],
        shuffle=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg_training["batch_size"],
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg_training["batch_size"],
        shuffle=False
    )

    return train_loader, val_loader, test_loader

# Load dataset folder or file
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

# Evaluation
def evaluate(model, loss_fn, dataloader, device):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for batch in dataloader:

            # Get data
            static_x = batch['static'].to(device)
            dynamic_x = batch['dynamic'].to(device)
            targets = batch['target'].to(device)

            # Predict
            preds = model(
                x_static=static_x,
                x_dynamic=dynamic_x
            )

            # Get loss
            loss = loss_fn(preds, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader) # Sum

    model.train() # Back to train mode

    return avg_loss

# Training
def train(model, loss_fn,
          train_dataloader, val_dataloader, test_dataloader, 
          cfg_training, device, writer, model_save_path):
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg_training["learning_rate"]) # TODO: Maybe replace with AdamW(lr, weight_decay)
    

    # Break conditions
    patience = cfg_training['patience']
    epochs_without_improvement = 0


    # Track the best loss to save the best model state
    best_loss = float('inf') 

    model.train() # Set training mode

    logging.info("Starting training loop...")

    for epoch in range(cfg_training['num_epochs']):
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_dataloader):
            
            logging.info(f'Epoch {epoch+1} | '
                f'Processing batch {batch_idx}'
            )
            
            # Get data
            static_x = batch['static'].to(device)
            dynamic_x = batch['dynamic'].to(device)
            targets = batch['target'].to(device)

            # Predict
            preds = model(x_static=static_x, x_dynamic=dynamic_x)

            # Calculate loss
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_dataloader)

        # Validate
        val_loss = evaluate(model = model,
                            loss_fn = loss_fn,
                            dataloader = val_dataloader,
                            device = device)

        # Log and TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        #writer.add_scalar('LearningRate')

        logging.info(
            f"Epoch [{epoch+1}/{cfg_training['num_epochs']}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )
        

        # Save model if validation loss improved
        if val_loss < best_loss:
            
            best_loss = val_loss # Update
            epochs_without_improvement = 0 # Reset

            torch.save(model.state_dict(),
                       model_save_path)
            logging.info(f"Validation improved. Saved model.")

        # Continue
        else:
            epochs_without_improvement += 1
            logging.info(f"No validation improvement for {epochs_without_improvement} epoch(s).")

        # Trigger stop after patience
        if epochs_without_improvement > patience:
            logging.info(f"Early stop triggered after {epoch + 1} epochs")
            break

    logging.info("Training complete.")

# ----------------------------
# Main
# ----------------------------
def main(config_path: Path, dataset_path: str, experiment_path: Path):
    
    # --- Setup Logging Directory and File ----
    exp_name = experiment_path.stem

    # ---- Python Logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(experiment_path / "training.log"), # Save to file
            logging.StreamHandler() # Print to console
        ]
    )
    
    # ---- TensorBoard Writer ----
    writer = SummaryWriter(log_dir=str(experiment_path))


    # ---- Load config ----
    cfg = load_config(config_path)
    cfg_model = cfg["model"]
    cfg_training = cfg["training"]
    logging.info(f'Using config file from: {config_path}')

    device = 'cpu' #torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logging.info(f"Using {device} device")


    # ---- Load Data ----
    logging.info(f"Loading dataset from: {dataset_path}")
    dataset_dict = load_dataset_dict(dataset_path)


    # ---- Build components ----
    train_dataloader, val_dataloader, test_dataloader = build_dataloaders(dataset_dict, cfg_training)
    ds_info = get_dataset_parameters(dataset_dict)

    model = build_model(cfg_model, device, ds_info)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg_training["pos_weight"]])).to(device)


    # --- Define Save Path ---
    experiment_path.mkdir(exist_ok=True)
    model_save_path = experiment_path / f"{exp_name}_model_best.pt"


    # ---- Train ----
    train(model = model,
          loss_fn = loss_fn,
          train_dataloader = train_dataloader,
          val_dataloader = val_dataloader,
          test_dataloader = test_dataloader,
          cfg_training = cfg_training,
          device = device,
          writer = writer,
          model_save_path = model_save_path)

    # Final test
    model.load_state_dict(
        torch.load(
            model_save_path,
            map_location=device
        )
    )
    

    # Evaluate on test data
    test_loss = evaluate(
        model = model,
        loss_fn = loss_fn,
        dataloader = test_dataloader,
        device = device
    )


    # Log and write
    logging.info(f"Final Test Loss: {test_loss:.6f}")
    writer.add_scalar("Loss/Test", test_loss)

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
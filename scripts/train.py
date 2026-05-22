# Internal
from src.core.utils import load_config
from src.models.vit.vit import STViT

from scripts.dataset import SpatialTemporalDataset, skip_missing_collate_fn

# External
from pathlib import Path
import time
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter


# Model
def build_model(cfg_model, cfg_data, device):
    model = STViT(
        num_static_channels = cfg_model['num_static_channels'] + cfg_model['num_single_dynamic_channels'],
        num_dynamic_channels = cfg_model['num_dynamic_channels'],
        num_timestamps_per_sample = cfg_data['temporal_extent']['sample_extent'],
        patch_size = cfg_model['patch_size'],
        embedding_dim = cfg_model['embedding_dim']
    )
    return model.to(device)


# Evaluation
def evaluate(model, loss_fn, cfg_training, cfg_path, mode: str, device):

    dataset = SpatialTemporalDataset(cfg_path, split_type=mode)

    set_dataloader = DataLoader(
        dataset, 
        batch_size=cfg_training['batch_size'], 
        shuffle=False,         
        num_workers=4,         
        pin_memory=True,       
        collate_fn=skip_missing_collate_fn
    )

    model.eval()

    total_loss = 0.0
    valid_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(set_dataloader):

            if batch is None:
                continue

            # Get data
            static_x = batch['static'].to(device)
            single_dynamic_x = batch['single_dynamic'].to(device)
            dynamic_x = batch['dynamic'].to(device)
            targets = batch['target'].to(device)

            # Predict
            preds = model(
                x_static=static_x,
                x_dynamic=dynamic_x,
                x_single_dynamic=single_dynamic_x
            )

            # Get loss
            loss = loss_fn(preds, targets)
            total_loss += loss.item()
            valid_batches += 1

    avg_loss = total_loss / valid_batches if valid_batches > 0 else 0.0
    model.train() 
    return avg_loss


# Training
def train(model, loss_fn, cfg_path: Path, cfg_training, device, writer, model_save_path):
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg_training["learning_rate"],
                            weight_decay=cfg_training['weight_decay']) 
    
    patience = cfg_training['patience']
    epochs_without_improvement = 0
    best_loss = float('inf') 

    model.train() 

    logging.info(f'Loading dataset from {cfg_path}')

    # Training dataloader
    train_dataset = SpatialTemporalDataset(cfg_path, split_type="train")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg_training['batch_size'], 
        shuffle=True,          
        num_workers=4,         
        pin_memory=True,       
        collate_fn=skip_missing_collate_fn
    )

    # Setup learning rate scheduler
    max_lr = cfg_training['learning_rate']
    steps_per_epoch = len(train_loader)
    total_steps = cfg_training['num_epochs'] * steps_per_epoch

    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=0.10,          
        anneal_strategy='cos',   
        div_factor=25.0,         
        final_div_factor=1000.0  
    )

    logging.info("Starting training loop...")

    for epoch in range(cfg_training['num_epochs']):
        
        epoch_loss = 0.0
        valid_batches = 0
        
        start_time = time.time()
        for batch_idx, batch in enumerate(train_loader):
            data_time = time.time() - start_time

            if batch is None: 
                continue 
            
            if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
                logging.info(f'Epoch {epoch+1} | Processing batch {batch_idx + 1}/{len(train_loader)}')
            
            # Get data
            static_x = batch['static'].to(device)
            single_dynamic_x = batch['single_dynamic'].to(device)
            dynamic_x = batch['dynamic'].to(device)
            targets = batch['target'].to(device)

            compute_start = time.time()

            # Predict
            preds = model(
                x_static=static_x,
                x_dynamic=dynamic_x, 
                x_single_dynamic=single_dynamic_x
            )

            # Calculate loss
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            valid_batches += 1

            compute_time = time.time() - compute_start
            start_time = time.time() 

        train_loss = epoch_loss / valid_batches if valid_batches > 0 else 0.0

        # Store time statistics (Scalar floats only - virtually zero memory)
        writer.add_scalar('Time/Data', data_time, epoch)
        writer.add_scalar('Time/Compute', compute_time, epoch)

        # Validate
        val_loss = evaluate(model=model,
                            loss_fn=loss_fn,
                            cfg_training=cfg_training,
                            cfg_path=cfg_path,
                            mode="val",
                            device=device)

        current_lr = optimizer.param_groups[0]['lr']
        
        # Log and TensorBoard scalars
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)

        logging.info(
            f"Epoch [{epoch+1}/{cfg_training['num_epochs']}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )
        
        # Save model if validation loss improved
        if val_loss < best_loss:
            best_loss = val_loss 
            epochs_without_improvement = 0 

            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Validation improved. Saved model.")
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
def main(config_path: Path, experiment_path: Path):
    
    exp_name = experiment_path.stem
    experiment_path.mkdir(exist_ok=True, parents=True)

    # ---- Python Logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(experiment_path / "training.log"), 
            logging.StreamHandler() 
        ]
    )
    
    # ---- TensorBoard Writer ----
    writer = SummaryWriter(log_dir=str(experiment_path))

    # ---- Load config ----
    cfg = load_config(config_path)
    cfg_model = cfg["model"]
    cfg_data = cfg["data"]
    cfg_training = cfg["training"]
    logging.info(f'Using config file from: {config_path}')

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logging.info(f"Using {device} device")

    # ---- Build components ----
    model = build_model(cfg_model=cfg_model, cfg_data=cfg_data, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg_training["pos_weight"]])).to(device)

    # --- Define Save Path ---
    model_save_path = experiment_path / f"{exp_name}_model_best.pt"

    # ---- Train ----
    train(model=model,
          loss_fn=loss_fn,
          cfg_training=cfg_training,
          cfg_path=config_path,
          device=device,
          writer=writer,
          model_save_path=model_save_path)

    # Final test using best checkpointed spatial states
    if model_save_path.exists():
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    
    # Evaluate on test data
    test_loss = evaluate(model=model,
                        loss_fn=loss_fn,
                        cfg_training=cfg_training,
                        cfg_path=config_path,
                        mode="test",
                        device=device)

    logging.info(f"Final Test Loss: {test_loss:.6f}")
    writer.add_scalar("Loss/Test", test_loss)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--experiment_path", type=Path, required=True)

    args = parser.parse_args()
    main(experiment_path=args.experiment_path, config_path=args.config_path)
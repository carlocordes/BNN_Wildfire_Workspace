# Internal
from src.models.vit.vit import STViT
from src.dataset_builder import create_pytorch_dataset

#External
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def weighted_BCE(preds, targets, pos_weight):
    pass

def train(data, model, loss_fn, optimizer):
    # TODO: Wrap into train function
    pass


if __name__ == '__main__':

    # Set up data
    paths_modis = Path('data', 'processed', 'LST_MODIS_8day')
    paths_dtm = Path('data', 'processed', 'slope', 'dtm_aligned.tif')
    paths_target = Path('data', 'processed', 'burn')

    # Dataset
    dataset = create_pytorch_dataset(paths_modis, paths_dtm, paths_target)
    batch_size = 4
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # Package as data loader and shuffle



    # Toy parameters
    EMBED_DIM = 32
    num_modules = 2
    patch_size = 16 # Rectangular, number of pixels

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device as compute")

    # Define Model
    model = STViT(num_modules = num_modules,
                  patch_size = patch_size,
                  embedding_dim = EMBED_DIM).to(device)


    # Loss function
    loss_fn = nn.MSELoss()

    # TODO: Define Binary Cross Entropy Loss function (custom)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)

    epochs = 20

    model.train() # Set training mode


    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to GPU
            inputs = inputs.to(device)
            targets = targets.to(device) # Shape: (B, 1, H, W)

            # 1. Forward pass
            # Note: targets is (B, 1, H, W), model output is (B, H, W)
            # We squeeze targets to (B, H, W) to match model output
            preds = model(inputs)
            loss = loss_fn(preds, targets.squeeze(1))

            # 2. Backward pass
            optimizer.zero_grad() # Clear previous gradients
            loss.backward()       # Compute gradients
            optimizer.step()      # Update weights

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")
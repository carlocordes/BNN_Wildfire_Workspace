# Internal
from ViT import STViT

#Internal
import torch
import torch.nn as nn


def train(data, model, loss_fn, optimizer):
    # TODO: Wrap into train function
    pass


if __name__ == '__main__':


    # Toy parameters
    EMBED_DIM = 32
    batch_size = 1
    num_modules = 2
    height = 256
    width = 256
    patch_size = 16 # Rectangular, number of pixels

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device as compute")

    # Define Model
    model = STViT(batch_size = batch_size,
                  num_modules = num_modules,
                  img_height = height,
                  img_width = width,
                  patch_size = patch_size,
                  embedding_dim = EMBED_DIM).to(device)

    # Model params
    #for name, param in model.named_parameters():
    #    print(name, param.shape)

    # Loss and Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)

    ## Train workflow

    # Create random tensor pair
    input_data = torch.randn(batch_size, num_modules, height, width, device = device)
    target = torch.rand(batch_size, height, width, device = device)

    # Predict
    pred = model(input_data)

    # Compute Loss
    loss = loss_fn(pred, target)

    # Backpropagate (opt. with optimizer)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
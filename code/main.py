#Internal
import torch
from ViT import STViT

if __name__ == '__main__':


    # Toy parameters
    EMBED_DIM = 32
    batch_size = 1
    num_modules = 2
    height = 128
    width = 128
    patch_size = 8 # Rectangular, number of pixels

    # Define Model
    model = STViT(batch_size = batch_size,
                  num_modules = num_modules,
                  img_height = height,
                  img_width = width,
                  patch_size = patch_size,
                  embedding_dim = EMBED_DIM)#.to(device)

    #print(model)

    # Some toy data
    input_data = torch.randn(batch_size, num_modules, height, width)
    target = torch.ones(batch_size, 1, height, width)

    #Feedforward
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    model(input_data).to(device)

    # Model params
    #for name, param in model.named_parameters():
    #    print(name, param.shape)
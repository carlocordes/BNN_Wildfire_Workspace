import torch
import torch.nn as nn

class STViT(nn.Module):
    def __init__(self,
                 batch_size : int,
                 num_modules : int,
                 patch_size : int,
                 ):
        super().__init__()




    def feedforward(self):
        pass

#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
#print(f"Using {device} device")

model = STViT()#.to(device)

print(model)
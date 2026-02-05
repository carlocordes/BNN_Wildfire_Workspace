import torch



device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using {device} for computation.')

tensor = torch.tensor([[1,2],
                       [3, 4]], device = device)

print(tensor)
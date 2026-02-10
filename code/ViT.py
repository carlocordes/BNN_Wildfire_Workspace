import torch
import torch.nn as nn
import numpy as np

def patch(input_tensor : torch.tensor, pixel_size : int) -> torch.tensor:
    """
    Def:

    Args:
    input_tensor: 


    TODO: Extend to temporally dependent datasets, here dim_t = 1
    """

    num_modules, num_pixels_height, num_pixels_width = input_tensor.shape


    print(f'Patching input tensor with {num_modules} modules to {pixel_size}x{pixel_size}')


    


static_data = np.zeros(64).reshape(8, 8)

dynamic_data = np.ones(64).reshape(8, 8)
target = np.arange(0, 64, 1).reshape(8, 8)


input_array = np.array([static_data, dynamic_data, target])
input_tensor = torch.from_numpy(input_array)

patch(input_tensor, pixel_size = 4)
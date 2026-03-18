# Internal

# External
import glob
from pathlib import Path

import rasterio
import torch
from torch.utils.data import TensorDataset

def create_pytorch_dataset(paths_a: Path,
                           paths_b: Path,
                           paths_t: Path,
                           ) -> TensorDataset:
    # Files at a (Input 1)
    files_a = glob.glob(str(paths_a) + '/*.tif')
    files_a.sort()
    
    # Files at t (Target)
    files_c = glob.glob(str(paths_t) + '/*.tif')
    files_c.sort()

    print(f'Creating dataset from {len(files_a)} file pairs')

    all_inputs = []
    all_targets = []

    # Iterate through the triplets
    for a_path, c_path in zip(files_a, files_c):
        # 1. Read Image A (Dynamic Input)
        with rasterio.open(a_path) as src:
            # .read(1) gets the first band; result is a numpy array
            img_a = torch.from_numpy(src.read(1)).float()

        # 2. Read Image B (Static Input)
        # Assuming paths_b is a single Path object to one static .tif file
        with rasterio.open(paths_b) as src:
            img_b = torch.from_numpy(src.read(1)).float()

        # 3. Read Image C (Target)
        with rasterio.open(c_path) as src:
            img_t = torch.from_numpy(src.read(1)).float()

        # Stack inputs into shape (2, height, width)
        stacked_input = torch.stack([img_a, img_b], dim=0)
        
        # Add to lists (Target becomes (1, height, width) to stay 3D)
        all_inputs.append(stacked_input)
        all_targets.append(img_t.unsqueeze(0))

    # Convert lists to massive 4D tensors: (N, C, H, W)
    final_input_tensor = torch.stack(all_inputs)
    final_target_tensor = torch.stack(all_targets)

    out_tensor = TensorDataset(final_input_tensor, final_target_tensor)

    # print some dataset info
    sample_input, sample_target = out_tensor[0]

    print(f"Input shape (C, H, W): {sample_input.shape}")   # Should be (2, 311, 609)
    print(f"Target shape (C, H, W): {sample_target.shape}")  # Should be (1, 311, 609)

    # 3. Access total memory footprint (in bytes)
    input_bytes = out_tensor.tensors[0].element_size() * out_tensor.tensors[0].nelement()
    print(f"Total RAM used by inputs: {input_bytes / 1e9:.2f} GB")

    return out_tensor




if __name__ == '__main__':

    paths_modis = Path('data', 'raw', 'LST_MODIS_8day')
    paths_dtm = Path('data', 'raw', 'dtm', 'dtm_aligned.tif')
    paths_target = Path('data', 'processed', 'burn')

    dataset = create_pytorch_dataset(paths_a=paths_modis,
                                     paths_b=paths_dtm,
                                     paths_t = paths_target)
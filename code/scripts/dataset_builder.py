# Internal

# External
import glob
from pathlib import Path
from omegaconf import OmegaConf
import argparse

import rasterio
import torch
from torch.utils.data import TensorDataset

class Dataset_Builder():
    def __init__(self, config_path : Path):
        self.config_path = config_path

    def save(self, dataset_name : str):
        # Read out config
        cfg = OmegaConf.load(self.config_path)
        cfg_data = cfg['data_paths']['raw']
        paths_a = cfg_data['MODIS']
        paths_b = cfg_data['DTM']
        paths_t = cfg_data['target']

        # Files at a (Input 1)
        files_a = glob.glob(str(paths_a) + '/*.tif')
        files_a.sort()

        # Files at t (Target)
        files_c = glob.glob(str(paths_t) + '/*.tif')
        files_c.sort()

        print(f'Creating dataset from {len(files_a)} file pairs')


        all_inputs = []
        all_targets = []

        # # Read static DTM first (once)
        path_b = glob.glob(str(paths_b) + '/*.tif')[0]
        with rasterio.open(path_b) as src:
            img_b = torch.from_numpy(src.read(1)).float()


        # Iterate through the triplets
        for a_path, c_path in zip(files_a, files_c):
            # Read Image A (Dynamic Input)
            with rasterio.open(a_path) as src:
                # .read(1) gets the first band; result is a numpy array
                img_a = torch.from_numpy(src.read(1)).float()

            # Read Image C (Target)
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

        out_path = Path(cfg['data_sets']['path'])
        torch.save(out_tensor, f = out_path / (dataset_name + '.pt'))

def main(config_path : Path, dataset_name : str):
    dataset = Dataset_Builder(config_path)
    dataset.save(dataset_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run full pipeline using a YAML config")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--datasetname",
        type=str,
        required = True,
        help="Name of dataset to be saved (excl. type e.g. '.pt')"
    )
    args = parser.parse_args()
    main(args.config, args.datasetname)

    # Example usage from /code : uv run -m scripts.dataset_builder --config configs/project.yaml --datasetname test
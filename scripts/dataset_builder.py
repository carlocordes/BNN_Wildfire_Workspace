# Internal
from src.core.systempaths import DATASETS, MODELS

# External
import glob
from pathlib import Path
from omegaconf import OmegaConf
import argparse
from datetime import datetime, timedelta
import pandas as pd

import rasterio
import torch
from torch.utils.data import TensorDataset

class Dataset_Builder():
    def __init__(self, config_path : Path, year_split : bool = False):
        self.config_path = config_path

        # Load config
        self.cfg = OmegaConf.load(self.config_path)
        self.cfg_data = self.cfg['data_paths']['processed']
        cfg_temporal = self.cfg['data']['temporal_extent']

        self.timeframes = self.get_timeframes(cfg_temporal = cfg_temporal, year_split = year_split)

    def get_timeframes(self, cfg_temporal, year_split):
        # Build timeframes
        sequence_extent = cfg_temporal['sample_extent'] + cfg_temporal['target_extent'] + cfg_temporal['lead_time'] # Length of entire sequence
        
        start_date = datetime.strptime(cfg_temporal['start_date'], "%Y-%m-%d")
        end_date = datetime.strptime(cfg_temporal['end_date'], "%Y-%m-%d")
        
        testing_extent = (end_date -start_date).days + 1 # Number of unique days considered
        #print(f'Sequence Extent: {sequence_extent}')

        num_sequences = (testing_extent - sequence_extent) // cfg_temporal['sequence_period'] + 1
        #print(f'Number of sequences: {num_sequences}')

        # Structure for storing sequences
        years_list = list(range(start_date.year, end_date.year +1))
        sequences = {year : [] for year in years_list}

        for i in range(num_sequences):
            current_sequence = {}

            # Sample Dates
            sample_start_date = start_date + timedelta(days = i*cfg_temporal['sequence_period'])
            sample_end_date = sample_start_date + timedelta(days = cfg_temporal['sample_extent'] - 1)

            sample = pd.date_range(start = sample_start_date, end = sample_end_date, freq = f'{cfg_temporal["day_interval"]}D')
            sample = sample.strftime("%Y-%m-%d").tolist()

            # Target dates
            target_start_date = sample_start_date + timedelta(days = cfg_temporal['sample_extent'] + cfg_temporal['lead_time']-1)
            target_end_date = target_start_date + timedelta(days = cfg_temporal['target_extent'] - 1)
            
            target = pd.date_range(start = target_start_date, end = target_end_date, freq = f'{cfg_temporal["day_interval"]}D')
            target = target.strftime("%Y-%m-%d").tolist()


            # Store and append
            current_sequence = {
                'sample' : sample,
                'target' : target,
            }

            # Append to sequence dict
            current_year = int(sample_start_date.year)
            sequences[current_year].append(current_sequence)


        if year_split:
            return sequences
        else:
            # Flatten into one
            sequences_combined = {'all' : [item for sublist in sequences.values() for item in sublist]}
            return sequences_combined

    def build(self, dataset_name : str):

        ## Prepare out paths
        if len(self.timeframes) > 1:
            out_dir = Path(self.cfg['data_sets']['path']) / dataset_name
            out_dir.mkdir(parents = True, exist_ok = True)
        else:
            out_dir = Path(self.cfg['data_sets']['path'])

        ## Define source paths
        sample_base_paths = [           
            Path(self.cfg_data['NDVI']),
            Path(self.cfg_data['wind_speed']),
            Path(self.cfg_data['wind_dir_v']),
            Path(self.cfg_data['wind_dir_u'])
        ]

        target_base_path = Path(self.cfg_data['target']) # TODO: Switch to other ground truth


        static_base_paths = [
            Path(self.cfg_data['aspect']),
            Path(self.cfg_data['slope']) # TODO: Add paths
        ]


        ## Load dynamic data
        sample_tensors = []
        target_tensors = []
        for key, timeframes in self.timeframes.items():
            for timeframe in timeframes: 
                try:
                    channel_sample_tensors = []

                    # Iterate through sample channels
                    for path in sample_base_paths:
                        time_sample_tensors = []

                        for t in timeframe['sample']:
                            fp = path / f"{t}.tif"

                            if not fp.exists():
                                raise FileNotFoundError(f"Missing sample file: {fp}")

                            with rasterio.open(fp) as src:
                                img = torch.from_numpy(src.read(1)).float()
                                time_sample_tensors.append(img)

                        channel_sample_data = torch.stack(time_sample_tensors, dim=0)
                        channel_sample_tensors.append(channel_sample_data)

                    sequence_sample_data = torch.stack(channel_sample_tensors, dim=0)
                    

                    # --- TARGET ---
                    time_target_tensors = []

                    for t in timeframe['target'][:1]:
                        fp = target_base_path / f"{t}.tif"

                        if not fp.exists():
                            raise FileNotFoundError(f"Missing target file: {fp}")

                        with rasterio.open(fp) as src:
                            img = torch.from_numpy(src.read(1)).float()
                            time_target_tensors.append(img)

                    time_target_data = torch.stack(time_target_tensors, dim=0)

                    # Only append if everything succeeded
                    sample_tensors.append(sequence_sample_data)
                    target_tensors.append(time_target_data)

                except FileNotFoundError as e:
                    print(f"Skipping timeframe due to missing data: {e}")
                    continue


            sample_data = torch.stack(sample_tensors, dim = 0)
            target_data = torch.stack(target_tensors, dim = 0)


            static_tensors = []
            for path in static_base_paths:
                tif_files = glob.glob(str(path) + '/*.tif')

                for fp in tif_files:

                    with rasterio.open(fp) as src:
                        img = torch.from_numpy(src.read(1)).float()
                        static_tensors.append(img)

            static_data = torch.stack(static_tensors, dim = 0)

            ## Concat
            tensors_dict = {
                'dynamic' : sample_data,
                'static' : static_data,
                'target' : target_data,
            }

            print(f"Produced dataset {dataset_name} with {tensors_dict['static'].shape[0]} static " \
                f"and {tensors_dict['dynamic'].shape[1]} with {tensors_dict['dynamic'].shape[2]} timesteps each")
            
            ## Save
            out_path = out_dir / f'{dataset_name}_{key}_ds.pt'
            torch.save(obj = tensors_dict, f = out_path)
            print(f'Saved to {out_path}')



def main(config_path : Path, dataset_name : str, year_split):
    dataset = Dataset_Builder(config_path, year_split)
    dataset.build(dataset_name)
    


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
    parser.add_argument('--year_split',
                        action='store_true',
                        help='Split saved datasets by year')
    args = parser.parse_args()
    main(args.config, args.datasetname, args.year_split)

    # Example usage from /code :
    # uv run -m scripts.dataset_builder --config configs/project.yaml --datasetname test
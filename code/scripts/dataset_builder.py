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
    def __init__(self, config_path : Path):
        self.config_path = config_path

        # Load config
        self.cfg = OmegaConf.load(self.config_path)
        self.cfg_data = self.cfg['data_paths']['processed']
        cfg_temporal = self.cfg['data']['temporal_extent']

        self.timeframes = self.get_timeframes(cfg_temporal = cfg_temporal)

    def get_timeframes(self, cfg_temporal):
         # Build timeframes
        sequence_extent = cfg_temporal['sample_extent'] + cfg_temporal['target_extent'] + cfg_temporal['lead_time'] # Length of entire sequence
        
        start_date = datetime.strptime(cfg_temporal['start_date'], "%Y-%m-%d")
        end_date = datetime.strptime(cfg_temporal['end_date'], "%Y-%m-%d")
        
        testing_extent = (end_date -start_date).days + 1 # Number of unique days considered
        #print(f'Sequence Extent: {sequence_extent}')

        num_sequences = (testing_extent - sequence_extent) // cfg_temporal['sequence_period'] + 1
        #print(f'Number of sequences: {num_sequences}')

        
        timeframes = []
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

            timeframes.append(current_sequence)

        return timeframes

    def build(self, dataset_name : str):

        ## Define source paths
        sample_base_paths = [           
            Path(self.cfg_data['NDVI']),
            Path(self.cfg_data['wind_speed']),
            Path(self.cfg_data['wind_dir_v']),
            Path(self.cfg_data['wind_dir_u'])
        ]

        target_base_path = Path(self.cfg_data['target']) # TODO: Switch to other ground truth

        static_base_paths = [
            Path(self.cfg_data['DTM']) # TODO: Add paths
        ]
        

        ## Load dynamic data
        sample_tensors = []
        target_tensors = []
        for timeframe in self.timeframes:
            
            channel_sample_tensors = []

            # Iterate through sample channels
            for path in sample_base_paths:

                time_sample_tensors = []

                # Iterate thorugh time stamps
                for t in timeframe['sample']:
                    filename = t + '.tif'
                    fp = path / filename
                    
                    with rasterio.open(fp) as src:
                        img = torch.from_numpy(src.read(1)).float()
                        time_sample_tensors.append(img)
                        
                channel_sample_data = torch.stack(time_sample_tensors, dim = 0)
                channel_sample_tensors.append(channel_sample_data)

            sequence__sample_data = torch.stack(channel_sample_tensors, dim = 0)
            sample_tensors.append(sequence__sample_data)


            time_target_tensors = []

            # Iterate through target
            for t in timeframe['target'][:1]: # Only take first of target sequence
                filename = t + '.tif'
                fp = target_base_path / filename

                with rasterio.open(fp) as src:
                    img = torch.from_numpy(src.read(1)).float()
                    time_target_tensors.append(img)

            time_target_data = torch.stack(time_target_tensors, dim = 0)
            target_tensors.append(time_target_data)


        sample_data = torch.stack(sample_tensors, dim = 0)
        target_data = torch.stack(target_tensors, dim = 0)


        ## Load static data
        static_tensors = []
        for path in static_base_paths:
            fp = glob.glob(str(path) + '/*.tif')[0]
            print(fp)
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

        ## Save
        out_path = Path(self.cfg['data_sets']['path'] ) / (dataset_name + '.pt')
        torch.save(obj = tensors_dict, f = out_path)



def main(config_path : Path, dataset_name : str):
    dataset = Dataset_Builder(config_path)
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
    args = parser.parse_args()
    main(args.config, args.datasetname)

    # Example usage from /code :
    # uv run -m scripts.dataset_builder --config configs/project.yaml --datasetname test
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

        self.batch_size = self.cfg['training']['batch_size']

        self.train_size = self.cfg['training']['train_size']
        self.val_size = self.cfg['training']['val_size']
        self.test_size = self.cfg['training']['test_size']

        cfg_temporal = self.cfg['data']['temporal_extent']

        self.train_frames, self.val_frames, self.test_frames = self.produce_batched_timeframes(cfg_temporal = cfg_temporal)

    def produce_batched_timeframes(self, cfg_temporal):
        # Build timeframes
        sequence_extent = cfg_temporal['sample_extent'] + cfg_temporal['target_extent'] + cfg_temporal['lead_time'] # Length of entire sequence
        
        start_date = datetime.strptime(cfg_temporal['start_date'], "%Y-%m-%d")
        end_date = datetime.strptime(cfg_temporal['end_date'], "%Y-%m-%d")
        
        testing_extent = (end_date -start_date).days + 1 # Number of unique days considered
        #print(f'Sequence Extent: {sequence_extent}')

        num_sequences = (testing_extent - sequence_extent) // cfg_temporal['sequence_period'] + 1
        #print(f'Number of sequences: {num_sequences}')

        sequences = []

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

            sequences.append(current_sequence)

        # Split into proportions
        n = len(sequences)

        idx1 = round(n * self.train_size)
        idx2 = round(n * (1.0 - self.val_size))

        train_frames = sequences[:idx1]
        val_frames = sequences[idx1:idx2]
        test_frames = sequences[idx2:]

        assert((len(train_frames) + len(val_frames) + len(test_frames)) == n)

        train_frames_batched = [train_frames[i : i + self.batch_size] for i in range(0, len(train_frames), self.batch_size)]
        val_frames_batched = [val_frames[i : i + self.batch_size] for i in range(0, len(val_frames), self.batch_size)]
        test_frames_batched = [test_frames[i : i + self.batch_size] for i in range(0, len(test_frames), self.batch_size)]

        return train_frames_batched, val_frames_batched, test_frames_batched

    def get_batches_from_dataset(self, dataset_type):

        # Select list to yield
        target_dataset = getattr(self, dataset_type, None)

        if target_dataset is None:
            raise ValueError(f"Category '{dataset_type}' does not exist.")


        ## Define source paths
        sample_base_paths = [           
            Path(self.cfg_data['NDVI']),
            Path(self.cfg_data['wind_speed']),
            Path(self.cfg_data['wind_dir_v']),
            Path(self.cfg_data['wind_dir_u']),
            Path(self.cfg_data['NDWI']),
            Path(self.cfg_data['LST'])
        ]

        target_base_path = Path(self.cfg_data['target']) # TODO: Switch to other ground truth


        static_base_paths = [
            Path(self.cfg_data['aspect']), # 2x in here
            Path(self.cfg_data['slope']),
            Path(self.cfg_data['roads']) # TODO: Add paths
        ]

        dynamic_single_base_paths = [
            Path(self.cfg_data['burn_history']),
            Path(self.cfg_data['precip']),
        ]


        # Pre-load static data ONCE
        static_channels = []
        for path in static_base_paths:
            tif_files = glob.glob(str(path) + '/*.tif')

            for fp in tif_files:

                with rasterio.open(fp) as src:
                    img = torch.from_numpy(src.read(1)).float()
                    static_channels.append(img)

        base_static_tensor = torch.stack(static_channels, dim = 0)


        ## Load dynamic data
        #print(len(target_dataset))
        for batch in target_dataset:
            sample_tensors = []
            target_tensors = []
            single_dynamic_tensors = []
            static_tensors = []
            for timeframe in batch:
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
                    

                    # --- Single dynamic --- #
                    time_single_dynamic_tensors = []

                    last_time_of_sample = timeframe['sample'][-1]
                    for path in dynamic_single_base_paths:
                        fp = path / f"{last_time_of_sample}.tif"
                        with rasterio.open(fp) as src:
                            img = torch.from_numpy(src.read(1)).float()
                            time_single_dynamic_tensors.append(img)

                    single_dynamic_data = torch.stack(time_single_dynamic_tensors, dim = 0)

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
                    static_tensors.append(base_static_tensor)

                    single_dynamic_tensors.append(single_dynamic_data)

                except FileNotFoundError as e:
                    print(f"Skipping timeframe due to missing data: {e}")
                    continue

            if sample_tensors and single_dynamic_tensors:

                sample_data = torch.stack(sample_tensors, dim = 0)
                target_data = torch.stack(target_tensors, dim = 0)
                single_dynamic_data = torch.stack(single_dynamic_tensors, dim = 0)
                static_data = torch.stack(static_tensors, dim = 0)


                ## Concat
                tensors_dict = {
                    'dynamic' : sample_data,
                    'static' : static_data,
                    'target' : target_data,
                    'single_dynamic' : single_dynamic_data
                    # append uncertainty here
                }
                """
                for s, value in tensors_dict.items():
                    print(f'{s} has shape: {value.shape}')
                """

                yield tensors_dict


def main(config_path : Path, dataset_name : str, year_split):
    dataset = Dataset_Builder(config_path, year_split)


    for batch in dataset.get_batches_from_dataset('train_frames'):
        for type, tensor in batch.items():
            print(type, ' : ', tensor.shape)



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
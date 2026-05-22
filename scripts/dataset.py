import glob
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from omegaconf import OmegaConf

def skip_missing_collate_fn(batch):
    """
    Filters out any 'None' entries caused by missing files during sampling.
    """
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


class SpatialTemporalDataset(Dataset):
    def __init__(self, config_path: Path, split_type: str = "train"):
        """
        Initializes dataset tracking by explicit Out-of-Time calendar blocks.
        """
        self.cfg = OmegaConf.load(config_path)
        self.cfg_data = self.cfg['data_paths']['processed']
        cfg_temporal = self.cfg['data']['temporal_extent']
        
        # Channel source paths
        self.sample_base_paths = [
            Path(self.cfg_data['NDVI']),
            Path(self.cfg_data['NDWI']),
            Path(self.cfg_data['wind_speed']),
            Path(self.cfg_data['wind_dir_v']),
            Path(self.cfg_data['wind_dir_u']),
            Path(self.cfg_data['LST'])
        ]
        self.target_base_path = Path(self.cfg_data['target'])
        
        self.dynamic_single_base_paths = [
            Path(self.cfg_data['burn_history']),
            Path(self.cfg_data['precip'])
        ]
        
        # Pre-cache static layers once globally per instantiated split
        self.base_static_tensor = self._load_static_layers([
            Path(self.cfg_data['aspect']),
            Path(self.cfg_data['slope']),
            Path(self.cfg_data['roads'])
        ])

        # Generate complete date sequences across the entire timeline extent
        all_sequences = self._produce_timeframes(cfg_temporal)
        
        # ==========================================
        # FIXED: HARD CHRONOLOGICAL BLOCK SPLITTING
        # ==========================================
        # Train: 2019-2023 | Val: 2024 | Test: 2025
        if split_type == "train":
            self.sequences = [s for s in all_sequences if s['year'] in [2019, 2020, 2021, 2022, 2023]]
        elif split_type == "val":
            self.sequences = [s for s in all_sequences if s['year'] == 2024]
        elif split_type == "test":
            self.sequences = [s for s in all_sequences if s['year'] == 2025]
        else:
            raise ValueError(f"Unknown split_type: {split_type}")

        print(f"[DATASET INFO] Created {split_type} split with {len(self.sequences)} samples.")

    def _produce_timeframes(self, cfg_temporal):
        sequence_extent = cfg_temporal['sample_extent'] + cfg_temporal['target_extent'] + cfg_temporal['lead_time']
        start_date = datetime.strptime(cfg_temporal['start_date'], "%Y-%m-%d")
        end_date = datetime.strptime(cfg_temporal['end_date'], "%Y-%m-%d")
        testing_extent = (end_date - start_date).days + 1
        num_sequences = (testing_extent - sequence_extent) // cfg_temporal['sequence_period'] + 1

        sequences = []
        for i in range(num_sequences):
            sample_start = start_date + timedelta(days=i * cfg_temporal['sequence_period'])
            sample_end = sample_start + timedelta(days=cfg_temporal['sample_extent'] - 1)
            sample = pd.date_range(start=sample_start, end=sample_end, freq=f'{cfg_temporal["day_interval"]}D')
            
            target_start = sample_start + timedelta(days=cfg_temporal['sample_extent'] + cfg_temporal['lead_time'] - 1)
            target_end = target_start + timedelta(days=cfg_temporal['target_extent'] - 1)
            target = pd.date_range(start=target_start, end=target_end, freq=f'{cfg_temporal["day_interval"]}D')

            # We use the beginning of the observation sequence to determine its block year assignment
            sequences.append({
                'sample': sample.strftime("%Y-%m-%d").tolist(),
                'target': target.strftime("%Y-%m-%d").tolist(),
                'year': sample_start.year 
            })
        return sequences

    def _load_static_layers(self, static_base_paths):
        static_channels = []
        for path in static_base_paths:
            for fp in glob.glob(str(path) + '/*.tif'):
                with rasterio.open(fp) as src:
                    static_channels.append(torch.from_numpy(src.read(1)).float())
        return torch.stack(static_channels, dim=0)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        timeframe = self.sequences[idx]
        try:
            # 1. Dynamic sequence channels
            channel_sample_tensors = []
            for path in self.sample_base_paths:
                time_sample_tensors = []
                for t in timeframe['sample']:
                    fp = path / f"{t}.tif"
                    if not fp.exists():
                        raise FileNotFoundError(f"Missing sample file: {fp}")
                    with rasterio.open(fp) as src:
                        time_sample_tensors.append(torch.from_numpy(src.read(1)).float())
                channel_sample_tensors.append(torch.stack(time_sample_tensors, dim=0))
            sequence_sample_data = torch.stack(channel_sample_tensors, dim=0)

            # 2. Single Dynamic
            time_single_dynamic_tensors = []
            last_time_of_sample = timeframe['sample'][-1]
            for path in self.dynamic_single_base_paths:
                fp = path / f"{last_time_of_sample}.tif"
                if not fp.exists():
                    raise FileNotFoundError(f"Missing single dynamic file: {fp}")
                with rasterio.open(fp) as src:
                    time_single_dynamic_tensors.append(torch.from_numpy(src.read(1)).float())
            single_dynamic_data = torch.stack(time_single_dynamic_tensors, dim=0)

            # 3. Target
            time_target_tensors = []
            for t in timeframe['target'][:1]:
                fp = self.target_base_path / f"{t}.tif"
                if not fp.exists():
                    raise FileNotFoundError(f"Missing target file: {fp}")
                with rasterio.open(fp) as src:
                    time_target_tensors.append(torch.from_numpy(src.read(1)).float())
            time_target_data = torch.stack(time_target_tensors, dim=0)

            return {
                'dynamic': sequence_sample_data,
                'static': self.base_static_tensor,
                'single_dynamic': single_dynamic_data,
                'target': time_target_data,
                'meta_first_date': timeframe['sample'][0]
            }
        except FileNotFoundError:
            return None
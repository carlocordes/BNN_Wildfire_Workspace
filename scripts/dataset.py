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
    def __init__(self, config_path: Path, split_type: str = "train", benchmark_mode = False):
        """
        Initializes dataset tracking by explicit Out-of-Time calendar blocks.
        """
        self.cfg = OmegaConf.load(config_path)
        self.cfg_data = self.cfg['data_paths']['processed']
        cfg_temporal = self.cfg['data']['temporal_extent']

        if benchmark_mode:
            print('[BENCHMARK MODE] Updating sequence period')
            cfg_temporal['sequence_period'] = 14
        
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
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys

    # Quick configuration path definition
    config_path = Path("files/configs/config_t007_1.yaml") # Adjust this if your config has a different path
    
    if not config_path.exists():
        print(f"[ERROR] Config file not found at {config_path}. Please check your path alignment.")
        sys.exit(1)

    print("=" * 60)
    print("        STARTING SPATIO-TEMPORAL DATASET DIAGNOSTIC          ")
    print("=" * 60)

    try:
        # 1. Initialize all three splits to verify out-of-time chronological blocks
        train_set = SpatialTemporalDataset(config_path, split_type="train")
        val_set = SpatialTemporalDataset(config_path, split_type="val")
        test_set = SpatialTemporalDataset(config_path, split_type="test")
        
        print("\n--- Split Allocations & Calendar Ranges ---")
        
        # Helper to extract the date bounds safely if samples exist
        def print_split_range(name, dataset):
            if len(dataset) > 0:
                start_date = dataset.sequences[0]['sample'][0]
                end_date = dataset.sequences[-1]['target'][-1]
                print(f"{name:<20} : {len(dataset):<5} samples | Covers: {start_date} to {end_date}")
            else:
                print(f"{name:<20} : 0     samples | No sequences generated.")

        print_split_range("Training Data", train_set)
        print_split_range("Validation Data", val_set)
        print_split_range("Testing Data", test_set)

        # 2. Extract a sample sequence from the training dataset
        if len(train_set) == 0:
            print("\n[WARNING] Dataset initialized but returned 0 samples. Check raw data folder paths or file dates.")
            sys.exit(0)
            
        print("\n--- Extracting Sample Item [0] from Train Split ---")
        sample = None
        for idx in range(len(train_set)):
            sample = train_set[idx]
            if sample is not None:
                print(f"Successfully retrieved valid sample at index {idx}")
                break
        
        if sample is None:
            print("[ERROR] Iterated through dataset but all samples returned None (missing .tif files).")
            sys.exit(1)

        # 3. Print structural shape and value range diagnostics
        print("\n--- Structural Dimension Diagnostics ---")
        print(f"Static Layers Tensor   : {sample['static'].shape}  | Min: {sample['static'].min():.2f}, Max: {sample['static'].max():.2f}")
        print(f"Single-Dynamic Tensor  : {sample['single_dynamic'].shape}  | Min: {sample['single_dynamic'].min():.2f}, Max: {sample['single_dynamic'].max():.2f}")
        print(f"Dynamic Timeline Tensor: {sample['dynamic'].shape} | Min: {sample['dynamic'].min():.2f}, Max: {sample['dynamic'].max():.2f}")
        print(f"Target Array Tensor    : {sample['target'].shape}  | Min: {sample['target'].min():.2f}, Max: {sample['target'].max():.2f}")

        # 4. Generate a quick, lightweight matplotlib visual summary
        print("\nRendering quick plotting visualization grid window...")
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Plot single static channel layer slice (e.g., Slope or Aspect index 0)
        axes[0].imshow(sample['static'][0].numpy(), cmap='terrain')
        axes[0].set_title(f"Static Layer [0]\n{sample['static'].shape[1:]}")
        axes[0].axis('off')
        
        # Plot single dynamic timeline slice (e.g., NDVI index 0, at step Day 0)
        axes[1].imshow(sample['dynamic'][0, 0].numpy(), cmap='YlGn')
        axes[1].set_title(f"Dynamic Var[0] Day[0]\n{sample['dynamic'].shape[2:]}")
        axes[1].axis('off')

        # Plot secondary dynamic variable snapshot layer
        axes[2].imshow(sample['single_dynamic'][0].numpy(), cmap='plasma')
        axes[2].set_title(f"Single Dynamic [0]\n{sample['single_dynamic'].shape[1:]}")
        axes[2].axis('off')
        
        # Plot target ground truth mask map
        axes[3].imshow(sample['target'][0].numpy(), cmap='gray')
        axes[3].set_title(f"Ground Truth Target\n{sample['target'].shape[1:]}")
        axes[3].axis('off')
        
        plt.suptitle(f"SpatialTemporalDataset Sample Verification (Sequence Start: {sample['meta_first_date']})", fontsize=14)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\n[CRITICAL FAILURE] Diagnostic loop failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
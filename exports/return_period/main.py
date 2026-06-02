# Internal
from src.core.utils import load_config
from src.models.vit.vit import STViT
from scripts.dataset import SpatialTemporalDataset, skip_missing_collate_fn

# External
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import math
import numpy as np
from pathlib import Path
from datetime import datetime
import rasterio

# ----- CONFIG ----- #
correction_logit = math.log(600)
# ------------------ #



def load_trained_model(cfg_model : nn.Module, cfg_data, model_path, device):
    model = STViT(
        num_dynamic_channels=cfg_model['num_dynamic_channels'],
        num_static_channels=cfg_model['num_static_channels'] + cfg_model['num_single_dynamic_channels'],
        num_timestamps_per_sample=cfg_data['temporal_extent']['sample_extent'],
        patch_size=cfg_model['patch_size'],
        embedding_dim=cfg_model['embedding_dim']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)


def predict_and_save_expected_frp(
    model, 
    dataloader: DataLoader, 
    device, 
    output_tiff_path: str, 
    reference_tiff_path: str = None,
    total_years: float = 10.0
):
    """
    Predicts wildfire probabilities, calculates the Expected Fire Return Period,
    and exports the result directly to a GeoTIFF.
    
    Args:
        model: Trained STViT model
        dataloader: PyTorch DataLoader
        device: Device to run inference on ('mps', 'cuda', 'cpu')
        output_tiff_path: Target path for the output GeoTIFF file
        reference_tiff_path: Optional path to an original MODIS input file to copy projection/metadata
        total_years: Baseline duration of the dataset if years cannot be parsed from metadata
    """
    model.eval()

    unique_years = set()
    spatial_shape = None 
    cumulative_expected_burns = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f'Processing test batch_idx {batch_idx + 1} / {len(dataloader)}')
            if batch is None:
                continue

            static_x = batch['static'].to(device)
            single_dynamic_x = batch['single_dynamic'].to(device)
            dynamic_x = batch['dynamic'].to(device)

            # Keep track of dates to compute exact total time footprint
            if 'meta_first_date' in batch:
                first_date = datetime.strptime(batch['meta_first_date'][0], '%Y-%m-%d')
                unique_years.add(first_date.year)

            ## Predict
            logits = model(x_static=static_x,
                           x_dynamic=dynamic_x,
                           x_single_dynamic=single_dynamic_x)

            probs = torch.sigmoid(logits - correction_logit).squeeze()
            probs_np = probs.cpu().numpy()
            
            if cumulative_expected_burns is None:
                spatial_shape = probs_np.shape
                cumulative_expected_burns = np.zeros(spatial_shape, dtype=np.float32)
                
            cumulative_expected_burns += probs_np

    if cumulative_expected_burns is None:
        print("No predictions generated.")
        return None

    # Determine real dataset tracking lifespan
    calculated_years = len(unique_years) if len(unique_years) > 0 else total_years
    print(f"Accumulated probabilities across {calculated_years} years.")

    # Calculate Expected Fire Return Period (Total Years / Expected Burns)
    expected_frp = np.where(
        cumulative_expected_burns > 0,
        calculated_years / cumulative_expected_burns,
        np.nan  # Areas with zero predicted fire risk over 10 years become NaN (transparent)
    )

    # --- Setup Rasterio Metadata Profile ---
    if reference_tiff_path and Path(reference_tiff_path).exists():
        print(f"Extracting spatial metadata profile from reference: {reference_tiff_path}")
        with rasterio.open(reference_tiff_path) as src:
            meta = src.meta.copy()
            # Ensure shape matches, otherwise use reference dimensions
            height, width = src.height, src.width
    else:
        print("No valid reference TIFF provided. Generating standard raster metadata profile...")
        height, width = spatial_shape[-2], spatial_shape[-1]
        meta = {
            'driver': 'GTiff',
            'crs': 'EPSG:4326', # Default to WGS84 if missing
            'transform': rasterio.transform.from_origin(0, 0, 1, 1), # Placeholder pixel size
        }

    # Force metadata update to match our computed float32 array
    meta.update({
        'dtype': 'float32',
        'count': 1,
        'height': height,
        'width': width,
        'nodata': np.nan  # Configures GIS engines to hide unburned pixels cleanly
    })

    # Reshape if array has extra dimensions (e.g., batch or channel dimension of 1)
    if expected_frp.ndim > 2:
        expected_frp = np.squeeze(expected_frp)

    # Write data to disk
    output_path = Path(output_tiff_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(expected_frp.astype(np.float32), 1)

    print(f"Successfully exported Predicted Expected FRP Map to: {output_path}")
    return expected_frp



if __name__ == '__main__':
    
    # Load cfg
    model_name = 't010_1'
    exp_name = 't010_1_rp' # 't009_1'

    cfg_file = 'config_' + exp_name + '.yaml'
    cfg_path = Path('files', 'configs', cfg_file)
    cfg = load_config(cfg_path)
    cfg_model = cfg['model']
    cfg_data = cfg['data']

    # Load model path
    model_path = Path('files', 'experiments', model_name, f'{model_name}_model_best.pt')

    # Device
    device = 'mps'

    # Select baseline model
    model = load_trained_model(
        cfg_model=cfg_model,
        cfg_data=cfg_data,
        model_path=model_path,
        device = device
    )

    # Data
    dataset = SpatialTemporalDataset(cfg_path, split_type="val", benchmark_mode=True)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1,
        pin_memory=False, collate_fn=skip_missing_collate_fn
    )

    output_tiff = Path('exports', 'return_period', 'model_rp.tiff')
    expected_frp = predict_and_save_expected_frp(model = model, dataloader = dataloader, device=device, output_tiff_path=output_tiff)


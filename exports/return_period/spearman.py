import numpy as np
import rasterio
from scipy.ndimage import generic_filter
from scipy.stats import spearmanr
import warnings
from pathlib import Path

# We declare global handles for the 2D arrays so our moving window 
# helper function can access them instantly without array-slicing overhead.
GLOBAL_OBS = None
GLOBAL_PRED = None

def local_spearman(window_indices):
    """
    Helper function applied to every moving 2D window.
    Instead of passing data values, generic_filter passes a window of flat coordinates/indices.
    """
    # Cast indices to integers to extract values from our global arrays
    idx = window_indices.astype(int)
    
    obs_win = GLOBAL_OBS[idx]
    pred_win = GLOBAL_PRED[idx]
    
    # --- ZERO & CONSTANT MASKING ---
    # 1. If the ground truth window has absolutely no recorded fires (all 0.0), mask it out.
    if np.all(obs_win == 0.0):
        return np.nan
        
    # 2. If either window is completely flat/constant, Spearman correlation is mathematically undefined.
    if np.all(obs_win == obs_win[0]) or np.all(pred_win == pred_win[0]):
        return np.nan
        
    # Compute local Spearman rank correlation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = spearmanr(obs_win, pred_win, nan_policy='omit')
        
    return rho


def generate_local_correlation_map(obs_prob, pred_prob, window_size=15):
    """
    Computes a localized Spearman rank correlation map using a 2D moving index footprint.
    """
    assert obs_prob.shape == pred_prob.shape, "Input maps must have the same shape."
    assert window_size % 2 == 1, "Window size must be an odd integer."
    
    # Assign arrays to the global scope for the filter helper function to read
    global GLOBAL_OBS, GLOBAL_PRED
    GLOBAL_OBS = obs_prob.flatten()
    GLOBAL_PRED = pred_prob.flatten()
    
    height, width = obs_prob.shape
    print(f"Calculating local Spearman correlation using a {window_size}x{window_size} window...")
    
    # Create a 2D grid containing the flattened 1D array index locations
    # shape: (height, width)
    index_grid = np.arange(height * width).reshape(height, width).astype(np.float64)
    
    # Run the moving window over the 2D index grid
    local_corr_map = generic_filter(
        index_grid, 
        function=local_spearman, 
        size=(window_size, window_size),
        mode='constant',
        cval=np.nan
    )
        
    return local_corr_map


# --- Main Execution ---
if __name__ == '__main__':

    obs_tiff = Path('exports', 'return_period', 'model_ap.tiff')
    pred_tiff = Path('exports', 'return_period', 'preds_ap.tiff')
    output_corr_tiff = Path('files', 'experiments', 'spearman.tif')

    with rasterio.open(obs_tiff) as src:
        obs_frp = src.read(1)
        meta = src.meta.copy()
        
    with rasterio.open(pred_tiff) as src:
        pred_frp = src.read(1)

    # 1. Transform to annual probability space
    # (Note: Since your files are already named '_ap.tiff', if they are already in probability space,
    # you can directly use the arrays. If they are still raw Return Periods, this step handles it.)
    obs_prob = np.where((obs_frp > 0) & (~np.isnan(obs_frp)), 1.0 / obs_frp, 0.0)
    pred_prob = np.where((pred_frp > 0) & (~np.isnan(pred_frp)), 1.0 / pred_frp, 0.0)

    # 2. Compute the local spatial correlation map
    corr_map = generate_local_correlation_map(obs_prob, pred_prob, window_size=15)

    # 3. Explicitly mask out areas where no fire ever occurred in the observations (zeros or NaNs)
    # This prevents the surrounding zero-risk matrix from bleeding into your validation stats.
    corr_map = np.where((obs_prob == 0.0) | np.isnan(obs_frp), np.nan, corr_map)

    # 4. Save to a new GeoTIFF
    meta.update({
        'dtype': 'float32',
        'count': 1,
        'nodata': np.nan
    })

    output_corr_tiff.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_corr_tiff, "w", **meta) as dst:
        dst.write(corr_map.astype(np.float32), 1)

    print(f"Local Spearman map saved successfully to {output_corr_tiff}!")
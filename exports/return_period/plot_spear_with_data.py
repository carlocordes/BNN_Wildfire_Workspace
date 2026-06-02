import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import rasterio

# Define file paths
obs_tiff = Path('exports', 'return_period', 'model_ap.tiff')
pred_tiff = Path('exports', 'return_period', 'preds_ap.tiff')
output_corr_tiff = Path('files', 'experiments', 'spearman.tif')

# 1. Read the raster data from all three files
with rasterio.open(obs_tiff) as src:
    obs_data = src.read(1)
    
with rasterio.open(pred_tiff) as src:
    pred_data = src.read(1)
    
with rasterio.open(output_corr_tiff) as src:
    corr_data = src.read(1)

# 2. Establish a shared colorbar range for the probability maps
# We dynamically scale the maximum value based on the data, but cap minimum at 0.0
max_prob = max(np.nanmax(obs_data), np.nanmax(pred_data))

# 3. Create the Matplotlib side-by-side subplots
# Using a wide aspect ratio layout (3 panels)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# --- Panel 1: Observed Annual Probability ---
im_obs = axes[0].imshow(
    obs_data, 
    cmap='YlOrRd', 
    vmin=0.0, 
    vmax=max_prob
)
axes[0].set_title('Observed Annual Burn Probability')
axes[0].axis('off')

# --- Panel 2: Predicted Annual Probability ---
im_pred = axes[1].imshow(
    pred_data, 
    cmap='YlOrRd', 
    vmin=0.0, 
    vmax=max_prob
)
axes[1].set_title('Predicted Annual Burn Probability')
axes[1].axis('off')

# Add the shared colorbar specifically for the first two subplots
cbar_prob = fig.colorbar(
    im_pred, 
    ax=[axes[0], axes[1]], 
    orientation='horizontal', 
    pad=0.08, 
    shrink=0.7
)
cbar_prob.set_label('Annual Burn Probability (0.0 - 1.0)')

# --- Panel 3: Local Spearman Correlation Map ---
# Center the diverging color map at 0 by matching vmin and vmax dynamically or explicitly (-1 to 1)
im_corr = axes[2].imshow(
    corr_data, 
    cmap='RdBu_r', 
    vmin=-1.0, 
    vmax=1.0
)
axes[2].set_title('Local Spearman Correlation ($\\rho$)')
axes[2].axis('off')

# Add a separate colorbar for the Spearman map
cbar_corr = fig.colorbar(
    im_corr, 
    ax=axes[2], 
    orientation='horizontal', 
    pad=0.08, 
    shrink=0.8
)
cbar_corr.set_label('Spearman Correlation Coefficient ($\\rho$)')

# Adjust layout and display
plt.suptitle('Spatial Wildfire Risk Performance Assessment', fontsize=16, weight='bold', y=0.98)
plt.tight_layout()
plt.show()
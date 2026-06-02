import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

def scale_rgb_image(rgb_array):
    """Normalizes raw 16-bit satellite bands using a 2%-98% stretch."""
    rgb_swapped = np.moveaxis(rgb_array, 0, -1).astype(np.float32)
    for c in range(3):
        channel = rgb_swapped[..., c]
        p2, p98 = np.percentile(channel, [2, 98])
        if p98 > p2:
            channel = (channel - p2) / (p98 - p2)
        rgb_swapped[..., c] = np.clip(channel, 0, 1)
    return np.moveaxis(rgb_swapped, -1, 0)

# --- 1. CONFIGURATION ---
tif_paths = [
    Path("exports/detailed_results/satellite/sentinel2_2025-08-16.tif"),
    Path("exports/detailed_results/preds/2025-09-02.tif"),
    Path("files/data/processed/target/2025-09-02.tif"),
]
titles = ["Satellite RGB", "Predicted Fire Prob", "Ground Truth"]
cmaps = [None, "gnuplot2", "inferno"]

zoom_configs = [
    {"name": "Zoom Region A", "xmin": 15000, "xmax": 77000, "ymin": 34000, "ymax": 110000},
    {"name": "Zoom Region B", "xmin": -80000, "xmax": -25000, "ymin": -300000, "ymax": -220000}
]

# Exact A4 dimensions in inches (Portrait)
A4_WIDTH, A4_HEIGHT = 8.27, 11.69

# --- 2. CACHE & OPTIMIZE DATA ---
data_cache = []
for path in tif_paths:
    with rasterio.open(path) as src:
        if len(data_cache) == 0:
            img_data = scale_rgb_image(src.read([1, 2, 3]))
            vmin, vmax = None, None
        else:
            img_data = src.read(1)
            valid = img_data[img_data != src.nodata] if src.nodata else img_data
            vmin, vmax = valid.min(), valid.max()
            
        data_cache.append({
            "data": img_data, "transform": src.transform, 
            "vmin": vmin, "vmax": vmax
        })

# --- Helper function for styling subtle ticks ---
def apply_subtle_ticks(ax):
    """Formats axes with tiny, muted, non-scientific notation ticks."""
    # Force labels to be flat integers instead of scientific notation (e.g., 20000 instead of 2e4)
    fmt = FuncFormatter(lambda x, p: f"{int(x)}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    
    # Cap the number of ticks so they don't crowd the subplots
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    
    # Style the ticks to be incredibly thin, short, and grey
    ax.tick_params(
        axis='both', 
        which='major', 
        labelsize=5, 
        colors='#555555', 
        length=2.5, 
        width=0.5, 
        pad=2
    )

# --- 3. PLOT SINGLE-PAGE A4 FIGURE ---
fig, axes = plt.subplots(3, 3, figsize=(A4_WIDTH, A4_HEIGHT))

for i, cache in enumerate(data_cache):
    # --- Row 1: Full Maps ---
    ax_full = axes[0, i]
    ax_full.set_title(f"Full:\n{titles[i]}", fontsize=8, fontweight='bold', pad=4)
    
    if i == 0:
        show(cache["data"], transform=cache["transform"], ax=ax_full)
    else:
        show(cache["data"], transform=cache["transform"], ax=ax_full, cmap=cmaps[i], vmin=cache["vmin"], vmax=cache["vmax"])
        divider = make_axes_locatable(ax_full)
        cax = divider.append_axes("right", size="5%", pad=0.03)
        cbar = fig.colorbar(ax_full.images[0], cax=cax)
        cbar.ax.tick_params(labelsize=5, width=0.5, length=2)

    apply_subtle_ticks(ax_full)

    # --- Rows 2 & 3: Zoom Maps ---
    for zoom_idx, cfg in enumerate(zoom_configs):
        row_idx = zoom_idx + 1
        ax_zoom = axes[row_idx, i]
        ax_zoom.set_title(f"{cfg['name']}:\n{titles[i]}", fontsize=8, fontweight='bold', pad=4)
        
        if i == 0:
            show(cache["data"], transform=cache["transform"], ax=ax_zoom)
        else:
            show(cache["data"], transform=cache["transform"], ax=ax_zoom, cmap=cmaps[i], vmin=cache["vmin"], vmax=cache["vmax"])
            divider = make_axes_locatable(ax_zoom)
            cax = divider.append_axes("right", size="5%", pad=0.03)
            cbar = fig.colorbar(ax_zoom.images[0], cax=cax)
            cbar.ax.tick_params(labelsize=5, width=0.5, length=2)
            
        # Apply crop window constraints
        ax_zoom.set_xlim(cfg["xmin"], cfg["xmax"])
        ax_zoom.set_ylim(cfg["ymin"], cfg["ymax"])
        
        apply_subtle_ticks(ax_zoom)

# Adjust margins to leave breathing room for the tick text on the margins
plt.tight_layout(pad=1.2, h_pad=1.5, w_pad=1.2)

# Save the final consolidated graphic 
output_fig = Path("exports/detailed_results/unified_comparison.png")
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
print(f"Unified one-page A4 figure with ticks saved to {output_fig}")
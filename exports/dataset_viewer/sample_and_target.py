import math
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from pathlib import Path
import numpy as np
import string

plt.rcParams['savefig.dpi'] = 300

# -----------------------------
# Paths
# -----------------------------
base_path = Path('files', 'data', 'processed')
day_tif = '2024-08-23.tif'
data_types = {
    'Aspect E-W': base_path / 'aspect' / 'aspect_EW.tif',
    'Aspect N-S': base_path / 'aspect' / 'aspect_NS.tif',
    'Slope': base_path / 'slope' / 'slope.tif',
    'Burn History': base_path / 'burn_history' / day_tif,
    'Road Proximity': base_path / 'roads' / 'proximity.tif'
}

dynamic_types = [
    'NDVI',
    'NDWI',
    'precip',
    'wind_dir_SN',
    'wind_dir_WE',
    'wind_speed',
    'LST'
]

for cat in dynamic_types:
    data_types[cat] = base_path / cat / day_tif


# -----------------------------
# Colormaps
# -----------------------------
cmaps = {
    'Aspect E-W': 'gist_yarg',
    'Aspect N-S': 'gist_yarg',
    'Slope': 'viridis',
    'Burn History': 'inferno',
    'Road Proximity': 'cividis',
    'NDVI': 'RdYlGn',
    'NDWI': 'BrBG',
    'precip': 'Blues',
    'wind_dir_SN': 'PuOr',
    'wind_dir_WE': 'PuOr',
    'wind_speed': 'YlGnBu',
    'LST': 'plasma',
    'target': 'magma'
}


# -----------------------------
# Figure setup (A4 portrait)
# -----------------------------
n_items = len(data_types)
n_cols = 4
n_rows = math.ceil(n_items / n_cols)

fig = plt.figure(
    figsize=(8.27, 11.69),  # A4 portrait
)

gs = GridSpec(
    n_rows,
    n_cols,
    figure=fig,
    hspace=0.12,
    wspace=0.23
)
letters = string.ascii_lowercase


# -----------------------------
# Plot layers
# -----------------------------
for i, (name, path) in enumerate(data_types.items()):

    row, col = divmod(i, 4)
    ax = fig.add_subplot(gs[row, col])

    try:
        with rasterio.open(path) as src:
            img = src.read(1)

        # ---------------------------------
        # Mask invalid values
        # ---------------------------------

        # Keep only finite values
        img = np.where(np.isfinite(img), img, np.nan)

        # Mask everything below 0 (nodata)
        img = np.where(img >= 0, img, np.nan)

        # Skip empty rasters
        if np.all(np.isnan(img)):
            ax.text(
                0.5, 0.5,
                "No valid data",
                ha='center',
                va='center',
                fontsize=10,
                transform=ax.transAxes
            )
            ax.axis('off')
            continue

        # ---------------------------------
        # Robust contrast scaling
        # ---------------------------------
        vmin = np.nanpercentile(img, 2)
        vmax = np.nanpercentile(img, 98)

        # Avoid edge case where min == max
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6

        # ---------------------------------
        # Colormap with white nodata
        # ---------------------------------
        cmap = plt.get_cmap(
            cmaps.get(name, 'viridis')
        ).copy()

        cmap.set_bad(color='white')

        # ---------------------------------
        # Plot
        # ---------------------------------
        im = ax.imshow(
            img,
            cmap=cmap,
            norm=Normalize(vmin=vmin, vmax=vmax)
        )

        # Clean look
        ax.set_xticks([])
        ax.set_yticks([])

        # Title with panel label
        ax.set_title(
            f"({letters[i]}) {name}",
            fontsize=10,
            pad=4
        )

        # Highlight target layer
        if name == 'target':
            for spine in ax.spines.values():
                spine.set_linewidth(2)

        # Small colorbar
        cbar = plt.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.02
        )
        cbar.ax.tick_params(labelsize=6)

    except Exception as e:
        ax.text(
            0.5, 0.5,
            f"Error\n{name}",
            ha='center',
            va='center',
            fontsize=9,
            transform=ax.transAxes
        )
        ax.axis('off')
        print(f"Failed loading {path}: {e}")

out_f = Path('exports', 'dataset_viewer', 'all_samples.png')
plt.savefig(
    out_f,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
    facecolor="white"
)

#plt.show()
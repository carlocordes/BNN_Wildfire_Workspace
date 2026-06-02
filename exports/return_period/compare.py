from pathlib import Path
import matplotlib.pyplot as plt
import rasterio


def plot_files(path_to_gt: Path, path_to_preds: Path):
    """Reads single-band TIFF files using rasterio and plots them side-by-side

    without any CRS or geographic metadata.
    """
    # Open and read the single band directly into numpy arrays
    with rasterio.open(path_to_gt) as src_gt:
        gt_arr = src_gt.read(1)

    with rasterio.open(path_to_preds) as src_preds:
        preds_arr = src_preds.read(1)

    # Set up the side-by-side plot layout
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Ground Truth (Raw array)
    im1 = axes[0].imshow(gt_arr, cmap="viridis")
    axes[0].set_title("Ground Truth (fire_return_pd)")
    axes[0].axis("off")  # Hides all coordinates and axes
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot Predictions (Raw array)
    im2 = axes[1].imshow(preds_arr, cmap="viridis")
    axes[1].set_title("Predictions (model_rp)")
    axes[1].axis("off")  # Hides all coordinates and axes
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Show the plot window
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    path_to_gt = Path('files', 'data', 'test', 'yearly_burn', 'model_rp.tif')
    path_to_preds = Path('exports', 'return_period', 'model_rp.tiff')
    #path_to_gt = Path('exports', 'return_period',  'model_ap.tiff')
    #path_to_preds = Path('exports', 'return_period',  'preds_ap.tiff')

    plot_files(path_to_gt, path_to_preds)
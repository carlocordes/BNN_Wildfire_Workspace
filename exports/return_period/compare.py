from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio

OUT_PATH = Path("exports/return_period/return_pd_compare.png")


def plot_files(path_to_gt: Path, path_to_preds: Path):
    """Reads single-band TIFF files using rasterio and plots them side-by-side.

    The color scale is unified based on the global minimum and maximum values
    across both datasets (ignoring NaNs).
    """
    # Open and read the single band directly into numpy arrays
    with rasterio.open(path_to_gt) as src_gt:
        gt_arr = src_gt.read(1).astype(float)

    with rasterio.open(path_to_preds) as src_preds:
        preds_arr = src_preds.read(1).astype(float)

    # Apply your masks
    # gt_arr[gt_arr > 26] = np.nan
    preds_arr[preds_arr > 150] = np.nan

    # Calculate global minimum and maximum across both arrays (ignoring NaNs)
    v_min = min(np.nanmin(gt_arr), np.nanmin(preds_arr))
    v_max = max(np.nanmax(gt_arr), np.nanmax(preds_arr))

    # Set up the side-by-side plot layout
    fig, axes = plt.subplots(1, 2, figsize=(8, 7))

    # Plot Ground Truth with unified color limits
    im1 = axes[0].imshow(gt_arr, cmap="viridis", vmin=v_min, vmax=v_max)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label("years", rotation=270, labelpad=15)  # Added unit label

    # Plot Predictions with identical unified color limits
    im2 = axes[1].imshow(preds_arr, cmap="viridis", vmin=v_min, vmax=v_max)
    axes[1].set_title("Predictions")
    axes[1].axis("off")
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label("years", rotation=270, labelpad=15)  # Added unit label

    # Save the plot window
    plt.tight_layout()

    # Ensure output directory exists before saving
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=300)


if __name__ == "__main__":
    path_to_gt = Path(
        "files", "data", "test", "yearly_burn", "fire_return_pd_25.tif"
    )
    path_to_preds = Path("exports", "return_period", "model_rp.tiff")

    plot_files(path_to_gt, path_to_preds)
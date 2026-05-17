# Internal
from src.core.goldengrid import GoldenGrid

# External
from datetime import datetime
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import xarray as xr
from pathlib import Path

def get_dates_since_last_burn(
    zarr_path: Path,
    golden_grid,
    target_date: str,
    out_tif: Path,
    nodata: int = -9999
):
    """
    Creates a GeoTIFF where each pixel contains:
    days since last burn event up to target_date.
    """

    # -----------------------------
    # 1. Load dataset
    # -----------------------------
    ds = xr.open_zarr(zarr_path)

    burn = ds["burn"].values  # (time, y, x)
    time_coords = ds["time"].values

    # Convert target date → index
    target_dt = np.datetime64(target_date)
    if target_dt not in time_coords:
        raise ValueError(f"{target_date} not found in Zarr time coordinate")

    t_index = int(np.where(time_coords == target_dt)[0][0])

    # Only consider data up to target date
    burn = burn[: t_index + 1]

    T, H, W = burn.shape

    # -----------------------------
    # 2. Compute last burn time index per pixel
    # -----------------------------
    # Convert burn events into time indices
    # Shape: (T, H*W)
    burn_flat = burn.reshape(T, -1)

    # Replace non-burns with -inf, burns with time index
    time_index = np.arange(T).reshape(T, 1)

    burn_time = np.where(burn_flat == 1, time_index, -1)

    # Running maximum over time axis → last burn time index
    last_burn_time = np.maximum.accumulate(burn_time, axis=0)

    last_burn_time = last_burn_time.reshape(H, W)

    # -----------------------------
    # 3. Compute days since last burn
    # -----------------------------
    days_since = np.full((H, W), nodata, dtype=np.int32)

    valid = last_burn_time >= 0
    days_since[valid] = t_index - last_burn_time[valid]

    # Optional: pixels that burned on target day → 0
    days_since[last_burn_time == t_index] = 0

    # -----------------------------
    # 4. Geo transform (from grid)
    # -----------------------------
    # Assumes golden_grid has bbox: [xmin, ymin, xmax, ymax]
    xmin, ymin, xmax, ymax = golden_grid.bbox
    height, width = H, W

    x_res = (xmax - xmin) / width
    y_res = (ymax - ymin) / height

    transform = from_origin(xmin, ymax, x_res, y_res)

    # -----------------------------
    # 5. Write GeoTIFF
    # -----------------------------
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        out_tif,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="int32",
        crs=golden_grid.crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(days_since, 1)

    print(f"Saved: {out_tif}")

if __name__ == '__main__':
    latmin, longmin = 36.812, -9.490
    latmax, longmax = 42.2724, -6.0234

    start_date = '2024-01-01'
    end_date = '2024-01-02'

    portugal_ggrid = GoldenGrid(
        crs='EPSG:3763',
        scale=1000,
        bbox=[longmin, latmin, longmax, latmax],
        start_date=start_date,
        end_date=end_date,
        day_interval=1
    )


    zarr_path = Path('data', 'raw', 'target_zarr')
    target_date = datetime(2024, 8, 1)

    get_dates_since_last_burn(zarr_path=zarr_path,
                              golden_grid= portugal_ggrid,
                              target_date=target_date,
                              out_tif=Path('data', 'processed', 'burn_history', 'test.tif'))
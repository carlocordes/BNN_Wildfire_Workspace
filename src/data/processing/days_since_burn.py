# Internal
from src.core.goldengrid import GoldenGrid

# External
from datetime import datetime
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin, from_bounds
from rasterio.warp import transform_bounds
import xarray as xr
from pathlib import Path

def get_dates_since_last_burn(
    zarr_path: Path,
    golden_grid,
    target_date: str,
    out_path: Path,
    nodata: int = 9999
):
    """
    Creates a GeoTIFF where each pixel contains:
    days since last burn event up to target_date.
    """

    # Load dataset
    ds = xr.open_zarr(zarr_path, mask_and_scale=False)

    target_dt = pd.to_datetime(target_date)

    if target_dt < pd.to_datetime(ds.time.values[0]):
        raise ValueError(f"Target date {target_date} is before the dataset start date.")

    # Slice up to target date
    ds_historical = ds.sel(time=slice(None, target_date))
    target_index = len(ds_historical.time) - 1

    num_days = len(ds_historical.time)
    time_indices = xr.DataArray(
            np.arange(num_days), 
            dims=['time'], 
            coords={'time': ds_historical.time}
        )
    
    # Binary mask for day interval
    burn_indices = ds_historical.burn * time_indices

    # Get days indices for every pixel
    last_fire_index = burn_indices.max(dim='time')

    days_since = target_index - last_fire_index

    # Case handling: pixels never burned
    ever_burned = ds_historical.burn.any(dim='time')
    days_since = days_since.where(ever_burned, nodata)

    # Cast to int and get values
    raster_data = days_since.values.astype(np.int16)


    # Dimension handling
    height, width = raster_data.shape
    west_4326, south_4326, east_4326, north_4326 = golden_grid.bbox
    west, south, east, north = transform_bounds(
        src_crs="EPSG:4326",
        dst_crs=golden_grid.crs,  # e.g., 'EPSG:3763'
        left=west_4326,
        bottom=south_4326,
        right=east_4326,
        top=north_4326
    )

    # Define transform
    transform = from_bounds(west, south, east, north, width, height)

    fname = out_path / f'{target_date}.tif'

    meta = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,                  
        'dtype': rasterio.int16,
        'crs': golden_grid.crs,      
        'transform': transform,
        'nodata': nodata,
        'compress': 'lzw'            
    }

    with rasterio.open(fname, 'w', **meta) as dst:
        dst.write(raster_data, 1) 

    print(f'Wrote burn raster for {target_date} to {out_path}')   


def process_burn_history_catalogue(zarr_path : Path, out_path: Path, golden_grid: GoldenGrid):
    date_strings = golden_grid.dates.strftime('%Y-%m-%d').tolist()
    for date in date_strings:
        get_dates_since_last_burn(
            zarr_path=zarr_path,
            out_path = out_path,
            golden_grid=golden_grid,
            target_date=date,
        )


if __name__ == '__main__':
    latmin, longmin = 36.812, -9.490
    latmax, longmax = 42.2724, -6.0234

    start_date = '2024-01-01'
    end_date = '2024-01-01'

    portugal_ggrid = GoldenGrid(
        crs='EPSG:3763',
        scale=1000,
        bbox=[longmin, latmin, longmax, latmax],
        start_date=start_date,
        end_date=end_date,
        day_interval=1
    )

    zarr_path = Path('data', 'raw', 'target_zarr')
    target_date = datetime(2025, 8, 1)
    """
    process_burn_history_catalogue(
        zarr_path=zarr_path,
        golden_grid= portugal_ggrid,
        out_path=Path('data', 'processed', 'burn_history')
    )"""


    get_dates_since_last_burn(
        zarr_path=zarr_path,
        golden_grid= portugal_ggrid,
        target_date='2025-08-01',
        out_path = Path('data', 'raw', 'test')
    )
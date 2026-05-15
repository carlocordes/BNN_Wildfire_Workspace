# Internal
from src.core.goldengrid import GoldenGrid

# External
import math
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

import rasterio
from rasterio.warp import transform_bounds
from rasterio.transform import from_bounds


def extract_zarr_window(zarr_path : Path, start_date : datetime, end_date : datetime, variable : str):
    """
    Accesses zarr and returns multiple images
    """
    # Open
    ds = xr.open_zarr(zarr_path)

    # Slice
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    subset = ds[variable].sel(time=slice(start_str, end_str))

    print(f'Returning {variable} from {start_str} to {end_str}')

    # Wrangle to numpy
    values = subset.values

    return values


def write_tiff(data, golden_grid : GoldenGrid, out_path : Path, start_date):
    ## Raster parameters
    height, width = data.shape
    lon_min, lat_min, lon_max, lat_max = golden_grid.bbox
    xmin, ymin, xmax, ymax = transform_bounds(
        src_crs="EPSG:4326",
        dst_crs=golden_grid.crs,  # 'EPSG:3763'
        left=lon_min,
        bottom=lat_min,
        right=lon_max,
        top=lat_max
    )
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    # Write
    out_name = start_date.strftime('%Y-%m-%d') + '.tif'
    out_file = out_path / out_name
    with rasterio.open(
        out_file,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=golden_grid.crs,
        transform=transform,
        compress="lzw"
    ) as dst:
        dst.write(data, 1)


def produce_targets(golden_grid : GoldenGrid, in_path : Path, out_path_target : Path, target_extent : int):
    """
    Make targets for golden grid configuration
    """


    ## Dates

    for start_date in golden_grid.dates:
        end_date = start_date + timedelta(days = target_extent -1) # total count (-1)

        ## Extract from zarr
        burn = extract_zarr_window(zarr_path=in_path, start_date=start_date, end_date=end_date,
                                variable='burn')

        # Compute union
        burn_max = np.max(burn, axis=0)

        ## Write
        write_tiff(burn_max, golden_grid=golden_grid, out_path = out_path_target, start_date=start_date)


if __name__ == '__main__':

    # Portugal Bounding Box
    latmin, longmin = 36.812, -9.490
    latmax, longmax = 42.2724, -6.0234

    start_date = '2015-08-01'
    end_date = '2015-08-31'

    portugal_ggrid = GoldenGrid(
        crs = 'EPSG:3763',
        scale = 1000,
        bbox = [longmin, latmin, longmax, latmax],
        start_date = start_date,
        end_date = end_date,
        day_interval = 1
    )

    produce_targets(golden_grid=portugal_ggrid,
                    in_path=Path('data', 'raw', 'target_alt'),
                    out_path_target=Path('data', 'processed', 'target_area'),
                    target_extent= 6)
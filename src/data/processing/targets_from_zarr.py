# Internal
from src.core.goldengrid import GoldenGrid

# External
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path



def extract_zarr_window(zarr_path : Path, start_date, end_date):

    # Open
    ds = xr.open_zarr(zarr_path)

    # Slice
    subset = ds['burn'].sel(time=slice(start_date, end_date))

    # Wrangle to numpy
    values = subset.values


    print(values)
    print(values[0].shape)


if __name__ == '__main__':

    extract_zarr_window(zarr_path = Path('data', 'raw', 'target_alt'),
                        start_date ='2025-08-01',
                        end_date = '2025-08-01')
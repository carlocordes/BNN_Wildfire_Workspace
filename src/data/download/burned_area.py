# Internal
from src.core.goldengrid import GoldenGrid

# External
from pathlib import Path
import ee
import requests
import mimetypes
from datetime import datetime

import xarray as xr
import math
import numpy as np
import pandas as pd
import zarr

def init_zarr(golden_grid : GoldenGrid, zarr_path : Path):
    ## Establish dimensions of output via test request
    test_ds = ee.ImageCollection('MODIS/061/MCD64A1') \
            .filterDate("2024-01-01", "2024-12-31") \
            .filterBounds(golden_grid.aoi) \
            .select(['BurnDate', 'Uncertainty'])
    burned_img = test_ds.max().unmask(0)

    processed_img = burned_img.reproject(
        crs=golden_grid.crs, 
        scale=golden_grid.scale
    ).clip(golden_grid.aoi)

    pixel_dict = processed_img.sampleRectangle(region=golden_grid.aoi, defaultValue=0)
    burn_date_array = np.array(pixel_dict.get('BurnDate').getInfo())
    height, width = burn_date_array.shape


    ## Initialize data structure
    num_dates = len(golden_grid.dates)
    ds = xr.Dataset(
        {
            "burn" : (['time', 'y', 'x'], np.zeros((num_dates, height, width), dtype = 'uint8')),
            'uncertainty' : (['time', 'y', 'x'], np.zeros((num_dates, height, width), dtype = 'uint8'))
        },
        coords = {
            "time" : golden_grid.dates,
            "y" : np.arange(height),
            "x" : np.arange(width),
        }
    )

    # Chunking
    ds = ds.chunk({'time' : 365, 'y' : 100, 'x' : 100})
    ds.to_zarr(zarr_path, mode = 'w')

    return ds

def download_annual_images(golden_grid: GoldenGrid, out_path: Path):
    """
    Iterates through each calendar year in the GoldenGrid date range
    and saves a separate TIFF for each.
    """
    # 1. Parse dates and identify the range of years
    start_year = datetime.strptime(golden_grid.start_date, '%Y-%m-%d').year
    end_year = datetime.strptime(golden_grid.end_date, '%Y-%m-%d').year
    
    out_path.mkdir(parents=True, exist_ok=True)

    for year in range(start_year, end_year + 1):
        print(f"--- Processing Year: {year} ---")
        
        # Define annual bounds
        year_start = f"{year}-01-01"
        year_end = f"{year}-12-31"
        
        # Filter for the specific year
        dataset = ee.ImageCollection('MODIS/061/MCD64A1') \
            .filterDate(year_start, year_end) \
            .filterBounds(golden_grid.aoi) \
            .select('BurnDate')

        # Get the maximum burn date per pixel for that year
        burned_img = dataset.max().unmask(0)

        processed_img = burned_img.reproject(
            crs=golden_grid.crs, 
            scale=golden_grid.scale
        ).clip(golden_grid.aoi)

        # 2. Generate the Download URL
        try:
            url = processed_img.getDownloadURL({
                'name': f'burned_area_{year}',
                'scale': golden_grid.scale,
                'crs': golden_grid.crs,
                'region': golden_grid.aoi,
                'format': 'GEO_TIFF'
            })
            
            response = requests.get(url, stream=True)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type')
                extension = mimetypes.guess_extension(content_type) or '.zip'
                
                if "text/html" in content_type:
                    print(f"Warning: EEE returned an error page for year {year}.")
                    continue

                target_file = out_path / f"{year}{extension}"
                
                with open(target_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"Successfully saved: {target_file}")
            else:
                print(f"Error {response.status_code} for year {year}: {response.text}")

        except Exception as e:
            print(f"Failed to process year {year}: {e}")

def download_area_with_uncertainty(
        golden_grid: GoldenGrid,
        out_path: Path):
    
    ## Gather applicable list of years
    start_year = datetime.strptime(golden_grid.start_date, '%Y-%m-%d').year
    end_year = datetime.strptime(golden_grid.end_date, '%Y-%m-%d').year
    years = list(range(start_year, end_year + 1))


    ## Initialize Zarr
    init_zarr(golden_grid=golden_grid, zarr_path=out_path)

    z_burn = zarr.open(str(out_path / 'burn'), mode='r+')
    z_uncertainty = zarr.open(str(out_path / 'uncertainty'), mode='r+')

    for year in range(start_year, end_year + 1):
        print(f"--- Processing Year: {year} ---")
        
        # Get the Collection (12 monthly images)
        collection = ee.ImageCollection('MODIS/061/MCD64A1') \
            .filterDate(f"{year}-01-01", f"{year}-12-31") \
            .filterBounds(golden_grid.aoi) \
            .select(['BurnDate', 'Uncertainty'])

        # Convert collection to a list to iterate through each month
        img_list = collection.toList(collection.size())
        num_images = img_list.size().getInfo()

        for i in range(num_images):
            print(f'Processing month {i+1}')
            img = ee.Image(img_list.get(i)).unmask(0)
            
            # Reproject and sample each month individually
            processed_img = img.reproject(
                crs=golden_grid.crs, 
                scale=golden_grid.scale
            ).clip(golden_grid.aoi)
            

            pixel_dict = processed_img.sampleRectangle(region=golden_grid.aoi, defaultValue=0).getInfo()

            if 'properties' in pixel_dict:
                props = pixel_dict['properties']

                # Isolate data
                burn_data = np.array(props['BurnDate'], dtype='uint16')
                unc_data = np.array(props['Uncertainty'], dtype='uint8')

                # Define time indices
                grid_start_dt = datetime.strptime(golden_grid.start_date, '%Y-%m-%d')
                year_start_dt = datetime(year, 1, 1)
                day_offset = (year_start_dt - grid_start_dt).days # Offset of xxxx-01-01 to start of zarr

                # Isolate burn pixels
                y_indices, x_indices = np.where(burn_data > 0)

                if len(y_indices) > 0:
                    print(f'Found {len(y_indices)} burn pixels for month {i+1}-{year}')
                    # Use batch assignment for speed
                    julian_days = burn_data[y_indices, x_indices].astype(int)
                    uncertainties = unc_data[y_indices, x_indices]
                    
                    time_indices = day_offset + (julian_days - 1)
                    
                    # Direct write to disk
                    # This only modifies the specific chunks affected by these pixels
                    for idx in range(len(time_indices)):
                        t = time_indices[idx]
                        yy, xx = y_indices[idx], x_indices[idx]
                        
                        if 0 <= t < z_burn.shape[0]:
                            z_burn[t, yy, xx] = 1
                            z_uncertainty[t, yy, xx] = uncertainties[idx]

        print(f"Finished Year {year}. Progress is saved to disk.")

if __name__ == '__main__':
    # Initialize with your project ID
    project_id = 'transformerwildfire'
    ee.Initialize(project=project_id)

    # Portugal Bounding Box
    latmin, longmin = 36.812, -9.490
    latmax, longmax = 42.2724, -6.0234

    start_date = '2015-01-01'
    end_date = '2026-12-31'

    portugal_ggrid = GoldenGrid(
        crs = 'EPSG:3763',
        scale = 1000,
        bbox = [longmin, latmin, longmax, latmax],
        start_date = start_date,
        end_date = end_date,
        day_interval = 1
    )

    download_area_with_uncertainty(
        golden_grid = portugal_ggrid, 
        out_path = Path('data', 'raw', 'target_alt')
    )
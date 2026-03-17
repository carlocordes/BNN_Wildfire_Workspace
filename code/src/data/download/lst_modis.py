# Internal
from src.core.goldengrid import GoldenGrid

# External
import time
from pathlib import Path

import pandas as pd
import ee
import geemap


def get_one_lst_image(date, golden_grid : GoldenGrid):
    collection = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterBounds(golden_grid.aoi)
        .filterDate(date, ee.Date(date).advance(8, 'day'))
    )

    image = collection.first()

    if image is None:
        raise ValueError(f"No image found for {date}")

    lst = (
        image.select("LST_Day_1km")
        .multiply(0.02)
        .subtract(273.15)
        .reproject(crs=golden_grid.target_proj)
    )

    # Build output path
    output_path = Path('data', 'processed', 'LST_MODIS_8day') / f'lst_{date}.tif'
    
    # Ensure the local folder exists before saving
    output_path.parent.mkdir(parents=True, exist_ok=True) 

    geemap.ee_export_image(
        lst,
        filename=str(output_path),
        scale=golden_grid.scale,
        region=golden_grid.aoi,
        crs=golden_grid.crs,
        file_per_band=False,
    )
    
    return None

def download_yearly_lst(year, golden_grid : GoldenGrid):
    print(f"--- Starting downloads for the year {year} ---")
    
    dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='8D')
    date_strings = dates.strftime('%Y-%m-%d').tolist()

    for date in date_strings:
        print(f"\nProcessing date: {date}")
        try:
            get_one_lst_image(date, golden_grid=golden_grid)
            time.sleep(1) 
            
        except ValueError as e:
            print(f"Skipped {date}: {e}")
        except Exception as e:
            print(f"An error occurred for {date}: {e}")
            
    print(f"\n--- Finished downloading data for {year} ---")

if __name__ == '__main__':
        download_yearly_lst(2024, portugal_ggrid)
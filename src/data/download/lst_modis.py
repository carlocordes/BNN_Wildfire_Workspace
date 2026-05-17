# Internal
from src.core.goldengrid import GoldenGrid

# External
import time
from pathlib import Path

import pandas as pd
import ee
import geemap


def get_one_lst_image(date, out_dir : Path, golden_grid : GoldenGrid):

    # Build output path
    output_path = out_dir / f'{date}.tif'

    if not output_path.exists():
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
    else:
        print(f'Skipping step {date}. File already exists.')
    
    return None

def download_lst(out_dir : Path, golden_grid : GoldenGrid):
  
    dates = golden_grid.dates
    date_strings = dates.strftime('%Y-%m-%d').tolist()

    for date in date_strings:
        print(f"\nProcessing date: {date}")
        try:
            get_one_lst_image(
                date = date,
                out_dir = out_dir,
                golden_grid=golden_grid)
            time.sleep(1) 
            
        except ValueError as e:
            print(f"Skipped {date}: {e}")
        except Exception as e:
            print(f"An error occurred for {date}: {e}")
     

if __name__ == '__main__':
        #download_yearly_lst(2024, portugal_ggrid)
        pass
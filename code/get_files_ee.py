import ee
import geemap
from pathlib import Path
from time import time
import pandas as pd

# ----------- CONFIG ----------- #
PROJECT_ID = 'transformerwildfire'

ee.Initialize(project=PROJECT_ID)

latmin, longmin = 36.812, -9.490
latmax, longmax = 43.046, -4.948

aoi = ee.Geometry.Rectangle([longmin, latmin, longmax, latmax])

# ------------------------------ #


def get_one_lst_image(date):
    collection = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterBounds(aoi)
        .filterDate(date, ee.Date(date).advance(8, 'day')) # Filtering between start_date and end_date, MODIS has 8 day averages
    )

    image = collection.first()

    if image is None:
        raise ValueError(f"No image found for {date}")

    # Scaling, conversion to Celcius
    lst = (
        image.select("LST_Day_1km")
        .multiply(0.02)
        .subtract(273.15)
    )

    # Build output path
    output_path = Path('data', 'raw', 'LST_MODIS_8day') / f'lst_{date}.tif'

    geemap.ee_export_image(
        lst,
        filename=output_path,
        scale=926.6254331383326, # MODIS native scale
        region=aoi,
        crs='EPSG:4326',
        file_per_band=False,
    )

    return lst

def download_yearly_lst(year):
    print(f"--- Starting downloads for the year {year} ---")
    
    # Generate dates every 8 days starting from Jan 1st of the target year
    # This perfectly aligns with the MOD11A2 8-day composite schedule
    dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='8D')
    date_strings = dates.strftime('%Y-%m-%d').tolist()

    for date in date_strings:
        print(f"\nProcessing date: {date}")
        try:
            # Call your download function
            get_one_lst_image(date)
            
            # Optional: Add a tiny sleep to avoid overwhelming Earth Engine's API limits
            time.sleep(1) 
            
        except ValueError as e:
            # If no image is found (e.g., collection hasn't updated yet), skip and continue
            print(f"Skipped {date}: {e}")
        except Exception as e:
            # Catch other potential Earth Engine/Geemap errors
            print(f"An error occurred for {date}: {e}")
            
    print(f"\n--- Finished downloading data for {year} ---")

if __name__ == '__main__':
    date = "2020-01-01"
    fname = f'surface_temp_test{date}.tif'
    file_target_path = Path('data', 'raw', 'surface_temp') / fname

    download_yearly_lst(2024)


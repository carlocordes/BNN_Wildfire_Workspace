import ee
import geemap
from pathlib import Path
import time  # <-- FIXED: changed from 'from time import time'
import pandas as pd

# ----------- CONFIG ----------- #
PROJECT_ID = 'transformerwildfire'

#ee.Authenticate()
ee.Initialize(project=PROJECT_ID)

latmin, longmin = 36.812, -9.490
#latmin, longmin = 40.8653,-8.5692
latmax, longmax = 42.2724, -6.0234



# ------- GOLDEN GRID ---------- #
TARGET_CRS = 'EPSG:3763' # ETRS89 Portugal TM06
TARGET_SCALE = 1000
target_proj = ee.Projection(TARGET_CRS).atScale(TARGET_SCALE)
# ------------------------------ #


class GoldenGrid():
    def __init__(self, crs, scale, bbox : list[float]):
        self.crs = crs
        self.scale = scale
        self.aoi = ee.Geometry.Rectangle(bbox)

        self.target_proj = ee.Projection(crs).atScale(scale)



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
        .reproject(crs=portugal_grid.target_proj)
    )

    # Build output path
    output_path = Path('data', 'raw', 'LST_MODIS_8day') / f'lst_{date}.tif'
    
    # <-- FIXED: Ensure the local folder exists before saving
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

def get_dtm(dtm_dir : Path, golden_grid : GoldenGrid) -> None:
    """
    dtm = (
        ee.ImageCollection("CGIAR/SRTM90_V4")
        .select('DEM')
        .mosaic()
        .setDefaultProjection(crs='EPSG:4326', scale=30.0)
    )
    """
    dtm = ee.Image("CGIAR/SRTM90_V4").select('elevation')

    dtm_aligned = (
        dtm
        .reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=2048
        )
        .reproject(crs=golden_grid.target_proj)
    )

    dtm_dir.mkdir(parents=True, exist_ok=True) # Assert that directory exists

    output_path = str(dtm_dir / 'dtm_aligned.tif')
    print(f"Exporting aligned DTM to {output_path}")

    geemap.ee_export_image(
        dtm_aligned,
        filename=output_path,
        scale=golden_grid.scale,
        region=golden_grid.aoi,
        crs=golden_grid.crs,
        file_per_band=False
    )

if __name__ == '__main__':
    portugal_grid = GoldenGrid(crs = 'EPSG:3763',
                               scale = 1000,
                               bbox = [longmin, latmin, longmax, latmax])

    #download_yearly_lst(2024)
    get_dtm(Path('data', 'raw', 'dtm'), portugal_grid)
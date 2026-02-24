# Internal

# External
import math

import numpy as np
import ee
import geemap
from pathlib import Path
import time
import pandas as pd
import geopandas as gpd
import glob
from shapely.geometry import box
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

# ----------- CONFIG ----------- #
PROJECT_ID = 'transformerwildfire'
#ee.Authenticate()
ee.Initialize(project=PROJECT_ID)
# ------------------------------ #


class GoldenGrid():
    """
    Defines project-unified grid via CRS, scale (m) and area of interest
    """
    def __init__(self, crs, scale, bbox : list[float]):
        self.crs = crs
        self.scale = scale
        self.bbox = bbox
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
        .reproject(crs=golden_grid.target_proj)
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

def get_one_dtm_image(dtm_dir : Path, golden_grid : GoldenGrid) -> None:

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

def grid_target_map(path_to_csv : Path,
                    out_path : Path,
                    golden_grid : GoldenGrid,
                    day_interval : str
                    ) -> None:

    load_path = glob.glob(str(path_to_csv) + '/*.csv')[0]

    # Load and disregard
    points = gpd.read_file(load_path).drop(columns = ["brightness", "scan", "track", "acq_time","satellite",
                                                  "instrument", "version", "frp", "daynight", "type"])
    
    # Some data wrangling
    points["acq_date"] = pd.to_datetime(points["acq_date"])

    points["month"] = points["acq_date"].dt.month
    points["day_of_year"] = points["acq_date"].dt.dayofyear

    points = gpd.GeoDataFrame(
        points,
        geometry=gpd.points_from_xy(points.longitude, points.latitude),
        crs="EPSG:4326",
    ).drop(columns = ["longitude", "latitude"])

    # Produce tifs at day interval
    dates = pd.date_range(start=f'2024-01-01', end=f'2024-12-31', freq=f'{day_interval}D') # TODO: either add to goldengrid or as function variable, dont hardcode

    for start_date in dates:
        end_date = start_date + pd.Timedelta(days = day_interval)

        mask = (points['acq_date'] >= start_date) & (points['acq_date'] < end_date)
        subset_gdf = points.loc[mask].copy()

        date_str = start_date.strftime('%Y_%m_%d')
        print(f"Interval {date_str} to {end_date.strftime('%Y-%m-%d')} | Points found: {len(subset_gdf)}")

        rasterize_points(points = subset_gdf,
                         date_str = date_str,
                         out_path  = out_path,
                         golden_grid = golden_grid)


def rasterize_points(points: gpd.GeoDataFrame,
                     date_str: str,
                     out_path: Path,
                     golden_grid) -> None:

    # Project the GoldenGrid bounding box to the target CRS
    grid_bbox_4326 = box(*golden_grid.bbox)
    grid_gdf = gpd.GeoDataFrame(geometry=[grid_bbox_4326], crs="EPSG:4326")
    grid_gdf_proj = grid_gdf.to_crs(golden_grid.crs)

    minx, miny, maxx, maxy = grid_gdf_proj.total_bounds

    # Snap coordinates
    minx_snapped = math.floor(minx / golden_grid.scale) * golden_grid.scale
    maxy_snapped = math.ceil(maxy / golden_grid.scale) * golden_grid.scale
    maxx_snapped = math.ceil(maxx / golden_grid.scale) * golden_grid.scale
    miny_snapped = math.floor(miny / golden_grid.scale) * golden_grid.scale

    # Calculate grid dimensions
    width = int((maxx_snapped - minx_snapped) / golden_grid.scale)
    height = int((maxy_snapped - miny_snapped) / golden_grid.scale)

    # from_origin takes (west, north, xsize, ysize)
    transform = from_origin(minx_snapped, maxy_snapped, golden_grid.scale, golden_grid.scale)

    #Project points and handle empty selections
    points_proj = points.to_crs(golden_grid.crs)
    
    if points_proj.empty:
        # If no fires occurred in this 8-day window, create a blank array of zeros
        binary_raster = np.zeros((height, width), dtype="uint8")
    else:
        # Create list of (geometry, 1) pairs for rasterio
        shapes = ((geom, 1) for geom in points_proj.geometry)
        
        binary_raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=False 
        )

    # Write the file safely
    out_path.mkdir(parents=True, exist_ok=True)
    output_file = out_path / f"anom_{date_str}.tif"

    with rasterio.open(
        output_file,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=golden_grid.crs,
        transform=transform,
        compress="lzw" # Highly recommended: drastically reduces file size for binary data
    ) as dst:
        dst.write(binary_raster, 1)
        
    print(f"Rasterized {len(points_proj)} points for {date_str} to {output_file.name}")


# Define local golden grid
latmin, longmin = 36.812, -9.490
latmax, longmax = 42.2724, -6.0234

portugal_ggrid = GoldenGrid(crs = 'EPSG:3763',
                            scale = 1000,
                            bbox = [longmin, latmin, longmax, latmax])

if __name__ == '__main__':

    grid_target_map(path_to_csv = Path('data', 'raw', 'burn'),
                    out_path = Path('data', 'processed', 'burn'),
                    golden_grid = portugal_ggrid,
                    day_interval = 8)


    #download_yearly_lst(2024, portugal_ggrid)
    #get_one_dtm_image(Path('data', 'raw', 'dtm'), portugal_ggrid)
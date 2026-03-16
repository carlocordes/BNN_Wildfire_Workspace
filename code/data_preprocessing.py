# Internal

# External
import math

import numpy as np
from shapely import from_wkb
import ee
import geemap
from pathlib import Path
import time
import datetime
import pandas as pd
import geopandas as gpd
import glob
from shapely.geometry import box
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import duckdb

# ----------- CONFIG ----------- #
PROJECT_ID = 'transformerwildfire'
#ee.Authenticate()
ee.Initialize(project=PROJECT_ID)

db_path = Path('data', 'raw', 'burn', 'burn_points.db')
# ------------------------------ #


class GoldenGrid():
    """
    Defines project-unified grid via CRS, scale (m) and area of interest
    """
    def __init__(self, crs, scale, bbox : list[float],
                start_date : str, end_date : str, day_interval: int):
        self.crs = crs
        self.scale = scale
        self.bbox = bbox
        self.aoi = ee.Geometry.Rectangle(bbox)

        self.target_proj = ee.Projection(crs).atScale(scale)
        
        self.start_date = start_date
        self.end_date = end_date

        self.day_interval = day_interval
        self.dates = pd.date_range(start = start_date,
                              end = end_date,
                              freq = f'{day_interval}D')


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

def ingest_burn_records(path_to_csv : Path) -> None:
    table_name = 'burn_points'
    db_name = 'burn_points.db'
    # DB config
    con = duckdb.connect(path_to_csv / db_name)
    con.execute("INSTALL spatial;"
                "LOAD spatial;")
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    first = True

    # csv file schema
    paths = glob.glob(str(path_to_csv) + '/*.csv')


    for load_path in paths:

        # Load and disregard
        points = gpd.read_file(load_path).\
            drop(columns = ["brightness", "scan", "track", "acq_time","satellite",
                            "instrument", "version", "frp", "daynight", "type"])
        
        # Some data wrangling
        points["acq_date"] = pd.to_datetime(points["acq_date"])
        points["year"] = points["acq_date"].dt.year
        points["month"] = points["acq_date"].dt.month
        points["day_of_year"] = points["acq_date"].dt.dayofyear


        points["geometry"] = gpd.points_from_xy(points['longitude'], points['latitude'])
        points = points.drop(columns = ['latitude', 'longitude'])
        points = gpd.GeoDataFrame(
            points,
            crs="EPSG:4326",
        )

        points["geom_wkb"] = points.geometry.to_wkb()
        points = points.drop(columns = "geometry")


        # Write to db
        view_name = 'gdf_view'
        con.register(view_name, points)
        rel = con.from_df(points)

        if first:
            # Create Mode
            con.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            con.execute(
                f"""
                CREATE TABLE {table_name} AS
                SELECT
                    * EXCLUDE geom_wkb,
                    ST_GeomFromWKB(geom_wkb) AS geom
                FROM {view_name};
                """
            )

            first = False
        else:
            # Append mode
            con.execute(
                f"""
                INSERT INTO {table_name}
                SELECT
                    * EXCLUDE geom_wkb,
                    ST_GeomFromWKB(geom_wkb) AS geom
                FROM {view_name};
                """
            )

        
    count = con.execute(f"""
            SELECT COUNT(*) FROM {table_name}
            """).fetchall()[0][0]
    print(f'Wrote {count} to {table_name}')

def grid_target_map(path_to_db : Path,
                    out_path : Path,
                    golden_grid : GoldenGrid
                    ) -> None:
    table_name = 'burn_points'
    # Produce tifs at day interval
    #dates = pd.date_range(start=f'2024-01-01', end=f'2024-12-31', freq=f'{day_interval}D') # TODO: either add to goldengrid or as function variable, dont hardcode

    con = duckdb.connect(path_to_db)
    con.execute("""
        INSTALL spatial;
        LOAD spatial;
    """)
    
    points = con.execute(f"""
        SELECT ST_AsWKB(geom) AS geom_wkb,
        * EXCLUDE geom
        FROM {table_name}
        WHERE acq_date BETWEEN '{golden_grid.start_date}' AND '{golden_grid.end_date}'
    """).fetch_df()

    points["geometry"] = points["geom_wkb"].apply(lambda g: from_wkb(bytes(g)))

    points = gpd.GeoDataFrame(points, geometry="geometry", crs="EPSG:4326")
    points = points.drop(columns="geom_wkb")

    for start_date in golden_grid.dates:
        end_date = start_date + pd.Timedelta(days = golden_grid.day_interval)

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



if __name__ == '__main__':

    # Define local golden grid
    latmin, longmin = 36.812, -9.490
    latmax, longmax = 42.2724, -6.0234

    start_date = '2024-01-01'
    end_date = '2024-12-31'

    portugal_ggrid = GoldenGrid(
        crs = 'EPSG:3763',
        scale = 1000,
        bbox = [longmin, latmin, longmax, latmax],
        start_date = start_date,
        end_date = end_date,
        day_interval = 7
    )

    #ingest_burn_records(path_to_csv = Path('data', 'raw', 'burn'))

    grid_target_map(path_to_db = db_path,
                    out_path = Path('data', 'processed', 'burn'),
                    golden_grid = portugal_ggrid)

    #get_one_dtm_image(Path('data', 'slope'), portugal_ggrid)
    #download_yearly_lst(2024, portugal_ggrid)
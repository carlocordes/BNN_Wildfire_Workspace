# Internal
from src.core.goldengrid import GoldenGrid

# External
import glob
import numpy as np
from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from shapely.geometry import box
from scipy.ndimage import distance_transform_edt

def produce_roads_distance_image(in_path : Path, out_dir : Path, golden_grid : GoldenGrid, max_distance = 13_000):

    ## Get dimensions from DTM
    with rasterio.open(Path('data', 'raw', 'elevation', 'dtm.tif')) as src:
        raster_height = src.height
        raster_width = src.width
        transform = src.transform
        raster_crs = src.crs


    ## Find input file
    filename = next(in_path.glob("*"), None)

    ## Load all roads
    all_roads = gpd.read_file(filename = filename)

    # bbox coordinates
    minx, miny, maxx, maxy = golden_grid.bbox
    bbox_geom = box(minx, miny, maxx, maxy)

    # Make bbox gdf
    bbox_gdf = gpd.GeoDataFrame(
        geometry=[bbox_geom],
        crs="EPSG:4326"
    )

    # Reproject if necessary
    if bbox_gdf.crs != all_roads.crs:
        bbox_gdf = bbox_gdf.to_crs(all_roads.crs)

    ## Clip and select
    selected_roads = gpd.clip(all_roads, bbox_gdf)
    print(f'Intersected {len(selected_roads)} features')


    # Reproject roads to raster CRS
    selected_roads = selected_roads.to_crs(raster_crs)

    # Optional:
    # buffer roads to avoid disappearance on coarse grid
    selected_roads.geometry = selected_roads.buffer(100)

    # Rasterize roads
    road_mask = rasterize(
        [(geom, 1) for geom in selected_roads.geometry],
        out_shape=(raster_height, raster_width),
        transform=transform,
        fill=0,
        all_touched=True,  # important
        dtype=np.uint8
    )

    distance_pixels = distance_transform_edt(road_mask == 0)
    distance_meters = distance_pixels * golden_grid.scale

    # Cap maximum distance (so values over water do not inflate)
    distance_meters = np.clip(
        distance_meters, 
        a_min = 0,
        a_max = max_distance
    )
    
    # Save output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "road_distance.tif"

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=raster_height,
        width=raster_width,
        count=1,
        dtype=np.float32,
        crs=golden_grid.crs,
        transform=transform,
        compress="lzw"
    ) as dst:
        dst.write(distance_meters.astype(np.float32), 1)

    print(f"Saved: {out_path}")


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
        day_interval = 8
    )

    produce_roads_distance_image(in_path = Path('data', 'raw', 'roads'),
                                 out_dir=Path('data', 'processed', 'roads'),
                                 golden_grid=portugal_ggrid)
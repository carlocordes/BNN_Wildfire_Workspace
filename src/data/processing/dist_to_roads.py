# Internal
from src.core.goldengrid import GoldenGrid

# External
import glob
import numpy as np
from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.transform import Affine
from rasterio.features import rasterize
from shapely.geometry import box
from scipy.ndimage import distance_transform_edt

def produce_roads_distance_image(in_path : Path, out_dir : Path, golden_grid : GoldenGrid, max_distance = 13_000):

    ## Get dimensions from DTM
    with rasterio.open(Path('files', 'data', 'raw', 'elevation', 'dtm.tif')) as src:
        raster_height = src.height
        raster_width = src.width
        transform = src.transform
        raster_crs = src.crs


    ## Find input file
    filename = next(in_path.glob("*"), None)

    ## Load all roads
    all_roads = gpd.read_file(filename = filename)

    # bbox coordinates
    bbox_buffer = 1 # degree buffer to alleviate edge case
    minx, miny, maxx, maxy = golden_grid.bbox
    bbox_geom = box(minx - bbox_buffer, miny - bbox_buffer, maxx + bbox_buffer, maxy + bbox_buffer)

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
    """
    # Cap maximum distance (so values over water do not inflate)
    distance_meters = np.clip(
        distance_meters, 
        a_min = 0,
        a_max = max_distance
    )
    """

    # Load binary water map
    path_to_water_binary = Path('files', 'data', 'resources', 'binary_water_projected.tif')
    with rasterio.open(path_to_water_binary) as src:
        binary_water = src.read(1)

    # Mask ocean
    #distance_meters = np.where(binary_water == 0, distance_meters, 1.0)

    # Create proximity
    land_max = np.max(distance_meters[binary_water == 0])
    land_min = 0.0

    proximity = (land_max - distance_meters) / (land_max - land_min)

    # Force ocean to -1
    proximity_masked = np.where(binary_water == 0, proximity, -1.0)

    # --- FIX: Synchronize Array Flip and Metadata Flip ---
    # 1. Flip the raw pixel array rows vertically
    proximity_flipped = np.flipud(proximity_masked)

    # 2. Invert the Y-axis transform parameters to reflect the physical array flip
    corrected_transform = Affine(
        transform.a, transform.b, transform.c,
        transform.d, -transform.e, transform.f + (raster_height * transform.e)
    )

    # Save output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "proximity.tif"

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=raster_height,
        width=raster_width,
        count=1,
        nodata=-1.0,
        dtype=np.float32,
        crs=golden_grid.crs,
        transform=corrected_transform, # Apply the updated transform
        compress="lzw"
    ) as dst:
        # Write out the vertically flipped array
        dst.write(proximity_flipped.astype(np.float32), 1)

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

    produce_roads_distance_image(in_path = Path('files', 'data', 'raw', 'roads'),
                                 out_dir=Path('files', 'data', 'test', 'roads'),
                                 golden_grid=portugal_ggrid)
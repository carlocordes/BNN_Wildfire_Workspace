# Internal
from src.core.goldengrid import GoldenGrid

# External
import math
import numpy as np
from pathlib import Path
import duckdb
import pandas as pd
import geopandas as gpd
from shapely import from_wkb, box

import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

# PARAMS # TODO: Later replace with config parameters passed to func
target_extent_days = 4
temporal_sigma = 3
spatial_sigma = 3000 # meters


def grid_target_map(path_to_db : Path,
                    out_path : Path,
                    golden_grid : GoldenGrid
                    ) -> None:
    table_name = 'burn_points'

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

        window_start = start_date - pd.Timedelta(days=target_extent_days)
        window_end   = end_date + pd.Timedelta(days=target_extent_days)

        mask = (points['acq_date'] >= window_start) & (points['acq_date'] < window_end)
        subset_gdf = points.loc[mask].copy()

        # Produce temporal weights
        center_date = start_date + pd.Timedelta(days=golden_grid.day_interval / 2)

        subset_gdf["time_diff"] = (subset_gdf["acq_date"] - center_date).dt.days.abs()

        subset_gdf["time_weight"] = np.exp(
            - (subset_gdf["time_diff"] ** 2) / (2 * temporal_sigma ** 2)
        )

        date_str = start_date.strftime('%Y-%m-%d')
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
    heatmap = np.zeros((height, width), dtype="float32")

    points_proj = points.to_crs(golden_grid.crs)

    if not points_proj.empty:

        for _, row in points_proj.iterrows():
            x, y = row.geometry.x, row.geometry.y
            time_weight = row["time_weight"]

            # Convert to pixel coordinates
            col = int((x - minx_snapped) / golden_grid.scale)
            row_pix = int((maxy_snapped - y) / golden_grid.scale)

            radius = int(3 * spatial_sigma / golden_grid.scale)

            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):

                    cx = col + dx
                    cy = row_pix + dy

                    if 0 <= cx < width and 0 <= cy < height:

                        dist = np.sqrt(
                            (dx * golden_grid.scale) ** 2 +
                            (dy * golden_grid.scale) ** 2
                        )

                        spatial_weight = np.exp(
                            - (dist ** 2) / (2 * spatial_sigma ** 2)
                        )

                        contribution = time_weight * spatial_weight

                        # probabilistic accumulation
                        heatmap[cy, cx] = 1 - (1 - heatmap[cy, cx]) * (1 - contribution)


    # Write the file safely
    out_path.mkdir(parents=True, exist_ok=True)
    output_file = out_path / f"{date_str}.tif"

    with rasterio.open(
        output_file,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=golden_grid.crs,
        transform=transform,
        compress="lzw" # Highly recommended: drastically reduces file size for binary data
    ) as dst:
        dst.write(heatmap, 1)
        
    print(f"Rasterized {len(points_proj)} points for {date_str} to {output_file.name}")



if __name__ == '__main__':

    # Define local golden grid
    latmin, longmin = 36.812, -9.490
    latmax, longmax = 42.2724, -6.0234

    start_date = '2024-01-01'
    end_date = '2024-08-31'

    portugal_ggrid = GoldenGrid(
        crs = 'EPSG:3763',
        scale = 1000,
        bbox = [longmin, latmin, longmax, latmax],
        start_date = start_date,
        end_date = end_date,
        day_interval = 8
    )

    #ingest_burn_records(path_to_csv = Path('data', 'raw', 'burn'))

    grid_target_map(path_to_db = Path('data', 'raw', 'burn', 'burn_points.db'),
                    out_path = Path('data', 'test', 'burned_area'),
                    golden_grid = portugal_ggrid)
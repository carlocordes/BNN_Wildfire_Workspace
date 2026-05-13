"""
Produces ground truth data from MODIS burned area product
"""


# Internal
from src.core.goldengrid import GoldenGrid
from src.core.utils import load_config


# External
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import rasterio
from rasterio.transform import from_origin


def produce_tiffs(golden_grid : GoldenGrid, target_extent : int, in_path : Path, out_path : Path):
    print(f'Processing ground truth area from {in_path} to {out_path}')
    # Gather years
    start_year = datetime.strptime(golden_grid.start_date, "%Y-%m-%d").year
    end_year = datetime.strptime(golden_grid.end_date, "%Y-%m-%d").year
    years = list(str(item) for item in range(start_year, end_year + 1))
    
    # Find and load applicable paths
    suffixes = tuple(years)
    file_paths = [
        Path(p.resolve()) for p in in_path.rglob("*")
        if any(s in p.name for s in suffixes)
    ]


    # Load files into data structure
    load_data = {}
    for path in file_paths:
        with rasterio.open(path) as src:
            year = datetime.strptime(path.stem, "%Y").year
            load_data[year] = {
                "array" : src.read(1),
                "profile" : src.profile
            }

    # Get target dates & extent from config
    for start_date in golden_grid.dates:
        first_year = next(iter(load_data))

        combined_mask = np.zeros_like(load_data[first_year]["array"], dtype=np.uint8)

        for i in range(target_extent + 1):
            current_day = start_date + timedelta(days = i)
            current_year = current_day.year
            current_doy = current_day.timetuple().tm_yday

            if current_year in load_data:
                day_mask = (load_data[current_year]["array"] == current_doy).astype(np.uint8)
                combined_mask = np.maximum(combined_mask, day_mask)

        out_profile = load_data[first_year]["profile"].copy()
        

        out_profile.update({
            "dtype": "uint8",
            "count": 1,
            "nodata": 0  # Standard for binary masks
        })
        # Write
        output_file = out_path / f"{start_date.strftime('%Y-%m-%d')}.tif"
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        out_profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "crs": golden_grid.crs, # 'EPSG:3763'
            "width": src.width,
            "height": src.height,
            "transform": src.transform, # Use the transform that matches the data
            #"nodata": 0,
            "compress": "lzw"       # Good practice for binary masks
        }

        with rasterio.open(output_file, "w", **out_profile) as dst:
            dst.write(combined_mask, 1)

        print(f"Created: {output_file.name}")




if __name__ == '__main__':
    out_path = Path('data', 'processed', 'target_area')
    in_path = Path('data', 'raw', 'burned_area')

    # Define local golden grid
    latmin, longmin = 36.812, -9.490
    latmax, longmax = 42.2724, -6.0234

    start_date = '2023-06-01'
    end_date = '2023-12-01'

    portugal_ggrid = GoldenGrid(
        crs = 'EPSG:3763',
        scale = 1000,
        bbox = [longmin, latmin, longmax, latmax],
        start_date = start_date,
        end_date = end_date,
        day_interval = 1
    )
    
    target_extent = 4

    produce_tiffs(
        golden_grid=portugal_ggrid,
        target_extent = target_extent,
        in_path=in_path,
        out_path=out_path
    )
# Internal
from src.core.goldengrid import GoldenGrid

# External
from pathlib import Path
import ee
import geemap

def download_ecostress_lst(date,
                           golden_grid,
                           lst_dir: Path):
    """
    Downloads ECOSTRESS Land Surface Temperature (LST) for a specific date.
    """
    lst_dir.mkdir(parents=True, exist_ok=True)

    start = ee.Date(date)
    end = start.advance(1, "day")
    
    # ECOSTRESS Collection 1 LST
    # 'LST' is the Land Surface Temperature band in Kelvin
    daily_coll = (
        ee.ImageCollection("NASA/ECOSTRESS/LSTE/001")
        .filterDate(start, end)
        .filterBounds(golden_grid.aoi)
        .select(["LST"])
    )

    # Check if any granules exist for this date/location
    count = daily_coll.size().getInfo()
    if count == 0:
        print(f"No ECOSTRESS data found for {date} in the specified AOI.")
        return

    # Since ECOSTRESS has irregular revisit times, we take the mean 
    # of any granules that overlapped the AOI on that calendar day.
    image = daily_coll.mean().rename('lst_kelvin')

    export_params = {
        'scale': golden_grid.scale,
        'region': golden_grid.aoi,
        'crs': golden_grid.crs,
        'file_per_band': False
    }

    fname = f"{date}.tif"

    # Export LST
    geemap.ee_export_image(
        image, 
        filename=str(lst_dir / fname), 
        **export_params
    )

    print(f"Finished downloading ECOSTRESS LST for {date} ({count} scenes found)")


def download_ecostress_catalogue(golden_grid: GoldenGrid,
                                 lst_dir: Path):
    date_strings = golden_grid.dates.strftime('%Y-%m-%d').tolist()
    for date in date_strings:
        download_ecostress_lst(
            date=date,
            golden_grid=golden_grid,
            lst_dir=lst_dir
        )

if __name__ == '__main__':
    # Initialize Earth Engine (Required if not already done)
    project_id = 'transformerwildfire'
    ee.Initialize(project = project_id)

    # Define local golden grid
    latmin, longmin = 36.812, -9.490
    latmax, longmax = 42.2724, -6.0234

    start_date = '2024-08-01'
    end_date = '2024-08-30'

    portugal_ggrid = GoldenGrid(
        crs='EPSG:3763',
        scale=1000,  # ECOSTRESS native resolution is ~70m, though 1000m works for consistency
        bbox=[longmin, latmin, longmax, latmax],
        start_date=start_date,
        end_date=end_date,
        day_interval=7
    )

    lst_output_path = Path('data', 'processed', 'ecostress_lst')
    
    download_ecostress_catalogue(
        golden_grid=portugal_ggrid,
        lst_dir=lst_output_path
    )
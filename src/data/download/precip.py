# Internal
from src.core.goldengrid import GoldenGrid

# External
from pathlib import Path
import geemap
import ee

def download_one_days_since_rain_image(
    date: str,
    golden_grid: GoldenGrid,
    out_path: Path
):
    
    days_window = 30

    fname = date + '.tif'
    file_path = out_path / fname

    if not file_path.exists():
        print(f'Gathering precipitation values for {date}')

        # Define window
        target_date = ee.Date(date)
        start_date = target_date.advance(-days_window, 'day')

        # Load filtered band
        precip_col = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
            .filterDate(start_date, target_date) \
            .select('precipitation')
        

        # Sum images over time axis
        total_precip_image = precip_col.sum()

        # Download
        try:
            geemap.ee_export_image(
                ee_object=total_precip_image,
                filename=str(file_path),
                scale=golden_grid.scale,
                crs=golden_grid.crs,
                region=golden_grid.aoi,
                file_per_band=False
            )
            print(f"Successfully downloaded: {file_path}")
        except Exception as e:
            print(f"Failed to download {date}. Error: {e}")

    else:
        print(f"File already exists: {file_path}")
        
def download_rain_catalogue(out_path: Path, golden_grid: GoldenGrid):
    date_strings = golden_grid.dates.strftime('%Y-%m-%d').tolist()
    for date in date_strings:
        download_one_days_since_rain_image(
            date=date,
            golden_grid=golden_grid,
            out_path=out_path
        )

if __name__ == '__main__':
    project_id = 'transformerwildfire'
    ee.Initialize(project = project_id)

    latmin, longmin = 36.812, -9.490
    latmax, longmax = 42.2724, -6.0234

    start_date = '2024-08-01'
    end_date = '2024-08-10'

    portugal_ggrid = GoldenGrid(
        crs = 'EPSG:3763',
        scale = 1000,
        bbox = [longmin, latmin, longmax, latmax],
        start_date = start_date,
        end_date = end_date,
        day_interval = 1
    )

    download_rain_catalogue(
        golden_grid=portugal_ggrid,
        out_path=Path('data', 'processed', 'precip')
    )
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
    fname = date + '.tif'
    file_path = out_path / fname

    if not file_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)

        target_date = ee.Date(date)
        start_lookback = target_date.advance(-60, "day")

        chirps = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterDate(start_lookback, target_date)
            .filterBounds(golden_grid.aoi)
            .select('precipitation')
        )

        if chirps.size().getInfo() == 0:
            print(f"No rainfall data found for {date}")
            return

        rain_threshold = 1.0

        def calculate_days_since(img):
            img_date = ee.Date(img.get('system:time_start'))
            days_diff = target_date.difference(img_date, 'days')
            is_rain = img.gte(rain_threshold)
            return ee.Image(days_diff).updateMask(is_rain)

        rain_days_collection = chirps.map(calculate_days_since)
        
        # Reduce to find the minimum days since rain
        days_since_rain = rain_days_collection.reduce(ee.Reducer.min())

        # 1. Fill unmasked pixels with 60 (max lookback)
        # 2. Force cast to Int32 so the exporter handles the data type properly
        # 3. Explicitly set the output CRS and nominal scale to prevent export alignment errors
        export_image = (
            days_since_rain.unmask(60)
            .toInt32()
            .rename('days_since_rain')
            .setDefaultProjection(crs=golden_grid.crs, scale=golden_grid.scale)
            .clip(golden_grid.aoi)
        )

        print(f"Downloading days since rain for {date} to {out_path}...")
        
        try:
            geemap.ee_export_image(
                export_image,
                filename=str(file_path),
                scale=golden_grid.scale,
                region=golden_grid.aoi,
                crs=golden_grid.crs,
                file_per_band=False
            )
        except Exception as e:
            print(f"Failed downloading {date}: {e}")
            if file_path.exists():
                file_path.unlink()  # Clean up partial/corrupted files
    else:
        print('File already exists. Skipping.')


        
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

    download_one_days_since_rain_image(date = '2024-08-01',
                             golden_grid=portugal_ggrid,
                             out_path=Path('data', 'processed', 'precip'))
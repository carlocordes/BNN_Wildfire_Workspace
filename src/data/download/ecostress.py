# Internal
from src.core.goldengrid import GoldenGrid

# External
from pathlib import Path
import geemap
import ee
def download_one_ecostress_image(date: str, golden_grid: GoldenGrid, out_path: Path):
    out_path.mkdir(parents=True, exist_ok=True)

    start = ee.Date(date)
    end = start.advance(1, "day")

    # Use Version 1, which is the stable public asset
    collection = (
        ee.ImageCollection("NASA/ECOSTRESS/LSTE_001")
        .filterDate(start, end)
        .filterBounds(golden_grid.aoi)
    )

    # Check if any images exist
    count = collection.size().getInfo()
    if count == 0:
        print(f"No ECOSTRESS image found for {date}")
        return
    
    # Select LST and multiply by scale factor (0.02) to get Kelvin
    # ECOSTRESS V1 stores LST as uint16 to save space.
    image = collection.mosaic().select('LST')
    
    # Scale to Kelvin: LST_real = LST_dn * 0.02
    lst_kelvin = image.multiply(0.02).updateMask(image.gt(0))

    print(f"Downloading ECOSTRESS LST for {date}...")
    
    fname = f"{date}_ecostress.tif"
    file_path = out_path / fname

    geemap.ee_export_image(
        lst_kelvin,
        filename=str(file_path),
        scale=70, 
        region=golden_grid.aoi,
        crs=golden_grid.crs,
        file_per_band=False
    )

    
def download_ecostress_catalogue(out_path: Path, golden_grid: GoldenGrid):
    date_strings = golden_grid.dates.strftime('%Y-%m-%d').tolist()

    for date in date_strings:
        download_one_ecostress_image(
            date=date,
            golden_grid=golden_grid,
            out_path=out_path
        )

if __name__ == '__main__':
    project_id = 'transformerwildfire'
    ee.Initialize(project=project_id)

    # Portugal AOI coordinates (as per your snippet)
    latmin, longmin = 36.812, -9.490
    latmax, longmax = 42.2724, -6.0234

    portugal_ggrid = GoldenGrid(
        crs='EPSG:3763',
        scale=70,  # Adjusted to match ECOSTRESS resolution
        bbox=[longmin, latmin, longmax, latmax],
        start_date='2024-01-01',
        end_date='2024-01-05',
        day_interval=1
    )

    eco_path = Path('data', 'processed', 'ECOSTRESS')
    download_ecostress_catalogue(eco_path, portugal_ggrid)
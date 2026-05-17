# Internal
from src.core.goldengrid import GoldenGrid

# External
from pathlib import Path
import geemap
import ee


def download_one_modis_ndwi_image(
    date: str,
    golden_grid: GoldenGrid,
    out_path: Path
):
    
    fname = date + '.tif'
    file_path = out_path / fname

    if not file_path.exists():

        out_path.mkdir(parents=True, exist_ok=True)

        start = ee.Date(date)
        end = start.advance(1, "day")

        modis_collection = (
            ee.ImageCollection("MODIS/061/MOD09GA")
            .filterDate(start, end)
            .filterBounds(golden_grid.aoi)
        )

        if modis_collection.size().getInfo() == 0:
            print(f"No MODIS image found for {date}")
            return

        image = modis_collection.first()

        # NDWI (vegetation moisture version)
        ndwi = image.normalizedDifference(
            ['sur_refl_b02', 'sur_refl_b06']
        ).rename('NDWI')

        # Cloud masking using state QA
        state_qa = image.select('state_1km')

        # Bits 0–1 = cloud state (0 = clear)
        cloud_mask = state_qa.bitwiseAnd(3).eq(0)

        ndwi = ndwi.updateMask(cloud_mask)

        print(f"Downloading NDWI for {date} to {out_path}...")

    else:
        print('File already exists. Skipping.')

    geemap.ee_export_image(
        ndwi,
        filename=str(file_path),
        scale=golden_grid.scale,
        region=golden_grid.aoi,
        crs=golden_grid.crs,
        file_per_band=False
    )


def download_ndwi_catalogue(out_path: Path, golden_grid: GoldenGrid):

    # Get strings
    date_strings = golden_grid.dates.strftime(
        '%Y-%m-%d'
    ).tolist()

    # Iterate over dates
    for date in date_strings:
        download_one_modis_ndwi_image(
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

    
    ndwi_path = Path('data', 'processed', 'NDWI')
    download_ndwi_catalogue(ndwi_path, portugal_ggrid)
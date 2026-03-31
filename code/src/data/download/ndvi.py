# Internal
from src.core.goldengrid import GoldenGrid

# External
from pathlib import Path
import geemap
import ee

def download_one_modis_ndvi_image(date : str, golden_grid :GoldenGrid, out_path : Path):
    start = ee.Date(date)
    end = start.advance(1, "day")

    modis_collection = (
            ee.ImageCollection("MODIS/061/MOD09GQ")
            .filterDate(start, end)
            .filterBounds(golden_grid.aoi)
        )

    if modis_collection.size().getInfo() == 0:
        print(f"No MODIS image found for {date}")
        return
    
    image = modis_collection.first()


    ndvi = image.normalizedDifference(['sur_refl_b02', 'sur_refl_b01']).rename('NDVI')

    # 5. Optional: Apply Cloud Masking
    # We need to pull the 'state_1km' band from the companion collection (MOD09GA) 
    # to mask clouds, as the 250m product doesn't include QA bands.
    qa_image = ee.ImageCollection("MODIS/061/MOD09GA") \
                .filterDate(start, end) \
                .filterBounds(golden_grid.aoi) \
                .first()
    
    if qa_image:
        state_qa = qa_image.select('state_1km')
        # Bits 0-1 are cloud state (0 = clear)
        cloud_mask = state_qa.bitwiseAnd(3).eq(0)
        ndvi = ndvi.updateMask(cloud_mask)

    # 5. Export locally using geemap
    print(f"Downloading NDVI for {date} to {out_path}...")
    
    fname = date + '.tif'
    file_path = out_path / fname

    geemap.ee_export_image(
        ndvi,
        filename=str(file_path),
        scale=golden_grid.scale,
        region=golden_grid.aoi,
        crs=golden_grid.crs,
        file_per_band=False
    )


def download_ndvi_catalogue(out_path : Path, golden_grid : GoldenGrid):

    date_strings = golden_grid.dates.strftime('%Y-%m-%d').tolist()

    for date in date_strings:
        download_one_modis_ndvi_image(
            date = date,
            golden_grid = golden_grid,
            out_path = out_path
        )

if __name__ == '__main__':

    project_id = 'transformerwildfire'
    ee.Initialize(project = project_id)

    latmin, longmin = 36.812, -9.490
    latmax, longmax = 42.2724, -6.0234

    start_date = '2024-01-01'
    end_date = '2024-01-02'

    portugal_ggrid = GoldenGrid(
        crs = 'EPSG:3763',
        scale = 1000,
        bbox = [longmin, latmin, longmax, latmax],
        start_date = start_date,
        end_date = end_date,
        day_interval = 1
    )
    """
    download_modis_ndvi_image('2024-08-08',
                              portugal_ggrid,
                              Path('data', 'processed', 'NDVI'))
                              """
    ndvi_path = Path('data', 'processed', 'NDVI')
    download_ndvi_catalogue(ndvi_path, portugal_ggrid)
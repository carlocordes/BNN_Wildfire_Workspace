# Internal
from src.core.goldengrid import GoldenGrid

# External
import ee
import geemap
from pathlib import Path

def get_one_dtm_image(dtm_dir : Path, golden_grid : GoldenGrid) -> None:

    dtm = ee.Image("CGIAR/SRTM90_V4").select('elevation')

    dtm_aligned = (
        dtm
        .reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=2048
        )
        .reproject(crs=golden_grid.target_proj)
    )

    dtm_dir.mkdir(parents=True, exist_ok=True) # Assert that directory exists

    output_path = str(dtm_dir / 'dtm_aligned.tif')
    print(f"Exporting aligned DTM to {output_path}")

    geemap.ee_export_image(
        dtm_aligned,
        filename=output_path,
        scale=golden_grid.scale,
        region=golden_grid.aoi,
        crs=golden_grid.crs,
        file_per_band=False
    )

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
        day_interval = 7
    )

    get_one_dtm_image(Path('data', 'slope'), portugal_ggrid)
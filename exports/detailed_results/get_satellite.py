# Internal
from src.core.goldengrid import GoldenGrid

# External
import ee
import geemap
from pathlib import Path

def get_single_date_satellite_image(sat_dir: Path, golden_grid: GoldenGrid, target_date: str) -> None:
    """
    Retrieves the sharpest, least-cloudy single Sentinel-2 image asset matching
    the targeted date slice and saves it locally at a safe, high-resolution 20m scale.
    """
    # 1. Access the Harmonized Sentinel-2 Surface Reflectance collection
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    
    # Create a tight window around the target date to ensure we capture a pass
    target = ee.Date(target_date)
    start_window = target.advance(-2, 'day')
    end_window = target.advance(3, 'day')
    
    # 2. Filter collection by bounding box, date window, and sort by cloud density
    aoi = golden_grid.aoi
    s2_selected = (
        s2.filterBounds(aoi)
          .filterDate(start_window, end_window)
          .sort('CLOUDY_PIXEL_PERCENTAGE') # Puts the clearest image at the front
    )
    
    # 3. Extract the top clear image and isolate the RGB channels
    clear_image = s2_selected.select(['B4', 'B3', 'B2']).mosaic()
    
    # 4. Reproject to your GoldenGrid coordinate system
    # Downscaling to 20m prevents the "Pixel grid dimensions exceed 32768" error
    export_scale = 220 
    sat_aligned = clear_image.reproject(
        crs=golden_grid.target_proj,
        scale=export_scale
    )

    sat_dir.mkdir(parents=True, exist_ok=True)
    output_path = sat_dir / f'sentinel2_{target_date}.tif'

    print(f"Downloading Sentinel-2 image for {target_date} to local path: {output_path}...")
    
    # 5. Export structural grid map directly to your local storage
    geemap.ee_export_image(
        sat_aligned,
        filename=str(output_path),
        scale=export_scale, 
        region=aoi,
        file_per_band=False
    )
    print("Local download complete and saved successfully!")

if __name__ == '__main__':
    latmin, longmin = 36.812, -9.490
    latmax, longmax = 42.2724, -6.0234

    # Keep your grid setup identical
    portugal_ggrid = GoldenGrid(
        crs='EPSG:3763',
        scale=1000,
        bbox=[longmin, latmin, longmax, latmax],
        start_date='2024-01-01',
        end_date='2024-12-31',
        day_interval=8
    )

    # Fetch and save the image locally targeting September 2nd, 2024
    get_single_date_satellite_image(
        sat_dir=Path('exports', 'detailed_results', 'satellite'), 
        golden_grid=portugal_ggrid, 
        target_date='2025-08-16'
    )
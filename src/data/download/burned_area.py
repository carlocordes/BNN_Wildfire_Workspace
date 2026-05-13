# Internal
from src.core.goldengrid import GoldenGrid

# External
from pathlib import Path
import ee
import requests
import mimetypes
from datetime import datetime

def download_annual_images(golden_grid: GoldenGrid, out_path: Path):
    """
    Iterates through each calendar year in the GoldenGrid date range
    and saves a separate TIFF for each.
    """
    # 1. Parse dates and identify the range of years
    start_year = datetime.strptime(golden_grid.start_date, '%Y-%m-%d').year
    end_year = datetime.strptime(golden_grid.end_date, '%Y-%m-%d').year
    
    out_path.mkdir(parents=True, exist_ok=True)

    for year in range(start_year, end_year + 1):
        print(f"--- Processing Year: {year} ---")
        
        # Define annual bounds
        year_start = f"{year}-01-01"
        year_end = f"{year}-12-31"
        
        # Filter for the specific year
        dataset = ee.ImageCollection('MODIS/061/MCD64A1') \
            .filterDate(year_start, year_end) \
            .filterBounds(golden_grid.aoi) \
            .select('BurnDate')

        # Get the maximum burn date per pixel for that year
        burned_img = dataset.max().unmask(0)

        processed_img = burned_img.reproject(
            crs=golden_grid.crs, 
            scale=golden_grid.scale
        ).clip(golden_grid.aoi)

        # 2. Generate the Download URL
        try:
            url = processed_img.getDownloadURL({
                'name': f'burned_area_{year}',
                'scale': golden_grid.scale,
                'crs': golden_grid.crs,
                'region': golden_grid.aoi,
                'format': 'GEO_TIFF'
            })
            
            response = requests.get(url, stream=True)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type')
                extension = mimetypes.guess_extension(content_type) or '.zip'
                
                if "text/html" in content_type:
                    print(f"Warning: EEE returned an error page for year {year}.")
                    continue

                target_file = out_path / f"{year}{extension}"
                
                with open(target_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"Successfully saved: {target_file}")
            else:
                print(f"Error {response.status_code} for year {year}: {response.text}")

        except Exception as e:
            print(f"Failed to process year {year}: {e}")

if __name__ == '__main__':
    # Initialize with your project ID
    project_id = 'transformerwildfire'
    ee.Initialize(project=project_id)

    # Portugal Bounding Box
    latmin, longmin = 36.812, -9.490
    latmax, longmax = 42.2724, -6.0234

    start_date = '2015-01-01'
    end_date = '2025-12-31'

    portugal_ggrid = GoldenGrid(
        crs = 'EPSG:3763',
        scale = 1000,
        bbox = [longmin, latmin, longmax, latmax],
        start_date = start_date,
        end_date = end_date,
        day_interval = 7
    )

    download_annual_images(
        golden_grid = portugal_ggrid, 
        out_path = Path('data', 'raw', 'burned_area')
    )
# Internal
from src.core.goldengrid import GoldenGrid

# External
from pathlib import Path
import ee
import requests
import mimetypes

def download_image(golden_grid: GoldenGrid, out_path: Path):
    # 1. Define Time Range and Collection
    start_date = ee.Date(golden_grid.start_date)
    end_date = ee.Date(golden_grid.end_date)
    
    dataset = ee.ImageCollection('MODIS/061/MCD64A1') \
        .filterDate(start_date, end_date) \
        .filterBounds(golden_grid.aoi) \
        .select('BurnDate')

    burned_img = dataset.max().unmask(0)

    processed_img = burned_img.reproject(
        crs=golden_grid.crs, 
        scale=golden_grid.scale
    ).clip(golden_grid.aoi)

    # 2. Generate the Download URL
    url = processed_img.getDownloadURL({
        'name': f'burned_area_{golden_grid.start_date}',
        'scale': golden_grid.scale,
        'crs': golden_grid.crs,
        'region': golden_grid.aoi,
        'format': 'GEO_TIFF'
    })

    # 3. Request and identify extension
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Requesting data from Google Earth Engine...")
    
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        # Determine the extension based on the 'Content-Type' header
        content_type = response.headers.get('content-type')
        extension = mimetypes.guess_extension(content_type) or '.zip'
        
        # If Earth Engine returns an error in a successful-looking stream, 
        # it's usually text/html. Let's catch that.
        if "text/html" in content_type:
            print("Warning: Earth Engine returned an HTML error page instead of a file.")
            extension = ".html"

        target_file = out_path / f"mcd64a1_{golden_grid.start_date}{extension}"
        
        with open(target_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Saved result to: {target_file} (Type: {content_type})")
    else:
        print(f"Error: Request failed with status {response.status_code}")
        print(response.text)

if __name__ == '__main__':
    # Initialize with your project ID
    project_id = 'transformerwildfire'
    ee.Initialize(project=project_id)

    # Portugal Bounding Box
    latmin, longmin = 36.812, -9.490
    latmax, longmax = 42.2724, -6.0234

    start_date = '2023-01-01'
    end_date = '2023-12-31'

    portugal_ggrid = GoldenGrid(
        crs = 'EPSG:3763',
        scale = 500,
        bbox = [longmin, latmin, longmax, latmax],
        start_date = start_date,
        end_date = end_date,
        day_interval = 7
    )

    download_image(
        golden_grid = portugal_ggrid, 
        out_path = Path('data', 'test', 'burned_area')
    )
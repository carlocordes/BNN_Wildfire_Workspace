# Internal

# External
import numpy as np
from pathlib import Path
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def produce_mask_from_tif(in_path : Path, out_path : Path):

    # Read EU water info file
    with rasterio.open(in_path) as src:
        data = src.read(1, masked=True)
        transform = src.transform
        profile = src.profile
        crs = src.crs


    # Create binary mask
    binary_mask = np.zeros_like(data, dtype = np.uint8)
    binary_mask[(data == 253) | (data ==255)] = 1


    # Update profile with new data type
    profile.update({
            'dtype': 'uint8',       # Smaller data type since values are just 0 and 1
            'count': 1,             # Single band output
            'nodata': None,         # Clear or set a nodata value if needed (e.g., 255 if 1 isn't max)
            'compress': 'lzw'       # Optional: Add compression to save disk space
        })
    
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(binary_mask, 1)
        
    print(f"Binary mask successfully saved to {out_path}")

def produce_local_binary(in_path: Path, out_path: Path):
    # 1. Read the target dimensions from the DTM
    dtm_path = Path('data', 'raw', 'elevation', 'dtm.tif')
    with rasterio.open(dtm_path) as dtm_src:
        raster_height = dtm_src.height
        raster_width = dtm_src.width
        raster_transform = dtm_src.transform
        raster_crs = dtm_src.crs
        dtm_profile = dtm_src.profile.copy()
        
    # 2. Read the input data
    with rasterio.open(in_path) as src:
        # Read band 1 as a standard array, filling nodata values with 0
        # This completely avoids the MaskedArray bug
        in_data = src.read(1)
        if src.nodata is not None:
            # If there's an existing nodata value, convert those pixels to 0 for your binary mask
            in_data = np.where(in_data == src.nodata, 0, in_data)
            
        in_transform = src.transform
        in_crs = src.crs

    # 3. Create a clean 2D destination array (Standard NumPy array)
    destination_data = np.zeros((raster_height, raster_width), dtype=in_data.dtype)

    # 4. Perform the reprojection (Both inputs are now pure 2D standard NumPy arrays)
    reproject(
        source=in_data,
        destination=destination_data,
        src_transform=in_transform,
        src_crs=in_crs,
        dst_transform=raster_transform,
        dst_crs=raster_crs,
        resampling=Resampling.nearest
    )

    # 5. Update the profile for the output file
    dtm_profile.update({
        'dtype': destination_data.dtype,
        'count': 1,
        'nodata': 255  
    })

    # 6. Write the aligned raster
    with rasterio.open(out_path, 'w', **dtm_profile) as dst:
        dst.write(destination_data, 1)


if __name__ == '__main__':
    dir = Path('data', 'resources') 
    file = '100m_wetness_classifier.tif'
    in_path = dir / file


    binary_water = dir / 'binary_water.tif'
    binary_water_projected = dir / 'binary_water_projected.tif'

    produce_local_binary(in_path = binary_water, out_path = binary_water_projected)
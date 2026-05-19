# Internal
from src.core.goldengrid import GoldenGrid

# External
import numpy as np
from pathlib import Path
import rasterio
from rasterio import warp

def get_range_from_dataset(path_to_ds : Path, mask = True):
    print(f'Gathering max/min data for dataset {path_to_ds.stem}')

    ds_min = np.inf
    ds_max = -np.inf

    # Loop over data (ignoring sidecar files)
    file_paths = sorted(f for f in path_to_ds.glob("*") if f.suffix.lower() != ".xml")

    for file_path in file_paths:
        with rasterio.open(file_path) as src:
            data = src.read(1, masked=True)

        # Use np.ma to strictly respect the rasterio mask
        img_min = np.ma.min(data)
        img_max = np.ma.max(data)

        if img_min < ds_min:
            ds_min = img_min
        if img_max > ds_max:
            ds_max = img_max

    print(f'Min for dataset {path_to_ds}: {ds_min}')
    print(f'Max for dataset {path_to_ds}: {ds_max}')

    return ds_max, ds_min


def normalize_dataset(input_dataset : Path, output_dataset : Path, mask = True):
    if not output_dataset.exists():
        output_dataset.mkdir(parents = True, exist_ok = False)
        print(f'Creating output dataset at {output_dataset}')

    # Get range of dataset
    ds_max, ds_min = get_range_from_dataset(path_to_ds=input_dataset, mask=True)

    # All files in dataset excl. helper files
    file_paths = sorted(f for f in input_dataset.glob("*") if f.suffix.lower() != ".xml")
    
    for file_path in file_paths:
        with rasterio.open(file_path) as src:
            data = src.read(1, masked=True)
            data_profile = src.profile

        nodata = data_profile['nodata']


        # Normalize
        normalized_data = (data - ds_min) / (ds_max - ds_min)

        if mask == True:
            print('Masking with binary water mask...')
            path_to_water_binary = Path('files', 'data', 'resources', 'binary_water_projected.tif')

            with rasterio.open(path_to_water_binary) as water_src:
                aligned_water = np.zeros(data.shape, dtype=water_src.dtypes[0])
                
                # Dynamically reproject/align the water mask to match the NDVI profile
                rasterio.warp.reproject(
                    source=rasterio.band(water_src, 1),
                    destination=aligned_water,
                    src_transform=water_src.transform,
                    src_crs=water_src.crs,
                    dst_transform=data_profile['transform'],
                    dst_crs=data_profile['crs'],
                    resampling=warp.Resampling.nearest # Use nearest for binary masks
                )

            # Use the aligned array for your mask logic
            water_mask = (aligned_water == 1)


            # Update the existing mask of our normalized data array
            normalized_data.mask = np.ma.mask_or(normalized_data.mask, water_mask)


        new_nodata_value = -1.0
        final_array = normalized_data.filled(new_nodata_value)


        # Set nodata values in profile
        profile = data_profile.copy()
        profile.update({
            'dtype': 'float32',       # Ensure float type for 0-1 values
            'nodata': new_nodata_value # Crucial: update metadata to -1
        })



        filename = file_path.stem
        out_path = output_dataset / f'{filename}.tif'

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(final_array.astype(np.float32), 1)
            
        print(f'Wrote normalized file from {input_dataset} to {out_path}')


if __name__ == '__main__':
    input_dataset = Path('files', 'data', 'raw', 'LST')
    output_dataset = Path('files', 'data', 'processed', 'LST')

    normalize_dataset(input_dataset=input_dataset, output_dataset=output_dataset, mask = True)
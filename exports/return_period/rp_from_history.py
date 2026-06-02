from pathlib import Path
import numpy as np
import rasterio


def calculate_local_frp(input_dir: str, output_path: str):
    """Calculates Fire Return Period from a directory of MODIS MCD64A1 GeoTIFFs.

    Args:
        input_dir: Path to folder containing the .tif files
        output_path: Path where the resulting FRP GeoTIFF will be saved
    """
    input_path = Path(input_dir)
    # Find all .tif files in the directory
    tif_files = sorted(list(input_path.glob("*.tiff")))

    if not tif_files:
        raise FileNotFoundError(
            f"No .tiff files found in directory: {input_dir}"
        )

    print(f"Found {len(tif_files)} files. Grouping by year...")

    # 1. Parse files and group by year to isolate annual burn footprints
    # Assumes filenames contain a 4-digit year (e.g., 'MCD64A1.A2021001...')
    years_dict = {}
    for f in tif_files:
        # Simple extraction of 4-digit year from filename
        # Adjust the string parsing if your naming convention is different
        year_parts = [p for p in f.name.replace("_", ".").split(".") if p.isdigit() and len(p) == 4]
        if not year_parts:
            # Fallback if parsing fails: look for any 4 digit sequence
            import re
            match = re.search(r"\b(20\d{2}|19\d{2})\b", f.name)
            year = match.group(0) if match else None
        else:
            year = year_parts[0]

        if year:
            years_dict.setdefault(year, []).append(f)

    unique_years = sorted(list(years_dict.keys()))
    total_years = len(unique_years)
    print(f"Processing {total_years} unique years: {unique_years}")

    # 2. Open the first image to establish spatial profile (metadata)
    with rasterio.open(tif_files[0]) as src:
        meta = src.meta.copy()
        height, width = src.height, src.width
        nodata_val = src.nodata

    # Initialize an array to store the cumulative count of fire years per pixel
    accumulated_burn_years = np.zeros((height, width), dtype=np.float32)

    # 3. Process year-by-year
    for year, files in years_dict.items():
        print(f"Processing year: {year}")
        # Initialize a blank canvas for the current year
        annual_burn_mask = np.zeros((height, width), dtype=np.uint8)

        for f in files:
            with rasterio.open(f) as src:
                data = src.read(1)

                # MCD64A1 standard Julian days are > 0.
                # Exclude standard invalid/missing data fills if applicable (e.g. values like -2)
                valid_burn = (data > 0) & (data != nodata_val)

                # If a pixel burned in *any* month of this year, mark it as 1
                annual_burn_mask = np.bitwise_or(annual_burn_mask, valid_burn.astype(np.uint8))

        # Add this year's footprint to the multi-year tally
        accumulated_burn_years += annual_burn_mask

    # 4. Calculate Fire Return Period (Total Years / Burn Tally)
    # Using np.where to prevent division-by-zero errors for unburned pixels
    frp = np.where(
        accumulated_burn_years > 0,
        total_years / accumulated_burn_years,
        np.nan,  # Mark unburned pixels as NaN
    )

    # 5. Export the final map
    meta.update(
        {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "nodata": np.nan,  # Unburned or background pixels will be transparent/masked
        }
    )

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(frp.astype(np.float32), 1)

    print(f"Successfully generated Fire Return Period map at: {output_path}")

if __name__ == '__main__':

    in_path = Path('files', 'data', 'test', 'yearly_burn')
    print(in_path.exists())
    out_path = in_path / 'fire_return_pd_10.tif'
    calculate_local_frp(in_path, out_path)
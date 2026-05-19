# Internal
from src.core.goldengrid import GoldenGrid

# External
import glob
import numpy as np
import rasterio
from pathlib import Path


def dtm_derivatives(
    golden_grid: GoldenGrid,
    path_to_dtm: Path,
    path_to_aspect: Path,
    path_to_slope: Path,
):
    """
    Loads DTM and generates slope + aspect maps.
    Aspect is stored as two separate rasters:
    - aspect_NS.tif
    - aspect_EW.tif
    """
    dtm_file = next(path_to_dtm.glob('*.tif'))
    with rasterio.open(dtm_file) as src:
        dtm = src.read(1, masked=True)
        transform = src.transform
        profile = src.profile
        crs = src.crs

    # Basic validation
    if crs.to_string() != golden_grid.crs:
        raise ValueError(f"CRS mismatch: {crs} vs {golden_grid.crs}")

    # Compute and write outputs
    slope_map(dtm, transform, profile, path_to_slope)
    aspect_map(dtm, transform, profile, path_to_aspect)


def slope_map(dtm, transform, profile, path_out: Path):
    """
    Computes slope and writes it as a single-band raster.
    Output is normalized to [0, 1].
    """

    dx = transform.a
    dy = -transform.e

    dz_dy, dz_dx = np.gradient(dtm, dy, dx)

    # Slope in radians
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))

    # Normalize to [0,1]
    slope = slope / np.max(slope)

    ## Mask with ocean
    # Load binary water map
    path_to_water_binary = Path('data', 'resources', 'binary_water_projected.tif')
    with rasterio.open(path_to_water_binary) as src:
        binary_water = src.read(1)

    slope = np.where(binary_water == 0, slope, -1.0)

    profile = profile.copy()
    profile.update(dtype="float32", count=1, compress="lzw", nodata = -1.0)

    # Ensure output directory exists
    path_out.mkdir(parents=True, exist_ok=True)

    slope_file = path_out / "slope.tif"

    with rasterio.open(slope_file, "w", **profile) as dst:
        dst.write(slope.astype(np.float32), 1)


def aspect_map(dtm, transform, profile, path_out: Path):
    """
    Computes aspect and stores it as two separate rasters:
    - aspect_NS.tif
    - aspect_EW.tif
    """

    dx = transform.a
    dy = -transform.e

    dz_dy, dz_dx = np.gradient(dtm, dy, dx)

    # Aspect in radians
    aspect = np.arctan2(-dz_dx, dz_dy)
    aspect = np.mod(aspect, 2 * np.pi)

    # Convert to ML-friendly components
    aspect_ns = np.sin(aspect)
    aspect_ew = np.cos(aspect)


    ## Masking

    max_val = np.max(aspect_ns)
    min_val = np.min(aspect_ew)

    aspect_ns_norm = (max_val - aspect_ns) / (max_val - min_val)
    aspect_ew_norm = (max_val - aspect_ew) / (max_val - min_val)

    # Load binary water map
    path_to_water_binary = Path('data', 'resources', 'binary_water_projected.tif')
    with rasterio.open(path_to_water_binary) as src:
        binary_water = src.read(1)

    aspect_ns_masked = np.where(binary_water == 0, aspect_ns_norm, -1.0)
    aspect_ew_masked = np.where(binary_water == 0, aspect_ew_norm, -1.0)


    profile = profile.copy()
    profile.update(dtype="float32", count=1, compress="lzw", nodata = -1.0)

    # Ensure output directory exists
    path_out.mkdir(parents=True, exist_ok=True)

    ns_file = path_out / "aspect_NS.tif"
    ew_file = path_out / "aspect_EW.tif"

    with rasterio.open(ns_file, "w", **profile) as dst:
        dst.write(aspect_ns_masked.astype(np.float32), 1)

    with rasterio.open(ew_file, "w", **profile) as dst:
        dst.write(aspect_ew_masked.astype(np.float32), 1)


if __name__ == '__main__':
    latmin, longmin = 36.812, -9.490
    latmax, longmax = 42.2724, -6.0234

    start_date = '2024-01-01'
    end_date = '2024-01-02'

    portugal_ggrid = GoldenGrid(
        crs='EPSG:3763',
        scale=1000,
        bbox=[longmin, latmin, longmax, latmax],
        start_date=start_date,
        end_date=end_date,
        day_interval=1
    )

    path_to_dtm = Path('data', 'raw', 'elevation')

    dtm_derivatives(
        golden_grid=portugal_ggrid,
        path_to_dtm=path_to_dtm,
        path_to_aspect=Path('data', 'processed', 'aspect'),
        path_to_slope=Path('data', 'processed', 'slope')
    )
import rasterio
import numpy as np
from pathlib import Path

def convert_frp_to_probability_no_nans(input_tiff_path: str, output_tiff_path: str):
    """
    Reads a Fire Return Period (FRP) GeoTIFF, converts it into an Annual 
    Burn Probability GeoTIFF (P_annual = 1 / FRP), and ensures that all 
    NaNs, zeros, and missing values are set strictly to 0.0.
    
    Args:
        input_tiff_path (str): Path to the source FRP GeoTIFF file.
        output_tiff_path (str): Target path where the Annual Probability GeoTIFF will be saved.
    """
    # 1. Open the original return period map
    with rasterio.open(input_tiff_path) as src:
        frp = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        nodata_val = src.nodata

    # 2. Identify pixels that have a valid, positive fire return period
    # Filter out NaNs, standard zeros, and dataset-specific nodata values
    if nodata_val is not None and not np.isnan(nodata_val):
        has_fire_record = (frp > 0) & (~np.isnan(frp)) & (frp != nodata_val)
    else:
        has_fire_record = (frp > 0) & (~np.isnan(frp))

    # 3. Apply the inversion formula: 
    # Where a fire was observed, compute 1.0 / FRP. 
    # Everywhere else (NaNs, unburned land, background borders) becomes strictly 0.0.
    prob_map = np.where(has_fire_record, 1.0 / frp, 0.0)

    # 4. Update the geospatial profile parameters
    meta.update({
        'dtype': 'float32',
        'count': 1,
        'nodata': None  # Removes the nodata flag since the whole raster is now filled with data (0.0 to 1.0)
    })

    # 5. Save the complete 0.0-filled annual probability raster to disk
    out_path = Path(output_tiff_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(prob_map.astype(np.float32), 1)

    print(f"Successfully generated 0-filled Annual Probability map at: {out_path}")
    return prob_map


if __name__ == '__main__':
    #path_to_rp = Path('exports', 'return_period', 'model_rp.tiff')
    #path_to_ap = Path('exports', 'return_period', 'model_ap.tiff')
    path_to_rp = Path('files', 'data', 'test', 'yearly_burn', 'fire_return_pd_25.tif')
    path_to_ap = Path('exports', 'return_period', 'preds_ap.tiff')
    convert_frp_to_probability_no_nans(input_tiff_path=path_to_rp, output_tiff_path=path_to_ap)
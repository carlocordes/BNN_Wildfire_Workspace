# External
import glob
from osgeo import gdal
from pathlib import Path

path_to_hdf = Path('data', 'raw', 'LST_MODIS_8day', 'MOD11A2.A2026033.h17v04.061.2026043145050.hdf')

path_to_tiff = Path('data', 'processed', 'LST_MODIS_8day')
fname_out = '20260431.tif'

hdf_layer = gdal.Open(str(path_to_hdf))
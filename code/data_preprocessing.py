# Internal

# External
import math

import numpy as np
from shapely import from_wkb
import ee
import geemap
from pathlib import Path
import time
import datetime
import pandas as pd
import geopandas as gpd
import glob
from shapely.geometry import box
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import duckdb

# ----------- CONFIG ----------- #
PROJECT_ID = 'transformerwildfire'
#ee.Authenticate()
ee.Initialize(project=PROJECT_ID)

db_path = Path('data', 'raw', 'burn', 'burn_points.db')
# ------------------------------ #



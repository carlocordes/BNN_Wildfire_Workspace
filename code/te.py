"""
Random off-topic trial error script
"""

import pandas as pd
from data_preprocessing import GoldenGrid

latmin, longmin = 36.812, -9.490
latmax, longmax = 42.2724, -6.0234

dr = pd.date_range(start = '2024-01-01', end = '2024-12-31', freq ='1D')

portugal_ggrid = GoldenGrid(crs = 'EPSG:3763',
                            scale = 1000,
                            bbox = [longmin, latmin, longmax, latmax],
                            start_date='2024-01-01', end_date='2024-12-31',
                            day_interval=1)
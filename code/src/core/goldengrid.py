# Internal

# External
import pandas as pd
import ee

PROJECT_ID = 'transformerwildfire'
#ee.Authenticate()
ee.Initialize(project=PROJECT_ID)

class GoldenGrid():
    """
    Defines project-unified grid via CRS, scale (m) and area of interest
    """
    def __init__(self, crs, scale, bbox : list[float],
                start_date : str, end_date : str, day_interval: int):
        self.crs = crs
        self.scale = scale
        self.bbox = bbox
        self.aoi = ee.Geometry.Rectangle(bbox)

        self.target_proj = ee.Projection(crs).atScale(scale)
        
        self.start_date = start_date
        self.end_date = end_date

        self.day_interval = day_interval
        self.dates = pd.date_range(start = start_date,
                              end = end_date,
                              freq = f'{day_interval}D')
        

if __name__ == '__main__':
    # Define local golden grid
    latmin, longmin = 36.812, -9.490
    latmax, longmax = 42.2724, -6.0234

    start_date = '2024-01-01'
    end_date = '2024-12-31'

    portugal_ggrid = GoldenGrid(
        crs = 'EPSG:3763',
        scale = 1000,
        bbox = [longmin, latmin, longmax, latmax],
        start_date = start_date,
        end_date = end_date,
        day_interval = 7
    )

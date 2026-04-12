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
    pass
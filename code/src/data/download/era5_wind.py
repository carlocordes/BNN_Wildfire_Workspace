# Internal
from src.core.goldengrid import GoldenGrid

# External
import ee

def download_era5_wind_image(date, golden_grid):
    """
    Download one ERA5-Land wind raster for a given timestamp.

    Parameters
    ----------
    date : str
        Timestamp (e.g. '2020-06-01T12:00')
    region : ee.Geometry
        Region to export
    filename : str
        Export task name
    """

    ee.Initialize()

    image = (
        ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        .filterDate(date, ee.Date(date).advance(1, "hour"))
        .first()
    )

    u = image.select("u_component_of_wind_10m")
    v = image.select("v_component_of_wind_10m")

    wind_speed = u.pow(2).add(v.pow(2)).sqrt()

    wind_direction = (
        v.atan2(u)
        .multiply(180 / 3.141592653589793)
    )

    speed_task = ee.batch.Export.image.toDrive(
        image=wind_speed,
        description=f"{date}_speed",
        folder="era5_wind",
        region=golden_grid.aoi,
        scale=11000,
        maxPixels=1e13,
    )

    direction_task = ee.batch.Export.image.toDrive(
        image=wind_direction,
        description=f"{date}_direction",
        folder="era5_wind",
        region=golden_grid.aoi,
        scale=11000,
        maxPixels=1e13,
    )

    speed_task.start()
    direction_task.start()

    print("Exports started:")
    print(f"{date}_speed")
    print(f"{date}_direction")


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

    download_era5_wind_image('2024-01-01', portugal_ggrid)
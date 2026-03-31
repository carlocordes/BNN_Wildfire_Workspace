# Internal
from src.core.goldengrid import GoldenGrid

# External
from pathlib import Path
import ee
import geemap



def download_era5_wind(date,
                       golden_grid,
                       speed_dir : Path,
                       dir_v_dir : Path,
                       dir_u_dir : Path
                        ):

    speed_dir.mkdir(parents=True, exist_ok=True)
    dir_v_dir.mkdir(parents=True, exist_ok=True)
    dir_u_dir.mkdir(parents=True, exist_ok=True)


    start = ee.Date(date)
    end = start.advance(1, "day")
    
    daily_coll = (
        ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        .filterDate(start, end)
        .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
    )

    if daily_coll.size().getInfo() == 0:
        print(f"No ERA5 data found for {date}")
        return

    image = daily_coll.mean() # Resulting U and V are daily averages

    # 2. Extract Components
    u_comp = image.select("u_component_of_wind_10m").rename('u_east_west')
    v_comp = image.select("v_component_of_wind_10m").rename('v_north_south')
    
    # Calculate Speed for the third feature
    wind_speed = u_comp.pow(2).add(v_comp.pow(2)).sqrt().rename('wind_speed')

    export_params = {
        'scale': golden_grid.scale,
        'region': golden_grid.aoi,
        'crs': golden_grid.crs,
        'file_per_band': False
    }

    fname = date + '.tif'

    # Export Speed
    geemap.ee_export_image(wind_speed, filename=str(speed_dir / fname), **export_params)
    
    # Export U (East-West)
    geemap.ee_export_image(u_comp, filename=str(dir_u_dir / fname), **export_params)
    
    # Export V (North-South)
    geemap.ee_export_image(v_comp, filename=str(dir_v_dir / fname), **export_params)

    print(f"Finished downloading wind features for {date}")


def download_era5_wind_catalogue(golden_grid : GoldenGrid,
                                 speed_dir : Path,
                                 dir_v_dir : Path,
                                 dir_u_dir : Path):
    date_strings = golden_grid.dates.strftime('%Y-%m-%d').tolist()
    for date in date_strings:
        download_era5_wind(
            date = date,
            golden_grid = golden_grid,
            speed_dir = speed_dir,
            dir_v_dir = dir_v_dir,
            dir_u_dir = dir_u_dir
        )

if __name__ == '__main__':
    # Define local golden grid
    latmin, longmin = 36.812, -9.490
    latmax, longmax = 42.2724, -6.0234

    start_date = '2024-08-01'
    end_date = '2024-08-30'

    portugal_ggrid = GoldenGrid(
        crs = 'EPSG:3763',
        scale = 1000,
        bbox = [longmin, latmin, longmax, latmax],
        start_date = start_date,
        end_date = end_date,
        day_interval = 7
    )

    dir_u_dir = Path('data', 'processed', 'wind_dir_WE')
    dir_v_dir = Path('data', 'processed', 'wind_dir_NS')
    speed_dir = Path('data', 'processed', 'wind_speed')
    download_era5_wind_catalogue(golden_grid= portugal_ggrid,
                                 speed_dir = speed_dir,
                                 dir_v_dir = dir_v_dir,
                                 dir_u_dir = dir_u_dir)
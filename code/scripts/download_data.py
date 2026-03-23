# Internal
from src.core.goldengrid import GoldenGrid
from src.data.download.dtm import get_one_dtm_image
from src.data.download.lst_modis import download_yearly_lst
from src.data.download.burn_targets import ingest_burn_records
from src.data.download.era5_wind import download_era5_wind_image

# External
import ee

def main():
    # Define local golden grid
    latmin, longmin = 36.812, -9.490
    latmax, longmax = 42.2724, -6.0234

    start_date = '2025-01-01'
    end_date = '2025-12-31'

    portugal_ggrid = GoldenGrid(
        crs = 'EPSG:3763',
        scale = 1000,
        bbox = [longmin, latmin, longmax, latmax],
        start_date = start_date,
        end_date = end_date,
        day_interval = 7
    )

    # MODIS LST
    #download_yearly_lst(Path('data', 'raw', 'LST_MODIS_8day'), portugal_ggrid)

    # DTM
    # get_one_dtm_image(Path('data', 'raw', 'dtm'), portugal_ggrid)

    # Targets
    # ingest_burn_records(Path('data', 'raw', 'burn'))

if __name__ == '__main__':
    main()
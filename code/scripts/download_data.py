# Internal
from src.core.goldengrid import GoldenGrid
from src.data.download.dtm import get_one_dtm_image
from src.data.download.lst_modis import download_yearly_lst
from src.data.download.burn_targets import ingest_burn_records
from src.data.download.era5_wind import download_era5_wind_image

# External
from pathlib import Path
from omegaconf import OmegaConf
import argparse
import ee




class DataDownloader():
    def __init__(self, config_path : Path):
        self.config_path = config_path


    def download(self):
        print('Fetching configs')

        cfg = OmegaConf.load(self.config_path)

        cfg_data = cfg['data']
        cfg_spatial_extent = cfg['data']['spatial_extent']
        temporal_extent = cfg['data']['temporal_extent']


        # Define local golden grid
        latmin, longmin = cfg_spatial_extent['latmin'], cfg_spatial_extent['longmin']
        latmax, longmax = cfg_spatial_extent['latmax'], cfg_spatial_extent['latmax']

        

        portugal_ggrid = GoldenGrid(
            crs = cfg_data['crs'],
            scale = cfg_data['scale'],
            bbox = [longmin, latmin, longmax, latmax],
            start_date = temporal_extent['start_date'],
            end_date = temporal_extent['end_date'],
            day_interval = temporal_extent['day_interval']
        )

        # MODIS LST
        #download_yearly_lst(Path('data', 'raw', 'LST_MODIS_8day'), portugal_ggrid)

        # DTM
        # get_one_dtm_image(Path('data', 'raw', 'dtm'), portugal_ggrid)

        # Targets
        # ingest_burn_records(Path('data', 'raw', 'burn'))

        print('Completed Download')


def main(config_path: Path):
    downloader = DataDownloader(config_path)
    downloader.download()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full pipeline using a YAML config")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    main(args.config)
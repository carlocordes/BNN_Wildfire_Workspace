# Internal
from src.core.utils import load_config

from src.core.goldengrid import GoldenGrid
from src.data.download.dtm import get_one_dtm_image
from src.data.download.lst_modis import download_lst
#from src.data.download.burn_targets import ingest_burn_records
from src.data.download.era5_wind import download_era5_wind_catalogue
from src.data.download.ndvi import download_ndvi_catalogue
from src.data.download.ndwi import download_ndwi_catalogue
from src.data.download.precip import download_rain_catalogue
from src.data.download.burned_area import download_area_with_uncertainty



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

        cfg = load_config(self.config_path)

        # Define local golden grid
        cfg_spatial_extent = cfg['data']['spatial_extent']
        latmin, longmin = cfg_spatial_extent['latmin'], cfg_spatial_extent['longmin']
        latmax, longmax = cfg_spatial_extent['latmax'], cfg_spatial_extent['longmax']


        # Define golden grid from config specs
        temporal_extent = cfg['data']['temporal_extent']
        cfg_data = cfg['data']

        ggrid = GoldenGrid(
            crs = cfg_data['crs'],
            scale = cfg_data['scale'],
            bbox = [longmin, latmin, longmax, latmax],
            start_date = temporal_extent['start_date'],
            end_date = temporal_extent['end_date'],
            day_interval = temporal_extent['day_interval']
        )


        # Perform downloads to paths
        cfg_data_paths = cfg['data_paths']

        """      
        # DTM
        cfg_dtm_path = Path(cfg_data_paths["raw"]['elevation'])
        get_one_dtm_image(cfg_dtm_path, ggrid)
        """

        # ERA-5 wind
        cfg_wind_speed_path = Path(cfg_data_paths['processed']['wind_speed'])
        cfg_wind_comp_u_path = Path(cfg_data_paths['processed']['wind_dir_u'])
        cfg_wind_comp_v_path = Path(cfg_data_paths['processed']['wind_dir_v'])
        download_era5_wind_catalogue(golden_grid = ggrid,
                                     speed_dir = cfg_wind_speed_path,
                                     dir_v_dir = cfg_wind_comp_v_path,
                                     dir_u_dir = cfg_wind_comp_u_path)

        """
        # MODIS NDVI (Normalized Difference Vegetation Index)
        cfg_ndvi_path = Path(cfg_data_paths['processed']['NDVI'])
        download_ndvi_catalogue(cfg_ndvi_path, ggrid)


        # CHIRPS Rain
        out_path_precip = Path(cfg_data_paths['processed']['precip'])
        download_rain_catalogue(out_path = out_path_precip, golden_grid=ggrid)

        # MODIS LST
        out_path_LST = Path(cfg_data_paths['processed']['LST'])
        download_lst(out_dir=out_path_LST, golden_grid=ggrid)

        # NDWI (Normalized Difference Water Index)
        out_path_NDWI = Path(cfg_data_paths['processed']['NDWI'])
        download_ndwi_catalogue(out_path = out_path_NDWI, golden_grid=ggrid)

        # New targets
        print('Downloading Targets')
        cfg_raw_target = Path(cfg_data_paths['raw']['target'])
        download_area_with_uncertainty(golden_grid=ggrid,
                                       out_path=cfg_raw_target)

        """
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
# Internal
from src.core.goldengrid import GoldenGrid
from src.core.utils import load_config
from src.data.processing.ground_truth import grid_target_map
from src.data.processing.dtm_derivatives import dtm_derivatives
from src.data.processing.dist_to_roads import produce_roads_distance_image
from src.data.processing.days_since_burn import process_burn_history_catalogue

from src.data.processing.targets_from_zarr import produce_targets

from src.data.processing.dataset_norm import normalize_dataset

# External
import argparse
from pathlib import Path

class DataProcessor():
    def __init__(self, config_path : Path):
        self.config_path = config_path


    def process(self):
        print(f'Fetching configs fom {self.config_path}')

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

        # Perform processing to paths
        cfg_data_paths = cfg['data_paths']
        
        """
        # Distance to roads
        cfg_raw_roads = Path(cfg_data_paths['raw']['roads'])
        cfg_raw_out = Path(cfg_data_paths['processed']['roads'])
        produce_roads_distance_image(in_path = cfg_raw_roads,
                                     out_dir=cfg_raw_out,
                                     golden_grid= ggrid)

        # Burn history from zarr
        cfg_raw_burn_history = Path(cfg_data_paths['raw']['target'])
        cfg_out_burn_history = Path(cfg_data_paths['processed']['burn_history'])
        process_burn_history_catalogue(zarr_path=cfg_raw_burn_history,
                                       out_path=cfg_out_burn_history,
                                       golden_grid=ggrid)


        # DTM derivatives
        print('Processing derivatives of digital terrain model')
        cfg_data_dtm = Path(cfg_data_paths['raw']['DTM'])
        cfg_data_slope = Path(cfg_data_paths['processed']['slope'])
        cfg_data_aspect = Path(cfg_data_paths['processed']['aspect'])
        dtm_derivatives(golden_grid = ggrid,
                        path_to_dtm = cfg_data_dtm,
                        path_to_aspect = cfg_data_aspect,
                        path_to_slope = cfg_data_slope)




        ## Target
        print('Processing targets.')

        cfg_raw_target = Path(cfg_data_paths['raw']['target'])
        cfg_processed_target = Path(cfg_data_paths['processed']['target'])
        cfg_target_extent = temporal_extent['target_extent']
        produce_targets(golden_grid=ggrid,
                      in_path = cfg_raw_target,
                      out_path_target = cfg_processed_target,
                      target_extent = cfg_target_extent)
        """


        # Add other processes here
        #
        #
        #


        ## Dataset Normalization
        print('Normalizing and masking datasets.')
        """
        # MODIS LST
        cfg_lst_in = Path(cfg_data_paths['raw']['LST'])
        cfg_lst_out = Path(cfg_data_paths['processed']['LST'])
        normalize_dataset(input_dataset=cfg_lst_in, output_dataset=cfg_lst_out)


        # NDVI
        cfg_ndvi_in = Path(cfg_data_paths['raw']['NDVI'])
        cfg_ndvi_out = Path(cfg_data_paths['processed']['NDVI'])
        normalize_dataset(input_dataset=cfg_ndvi_in, output_dataset=cfg_ndvi_out)


        # NDWI
        cfg_ndwi_in = Path(cfg_data_paths['raw']['NDWI'])
        cfg_ndwi_out = Path(cfg_data_paths['processed']['NDWI'])
        normalize_dataset(input_dataset=cfg_ndwi_in, output_dataset=cfg_ndwi_out)


        # CHIRPS precipitation
        cfg_precip_in = Path(cfg_data_paths['raw']['precip'])
        cfg_precip_out = Path(cfg_data_paths['processed']['precip'])
        normalize_dataset(input_dataset=cfg_precip_in, output_dataset=cfg_precip_out)
        """

        # Wind 
        cfg_wspeed_in = Path(cfg_data_paths['raw']['wind_speed'])
        cfg_wspeed_out = Path(cfg_data_paths['processed']['wind_speed'])
        normalize_dataset(input_dataset=cfg_wspeed_in, output_dataset=cfg_wspeed_out)

        cfg_wu_in = Path(cfg_data_paths['raw']['wind_dir_u'])
        cfg_wu_out = Path(cfg_data_paths['processed']['wind_dir_u'])
        normalize_dataset(input_dataset=cfg_wu_in, output_dataset=cfg_wu_out)


        cfg_wv_in = Path(cfg_data_paths['raw']['wind_dir_v'])
        cfg_wv_out = Path(cfg_data_paths['processed']['wind_dir_v'])
        normalize_dataset(input_dataset=cfg_wv_in, output_dataset=cfg_wv_out)

def main(config_path: Path):
    downloader = DataProcessor(config_path)
    downloader.process()

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

    # Example usage from /code:
    # uv run -m scripts.process_data --config configs/project.yaml
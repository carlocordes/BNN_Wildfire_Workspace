# Internal
from src.core.goldengrid import GoldenGrid
from src.core.utils import load_config
from src.data.processing.ground_truth import grid_target_map


# External
import argparse
from pathlib import Path

class DataProcessor():
    def __init__(self, config_path : Path):
        self.config_path = config_path


    def process(self):
        print('Fetching configs')

        cfg = load_config(self.config_path)

        # Define local golden grid
        cfg_spatial_extent = cfg['data']['spatial_extent']
        latmin, longmin = cfg_spatial_extent['latmin'], cfg_spatial_extent['longmin']
        latmax, longmax = cfg_spatial_extent['latmax'], cfg_spatial_extent['longmax']


        # Define golden grid from config specs
        temporal_extent = cfg['data']['temporal_extent']
        cfg_data = cfg['data']

        portugal_ggrid = GoldenGrid(
            crs = cfg_data['crs'],
            scale = cfg_data['scale'],
            bbox = [longmin, latmin, longmax, latmax],
            start_date = temporal_extent['start_date'],
            end_date = temporal_extent['end_date'],
            day_interval = temporal_extent['day_interval']
        )

        # Perform processing to paths
        cfg_data_paths = cfg['data_paths']

        # 1. Targets
        cfg_target_path = Path(cfg_data_paths["processed"]['target'])
        cfg_target_out = cfg_data_paths['raw']['target']
        cfg_db_path = Path(cfg_target_out['csv']) / cfg_target_out['db_name']
        grid_target_map(path_to_db=cfg_db_path,
                        out_path=cfg_target_path,
                        golden_grid=portugal_ggrid)
        
        # Add other processes here
        #
        #
        #

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
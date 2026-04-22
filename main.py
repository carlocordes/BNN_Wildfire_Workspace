"""
From here training runs are controlled
"""

# Internal
from scripts.train import main as run_training
#from src.core.s3_data_download import download_single_file as download_dataset
#from src.core.s3_data_upload import upload_directory as upload_results

# External
import argparse
from pathlib import Path

OUTPUT_FOLDER = Path('files')


def main(config : str, dataset : str, exp_name : str):
    """
    ## 1. Retrieve dataset
    s3_path = 'test_sets/small.pt' # 'alpha/small.pt'
    local_dest = 'files/datasets'
    Path(local_dest).mkdir(exist_ok=True, parents = True)
    download_dataset(s3_path = s3_path, local_folder = local_dest)
    """

    ## 2. Set up experiment structure
    path_to_config = OUTPUT_FOLDER / 'configs' / config
    path_to_dataset = OUTPUT_FOLDER / 'datasets' / dataset
    
    output_dir = OUTPUT_FOLDER / 'experiments' / exp_name
    try: 
        output_dir.mkdir(parents = True, exist_ok = False) # Define output dir

        # 3. Train
        run_training(config_path=path_to_config,
                    dataset_path=path_to_dataset,
                    experiment_path=output_dir)
        print(f'Results were stored to {output_dir}')

    except FileExistsError:
        print('Output folder name already exists. Choose again.')




    ## 4.  Upload


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        help='Name of config in files/configs/',
                        required=True)
    parser.add_argument('--dataset',
                        type=str,
                        help='Name of dataset in files/datasets',
                        required=True
                        )
    parser.add_argument('--exp_name',
                        type=str,
                        help='Output name of experiment',
                        required=True)
    args = parser.parse_args()
    main(config = args.config,
         dataset = args.dataset,
         exp_name = args.exp_name)

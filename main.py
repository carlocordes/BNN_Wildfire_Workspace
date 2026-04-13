"""
From here training runs are controlled
"""

# Internal
from scripts.train import main as run_training
from src.core.s3_data_download import download_single_file as download_dataset
from src.core.s3_data_upload import upload_directory as upload_results
# External
from pathlib import Path


def main():

    # 0. Retrieve dataset
    s3_path = 'test_sets/small.pt' # 'alpha/small.pt'
    local_dest = 'data/datasets'
    Path(local_dest).mkdir(exist_ok=True, parents = True)
    download_dataset(s3_path = s3_path, local_folder = local_dest)




    ## 1. Small experiment
    experiment_path = Path('experiments', 'test_small') # Define experiment direcotry
    config_path = experiment_path / 'config_small.yaml' # Define config path
    dataset = 'small.pt'
    run_training(experiment_path = experiment_path,
                config_path = config_path,
                dataset_name = dataset)
    upload_results(local_directory_path = experiment_path,
                   s3_prefix = 'results/small')


    ## 2. Large Experiment
    experiment_path = Path('experiments', 'test_large')
    config_path = experiment_path / 'config_large.yaml'
    print(experiment_path.exists())
    print(config_path.exists())
    dataset = 'small.pt'
    run_training(experiment_path = experiment_path,
                 config_path = config_path,
                 dataset_name = dataset)
    upload_results(local_directory_path = experiment_path,
                   s3_prefix = 'results/large')




if __name__ == '__main__':
    main()
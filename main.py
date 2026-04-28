"""
From here training runs are controlled
"""

# Internal
from scripts.train import main as run_training
from scripts.s3_data_upload import upload_directory, BUCKET_NAME, S3_ENDPOINT


# External
import os
import boto3
import argparse
from pathlib import Path
from dotenv import dotenv_values

OUTPUT_FOLDER = Path('files')

def get_s3_client():
    """Helper to initialize the S3 client."""
    config = dotenv_values(".env")
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=config.get('ACCESS_KEY'),
        aws_secret_access_key=config.get('SECRET_KEY'),
    )

def main(config : str, dataset : str, exp_name : str, upload : bool):
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
    print(f'output dir:', output_dir)

    if not output_dir.exists() or not any(output_dir.iterdir()): # Folder doesnt exist or isn't empty
        output_dir.mkdir(parents = True, exist_ok = False) # Define output dir
        # 3. Train
        run_training(config_path=path_to_config,
                    dataset_path=path_to_dataset,
                    experiment_path=output_dir)
        print(f'Results were stored to {output_dir}')

        ## 4.  Upload
        if upload:
            client = get_s3_client()
            s3_path = 'results'
            print(s3_path)

            upload_directory(s3_client=client,
                            local_directory_path=output_dir,
                            s3_prefix=s3_path)

    else:
        print('Output folder name already exists. Choose again.')
    
        




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
    parser.add_argument('--auto_upload',
                        action='store_true',
                        help='Upload results if flag is present')
    args = parser.parse_args()
    main(config = args.config,
         dataset = args.dataset,
         exp_name = args.exp_name,
         upload = args.auto_upload)

# Code Instructions

## Running using Docker



Create bind mound folders locally:
````
mkdir -p ./mnt/datasets
mkdir -p ./mnt/configs
mkdir -p ./mnt/experiments
````

Config files are shipeed in `files/configs/`. However, both datasets and experiments folders need to be added:

````
mkdir -p ./files/datasets
mkdir -p .files/experiments
````


For communicatio with Hetzner object storage and retrieval of datasets create  .env file & fill credentials:
````
touch .env
----------
ACCESS_KEY = '...'
SECRET_KEY = '...'
````


Build docker container from Dockerfile:
````
docker buildx build --platform linux/amd64 -t wildfire-model .
````

Download data:
````
docker run --platform linux/amd64 -v "$(pwd)"/mnt:/app/files wildfire-model python scripts/s3_data_download.py --s3_path test_sets/small.pt --local_dest files/datasets
````

Run training (args: config file, datasetname):
````
docker run --platform linux/amd64 -v "$(pwd)"/mnt:/app/files wildfire-model python main.py --config project.yaml --dataset validator_2020.pt --exp_name test
````


Upload data (will upload entire results directory, name accordingly):
````
docker run --platform linux/amd64 -v "$(pwd)"/mnt:/app/files wildfire-model python scripts.s3_data_upload --path files/experiments --s3_prefix results/
````
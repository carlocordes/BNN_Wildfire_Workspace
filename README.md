# Code Instructions

After cloning, from the project root, set:

```
export ROOT_DIR=$(pwd)
```

## Running using Docker

Config files are shipeed in `files/configs/`. However, both datasets and experiments folders need to be added:

````
mkdir -p ./files/datasets
mkdir -p ./files/experiments
````

For communication with Hetzner object storage and retrieval of datasets create  .env file & fill credentials:
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

## Manual Upload

Upload data (will upload entire results directory, name accordingly):
````
docker run -v "$(pwd)"/files:/app/files wildfire-model python scripts.s3_data_upload --path files/experiments --s3_prefix results
````

## t002 testing routine:
Running larger datasets on larger model in two variants
1. Patchsize: 16
2. Patchsize: 8


### Download data:
```
docker run -v "$(pwd)"/files:/app/files:Z wildfire-model python scripts/s3_data_download.py dir --s3_path datasets/t003 --local_dest files/datasets/
```

### Patch size = 16: 
```
docker run  --device nvidia.com/gpu=all -v "$(pwd)"/files:/app/files:Z wildfire-model python main.py --config config_t003_p200.yaml --dataset t003_dataset --exp_name t003_pos200 --auto_upload
```

### pos weight = 4000
```
docker run  --device nvidia.com/gpu=all -v "$(pwd)"/files:/app/files:Z wildfire-model python main.py --config config_t003_p4000.yaml --dataset t003_dataset --exp_name t003_pos4000 --auto_upload
```
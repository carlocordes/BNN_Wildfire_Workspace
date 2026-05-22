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

# t007 Regimen

### 1 & 2

```
docker run  --device nvidia.com/gpu=all --ipc=host -v "$(pwd)"/files:/app/files:Z wildfire-model python main.py --config config_t007_1.yaml --exp_name t007_1 --auto_upload;docker run  --device nvidia.com/gpu=all --ipc=host -v "$(pwd)"/files:/app/files:Z wildfire-model python main.py --config config_t007_2.yaml --exp_name t007_2 --auto_upload;
```



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

### Download data:
```
docker run -v "$(pwd)"/files:/app/files:Z wildfire-model python scripts/s3_data_download.py dir --s3_path datasets/t004 --local_dest files/datasets/
```
### 1
```
docker run  --device nvidia.com/gpu=all --ipc=host -v "$(pwd)"/files:/app/files:Z wildfire-model python main.py --config config_t005_1.yaml --exp_name t005_1 --auto_upload
```

### 2
```
docker run  --device nvidia.com/gpu=all --ipc=host -v "$(pwd)"/files:/app/files:Z wildfire-model python main.py --config config_t005_2.yaml --exp_name t005_2 --auto_upload
```


### 3
```
docker run  --device nvidia.com/gpu=all --ipc=host -v "$(pwd)"/files:/app/files:Z wildfire-model python main.py --config config_t005_3.yaml --exp_name t005_3 --auto_upload
```


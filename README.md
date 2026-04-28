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

Download data:
````
docker run -v "$(pwd)"/files:/app/files wildfire-model python scripts/s3_data_download.py --s3_path test_sets/small.pt --local_dest files/datasets
````

Run training with --auto_upload flag (args: config file, datasetname):
````
docker run  -v "$(pwd)"/files:/app/files wildfire-model python main.py --config project.yaml --dataset small.pt --exp_name test --auto_upload
````

## Manual Upload

Upload data (will upload entire results directory, name accordingly):
````
docker run -v "$(pwd)"/files:/app/files wildfire-model python scripts.s3_data_upload --path files/experiments --s3_prefix results
````

## t001 testing routine:
Focus: Trying to get a sense of how system parameters **embedding dimension** and **patch size** influence computational capacity on remote machine.


### Download data:
```
docker run -v "$(pwd)"/files:/app/files:Z wildfire-model python scripts/s3_data_download.py --s3_path datasets/t001/3year_0lead.pt --local_dest files/datasets
```

### 1st. Variant: 
```
docker run  --gpus all -v "$(pwd)"/files:/app/files:Z wildfire-model python main.py --config 3year_0lead.yaml --dataset 3year_0lead.pt --exp_name t001_small --auto_upload
```

### 2nd. Variant: larger embedding dimension
```
docker run  --gpus all -v "$(pwd)"/files:/app/files:Z wildfire-model python main.py --config 3year_0lead_embed.yaml --dataset 3year_0lead.pt --exp_name t001_embedding --auto_upload
```

### 3rd. Variant: smaller patch size
```
docker run  --gpus all -v "$(pwd)"/files:/app/files:Z wildfire-model python main.py --config 3year_0lead_patch.yaml --dataset 3year_0lead.pt --exp_name t001_patch --auto_upload
```

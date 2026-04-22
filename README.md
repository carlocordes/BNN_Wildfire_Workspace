# Code Instructions

## 1. Docker

Create volume/mount:
```
docker volume create datasets
```


Inspect it:
```
docker volume inspect datasets
````


Create bind mound folders locally:
````
mkdir -p ./mnt/datasets
mkdir -p ./mnt/configs
mkdir -p ./mnt/experiments
````


Create .env file & fill credentials:
````
touch .env
----------
ACCESS_KEY = '...'
SECRET_KEY = '...'
````


Build docker container:
````
docker buildx build --platform linux/amd64 -t wildfire-model .
````

Download data:
````
docker run --platform linux/amd64 -v "$(pwd)"/mnt:/app/files wildfire-model python scripts/s3_data_download.py --s3_path test_sets/small.pt --local_dest files/datasets
````

Run training:
````
docker run --platform linux/amd64 -v "$(pwd)"/mnt:/app/files wildfire-model python main.py --config project.yaml --dataset validator_2020.pt --exp_name test
````

Upload data:
````

````
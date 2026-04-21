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
````
Todo: add other folders


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


Run it using bind mount:
````
docker run --platform linux/amd64 -v "$(pwd)"/mnt:/app/files wildfire-model python main.py --config project.yaml --dataset validator_2020.pt --exp_name test
````

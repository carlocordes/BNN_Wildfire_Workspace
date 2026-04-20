# Code Instructions

## 1. Docker
* Change to directory `code/`
* Run `docker build -t wildfire-model .`
* Configure configs in `configs/project.yaml`
* Runs scripts in `scripts/`

docker buildx build --platform linux/amd64 -t wildfire-model

CPU-based:
```
docker run wildfire-model scripts.train --config configs/project.yaml --datasetname dataset
```

GPU accelerated:
```
docker run --gpus all wildfire-model scripts.train --config configs/project.yaml --datasetname dataset
```
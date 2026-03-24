# Code Instructions

## 1. Docker
* Change to directory `code/`
* Run `docker build wildfire-model .`
* Configure configs in `configs/project.yaml`
* Runs scripts in `scripts/`

CPU-based:
```
docker run wildfire-model scripts.train --config configs/project.yaml --datasetname dataset
```

GPU accelerated:
```
docker run --gpus all wildfire-model scripts.train --config configs/project.yaml --datasetname dataset
```
# Spatio-Temporal Transformers for Wildfire Prediction

This repository contains the complete codebase, documentation, and thesis report for my master's thesis titled **"Spatio-Temporal Transformers for Wildfire Prediction"** at **TU Delft**. 

This project leverages multi-modal satellite data within a spatio-temporal vision transformer architecture to infer a probabilistic risk of wildfire from historical burn records. The study investigates the effect of altering data contextualization and lead-time projection of predictions into the future.

![Alt Text](exports/seasonal_prediction/season_pred_animation.gif)

---

## 📌 Table of Contents
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)
---

## Repository Structure
```text
Transformer_Wildfire_Workspace/
│
├── files/                        # Data and Models
│   ├── configs/                  # Configuration files for training
│   ├── data/
│   │   ├── raw/                  # Raw EE data container
│   │   ├── processed/            # Harmonized in CRS, scale
│   └── experiments/              # Model run output location
│
├── scripts/                      # Functional workflows
│   ├── download_data.py          # EE endpoint retrievals
│   ├── process_data.py           # Data harmonization
│   ├── dataset_builder.py        # Constructs batched model inputs
│   ├── train.py                  # Training orchestration scripts
│
├── src/                          # Core Python source code
│   ├── core/
│   ├── data/                     # Data download and processing logic
│   └── models/                   # STViT Transformer architecture
│
├── exports/                      # Visuals
├── report/                       # Thesis materials and documentation
│   ├── thesis/                   # MSc thesis manuscript source
│
├── Dockerfile                    # Containerized runtime environment
├── .dockerignore                 # Docker build exclusions
├── pyproject.toml                # Python dependencies and project metadata
├── .gitignore                    # Ignored files and directories
├── README.md                     # Project overview and usage instructions
└──  main.py                      # Main project entry point

````


## Installation & Setup

Follow these steps to set up the project locally. Instructions are here provided mainly for macOS and may vary by system. The repositories functionalities can be run via two methods:
- uv: python project manager
- Docker: containerized application

### 1. Clone the Repository

```bash
git clone https://github.com/carlocordes/Transformer_Wildfire_Workspace.git
cd Transformer_Wildifire_Workspace
```

### 2. Install `uv`
Install `uv` (Python package and environment manager) with curl or pip:

#### curl
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### pip
```bash
pip install uv
```


Verify installation:

```bash
uv --version
```

### 3. Create the Environment from `pyproject.toml`

Create and sync the virtual environment using the project's dependencies:

```bash
uv sync
```

Activate the environment:

#### macOS / Linux
```bash
source .venv/bin/activate
```

#### Windows
```powershell
.venv\Scripts\activate
```

### 4. Install Docker (Optional)

Download and install Docker Desktop for your operating system:

- macOS / Windows: Install Docker Desktop
- Linux: Install Docker Engine

Verify installation:

```bash
docker --version
docker compose version
```



## Usage



### 1. Build docker container from Dockerfile:
````
docker buildx build --platform linux/amd64 -t wildfire-model .
````

### 2. Set project root:
```
export ROOT_DIR=$(pwd)
```


### 3. Set System Configuration
In order to harmonize datasets, project defining parameters have to be set in `files/configs/project.yaml` prior to data retrieval. Parameters by category include:

- Geography
  - Spatial extent: minima and maxima of longitude and latitude
  - Coordinate Reference System (EPSG code)
  - Scale (Resolution in Meters)
Temporal
  - Global temporal extent (start-date & end_date)
  - Day Interval (Sampling period, daily is reccommended)


Parameters for model and training can be changed at a later point.


### 4. Retrieve Data from Endpoint
Download data from earth engine according to project parameters set in the configuration file:
````bash
uv run -m scripts.download_data --config files/configs/project.yaml
````

### 5. Process Data
Process data from `files/data/raw` to `files/data/processed`, according to project parameters set in the configuration file:

````bash
uv run -m scripts.process_data --config files/configs/project.yaml
````


### 6. Run Model
Model trainings are run according to specifications passed under `--config`. To keep all experiments unique an experiment name has to be chosen and passed via `--exp_name`. 

This part can be carried out via uv or a Docker container.

#### Run using uv
```
uv run -v "$(pwd)"/files:/app/files:Z wildfire-model python main.py --config project.yaml --exp_name my_experiment_name;
```


#### Run in Docker container

```
docker run  --device nvidia.com/gpu=all --ipc=host -v "$(pwd)"/files:/app/files:Z wildfire-model python main.py --config project.yaml --exp_name my_experiment_name;
```


## License


## Contact


## Citation
````
@mastersthesis{cordes2026wildfire,
              author = {Carlo Cordes},
              title = {Spatio-Temporal Transformers for Wildfire Prediction},
              school = {Delft University of Technology (TU Delft)},
              type = {Master's Thesis},
              year = {2026}, address = {Delft, The Netherlands},
              url = {https://github.com/carlocordes/Transformer_Wildfire_Workspace},
              note = {Codebase and thesis repository}
}
````

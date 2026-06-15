# Spatio-Temporal Transformers for Wildfire Prediction

This repository contains the complete codebase, documentation, and thesis report for my master's thesis titled **"Spatio-Temporal Transformers for Wildfire Prediction"** at **TU Delft**. 

This project leverages multi-modal satellite data within a spatio-temporal vision transformer architecture to infer a probabilistic risk of wildfire from historical burn records. The study investigates the effect of altering data contextualization and lead-time projection of predictions into the future.


---

## 📌 Table of Contents
- [Repository Structure](#-repository-structure)
- [Key Features](#-key-features)
- [Dataset Information](#-dataset-information)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Training](#2-model-taining)
  - [3. Evaluation](#3-evaluation)
- [Thesis Report & Documentation](#-thesis-report--documentation)
- [Results Summary](#-results-summary)
- [License](#-license)
- [Citation](#-citation)
- [Contact](#-contact)

---

## 📁 Repository Structure
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
- Clone
- 



## Usage

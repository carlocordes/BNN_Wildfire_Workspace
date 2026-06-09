# Spatio-Temporal Transformers for Wildfire Prediction

This repository contains the complete codebase, documentation, and thesis report for my master's thesis titled **"Spatio-Temporal Transformers for Wildfire Prediction"** at **TU Delft**. 

This project leverages multi-modal satellite data within a spatio-temporal vision transformer architecture to infer a probabilistic risk of wildfire from historical burn records. The study investigates the effect of altering data contextualization and lead-time projection of predictions into the future.


---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
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
├── files/
│   └── configs/                  # Configuration files for training
│
├── report/                       # Thesis materials and academic documentation
│   ├── thesis/                   # MSc thesis manuscript source
│   ├── figures/                  # Images and diagrams used in the report
│   └── references/               # Bibliography and supporting material
│
├── scripts/                      # Utility and automation scripts
│   ├── preprocessing/            # Dataset preparation and cleaning
│   ├── train.py                  # Training orchestration scripts
│   └── evaluation/               # Validation and metrics pipelines
│
├── src/                          # Core Python source code
│   ├── data/                     # Dataset loading and preprocessing logic
│   ├── models/                   # Transformer architectures and model components
│   ├── training/                 # Training loops, optimization, callbacks
│   └── utils/                    # Shared helper utilities
│
├── main.py                       # Main project entry point
├── Dockerfile                    # Containerized runtime environment
├── pyproject.toml                # Python dependencies and project metadata
├── .gitignore                    # Ignored files and directories
├── .dockerignore                 # Docker build exclusions
└── README.md                     # Project overview and usage instructions
````


## Installation & Setup


## Usage

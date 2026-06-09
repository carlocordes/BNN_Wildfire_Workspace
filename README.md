# Spatio-Temporal Transformers for Wildfire Prediction

This repository contains the complete codebase, documentation, and thesis report for my master's/bachelor's thesis titled **"patio-Temporal Transformers for Wildfire Prediction"** at **TU Delft**. 

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

## Project Overview
Provide a concise abstract of your thesis here. 
* **The Problem:** Why is predicting wildfires difficult? Why are traditional methods lacking?
* **The Solution:** How does the Transformer architecture solve this? (e.g., handling long-range spatial-temporal dependencies).
* **Objective:** What did this thesis set out to achieve and what were the key findings?

---

## 📁 Repository Structure
```text
├── docs/                     # Thesis report, presentations, and diagrams
│   ├── thesis_final.pdf      # The complete written thesis report
│   └── presentation.pdf      # Defense slides
├── data/                     # Data directory (Note: heavy data should be gitignored)
│   ├── raw/                  # Original datasets
│   └── processed/            # Tokenized or preprocessed patches/tensors
├── models/                   # Model definitions (Transformer variants)
│   ├── layers.py
│   └── transformer_fire.py
├── notebooks/                # Jupyter notebooks for EDA and quick prototyping
├── src/                      # Core source code modules
│   ├── data_loader.py        # Custom dataset classes and batching
│   ├── train.py              # Main training script
│   ├── evaluate.py           # Evaluation metrics (F1, IoU, RMSE, etc.)
│   └── utils.py              # Helper functions
├── weights/                  # Saved model checkpoints (.pt, .pth, or .h5)
├── requirements.txt          # Python dependencies
└── README.md                 # Project navigation guide
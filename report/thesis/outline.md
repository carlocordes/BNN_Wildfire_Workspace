# Thesis Structure: Wildfire Transformer

## 1. Introduction

| Section | Subsection | Content Description | Key Sources / Citations | Figures / Tables |
|---------|------------|---------------------|-------------------------|------------------|
| 1.1 Background |  | Natural Hazards & their significance |  |  |
| 1.2 Catastrophe Modeling | | Context of geospatial modeling, environmental data, and ML pipelines |  |  |
| 1.3 ML & geospatial data |  | Data-driven ML models aid natural hazard understanding  |  |  |
| 1.2 Problem Statement | | Define core problem (e.g., spatio-temporal prediction, terrain-informed modeling) | Domain-specific papers | Problem illustration |
| 1.4 Research Objectives | | RQs, Sub-RQs, hypothesis | — | — |
| 1.5 Thesis Structure | | Overview of chapters | — | — |

## 2. Related Work

| Section | Subsection | Content Description | Key Sources / Citations | Figures / Tables |
|---------|------------|---------------------|-------------------------|------------------|
| 2.1 Wildfire Risk Management |  |  |  |  |
|  | 2.1.1 Drivers: Susceptibility & Ignition |  |  |  |
|  | 2.1.2 Historical Approaches |  |  |  |
|  | 2.1.3 Preparedness & Detection |  |  |  |
|  | 2.1 4 ML approaches |  |  |  |
| 2.2 Transformers |  |  |  |  |
|  | 2.2.1 Background & Motivation | Bottleneck of sequential models (LSTM & Recurrent NN) & convolution, encoder-decoder frameworks, attention for parallelization | Bahdanau Attention, Attention is All You Need (Vaswani et al.) |  |
|  | 2.2.2 Architecture | Structure, tokens, embedding, positional encoding |  |  |
|  | 2.2.3 Attention Mechanism | Key, Query, Value, attention weights via scaled dot-product, self- and cross-attention |  |  |
|  | 2.2.4 Johnson-Lindenstrauss Lemma & Vector Representations| Representation of unique vectors in high-D space | JL Lemma 1984 | Embedding dim. vs. no. of distinguishable items |
|  | 2.2.5 Vision Transformers | Extension to image vs. semantic tokens, patch embedding, pos. encoding, CNN comparison, local vs. distant features| Dosovitskiy 2020 |  |
| 2.3 Memory Acceleration |  | GPU as a driver of better memory handling for parallel tasks |  |  |  |
|  | 2.3.1 Computational Complexity | Scaling model parameters and bottlenecks |  |  |
|  | 2.3.2 Graphics Processing Units (GPU) | Contrast CPU, parallelization, CUDA |  | Wang 2023: CPU for DL Earth Science, Cha 2026: GPU-Accelerated Transformer |


## 3. Methodology

| Section | Subsection | Content Description | Key Sources / Citations | Figures / Tables |
|---------|------------|---------------------|-------------------------|------------------|
| 3.1 Workflow & System Overview |  | End-to-end pipeline, config system, focus: modelling pure risk |  | Tech stack/flow diagram, python dependencies |
| 3.2 Data |  | Outline of used data |  | Snapshot of map of all data modules |
| 3.3 Data Acquisition & Processing |  |  |  |  |
|  | 3.3.1 Earth Engine & Harmonization |  |  |  |
|  | 3.3.2 Feature Engineering | Derived data, e.g. wind split & DTM |  | Map |
|  | 3.3.3 Ground Truth |  |  |  |
| 3.4 Dataset Construction |  | Static/dynamic data, WildfireDataset class,  |  | Sequence/target chart |
|  | 3.4.1 Timeframe Configuration | Reading and constructing config instructions into loading sequence |  | Sequence, Target time graph |
|  | 3.4.2 Torch Datasets & Tensors | Storing static & dynamic data into 5D pytorch tensors & storing as datast |  |  |
| 3.5 Model Architecture |  |  |  |  |
|  | 3.5.1 Encoding | Patch construction, reflective padding, flattening |  | Figure of reflective padding, patches, flattening operation |
|  | 3.5.2 Temporal Embedding | Tubelets, implicit embedding, 3D convolution |  | 3D convolution figure,  |
|  | 3.5.3 Spatial Embedding | Cosine embedding, static tags |  | Cosine similarity of patches, Visual encoding dimensions |
|  | 3.5.4 Attention Layers |  |  |  |
|  | 3.5.5 Fusion Layer | Implicit Cross-Attention, token stacking |  |  |
|  | 3.5.6 Decoder Layer | Convolution ReLu series, batch-norm, upsample to $h \times w$ |  |  |
| 3.6 Training Routine |  |  |  |  |
|  | 3.6.1 Model parameters | Distribution of parameters through model |  | Tabular view of parameters |
|  | 3.6.2 Loss Function | Class imbalance, positive weight (Todo: Error infusion) |  |  |
|  | 3.6.3 Back propagation & Optimizer | Handling weights, gradient and loss computation, contiguous memory |  | UML of class inheritance from nn.Module |

## 4. Experiments & Results

| Section | Subsection | Content Description | Key Sources / Citations | Figures / Tables |
|---------|------------|---------------------|-------------------------|------------------|
|  |  |  |  |  |

## 5. Discussion

| Section | Subsection | Content Description | Key Sources / Citations | Figures / Tables |
|---------|------------|---------------------|-------------------------|------------------|
|  |  |  |  |  |

## 6. Conclusion

| Section | Subsection | Content Description | Key Sources / Citations | Figures / Tables |
|---------|------------|---------------------|-------------------------|------------------|
|  |  |  |  |  |


## Appendices
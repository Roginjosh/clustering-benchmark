# Clustering Benchmark

A benchmarking project for evaluating clustering algorithms across datasets with different size and dimensionality characteristics.

The project converts image datasets into numerical feature representations using a pretrained ResNet-50 model. These fixed feature datasets are then used to compare multiple clustering algorithms under identical input conditions.

## Project Overview

The project consists of two main phases:

### 1. Dataset and Feature Extraction

Three image datasets are processed using a frozen ImageNet-pretrained ResNet-50.

Features can be extracted from several stages of the network and converted into fixed-length vectors using global average pooling.

| ResNet-50 Stage | Feature Dimensions |
|---|---:|
| Stem | 64 |
| Layer 1 | 256 |
| Layer 2 | 512 |
| Layer 3 | 1024 |
| Layer 4 | 2048 |

The goal is to construct three numerical datasets representing different clustering regimes:

| Dataset Regime | Requirement |
|---|---|
| Standard Multivariate | ≥ 5,000 observations, ≥ 20 variables |
| Large-Scale | ≥ 2,000,000 observations, ≥ 10 variables |
| High-Dimensional | ≥ 500 observations, ≥ 700 variables |

The current candidate image datasets are:

- **CAMELYON16** — standard multivariate dataset
- **MINC** — large-scale dataset
- **DTD** — high-dimensional dataset

The exact feature extraction configuration for each dataset will be documented as the project develops.

### 2. Clustering

Once feature extraction is complete, the resulting numerical datasets will be held fixed and used as common inputs for all clustering methods.

Each clustering implementation should operate on the same feature matrices so differences in results can be attributed to the clustering methods rather than different preprocessing pipelines.

Clustering methods will be implemented independently by project team members and evaluated using a common benchmarking framework.

## Pipeline

```text
Image Dataset
     |
     v
Dataset Preprocessing
     |
     v
Frozen ResNet-50
     |
     v
Intermediate Feature Map
     |
     v
Global Average Pooling
     |
     v
Numerical Feature Matrix
     |
     +-------------------+
     |         |         |
     v         v         v
Algorithm A Algorithm B Algorithm C
     |         |         |
     +---------+---------+
               |
               v
            Evaluation
```

## Repository Structure

```text
clustering-benchmark/
├── scripts/
│   └── test_resnet50.py
│
├── src/
│   ├── __init__.py
│   └── features/
│       ├── __init__.py
│       └── resnet50_extractor.py
│
├── test_images/
│   └── .gitkeep
│
├── .gitignore
└── README.md
```

The repository structure will expand as dataset loaders, clustering algorithms, and evaluation tools are added.

## Setup

Clone the repository and create a Python virtual environment:

```bash
git clone <repository-url>
cd clustering-benchmark

python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install torch torchvision pillow
```

## ResNet-50 Feature Extractor

The current feature extractor uses the pretrained ResNet-50 implementation provided by `torchvision`.

The model:

- Uses ImageNet pretrained weights
- Runs in evaluation mode
- Has all model parameters frozen
- Uses the preprocessing associated with the pretrained weights
- Supports CUDA, Apple MPS, and CPU
- Can extract representations from multiple ResNet-50 stages

To test the extractor, place a JPG or PNG image in:

```text
test_images/
```

Then run:

```bash
python -m scripts.test_resnet50
```

The test should produce representations with dimensions:

```text
stem      64
layer1   256
layer2   512
layer3  1024
layer4  2048
```

## Collaboration

Development should be performed on feature branches rather than directly on `main`.

Example:

```bash
git switch -c feature/algorithm-name
```

After completing and testing a change:

```bash
git add .
git commit -m "Implement algorithm name"
git push -u origin feature/algorithm-name
```

Changes can then be merged into `main` through a pull request.

Large datasets, generated embeddings, virtual environments, and experiment outputs should **not** be committed to the repository.

## Status

The project is currently in the **dataset and feature extraction phase**.

Current work:

- [x] Initialize project structure
- [x] Create frozen ResNet-50 feature extractor
- [x] Support multiple ResNet-50 feature dimensions
- [ ] Validate feature extractor
- [ ] Implement DTD extraction
- [ ] Implement CAMELYON16 extraction
- [ ] Implement MINC extraction
- [ ] Finalize numerical dataset format
- [ ] Implement clustering algorithms
- [ ] Implement common evaluation framework
- [ ] Run benchmark experiments
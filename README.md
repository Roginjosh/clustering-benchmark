# Clustering Benchmark

A benchmarking project for evaluating clustering algorithms across numerical datasets with different size and dimensionality characteristics.

The project converts image datasets into numerical feature representations using a frozen pretrained ResNet-50. These fixed feature matrices are then used as common inputs for multiple clustering algorithms.

Ground-truth labels are retained separately and are used **only for evaluation**. They must not be provided to clustering algorithms during fitting or cluster assignment.

## Project Overview

The project consists of three main stages:

1. Image dataset preparation
2. Frozen ResNet-50 feature extraction
3. Clustering and evaluation

### Dataset Requirements

The project targets three numerical dataset regimes:

| Dataset Regime | Requirement | Planned Dataset |
|---|---:|---|
| Standard Multivariate | ≥ 5,000 observations, ≥ 20 variables | CAMELYON16 |
| Large-Scale | ≥ 2,000,000 observations, ≥ 10 variables | MINC |
| High-Dimensional | ≥ 500 observations, ≥ 700 variables | DTD |

Because these datasets come from different image domains, results should be interpreted as benchmarking across different dataset regimes rather than as a controlled experiment isolating dimensionality or dataset size.

## ResNet-50 Feature Extraction

All image datasets use the same frozen ImageNet-pretrained ResNet-50 architecture provided by `torchvision`.

The preprocessing associated with `ResNet50_Weights.DEFAULT` is used.

Representations can be extracted from five stages of the network:

| ResNet-50 Stage | Feature Dimensions |
|---|---:|
| Stem | 64 |
| Layer 1 | 256 |
| Layer 2 | 512 |
| Layer 3 | 1024 |
| Layer 4 | 2048 |

Spatial feature maps are converted into fixed-length vectors using global average pooling.

Multiple representations can be extracted during the same forward pass.

The extractor:

- Uses ImageNet-pretrained ResNet-50 weights
- Runs in evaluation mode
- Keeps all model parameters frozen
- Uses the preprocessing associated with the pretrained weights
- Supports CUDA, Apple MPS, and CPU
- Supports extraction from multiple network depths
- Writes large feature matrices directly to disk using NumPy memory-mapped `.npy` files

The disk-backed extraction pipeline allows memory usage to scale primarily with batch size rather than total dataset size.

## Pipeline

```text
Image Dataset
     |
     v
Dataset Loader
     |
     v
ResNet-50 Preprocessing
     |
     v
Frozen ResNet-50
     |
     +-------> Stem ------> Global Average Pooling ---> 64D
     |
     +-------> Layer 1 ---> Global Average Pooling ---> 256D
     |
     +-------> Layer 2 ---> Global Average Pooling ---> 512D
     |
     +-------> Layer 3 ---> Global Average Pooling ---> 1024D
     |
     +-------> Layer 4 ---> Global Average Pooling ---> 2048D
                              |
                              v
                    Numerical Feature Matrix
                              |
                +-------------+-------------+
                |             |             |
                v             v             v
          Algorithm A   Algorithm B   Algorithm C
                |             |             |
                +-------------+-------------+
                              |
                              v
                          Evaluation
                              ^
                              |
                    Ground-Truth Labels
                    (evaluation only)
```

## Standard Dataset Interface

Image dataset loaders use the common interface:

```python
image, sample_id, label
```

where:

- `image` is the preprocessed image tensor
- `sample_id` is a stable identifier for the observation
- `label` is the integer ground-truth class label

The extraction pipeline preserves row alignment between features and metadata.

For any observation `i`:

```text
features[i] <-> labels[i] <-> sample_ids[i]
```

Ground-truth labels must not be included in the feature matrix supplied to a clustering algorithm.

## Numerical Dataset Format

Generated datasets use the following general format:

```text
dataset/
├── stem.npy
├── layer1.npy
├── layer2.npy
├── layer3.npy
├── layer4.npy
├── labels.npy
├── sample_ids.npy
└── classes.json
```

Not every dataset is required to contain every ResNet representation.

Feature matrices use `float32`.

The general clustering workflow is:

```python
import numpy as np

X = np.load("dataset/layer4.npy")
y_true = np.load("dataset/labels.npy")

labels_pred = cluster(X, n_clusters=...)

# y_true is used only after clustering for evaluation.
```

## Available Numerical Datasets

### DTD — Describable Textures Dataset

DTD extraction is complete.

| Property | Value |
|---|---:|
| Observations | 5,640 |
| Ground-truth classes | 47 |
| Samples per class | 120 |

Available representations:

| File | Shape |
|---|---:|
| `stem.npy` | `(5640, 64)` |
| `layer1.npy` | `(5640, 256)` |
| `layer2.npy` | `(5640, 512)` |
| `layer3.npy` | `(5640, 1024)` |
| `layer4.npy` | `(5640, 2048)` |

`layer3.npy` and `layer4.npy` satisfy the project's high-dimensional requirement of at least 700 variables.

The DTD numerical dataset is distributed as the GitHub release:

**DTD ResNet-50 Embeddings v1**

Release tag:

```text
dtd-resnet50-v1
```

With the GitHub CLI installed, download it using:

```bash
gh release download dtd-resnet50-v1
tar -xzf dtd-resnet50-v1.tar.gz
```

The extracted directory contains:

```text
dtd/
├── stem.npy
├── layer1.npy
├── layer2.npy
├── layer3.npy
├── layer4.npy
├── labels.npy
├── sample_ids.npy
├── classes.json
└── README.txt
```

`classes.json` maps numeric ground-truth labels to DTD class names.

The included `README.txt` contains additional information about the distributed dataset.

## Clustering Interface

Clustering implementations should operate only on numerical feature matrices.

The common API will approximately follow:

```python
def cluster(X, n_clusters, **kwargs):
    ...
    return labels
```

where:

- `X` is an `n_samples × n_features` numerical feature matrix
- `n_clusters` is the requested number of clusters when required by the algorithm
- the returned array contains the predicted cluster assignment for each observation

Ground-truth labels must **not** be passed to the clustering method.

Each team member can implement a clustering method independently while using the same numerical feature matrices.

## Repository Structure

The repository currently follows approximately:

```text
clustering-benchmark/
├── scripts/
│   ├── extract_dtd.py
│   ├── test_extraction_pipeline.py
│   └── test_resnet50.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── dtd.py
│   │
│   └── extraction/
│       ├── __init__.py
│       ├── resnet50.py
│       └── runner.py
│
├── test_images/
│
├── data/                   # ignored by Git
├── outputs/                # ignored by Git
│
├── .gitignore
└── README.md
```

Additional dataset loaders, clustering algorithms, and evaluation tools will be added as the project develops.

## Setup

Clone the repository:

```bash
git clone <repository-url>
cd clustering-benchmark
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install torch torchvision pillow numpy
```

## Testing the ResNet-50 Extractor

To test the extractor, place one or more JPG or PNG images in:

```text
test_images/
```

Then run:

```bash
python -m scripts.test_resnet50
```

Expected feature dimensions are:

```text
stem      64
layer1   256
layer2   512
layer3  1024
layer4  2048
```

The complete disk-backed extraction pipeline can also be tested with:

```bash
python -m scripts.test_extraction_pipeline
```

## Collaboration

Development should be performed on feature branches rather than directly on `main`.

Create a branch:

```bash
git switch -c feature/algorithm-name
```

After implementing and testing a change:

```bash
git add .
git commit -m "Implement algorithm name"
git push -u origin feature/algorithm-name
```

Changes can then be merged into `main` through a pull request.

Large datasets, generated embeddings, model files, virtual environments, experiment outputs, and generated dataset archives should **not** be committed to the repository.

Precomputed numerical datasets should instead be distributed separately, such as through GitHub Releases.

## Status

The project is currently in the **dataset extraction and clustering implementation phase**.

Current progress:

- [x] Initialize project structure
- [x] Create frozen ResNet-50 feature extractor
- [x] Support stem, layer1, layer2, layer3, and layer4 representations
- [x] Validate ResNet-50 feature extractor
- [x] Implement scalable disk-backed feature extraction
- [x] Validate end-to-end extraction pipeline
- [x] Implement DTD dataset loader
- [x] Extract all five DTD representations
- [x] Validate DTD numerical datasets
- [x] Publish DTD numerical dataset for team use
- [ ] Implement CAMELYON16 extraction
- [ ] Implement MINC extraction
- [ ] Implement clustering algorithms
- [ ] Implement common evaluation framework
- [ ] Run benchmark experiments
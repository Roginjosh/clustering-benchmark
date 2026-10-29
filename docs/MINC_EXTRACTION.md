# MINC-3M Feature Extraction

## Overview

This document describes the preparation and feature extraction pipeline for the MINC-3M dataset used in the clustering benchmark project.

The objective is to construct large-scale numerical datasets suitable for comparing K-means, Gaussian Mixture Models (GMM), Self-Organizing Maps (SOM), and Gaussian Cluster Embedding in Autoencoder Latent Space (G-CEALS).

A frozen, ImageNet-pretrained ResNet-50 is used to convert image patches into fixed-length numerical feature vectors. Four intermediate representations are extracted to provide datasets with different feature dimensionalities.

The resulting MINC datasets contain **2,000,000 observations across 23 material classes**, with feature dimensions of 64, 256, 512, and 1024.

## 1. Source Dataset

**Dataset:** Materials in Context Database (MINC-3M)

**Local dataset directory:** `~/datasets/MINC/minc-3m-224`

The dataset consists of 224 × 224 RGB image patches organized into training, validation, and test splits.

Only the original training split is used for the clustering benchmark.

| Property | Value |
|---|---:|
| Available training patches | 2,517,205 |
| Selected training patches | 2,000,000 |
| Material classes | 23 |
| Input image resolution | 224 × 224 |
| Original validation and test splits | Not included |

Each image patch constitutes one observation in the numerical clustering datasets.

Multiple patches may originate from the same source photograph. Consequently, the 2,000,000 observations should not be interpreted as 2,000,000 independent photographs.

## 2. Sample Selection

The sample manifest is generated using:

```bash
python -m scripts.build_minc_manifest
```

The manifest-generation procedure:

1. Enumerates the available image patches in the training split.
2. Associates each patch with its material class.
3. Sorts the image paths to establish a deterministic initial ordering.
4. Shuffles the observations using Python's pseudorandom generator with seed 42.
5. Selects the first 2,000,000 observations without replacement.
6. Writes the selected observations to a CSV manifest.

Sampling is not stratified. Therefore, the selected dataset approximately preserves the natural class proportions of the original training split.

### Manifest

**Path:** `data/minc/minc_2m_manifest.csv`

**Columns:**

| Column | Description |
|---|---|
| `index` | Sequential observation index |
| `path` | Image path relative to the dataset root |
| `label` | Integer class label (0–22) |
| `class_name` | Material class name |

The manifest contains exactly 2,000,000 observations and was verified to contain no duplicate image paths.

### Manifest integrity

SHA-256:

```text
1aa8a69aaf48090d221a1fab7ac63aa15e37e31873baaf1231b6f1d0e1d6c19f
```

To verify a regenerated or transferred manifest:

```bash
shasum -a 256 data/minc/minc_2m_manifest.csv
```

An identical checksum confirms byte-for-byte equality with the manifest used for the original extraction.

## 3. Class Distribution

The sampled dataset contains all 23 material classes.

| Label | Class | Samples |
|---:|---|---:|
| 0 | brick | 42,669 |
| 1 | carpet | 66,575 |
| 2 | ceramic | 18,693 |
| 3 | fabric | 264,911 |
| 4 | foliage | 80,984 |
| 5 | food | 16,888 |
| 6 | glass | 144,639 |
| 7 | hair | 17,231 |
| 8 | leather | 56,015 |
| 9 | metal | 126,062 |
| 10 | mirror | 50,159 |
| 11 | other | 26,543 |
| 12 | painted | 311,360 |
| 13 | paper | 15,863 |
| 14 | plastic | 26,096 |
| 15 | polishedstone | 76,704 |
| 16 | skin | 23,213 |
| 17 | sky | 94,359 |
| 18 | stone | 19,683 |
| 19 | tile | 98,565 |
| 20 | wallpaper | 9,341 |
| 21 | water | 35,780 |
| 22 | wood | 377,667 |

Class imbalance is retained rather than corrected through oversampling or undersampling.

## 4. Feature Extraction

### Model

A ResNet-50 pretrained on ImageNet is used as a fixed feature extractor.

The model is placed in evaluation mode, and its pretrained parameters are frozen. No fine-tuning or additional training is performed on MINC.

Images are processed using the preprocessing transforms associated with the selected torchvision ResNet-50 pretrained weights.

### Feature representations

Features are extracted from four stages of ResNet-50. Spatial activations are converted to fixed-length vectors using global average pooling.

| Representation | Feature dimensions |
|---|---:|
| Stem | 64 |
| Layer 1 | 256 |
| Layer 2 | 512 |
| Layer 3 | 1024 |

Layer 4 (2048 dimensions) is excluded from the MINC extraction to reduce storage requirements.

All four representations are derived from the same images in the same manifest order, allowing clustering methods to be compared across consistent observation sets.

**Important limitation:** ResNet stages differ in both feature dimensionality and learned representation. Differences in clustering performance cannot therefore be attributed exclusively to dimensionality.

### Extraction configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch / torchvision |
| Architecture | ResNet-50 |
| Weights | `ResNet50_Weights.DEFAULT` |
| Training during extraction | None |
| Batch size | 16 |
| DataLoader workers | 0 |
| Device | Apple Metal Performance Shaders (MPS) |
| Output dtype | float32 |
| Dataset ordering | Manifest order; no DataLoader shuffling |

The shared extraction implementation is located in:

- `src/extraction/resnet50.py`
- `src/extraction/runner.py`

The MINC-specific dataset loader and extraction entry point are:

- `src/datasets/minc.py`
- `scripts/extract_minc.py`

## 5. Running the Extraction

### Smoke test

Before performing the complete extraction, run:

```bash
python -m scripts.extract_minc --limit 1000
```

This extracts the first 1,000 observations from the manifest and writes the results to:

```text
outputs/minc_smoke/
```

The smoke test uses the same model, preprocessing, feature extraction implementation, and output-writing procedure as the full extraction.

### Full extraction

Run:

```bash
python -m scripts.extract_minc
```

The full extraction processes all 2,000,000 manifest observations.

For unattended execution on macOS:

```bash
mkdir -p logs
nohup python -u -m scripts.extract_minc > logs/minc_extraction.log 2>&1 &
```

Progress can be inspected using:

```bash
tail -20 logs/minc_extraction.log
```

## 6. Output Files

The complete extraction produces:

```text
outputs/minc/
├── stem.npy
├── layer1.npy
├── layer2.npy
├── layer3.npy
├── labels.npy
├── sample_ids.npy
└── classes.json
```

### Feature arrays

| File | Shape | Approximate size |
|---|---|---:|
| `stem.npy` | (2,000,000, 64) | 0.48 GiB |
| `layer1.npy` | (2,000,000, 256) | 1.91 GiB |
| `layer2.npy` | (2,000,000, 512) | 3.81 GiB |
| `layer3.npy` | (2,000,000, 1024) | 7.63 GiB |

Total feature storage is approximately 13.83 GiB, excluding labels, sample IDs, and metadata.

Feature matrices are stored in NumPy `.npy` format as float32 arrays.

The extraction runner uses NumPy memory-mapped output files to avoid holding the complete feature matrices in RAM during extraction.

### Labels and identifiers

- `labels.npy`: Integer material-class labels in manifest order.
- `sample_ids.npy`: Relative image paths corresponding to each observation.
- `classes.json`: Mapping between class indices and material names.

The manifest provides the canonical sample ordering.

## 7. Validation

The completed extraction was validated on October 28, 2026.

### Extraction completion

The extraction log confirmed that all 2,000,000 observations were processed and that the four feature arrays, labels, sample identifiers, and class mapping were saved.

### Numerical validation

All four feature arrays were loaded using NumPy memory mapping and checked in chunks of 10,000 observations.

The validation confirmed:

- Every feature matrix has the expected shape.
- Every feature matrix uses float32 values.
- Every feature value is finite (no NaN or infinity).
- The label array contains 2,000,000 entries.
- All labels are within the valid range of 0–22.
- All 23 material classes are represented.
- The sample ID array contains 2,000,000 entries.

**Result: VALIDATION PASSED**

This validation establishes numerical integrity and expected array structure. It does not independently establish clustering quality, verify every source image, or guarantee that every identifier matches the manifest.

## 8. Benchmarking Considerations

The four MINC feature datasets are intended to support controlled comparisons of clustering methods under increasing dataset size and feature dimensionality.

Potential experimental sample counts include:

- 5,000 observations
- 20,000 observations
- 100,000 observations
- 500,000 observations
- 2,000,000 observations

These are proposed experimental scales, not completed clustering experiments.

For reproducibility, experimental subsets should be selected deterministically and reused across algorithms when comparing methods at the same sample count.

Clustering evaluations should record:

- Clustering quality using external labels, such as ARI and NMI.
- Wall-clock execution time.
- Peak memory consumption where measurable.
- Independent random seeds and repeated-run variation.
- Algorithm configurations and convergence criteria.

All methods should receive the same numerical inputs for a given dataset configuration. Any normalization, dimensionality reduction, or other preprocessing must be documented and applied consistently where appropriate.

The benchmark does not assume that all four clustering methods will be computationally feasible at the maximum dataset size. Infeasible configurations should be reported explicitly rather than silently omitted.

## 9. Reproducibility and Repository Policy

The source code for manifest generation, dataset loading, and feature extraction is version-controlled in the project repository.

The original MINC images, generated manifest, and extracted feature arrays are not committed to Git because of their size.

The manifest SHA-256 checksum is provided above to identify the exact sample selection used.

The final extraction configuration was committed on branch `feature/minc-extraction` in commit:

```text
d806273
```

For exact reproduction, future work should also record the Python, PyTorch, torchvision, and NumPy versions, as well as the resolved pretrained weight identifier and model preprocessing configuration.

## 10. Current Status

As of October 28, 2026:

- MINC training patches indexed: complete.
- Two-million-sample manifest generated: complete.
- Manifest uniqueness verified: complete.
- ResNet-50 extraction completed: complete.
- Four feature representations saved: complete.
- Numerical validation: passed.
- Clustering experiments: pending.

The MINC numerical datasets are ready for clustering benchmark preparation and experimental evaluation.

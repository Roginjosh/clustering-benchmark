Clustering Benchmark

A benchmarking project evaluating clustering algorithms across numerical datasets with different observation counts, feature dimensionalities, and underlying data characteristics.

The project converts image datasets into numerical feature representations using a frozen, pretrained ResNet-50. These fixed feature matrices serve as common inputs for multiple clustering algorithms, allowing comparisons of clustering quality, runtime, memory consumption, and computational scalability.

The benchmark evaluates five clustering approaches:

1. K-means
2. Gaussian Mixture Models (GMM)
3. Self-Organizing Maps (SOM)
4. Gaussian Cluster Embedding in Autoencoder Latent Space (G-CEALS)
5. Autoencoder + K-means (AE + K-means)

The primary benchmark uses four clustering algorithms across three datasets. AE + K-means is included as an additional deep-representation baseline on DTD.

Ground-truth labels are retained separately and used only for evaluation. They are never supplied to clustering algorithms during fitting or cluster assignment.

Project Overview

The project investigates the following research question:

How do clustering quality and computational cost differ among clustering algorithms when applied to numerical datasets with different numbers of observations and feature dimensions?

Two additional questions guide the analysis:

* Does the preferred clustering algorithm change across different numerical dataset regimes?
* When a computationally expensive clustering algorithm improves clustering quality, is the improvement sufficient to justify its additional runtime and memory requirements?

The project consists of four main stages:

1. Image dataset preparation
2. Frozen ResNet-50 feature extraction
3. Clustering and evaluation
4. Benchmark aggregation and analysis

All four stages have been implemented.

Dataset Requirements

The project targets three numerical dataset regimes:

Dataset Regime	Requirement	Dataset	Primary Configuration
Standard Multivariate	≥ 5,000 observations, ≥ 20 variables	CAMELYON16	8,580 × 64
Large-Scale	≥ 2,000,000 observations, ≥ 10 variables	MINC	2,000,000 × 64
High-Dimensional	≥ 500 observations, ≥ 700 variables	DTD	5,640 × 2,048

The numerical datasets originate from different image domains. Consequently, cross-dataset comparisons should be interpreted as empirical benchmarking rather than controlled experiments isolating dataset size or dimensionality.

Comparisons between clustering algorithms operating on identical numerical feature matrices provide the strongest basis for evaluating relative performance.

ResNet-50 Feature Extraction

All three image datasets use the same frozen ImageNet-pretrained ResNet-50 architecture provided by torchvision.

The preprocessing associated with ResNet50_Weights.DEFAULT is used.

Representations can be extracted from five stages of the network:

ResNet-50 Stage	Feature Dimensions
Stem	64
Layer 1	256
Layer 2	512
Layer 3	1,024
Layer 4	2,048

Spatial feature maps are converted into fixed-length vectors using global average pooling.

Multiple representations can be extracted during the same forward pass.

The feature extraction pipeline:

* Uses ImageNet-pretrained ResNet-50 weights
* Runs in evaluation mode
* Keeps all model parameters frozen
* Uses the preprocessing associated with the pretrained weights
* Supports CUDA, Apple MPS, and CPU
* Supports extraction from multiple network depths
* Writes large feature matrices directly to disk using NumPy memory-mapped .npy files
* Preserves row alignment between features, labels, and sample identifiers

The disk-backed extraction pipeline allows memory usage to scale primarily with batch size rather than total dataset size.

Pipeline

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
             K-means         GMM            SOM
                |             |             |
                +-------------+-------------+
                              |
                    +---------+---------+
                    |                   |
                    v                   v
                  G-CEALS         AE + K-means
                    |                   |
                    +---------+---------+
                              |
                              v
                          Evaluation
                              ^
                              |
                    Ground-Truth Labels
                    (evaluation only)

Standard Dataset Interface

Image dataset loaders use the common interface:

image, sample_id, label

where:

* image is the preprocessed image tensor
* sample_id is a stable identifier for the observation
* label is the integer ground-truth class label

For any observation i:

features[i] <-> labels[i] <-> sample_ids[i]

Ground-truth labels must not be included in the feature matrix supplied to a clustering algorithm.

Numerical Dataset Format

Generated datasets generally use the following structure:

dataset/
├── stem.npy
├── layer1.npy
├── layer2.npy
├── layer3.npy
├── layer4.npy
├── labels.npy
├── sample_ids.npy
└── classes.json

Not every dataset contains every ResNet representation.

Feature matrices are stored using float32.

A typical clustering workflow is:

import numpy as np
X = np.load("dataset/layer4.npy")
y_true = np.load("dataset/labels.npy")
labels_pred = cluster(X, n_clusters=...)
# Ground-truth labels are used only after clustering.

Available Numerical Datasets

DTD — Describable Textures Dataset

DTD provides the high-dimensional benchmark condition.

Property	Value
Observations	5,640
Ground-truth classes	47
Samples per class	120
Primary feature dimensions	2,048

Available representations:

File	Shape
stem.npy	(5640, 64)
layer1.npy	(5640, 256)
layer2.npy	(5640, 512)
layer3.npy	(5640, 1024)
layer4.npy	(5640, 2048)

The 1,024-dimensional and 2,048-dimensional representations satisfy the project’s high-dimensional requirement.

The primary benchmark compares K-means, GMM, SOM, and G-CEALS using the stem and layer4 representations.

An additional AE + K-means experiment uses the 2,048-dimensional layer4 representation.

Dataset Distribution

The DTD numerical dataset is available through the GitHub release:

DTD ResNet-50 Embeddings v1

Release tag:

dtd-resnet50-v1

Download using the GitHub CLI:

gh release download dtd-resnet50-v1
tar -xzf dtd-resnet50-v1.tar.gz

The extracted directory contains the numerical features, labels, sample identifiers, class metadata, and dataset documentation.

CAMELYON16

CAMELYON16 provides the standard multivariate benchmark condition.

The dataset consists of image patches extracted from histopathology whole-slide images.

Property	Value
Primary benchmark observations	8,580
Ground-truth categories	3
Primary feature dimensions	64
Feature representation	ResNet-50 stem
Source whole-slide images	110

The three sampling categories are:

Category	Observations
Exterior	3,376
Boundary	3,123
Interior	2,081
Total	8,580

The primary experiment evaluates K-means, GMM, SOM, and G-CEALS on the same 8,580-observation numerical feature matrix.

Additional K-means experiments were performed using 3,728 observations and multiple ResNet representations:

* Stem: 64 dimensions
* Layer1: 256 dimensions
* Layer2: 512 dimensions
* Layer3: 1,024 dimensions
* Layer4: 2,048 dimensions

These additional experiments are retained separately from the primary 8,580-observation comparison.

MINC — Materials in Context

MINC provides the large-scale benchmark condition.

Property	Value
Maximum observations	2,000,000
Ground-truth classes	23
Primary feature dimensions	64
Feature representation	ResNet-50 stem

A reproducible manifest containing 2,000,000 observations was constructed using a fixed random seed.

The benchmark uses deterministic nested subsets of the same numerical dataset.

Four observation counts are evaluated:

Observations	Feature Dimensions	Target Clusters
5,000	64	23
50,000	64	23
500,000	64	23
2,000,000	64	23

This design allows computational scaling to be investigated while holding the source dataset, feature representation, and number of target clusters fixed.

Completed MINC Experiments

Algorithm	5k	50k	500k	2M
K-means	Complete	Complete	Complete	Complete
GMM	Complete	Complete	Complete	Complete
SOM	Complete	Complete	Complete	Complete
G-CEALS	Complete	Complete	Complete	Not completed

Each completed condition contains five successful independent runs.

G-CEALS was evaluated through 500,000 observations. A successful full-scale G-CEALS experiment at 2,000,000 observations is not available.

Clustering Algorithms

K-means

K-means is the centroid-based clustering baseline.

The algorithm partitions observations by minimizing within-cluster squared Euclidean distance.

The number of clusters is set equal to the number of known ground-truth classes, without supplying individual ground-truth labels during fitting.

The primary benchmark uses K-means++ initialization and independent random seeds.

Additional experiments investigate different initialization counts.

Gaussian Mixture Models

GMM provides a probabilistic clustering baseline.

The benchmark uses Gaussian mixture components with diagonal covariance matrices.

The number of mixture components is set equal to the number of ground-truth classes.

A covariance regularization parameter is used to improve numerical stability.

Self-Organizing Maps

SOM provides a neural, topology-preserving clustering approach.

The benchmark uses a two-dimensional grid of prototype units.

After SOM training, the learned prototypes are grouped into the required number of clusters using K-means.

Each observation receives the final cluster assignment associated with its best-matching prototype.

The final DTD experiments use 10,000 SOM training iterations.

Earlier DTD experiments using 100,000 iterations are retained as separate experimental results and are not mixed into the final five-seed summaries.

G-CEALS

Gaussian Cluster Embedding in Autoencoder Latent Space is a deep clustering algorithm proposed by Rabbani, Medri, and Samad.

G-CEALS combines:

* Autoencoder-based representation learning
* Gaussian cluster modeling in latent space
* Learned cluster weights
* Joint reconstruction and clustering optimization

The algorithm first learns a compressed representation of the input data through autoencoder pretraining.

It subsequently optimizes the representation and cluster assignments jointly.

G-CEALS is evaluated on DTD, CAMELYON16, and multiple MINC observation counts.

AE + K-means

AE + K-means provides an additional baseline for evaluating the effect of learned representations.

An autoencoder is pretrained to compress the original numerical features into a lower-dimensional latent space.

K-means is then applied to the learned latent representations.

Unlike G-CEALS, the baseline does not jointly optimize the representation using a clustering objective.

The DTD configuration uses:

Parameter	Value
Input observations	5,640
Input dimensions	2,048
Latent dimensions	10
Target clusters	47
Pretraining epochs	100
K-means initializations	10
Independent runs	5

The autoencoder is pretrained independently for each experiment.

This comparison was added to evaluate whether G-CEALS provides a benefit beyond learning a compressed representation and subsequently applying K-means.

Experimental Methodology

Preprocessing

Clustering inputs are standardized using StandardScaler.

Each feature is transformed to have zero mean and unit variance.

The transformation is fitted to the numerical feature matrix associated with the experimental condition.

Ground-truth labels are not used during preprocessing or clustering.

Random Seeds

The primary repeated-run experiments use five independent random seeds:

0, 1, 2, 3, 4

Some supplementary K-means experiments use different documented seed configurations.

Repeated runs are used to measure variation in clustering quality and computational requirements.

Evaluation Metrics

Clustering quality is evaluated using:

Metric	Description
ARI	Adjusted Rand Index
NMI	Normalized Mutual Information
AMI	Adjusted Mutual Information
Homogeneity	Measures whether clusters contain observations from a single class
Completeness	Measures whether observations from the same class are assigned together
V-measure	Harmonic mean of homogeneity and completeness

ARI, NMI, and AMI are the primary metrics reported in the benchmark comparisons.

Additional metrics are recorded where available.

Computational performance is evaluated using:

Metric	Description
Runtime	Time required to execute the clustering procedure
Peak RSS	Peak resident memory usage of the process

Runtime includes method-specific training and clustering operations.

The common ResNet-50 feature extraction stage is performed separately and is not included in the clustering runtime.

Peak RSS measures the process-wide memory footprint rather than memory allocated exclusively by the clustering algorithm.

Experimental Comparability

Comparisons are most meaningful when algorithms use:

* The same observations
* The same feature representation
* The same preprocessing
* The same target number of clusters
* Documented experimental configurations

Different feature representations may contain different semantic information in addition to having different dimensionalities.

Therefore, observed performance differences across ResNet layers cannot be attributed solely to dimensionality.

Benchmark Results

The consolidated benchmark currently contains:

Dataset	Successful Runs
DTD	45
CAMELYON16	35
MINC	75
Total	155

These results cover 33 experimental configurations.

All 155 included runs completed successfully.

DTD Results

The following table summarizes the primary 2,048-dimensional DTD comparison.

Values are means across five independent runs.

Algorithm	Mean ARI	Mean NMI	Mean AMI
K-means	0.0979	0.3878	0.3627
GMM	0.1394	0.4005	0.3770
SOM	0.0157	0.2184	0.1764
G-CEALS	0.1071	0.3649	0.3282

GMM achieved the highest mean ARI, NMI, and AMI in the primary high-dimensional DTD comparison.

G-CEALS did not outperform GMM in mean clustering quality, despite requiring substantially more computation.

AE + K-means results are included in the consolidated benchmark and provide an additional representation-learning comparison.

CAMELYON16 Results

The primary CAMELYON16 experiment uses 8,580 observations and 64-dimensional stem features.

Algorithm	Mean ARI	Mean NMI	Mean AMI
K-means	0.0804	0.1145	0.1143
GMM	0.0785	0.1083	0.1081
SOM	0.0739	0.1079	0.1077
G-CEALS	0.0203	0.0285	0.0283

K-means achieved the highest mean clustering quality in the primary CAMELYON16 comparison.

MINC Results

The primary large-scale MINC experiment uses 2,000,000 observations and 64-dimensional stem features.

Algorithm	Mean ARI	Mean NMI	Mean AMI
K-means	0.0640	0.1252	0.1252
GMM	0.0399	0.1185	0.1184
SOM	0.0736	0.1227	0.1227

SOM achieved the highest mean ARI on the full MINC dataset.

K-means achieved the highest mean NMI and AMI while requiring less runtime.

Additional MINC experiments at 5,000, 50,000, and 500,000 observations are included in the consolidated results.

These experiments allow clustering quality, runtime, and memory usage to be analyzed as observation count increases.

Main Findings

The completed primary benchmark demonstrates that the highest-performing algorithm changes across numerical dataset conditions.

Dataset	Highest Mean ARI
DTD, 2,048D	GMM
CAMELYON16, 64D	K-means
MINC, 2M observations	SOM

No single algorithm achieved the highest mean ARI across all three primary benchmark conditions.

The experiments also demonstrate that increased computational complexity does not necessarily produce improved clustering quality.

These findings are specific to the numerical datasets and configurations evaluated and should not be interpreted as universal rankings of the algorithms.

Consolidated Benchmark Results

Individual benchmark runs are stored as CSV files.

The consolidation script combines the selected experimental outputs into a common results dataset.

Run:

python3 scripts/collect_final_results.py

The script generates:

outputs/
└── final_results/
    ├── all_results.csv
    ├── dtd_results.csv
    ├── camelyon16_results.csv
    ├── minc_results.csv
    └── summary_results.csv

Output Files

File	Description
all_results.csv	All selected individual benchmark runs
dtd_results.csv	Individual DTD experiment results
camelyon16_results.csv	Individual CAMELYON16 experiment results
minc_results.csv	Individual MINC experiment results
summary_results.csv	Aggregated statistics for experimental configurations

The summary file contains mean and standard deviation statistics calculated across successful runs.

Experimental configurations are kept separate when they differ in relevant settings, such as dataset size, feature representation, or algorithm parameters.

Result Validation

The consolidated results have been compared against the original experiment summaries used in the project manuscript.

Validated comparisons include:

* DTD K-means, GMM, and G-CEALS
* DTD SOM using the final 10,000-iteration configuration
* CAMELYON16 primary four-algorithm comparison
* MINC K-means, GMM, and SOM at 2,000,000 observations
* MINC G-CEALS at 5,000 observations

Additional MINC scaling experiments and AE + K-means results are included in the consolidated files.

Their successful-run counts have been verified, although their metric values have not all been independently cross-checked against separate summary files.

Sharing Results

The outputs/ directory is excluded from Git tracking because it may contain large generated files, embeddings, and experimental artifacts.

Consequently, the consolidated CSV files are not automatically included when committing the consolidation script.

To share the results, distribute the consolidated CSV files separately or publish them as a versioned GitHub release.

The consolidation script can be committed to the repository so that results can be regenerated from the original experiment outputs.

Repository Structure

The repository contains dataset loaders, feature extraction utilities, clustering implementations, benchmark scripts, and experiment outputs.

A simplified overview is:

clustering-benchmark/
├── scripts/
│   ├── extract_dtd.py
│   ├── test_extraction_pipeline.py
│   ├── test_resnet50.py
│   ├── collect_final_results.py
│   ├── experiments/
│   │   └── ae_kmeans.py
│   └── ...
│
├── src/
│   ├── datasets/
│   ├── extraction/
│   ├── clustering/
│   └── ...
│
├── data/                   # Ignored by Git
│
├── outputs/                # Ignored by Git
│   ├── benchmark/
│   └── final_results/
│
├── test_images/
├── .gitignore
└── README.md

The diagram is illustrative rather than an exhaustive listing of repository files.

Setup

Clone the repository:

git clone <repository-url>
cd clustering-benchmark

Create and activate a Python virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install the project dependencies.

Core packages include:

pip install torch torchvision pillow numpy scipy scikit-learn pandas

Additional dependencies may be required for specific clustering implementations or dataset preparation scripts.

Hardware Support

The extraction pipeline supports:

* NVIDIA CUDA
* Apple Metal Performance Shaders (MPS)
* CPU

The primary benchmark experiments were executed on an Apple Mac mini with an Apple M4 processor and 16 GB of unified memory.

Traditional clustering methods primarily execute on the CPU.

Deep clustering experiments use PyTorch and may use MPS acceleration.

Testing

ResNet-50 Extractor

Place one or more JPG or PNG images in:

test_images/

Run:

python -m scripts.test_resnet50

Expected feature dimensions:

stem      64
layer1   256
layer2   512
layer3  1024
layer4  2048

Disk-Backed Extraction

Run:

python -m scripts.test_extraction_pipeline

This tests the end-to-end extraction pipeline and verifies the generated numerical feature representations.

Collaboration

The project is developed collaboratively by:

* Joshua Rogge — K-means, feature extraction, and benchmark integration
* Keith Miller — G-CEALS
* Qiang Fu — SOM
* Robert Crissman — GMM

Development should generally occur on feature branches rather than directly on main.

Create a branch:

git switch -c feature/algorithm-name

After implementing and testing a change:

git add <files>
git commit -m "Describe implemented change"
git push -u origin feature/algorithm-name

Changes can then be merged into main through a pull request.

Large datasets, generated embeddings, model weights, virtual environments, and experimental outputs should not be committed directly to the repository.

Precomputed numerical datasets and finalized benchmark results should be distributed separately when appropriate.

Project Status

Current phase: Benchmark analysis and final reporting

Dataset Preparation

* Initialize project structure
* Implement frozen ResNet-50 feature extraction
* Support all five ResNet-50 representations
* Validate the extraction pipeline
* Implement disk-backed feature extraction
* Prepare and extract DTD
* Prepare and extract CAMELYON16
* Prepare and extract MINC
* Publish DTD numerical embeddings

Clustering Implementation

* Implement K-means
* Implement GMM
* Implement SOM
* Implement G-CEALS
* Implement AE + K-means
* Implement common evaluation metrics
* Implement benchmark result recording

Experiments

* Complete five-seed DTD benchmark
* Complete five-seed CAMELYON16 primary benchmark
* Complete MINC classical clustering at 2 million observations
* Complete MINC scaling experiments at 5k, 50k, and 500k
* Complete MINC G-CEALS experiments through 500k observations
* Complete DTD AE + K-means comparison
* Consolidate 155 successful benchmark runs
* Validate primary results against original experiment summaries

Final Reporting

* Complete independent validation of supplementary experiment summaries
* Generate final benchmark tables and figures
* Integrate AE + K-means and MINC scaling results into the manuscript
* Complete final results discussion and conclusions
* Finalize project presentation
* Publish or distribute consolidated benchmark results

References

The project is motivated in part by:

Rabbani, S. B., Medri, I. V., and Samad, M. D. (2025). Deep clustering of tabular data by weighted Gaussian distribution learning. Neurocomputing, 623, 129359.

The project extends the experimental comparison beyond the datasets used in the original G-CEALS study by evaluating clustering quality and computational performance across numerical representations derived from three image datasets.
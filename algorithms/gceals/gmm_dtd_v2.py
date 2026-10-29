#!/usr/bin/env python3
"""
gmm_dtd_beginner.py
===================

Beginner-friendly Gaussian Mixture Model (GMM) clustering script for:

    DTD ResNet-50 Embeddings v1

This script follows the README included with the official release.

README rules followed:
- Clustering uses ONLY one feature matrix (stem/layer1/layer2/layer3/layer4).
- labels.npy is NEVER given to the GMM during fitting or cluster assignment.
- labels.npy is used ONLY AFTER clustering for evaluation metrics.
- All rows stay aligned with sample_ids.npy and labels.npy.

DEFAULT EXPERIMENT
------------------
Feature matrix: layer4.npy
Shape:          5,640 samples x 2,048 variables
Clusters:       47
Covariance:     diagonal

The script can use either:
1. An EXTRACTED release folder containing:
       dtd/
           layer4.npy
           labels.npy
           sample_ids.npy
           classes.json
           ...

OR

2. The original:
       dtd-resnet50-v1.tar.gz

If this file is placed somewhere inside the same VS Code project as the
downloaded release, it will try to find the data automatically.

Python 3.14 install command:
    py -3.14 -m pip install -U numpy scipy scikit-learn matplotlib

FIRST TEST:
    py -3.14 gmm_dtd_beginner.py --check-only

NORMAL RUN:
    py -3.14 gmm_dtd_beginner.py

If auto-detection does not find your data:
    py -3.14 gmm_dtd_beginner.py --data "C:\\path\\to\\dtd"

or:
    py -3.14 gmm_dtd_beginner.py --data "C:\\path\\to\\dtd-resnet50-v1.tar.gz"
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ============================================================
# BEGINNER SETTINGS
# ============================================================
# You normally do NOT need to change anything below this line.
#
# layer3 and layer4 meet the course high-dimensional requirement
# because they contain >= 700 variables.
#
# layer4 is the default because it is the deepest supplied
# ResNet-50 representation.
FEATURE_LAYER = "layer4"

# The README says the DTD dataset has 47 ground-truth classes.
# Knowing the NUMBER of clusters is okay here; the actual labels
# are still not given to GMM.
NUMBER_OF_CLUSTERS = 47

# "diag" is much more practical than "full" for 2,048 features.
COVARIANCE_TYPE = "diag"

RANDOM_SEED = 42
N_INITIALIZATIONS = 3
MAX_ITERATIONS = 300

# Silhouette score is expensive, so evaluate it on at most this
# many randomly selected observations.
SILHOUETTE_SAMPLE_SIZE = 3000

VALID_LAYERS = ("stem", "layer1", "layer2", "layer3", "layer4")

REQUIRED_METADATA_FILES = (
    "labels.npy",
    "sample_ids.npy",
    "classes.json",
)


# ============================================================
# DATA LOCATION HELPERS
# ============================================================

def folder_is_dtd_data(folder: Path, layer: str) -> bool:
    """Return True if this folder looks like the extracted dtd data folder."""
    if not folder.is_dir():
        return False

    required = [
        folder / f"{layer}.npy",
        folder / "labels.npy",
        folder / "sample_ids.npy",
        folder / "classes.json",
    ]

    return all(path.is_file() for path in required)


def normalize_user_data_path(path: Path, layer: str) -> tuple[str, Path]:
    """
    Convert a user-supplied path into:
        ("folder", path_to_dtd_folder)
    or
        ("archive", path_to_tar_gz)
    """
    path = path.expanduser().resolve()

    if path.is_file():
        name = path.name.lower()

        if name.endswith(".tar.gz") or name.endswith(".tgz"):
            return "archive", path

        raise ValueError(
            f"\nThe --data path is a file, but it is not a .tar.gz archive:\n{path}"
        )

    if path.is_dir():
        # User pointed directly at the dtd folder.
        if folder_is_dtd_data(path, layer):
            return "folder", path

        # User pointed at a release/repository folder that contains dtd/.
        child = path / "dtd"

        if folder_is_dtd_data(child, layer):
            return "folder", child

        raise ValueError(
            "\nI found the folder you gave me, but I could not find the DTD "
            "release files inside it.\n\n"
            f"Path checked:\n{path}\n\n"
            "I expected either the files directly in that folder or inside "
            "a child folder named 'dtd'."
        )

    raise FileNotFoundError(
        f"\nThe --data path does not exist:\n{path}"
    )


def find_data_automatically(layer: str) -> tuple[str, Path] | None:
    """
    Look near the script and current VS Code working directory.

    We deliberately avoid crawling .git, virtual environments, and other
    large unrelated folders.
    """
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()

    roots = []
    for root in (script_dir, cwd):
        if root not in roots:
            roots.append(root)

    # 1. Very common/simple locations first.
    for root in roots:
        simple_folder_candidates = (
            root,
            root / "dtd",
            root / "dtd-resnet50-v1",
            root / "dtd-resnet50-v1" / "dtd",
            root / "data",
            root / "data" / "dtd",
        )

        for candidate in simple_folder_candidates:
            if folder_is_dtd_data(candidate, layer):
                return "folder", candidate

        simple_archive_candidates = (
            root / "dtd-resnet50-v1.tar.gz",
            root / "data" / "dtd-resnet50-v1.tar.gz",
        )

        for candidate in simple_archive_candidates:
            if candidate.is_file():
                return "archive", candidate

    # 2. Search a few directory levels inside the workspace.
    skip_names = {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        ".idea",
    }

    max_depth = 4

    for root in roots:
        root_parts = len(root.parts)

        try:
            for current, dirs, files in os.walk(root):
                current_path = Path(current)
                depth = len(current_path.parts) - root_parts

                # Do not search huge/unrelated folders.
                dirs[:] = [
                    d for d in dirs
                    if d not in skip_names and not d.startswith(".")
                ]

                if depth >= max_depth:
                    dirs[:] = []

                if folder_is_dtd_data(current_path, layer):
                    return "folder", current_path

                if "dtd-resnet50-v1.tar.gz" in files:
                    return "archive", current_path / "dtd-resnet50-v1.tar.gz"

        except PermissionError:
            # Ignore folders Windows does not let us inspect.
            continue

    return None


# ============================================================
# LOAD THE OFFICIAL RELEASE
# ============================================================

def load_npy_from_tar(
    tar: tarfile.TarFile,
    member_name: str,
) -> np.ndarray:
    """Read one .npy file directly from the tar.gz archive."""
    file_obj = tar.extractfile(member_name)

    if file_obj is None:
        raise FileNotFoundError(
            f"Could not find '{member_name}' inside the archive."
        )

    return np.load(
        io.BytesIO(file_obj.read()),
        allow_pickle=False,
    )


def find_tar_member_name(
    names: set[str],
    filename: str,
) -> str:
    """
    Find dtd/filename in the archive.

    The official release uses dtd/<filename>, but this also tolerates
    an archive with an extra top-level release directory.
    """
    preferred = f"dtd/{filename}"

    if preferred in names:
        return preferred

    matches = [
        name
        for name in names
        if name.endswith(f"/dtd/{filename}") or name == filename
    ]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"Could not find '{filename}' in the archive."
        )

    raise RuntimeError(
        f"More than one possible '{filename}' was found in the archive: "
        + ", ".join(matches)
    )


def load_from_archive(
    archive_path: Path,
    layer: str,
):
    """Load the official data directly from dtd-resnet50-v1.tar.gz."""
    print(f"\nReading archive:\n  {archive_path}")

    with tarfile.open(archive_path, "r:gz") as tar:
        names = set(tar.getnames())

        feature_member = find_tar_member_name(
            names,
            f"{layer}.npy",
        )
        labels_member = find_tar_member_name(
            names,
            "labels.npy",
        )
        sample_ids_member = find_tar_member_name(
            names,
            "sample_ids.npy",
        )
        classes_member = find_tar_member_name(
            names,
            "classes.json",
        )

        X = load_npy_from_tar(
            tar,
            feature_member,
        )

        labels_true = load_npy_from_tar(
            tar,
            labels_member,
        )

        sample_ids = load_npy_from_tar(
            tar,
            sample_ids_member,
        )

        classes_file = tar.extractfile(classes_member)

        if classes_file is None:
            raise FileNotFoundError(
                "Could not read classes.json from the archive."
            )

        classes = json.load(classes_file)

    return X, labels_true, sample_ids, classes


def load_from_folder(
    dtd_folder: Path,
    layer: str,
):
    """Load the extracted official release using normal np.load calls."""
    print(f"\nReading extracted DTD folder:\n  {dtd_folder}")

    X = np.load(
        dtd_folder / f"{layer}.npy",
        allow_pickle=False,
    )

    labels_true = np.load(
        dtd_folder / "labels.npy",
        allow_pickle=False,
    )

    sample_ids = np.load(
        dtd_folder / "sample_ids.npy",
        allow_pickle=False,
    )

    with (dtd_folder / "classes.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        classes = json.load(file)

    return X, labels_true, sample_ids, classes


def validate_release(
    X: np.ndarray,
    labels_true: np.ndarray,
    sample_ids: np.ndarray,
    classes: dict,
    layer: str,
):
    """Make sure the loaded release matches what the README describes."""
    expected_dimensions = {
        "stem": 64,
        "layer1": 256,
        "layer2": 512,
        "layer3": 1024,
        "layer4": 2048,
    }

    expected_features = expected_dimensions[layer]

    if X.ndim != 2:
        raise ValueError(
            f"{layer}.npy should be a 2-D matrix, but its shape is {X.shape}."
        )

    if X.shape[0] != 5640:
        raise ValueError(
            f"The README says there should be 5,640 observations, "
            f"but {layer}.npy contains {X.shape[0]}."
        )

    if X.shape[1] != expected_features:
        raise ValueError(
            f"The README says {layer}.npy should contain "
            f"{expected_features} variables, but I found {X.shape[1]}."
        )

    if labels_true.shape != (5640,):
        raise ValueError(
            f"labels.npy should have shape (5640,), "
            f"but I found {labels_true.shape}."
        )

    if sample_ids.shape != (5640,):
        raise ValueError(
            f"sample_ids.npy should have shape (5640,), "
            f"but I found {sample_ids.shape}."
        )

    if len(classes) != 47:
        raise ValueError(
            f"The README says there should be 47 classes, "
            f"but classes.json contains {len(classes)}."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{layer}.npy contains NaN or infinite values."
        )


# ============================================================
# GMM
# ============================================================

def fit_gmm(
    X: np.ndarray,
    n_clusters: int,
    covariance_type: str,
    random_seed: int,
    n_init: int,
    max_iterations: int,
):
    """
    Fit GMM ONLY on X.

    Notice that labels_true is not an argument to this function.
    That is intentional and follows the README.
    """
    print("\nSTEP 1 OF 3: Standardizing the feature matrix...")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("  Done.")
    print(f"  GMM sees ONLY X with shape {X_scaled.shape}.")
    print("  Ground-truth labels are NOT being passed to GMM.")

    print("\nSTEP 2 OF 3: Fitting Gaussian Mixture Model...")
    print(f"  Components:      {n_clusters}")
    print(f"  Covariance type: {covariance_type}")
    print(f"  Initializations: {n_init}")

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        reg_covar=1e-6,
        n_init=n_init,
        max_iter=max_iterations,
        random_state=random_seed,
        verbose=1,
    )

    start = time.perf_counter()

    labels_pred = gmm.fit_predict(X_scaled)

    runtime_seconds = time.perf_counter() - start

    probabilities = gmm.predict_proba(X_scaled)
    confidence = probabilities.max(axis=1)

    print("  Done.")

    return (
        scaler,
        X_scaled,
        gmm,
        labels_pred,
        confidence,
        runtime_seconds,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_after_clustering(
    X_scaled: np.ndarray,
    gmm: GaussianMixture,
    labels_pred: np.ndarray,
    labels_true: np.ndarray,
    silhouette_sample_size: int,
    random_seed: int,
):
    """
    Ground-truth labels FIRST appear here, after GMM has already finished.
    """
    print("\nSTEP 3 OF 3: Evaluating the finished clusters...")

    metrics = {
        "ARI": adjusted_rand_score(
            labels_true,
            labels_pred,
        ),
        "NMI": normalized_mutual_info_score(
            labels_true,
            labels_pred,
        ),
        "AMI": adjusted_mutual_info_score(
            labels_true,
            labels_pred,
        ),
        "AIC": gmm.aic(X_scaled),
        "BIC": gmm.bic(X_scaled),
    }

    unique_clusters = np.unique(labels_pred)

    if 1 < len(unique_clusters) < len(labels_pred):
        metrics["Silhouette"] = silhouette_score(
            X_scaled,
            labels_pred,
            sample_size=min(
                silhouette_sample_size,
                len(X_scaled),
            ),
            random_state=random_seed,
        )
    else:
        metrics["Silhouette"] = float("nan")

    print("  Done.")

    return metrics


def make_cluster_summary(
    labels_pred: np.ndarray,
    labels_true: np.ndarray,
    classes: dict,
):
    """
    Evaluation-only table that shows which real DTD class appears most
    frequently inside each predicted GMM cluster.
    """
    rows = []

    for cluster_id in sorted(np.unique(labels_pred)):
        mask = labels_pred == cluster_id
        cluster_true_labels = labels_true[mask]

        values, counts = np.unique(
            cluster_true_labels,
            return_counts=True,
        )

        best_index = int(np.argmax(counts))
        dominant_label = int(values[best_index])
        dominant_count = int(counts[best_index])
        cluster_size = int(mask.sum())

        dominant_class = classes.get(
            str(dominant_label),
            f"class_{dominant_label}",
        )

        purity = dominant_count / cluster_size

        rows.append(
            {
                "gmm_cluster": int(cluster_id),
                "cluster_size": cluster_size,
                "dominant_true_label": dominant_label,
                "dominant_true_class": dominant_class,
                "dominant_class_count": dominant_count,
                "cluster_purity": purity,
            }
        )

    return rows


# ============================================================
# SAVE RESULTS
# ============================================================

def save_assignments_csv(
    output_path: Path,
    sample_ids: np.ndarray,
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    confidence: np.ndarray,
    classes: dict,
):
    """Save one row per DTD observation."""
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "sample_id",
                "true_label",
                "true_class",
                "gmm_cluster",
                "gmm_confidence",
            ]
        )

        for sample_id, true_label, pred_label, conf in zip(
            sample_ids,
            labels_true,
            labels_pred,
            confidence,
        ):
            true_label_int = int(true_label)

            writer.writerow(
                [
                    str(sample_id),
                    true_label_int,
                    classes.get(
                        str(true_label_int),
                        f"class_{true_label_int}",
                    ),
                    int(pred_label),
                    float(conf),
                ]
            )


def save_cluster_summary_csv(
    output_path: Path,
    rows: list[dict],
):
    """Save one row per GMM cluster."""
    fieldnames = [
        "gmm_cluster",
        "cluster_size",
        "dominant_true_label",
        "dominant_true_class",
        "dominant_class_count",
        "cluster_purity",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def save_metrics_txt(
    output_path: Path,
    data_description: str,
    layer: str,
    X_shape: tuple[int, int],
    gmm: GaussianMixture,
    metrics: dict,
    confidence: np.ndarray,
    runtime_seconds: float,
):
    """Save a plain-English experiment summary."""
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write("DTD ResNet-50 Gaussian Mixture Model Results\n")
        file.write("=" * 55 + "\n\n")

        file.write(f"Data source: {data_description}\n")
        file.write(f"Feature matrix: {layer}.npy\n")
        file.write(f"Observations: {X_shape[0]}\n")
        file.write(f"Variables: {X_shape[1]}\n")
        file.write(f"GMM components: {gmm.n_components}\n")
        file.write(f"Covariance type: {gmm.covariance_type}\n")
        file.write(f"GMM converged: {gmm.converged_}\n")
        file.write(f"EM iterations: {gmm.n_iter_}\n")
        file.write(f"Runtime seconds: {runtime_seconds:.3f}\n")
        file.write(
            f"Mean cluster-assignment confidence: "
            f"{np.mean(confidence):.6f}\n"
        )

        file.write("\nEvaluation Metrics\n")
        file.write("-" * 55 + "\n")
        file.write(
            f"Adjusted Rand Index (ARI): "
            f"{metrics['ARI']:.6f}\n"
        )
        file.write(
            f"Normalized Mutual Information (NMI): "
            f"{metrics['NMI']:.6f}\n"
        )
        file.write(
            f"Adjusted Mutual Information (AMI): "
            f"{metrics['AMI']:.6f}\n"
        )
        file.write(
            f"Silhouette Score: "
            f"{metrics['Silhouette']:.6f}\n"
        )
        file.write(f"AIC: {metrics['AIC']:.3f}\n")
        file.write(f"BIC: {metrics['BIC']:.3f}\n")

        file.write("\nIMPORTANT METHOD NOTE\n")
        file.write("-" * 55 + "\n")
        file.write(
            "The Gaussian Mixture Model was fitted ONLY to the selected "
            "ResNet-50 feature matrix X. labels.npy was NOT used during "
            "model fitting or cluster assignment. Ground-truth labels were "
            "used only afterward to calculate ARI, NMI, AMI, and the "
            "evaluation-only cluster summary.\n"
        )


def save_pca_plot(
    output_path: Path,
    X_scaled: np.ndarray,
    labels_pred: np.ndarray,
    layer: str,
):
    """
    PCA is used ONLY to draw a 2-D picture.
    It does not replace the data used by the GMM.
    """
    pca = PCA(
        n_components=2,
        random_state=RANDOM_SEED,
    )

    X_2d = pca.fit_transform(X_scaled)

    explained = (
        pca.explained_variance_ratio_.sum() * 100.0
    )

    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels_pred,
        s=8,
        alpha=0.65,
        cmap="nipy_spectral",
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.title(
        f"DTD {layer} - Gaussian Mixture Model\n"
        f"2-D PCA visualization ({explained:.1f}% variance shown)"
    )

    plt.colorbar(
        scatter,
        label="GMM cluster",
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


# ============================================================
# COMMAND LINE
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Beginner-friendly GMM clustering for the "
            "DTD ResNet-50 Embeddings v1 release."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=(
            "Optional path to the extracted dtd folder, a release folder "
            "containing dtd/, or dtd-resnet50-v1.tar.gz. "
            "If omitted, the script searches the VS Code project."
        ),
    )

    parser.add_argument(
        "--layer",
        choices=VALID_LAYERS,
        default=FEATURE_LAYER,
        help=f"Feature matrix to use. Default: {FEATURE_LAYER}",
    )

    parser.add_argument(
        "--clusters",
        type=int,
        default=NUMBER_OF_CLUSTERS,
        help=f"Number of GMM components. Default: {NUMBER_OF_CLUSTERS}",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Only find and validate the release files. "
            "Do not run the GMM."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output directory. "
            "Default: gmm_results next to this script."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    args = parse_arguments()

    print("\n" + "=" * 68)
    print("DTD RESNET-50 - GAUSSIAN MIXTURE MODEL")
    print("=" * 68)

    print(f"\nPython version: {sys.version.split()[0]}")
    print(f"Selected feature layer: {args.layer}")
    print(f"Requested GMM clusters: {args.clusters}")

    # --------------------------------------------------------
    # Find data.
    # --------------------------------------------------------
    if args.data is not None:
        source_type, source_path = normalize_user_data_path(
            args.data,
            args.layer,
        )
    else:
        found = find_data_automatically(args.layer)

        if found is None:
            print(
                "\nERROR: I could not automatically find the DTD release.\n"
                "\nEasiest fix:\n"
                "1. Find the 'dtd' folder in VS Code.\n"
                "2. Right-click it and choose 'Copy Path'.\n"
                "3. Run this command in the VS Code terminal:\n\n"
                '   py -3.14 gmm_dtd_beginner.py --data "PASTE_PATH_HERE"\n'
                "\nYou can also point --data at dtd-resnet50-v1.tar.gz."
            )
            sys.exit(1)

        source_type, source_path = found

    print(f"\nFound data source ({source_type}):")
    print(f"  {source_path}")

    # --------------------------------------------------------
    # Load data.
    # --------------------------------------------------------
    if source_type == "folder":
        X, labels_true, sample_ids, classes = load_from_folder(
            source_path,
            args.layer,
        )
    else:
        X, labels_true, sample_ids, classes = load_from_archive(
            source_path,
            args.layer,
        )

    # --------------------------------------------------------
    # Validate against README.
    # --------------------------------------------------------
    validate_release(
        X,
        labels_true,
        sample_ids,
        classes,
        args.layer,
    )

    print("\nREADME CHECK: PASSED")
    print(f"  Observations:       {X.shape[0]:,}")
    print(f"  Feature variables:  {X.shape[1]:,}")
    print(f"  Ground-truth labels:{len(np.unique(labels_true)):>4}")
    print(f"  Class names:        {len(classes):>4}")
    print("  Row alignment:      lengths agree")

    if args.layer in ("layer3", "layer4"):
        print(
            "  Course >=700-variable high-dimensional requirement: YES"
        )
    else:
        print(
            "  Course >=700-variable high-dimensional requirement: NO"
        )

    if args.check_only:
        print(
            "\nCHECK-ONLY MODE COMPLETE.\n"
            "Your data loaded correctly. Nothing was clustered yet."
        )
        return

    # --------------------------------------------------------
    # Fit GMM.
    # labels_true is deliberately NOT passed into fit_gmm().
    # --------------------------------------------------------
    (
        scaler,
        X_scaled,
        gmm,
        labels_pred,
        confidence,
        runtime_seconds,
    ) = fit_gmm(
        X=X,
        n_clusters=args.clusters,
        covariance_type=COVARIANCE_TYPE,
        random_seed=RANDOM_SEED,
        n_init=N_INITIALIZATIONS,
        max_iterations=MAX_ITERATIONS,
    )

    # --------------------------------------------------------
    # Evaluation happens only AFTER GMM is finished.
    # --------------------------------------------------------
    metrics = evaluate_after_clustering(
        X_scaled=X_scaled,
        gmm=gmm,
        labels_pred=labels_pred,
        labels_true=labels_true,
        silhouette_sample_size=SILHOUETTE_SAMPLE_SIZE,
        random_seed=RANDOM_SEED,
    )

    cluster_summary = make_cluster_summary(
        labels_pred=labels_pred,
        labels_true=labels_true,
        classes=classes,
    )

    # --------------------------------------------------------
    # Print key results.
    # --------------------------------------------------------
    print("\n" + "=" * 68)
    print("RESULTS")
    print("=" * 68)

    print(f"Converged:                    {gmm.converged_}")
    print(f"EM iterations:                {gmm.n_iter_}")
    print(f"Adjusted Rand Index (ARI):    {metrics['ARI']:.4f}")
    print(f"Normalized Mutual Info (NMI): {metrics['NMI']:.4f}")
    print(f"Adjusted Mutual Info (AMI):   {metrics['AMI']:.4f}")
    print(f"Silhouette Score:             {metrics['Silhouette']:.4f}")
    print(f"Mean GMM confidence:          {np.mean(confidence):.4f}")

    # --------------------------------------------------------
    # Output directory.
    # --------------------------------------------------------
    if args.output is None:
        output_dir = (
            Path(__file__).resolve().parent
            / "gmm_results"
        )
    else:
        output_dir = args.output.expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        f"dtd_{args.layer}_gmm_{args.clusters}"
    )

    assignments_path = (
        output_dir
        / f"{prefix}_assignments.csv"
    )

    cluster_summary_path = (
        output_dir
        / f"{prefix}_cluster_summary.csv"
    )

    metrics_path = (
        output_dir
        / f"{prefix}_metrics.txt"
    )

    plot_path = (
        output_dir
        / f"{prefix}_pca.png"
    )

    save_assignments_csv(
        output_path=assignments_path,
        sample_ids=sample_ids,
        labels_true=labels_true,
        labels_pred=labels_pred,
        confidence=confidence,
        classes=classes,
    )

    save_cluster_summary_csv(
        output_path=cluster_summary_path,
        rows=cluster_summary,
    )

    save_metrics_txt(
        output_path=metrics_path,
        data_description=str(source_path),
        layer=args.layer,
        X_shape=X.shape,
        gmm=gmm,
        metrics=metrics,
        confidence=confidence,
        runtime_seconds=runtime_seconds,
    )

    print("\nCreating PCA picture...")

    save_pca_plot(
        output_path=plot_path,
        X_scaled=X_scaled,
        labels_pred=labels_pred,
        layer=args.layer,
    )

    print("\n" + "=" * 68)
    print("FINISHED")
    print("=" * 68)

    print("\nYour output folder is:")
    print(f"  {output_dir}")

    print("\nFiles created:")
    print(f"  1. {assignments_path.name}")
    print(f"  2. {cluster_summary_path.name}")
    print(f"  3. {metrics_path.name}")
    print(f"  4. {plot_path.name}")

    print(
        "\nReminder: labels.npy was used only AFTER clustering "
        "for evaluation."
    )


if __name__ == "__main__":
    main()

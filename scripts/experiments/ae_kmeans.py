#!/usr/bin/env python3
"""AE + K-means baseline using the pretrained G-CEALS autoencoder."""

from __future__ import annotations

import argparse
import contextlib
import csv
import random
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


FIELDS = [
    "dataset",
    "representation",
    "algorithm",
    "seed",
    "samples",
    "dimensions",
    "latent_dim",
    "n_clusters",
    "pretrain_epochs",
    "n_init",
    "preprocessing",
    "ari",
    "nmi",
    "ami",
    "fit_seconds",
    "peak_rss_mb",
    "status",
    "error",
]


def load_gceals_pretrain():
    """Import Keith's existing G-CEALS pretraining function without modifying it."""
    repo_root = Path(__file__).resolve().parents[2]
    gceals_dir = repo_root / "algorithms" / "gceals"

    sys.path.insert(0, str(gceals_dir))
    try:
        from gceals import pretrain
    finally:
        sys.path.pop(0)

    return pretrain


def run_one(X, y, args, seed, pretrain):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    n_clusters = len(np.unique(y))

    result = {
        "dataset": "dtd",
        "representation": "layer4",
        "algorithm": "ae_kmeans",
        "seed": seed,
        "samples": len(y),
        "dimensions": X.shape[1],
        "latent_dim": args.latent_dim,
        "n_clusters": n_clusters,
        "pretrain_epochs": args.pretrain_epochs,
        "n_init": args.n_init,
        "preprocessing": "standard",
    }

    start = time.perf_counter()

    try:
        data_tensor = torch.as_tensor(
            np.asarray(X, dtype=np.float32),
            device=args.device,
        )

        gceals_args = SimpleNamespace(
            dataset="benchmark",
            name=f"ae_kmeans_seed_{seed}",
            device=args.device,
            latent_dim=args.latent_dim,
            n_clusters=n_clusters,
            batch_size=args.batch_size,
            batch_count=int(np.ceil(len(X) / args.batch_size)),
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=0,
            l_rate=args.learning_rate,
            gamma=0.1,
            update_interval=1,
            stop_w_factor=0.1,
            timed=True,
            redo_pretrain=True,
            plot_interval=100,
            random_state=seed,
        )

        Path("debug").mkdir(parents=True, exist_ok=True)

        # Use the exact AE pretraining routine used by G-CEALS.
        with contextlib.redirect_stdout(sys.stderr):
            ae_model = pretrain(data_tensor, gceals_args)

        # Extract Z immediately after AE pretraining.
        ae_model.eval()
        with torch.no_grad():
            Z_tensor, _ = ae_model(data_tensor)

        Z = Z_tensor.cpu().numpy()

        # Baseline: pretrained AE representation -> K-means.
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=args.n_init,
            max_iter=args.max_iter,
        )

        with threadpool_limits(limits=args.threads):
            pred = kmeans.fit_predict(Z)

        elapsed = time.perf_counter() - start

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            peak_mb = rss / (1024 * 1024)
        else:
            peak_mb = rss / 1024

        result.update(
            ari=float(adjusted_rand_score(y, pred)),
            nmi=float(normalized_mutual_info_score(y, pred)),
            ami=float(adjusted_mutual_info_score(y, pred)),
            fit_seconds=round(elapsed, 6),
            peak_rss_mb=round(peak_mb, 2),
            status="ok",
            error="",
        )

    except Exception as exc:
        result.update(
            ari="",
            nmi="",
            ami="",
            fit_seconds=round(time.perf_counter() - start, 6),
            peak_rss_mb="",
            status="error",
            error=repr(exc),
        )

    return result


def write_results(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="outputs/dtd",
    )
    parser.add_argument(
        "--output",
        default="outputs/benchmark/dtd_ae_kmeans_layer4.csv",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--threads", type=int, default=4)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    X = np.load(data_dir / "layer4.npy")
    y = np.load(data_dir / "labels.npy")

    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("Incompatible features and labels")

    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN or infinity")

    # Identical preprocessing to run_benchmark.py.
    X = StandardScaler().fit_transform(X)
    X = X.astype(np.float64, copy=False)

    print("AE + K-means DTD Benchmark")
    print(f"Samples: {len(y)}")
    print(f"Input dimensions: {X.shape[1]}")
    print(f"Clusters: {len(np.unique(y))}")
    print(f"Latent dimensions: {args.latent_dim}")
    print(f"Pretrain epochs: {args.pretrain_epochs}")
    print(f"Device: {args.device}")
    print(f"Seeds: {args.seeds}")
    print()

    pretrain = load_gceals_pretrain()

    rows = []

    for i, seed in enumerate(args.seeds, 1):
        print(f"[{i}/{len(args.seeds)}] seed={seed}: ", end="", flush=True)

        result = run_one(X, y, args, seed, pretrain)
        rows.append(result)

        # Save after every seed so an interruption does not lose completed work.
        write_results(Path(args.output), rows)

        if result["status"] == "ok":
            print(
                f"ARI={result['ari']:.4f} "
                f"NMI={result['nmi']:.4f} "
                f"AMI={result['ami']:.4f} "
                f"time={result['fit_seconds']:.2f}s "
                f"peakRSS={result['peak_rss_mb']:.0f}MB"
            )
        else:
            print(f"ERROR: {result['error']}")

    print()
    print(f"Results saved: {args.output}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Shared numerical-feature clustering benchmark (K-means and GMM)."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler

from src.clustering.gmm import run_gmm
from src.clustering.gceals import run_gceals
from src.clustering.som import run_som


FIELDS = [
    "dataset",
    "representation",
    "algorithm",
    "seed",
    "samples",
    "dimensions",
    "n_clusters",
    "n_init",
    "preprocessing",
    "covariance_type",
    "reg_covar",
    "max_iter",
    "threads",
    "ari",
    "nmi",
    "ami",
    "homogeneity",
    "completeness",
    "v_measure",
    "clusters_used",
    "converged",
    "iterations",
    "fit_seconds",
    "peak_rss_mb",
    "status",
    "error",
]


def worker(args):
    """Execute one algorithm/representation/seed combination."""

    from threadpoolctl import threadpool_limits

    # The parent passes one value per worker, but argparse
    # stores these arguments as lists because of nargs="+".
    representation = args.representations[0]
    algorithm = args.algorithms[0]
    seed = args.seeds[0]

    data_dir = Path(args.data_dir)

    y = np.load(data_dir / "labels.npy")
    X = np.load(data_dir / f"{representation}.npy")

    if X.ndim != 2 or y.ndim != 1 or len(y) != len(X):
        raise ValueError("Incompatible features and labels")

    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN or infinity")

    # Shared preprocessing for every algorithm.
    X = StandardScaler().fit_transform(X)
    X = X.astype(np.float64, copy=False)

    n_clusters = len(np.unique(y))

    result = {
        "dataset": args.dataset,
        "representation": representation,
        "algorithm": algorithm,
        "seed": seed,
        "samples": len(y),
        "dimensions": X.shape[1],
        "n_clusters": n_clusters,
        "n_init": args.n_init,
        "preprocessing": "standard",
        "covariance_type": (
            args.covariance_type if algorithm == "gmm" else ""
        ),
        "reg_covar": (
            args.reg_covar if algorithm == "gmm" else ""
        ),
        "max_iter": args.max_iter,
        "threads": args.threads,
    }

    # Restrict native numerical-library threads consistently.
    with threadpool_limits(limits=args.threads):

        start = time.perf_counter()

        if algorithm == "kmeans":
            model = KMeans(
                n_clusters=n_clusters,
                random_state=seed,
                n_init=args.n_init,
                max_iter=args.max_iter,
            )

            pred = model.fit_predict(X)

            info = {
                "converged": "",
                "iterations": int(model.n_iter_),
            }

        elif algorithm == "gmm":
            pred, details = run_gmm(
                X,
                n_clusters=n_clusters,
                covariance_type=args.covariance_type,
                n_init=args.n_init,
                reg_covar=args.reg_covar,
                max_iter=args.max_iter,
                random_state=seed,
                return_info=True,
            )

            info = {
                "converged": details["converged"],
                "iterations": details["n_iter"],
            }

        elif algorithm == "som":
            pred = run_som(
                X,
                n_clusters=n_clusters,
                random_state=seed,
                grid_x=args.som_grid_x,
                grid_y=args.som_grid_y,
                sigma=args.som_sigma,
                learning_rate=args.som_learning_rate,
                iterations=args.som_iterations,
            )

            info = {
                "converged": "",
                "iterations": args.som_iterations,
            }

        elif algorithm == "gceals":
            pred = run_gceals(
                X,
                y,
                n_clusters=n_clusters,
                random_state=seed,
                device=args.gceals_device,
                latent_dim=args.gceals_latent_dim,
                batch_size=args.gceals_batch_size,
                pretrain_epochs=args.gceals_pretrain_epochs,
                finetune_epochs=args.gceals_finetune_epochs,
                learning_rate=args.gceals_lr,
                gamma=args.gceals_gamma,
                stop_w_factor=args.gceals_stop_w_factor,
            )

            info = {
                "converged": "",
                "iterations": "",
            }

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        elapsed = time.perf_counter() - start

    # Peak resident memory of this dedicated subprocess.
    # Includes interpreter, imports, data loading, preprocessing,
    # and fitting. It is not incremental fit-only memory.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if sys.platform == "darwin":
        peak_mb = rss / (1024 * 1024)
    else:
        peak_mb = rss / 1024

    result.update(
        info,
        fit_seconds=round(elapsed, 6),
        peak_rss_mb=round(peak_mb, 2),
        ari=float(adjusted_rand_score(y, pred)),
        nmi=float(normalized_mutual_info_score(y, pred)),
        ami=float(adjusted_mutual_info_score(y, pred)),
        homogeneity=float(homogeneity_score(y, pred)),
        completeness=float(completeness_score(y, pred)),
        v_measure=float(v_measure_score(y, pred)),
        clusters_used=int(np.unique(pred).size),
        status="ok",
        error="",
    )

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-dir",
        default="outputs/dtd",
    )

    parser.add_argument(
        "--dataset",
        default="dtd",
    )

    parser.add_argument(
        "--representations",
        nargs="+",
        default=["stem", "layer4"],
    )

    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["kmeans", "gmm", "gceals", "som"],
        default=["kmeans", "gmm"],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--n-init",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--covariance-type",
        choices=["diag", "spherical", "tied", "full"],
        default="diag",
    )

    parser.add_argument(
        "--reg-covar",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--output",
        default="outputs/benchmark/dtd_results.csv",
    )

    parser.add_argument(
        "--_worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--gceals-device", default="cpu")
    parser.add_argument("--gceals-latent-dim", type=int, default=10)
    parser.add_argument("--gceals-batch-size", type=int, default=256)
    parser.add_argument("--gceals-pretrain-epochs", type=int, default=100)
    parser.add_argument("--gceals-finetune-epochs", type=int, default=100)
    parser.add_argument("--gceals-lr", type=float, default=1e-5)
    parser.add_argument("--gceals-gamma", type=float, default=0.1)
    parser.add_argument("--gceals-stop-w-factor", type=float, default=0.1)

    parser.add_argument("--som-grid-x", type=int, default=10)
    parser.add_argument("--som-grid-y", type=int, default=10)
    parser.add_argument("--som-sigma", type=float, default=1.0)
    parser.add_argument("--som-learning-rate", type=float, default=0.5)
    parser.add_argument("--som-iterations", type=int, default=10000)

    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads must be positive")

    if args.n_init < 1:
        parser.error("--n-init must be positive")

    if args.max_iter < 1:
        parser.error("--max-iter must be positive")

    if args._worker:
        if (
            len(args.representations) != 1
            or len(args.algorithms) != 1
            or len(args.seeds) != 1
        ):
            parser.error(
                "Worker requires exactly one representation, "
                "algorithm, and seed"
            )

        print(json.dumps(worker(args)))
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_runs = (
        len(args.representations)
        * len(args.algorithms)
        * len(args.seeds)
    )

    print("Unified Clustering Benchmark")
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"Representations: {args.representations}")
    print(f"Algorithms: {args.algorithms}")
    print(f"Seeds: {args.seeds}")
    print(f"Threads per fit: {args.threads}")
    print(f"Total runs: {total_runs}")
    print()

    run_number = 0

    with output_path.open("w", newline="") as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
        )

        writer.writeheader()

        for representation in args.representations:
            for algorithm in args.algorithms:
                for seed in args.seeds:

                    run_number += 1

                    command = [
                        sys.executable,
                        "-m",
                        "scripts.experiments.run_benchmark",
                        "--_worker",
                        "--data-dir",
                        args.data_dir,
                        "--dataset",
                        args.dataset,
                        "--representations",
                        representation,
                        "--algorithms",
                        algorithm,
                        "--seeds",
                        str(seed),
                        "--threads",
                        str(args.threads),
                        "--n-init",
                        str(args.n_init),
                        "--max-iter",
                        str(args.max_iter),
                        "--covariance-type",
                        args.covariance_type,
                        "--reg-covar",
                        str(args.reg_covar),
                        "--gceals-device", args.gceals_device,
                        "--gceals-latent-dim", str(args.gceals_latent_dim),
                        "--gceals-batch-size", str(args.gceals_batch_size),
                        "--gceals-pretrain-epochs", str(args.gceals_pretrain_epochs),
                        "--gceals-finetune-epochs", str(args.gceals_finetune_epochs),
                        "--gceals-lr", str(args.gceals_lr),
                        "--gceals-gamma", str(args.gceals_gamma),
                        "--gceals-stop-w-factor", str(args.gceals_stop_w_factor),
                        "--som-grid-x", str(args.som_grid_x),
                        "--som-grid-y", str(args.som_grid_y),
                        "--som-sigma", str(args.som_sigma),
                        "--som-learning-rate", str(args.som_learning_rate),
                        "--som-iterations", str(args.som_iterations),
                    ]

                    print(
                        f"[{run_number}/{total_runs}] "
                        f"{representation:7s} "
                        f"{algorithm:6s} "
                        f"seed={seed}: ",
                        end="",
                        flush=True,
                    )

                    completed = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                    )

                    if completed.returncode == 0:

                        try:
                            row = json.loads(completed.stdout)
                        except json.JSONDecodeError as exc:
                            row = {
                                "dataset": args.dataset,
                                "representation": representation,
                                "algorithm": algorithm,
                                "seed": seed,
                                "status": "error",
                                "error": f"Invalid worker JSON: {exc}",
                            }

                        if row.get("status") == "ok":
                            print(
                                f"ARI={row['ari']:.4f} "
                                f"NMI={row['nmi']:.4f} "
                                f"AMI={row['ami']:.4f} "
                                f"time={row['fit_seconds']:.2f}s "
                                f"peakRSS={row['peak_rss_mb']:.0f}MB"
                            )
                        else:
                            print("ERROR", row.get("error", ""))

                    else:
                        row = {
                            "dataset": args.dataset,
                            "representation": representation,
                            "algorithm": algorithm,
                            "seed": seed,
                            "status": "error",
                            "error": completed.stderr.strip()[-1500:],
                        }

                        print("ERROR", row["error"][-250:])

                    writer.writerow(row)
                    handle.flush()

    print()
    print(f"Results saved: {output_path}")


if __name__ == "__main__":
    main()

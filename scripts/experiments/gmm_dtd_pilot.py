"""
Pilot: Gaussian Mixture Model on DTD as feature dimensionality D changes.

Design (N fixed at 5,640 DTD images, D varied):

1. Load one fixed representation (ResNet-50 layer4, 2048D by default).
2. Standardize each feature to zero mean / unit variance.
3. Fit PCA once and keep the first D components for each D in --dims.
   The original (unreduced) representation can be included with D = 0.
4. Fit a GMM for every (D, covariance type, seed) with a wall-time limit.
   Fits that exceed the limit, or fail, are recorded rather than retried.

Ground-truth labels are used only after fitting, for ARI / NMI / AMI.

Usage:
    python -m scripts.experiments.gmm_dtd_pilot --data-dir data/dtdres50
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import time
import tracemalloc
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.clustering.gmm import COVARIANCE_TYPES, n_gmm_parameters, run_gmm

N_CLUSTERS = 47

FIELDS = [
    "representation",
    "dimensions",
    "pca_explained_variance",
    "covariance_type",
    "seed",
    "status",
    "ari",
    "nmi",
    "ami",
    "n_clusters_used",
    "bic",
    "mean_log_likelihood",
    "converged",
    "n_iter",
    "n_parameters",
    "fit_time_s",
    "peak_mem_mb",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/dtd"))
    parser.add_argument("--representation", default="layer4")
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256, 512, 1024],
        help="PCA dimensions to evaluate; 0 means the unreduced representation.",
    )
    parser.add_argument(
        "--cov-types", nargs="+", default=list(COVARIANCE_TYPES), choices=COVARIANCE_TYPES
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--reg-covar", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--n-init", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--time-limit", type=float, default=600.0, help="Wall-time limit per fit (seconds)."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gmm_dtd_pilot"))
    return parser.parse_args()


def _fit_worker(X, kwargs, threads, queue) -> None:
    """Run one GMM fit in a child process so it can be stopped at the time limit."""
    try:
        with threadpool_limits(limits=threads):
            tracemalloc.start()
            start = time.perf_counter()
            labels, info = run_gmm(X, return_info=True, **kwargs)
            fit_time = time.perf_counter() - start
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        queue.put(("ok", labels, info, fit_time, peak / 2**20, ""))
    except Exception as exc:  # recorded as a failed configuration
        queue.put(("error", None, None, None, None, f"{type(exc).__name__}: {exc}"))


def fit_with_limit(X, kwargs, threads, time_limit):
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_fit_worker, args=(X, kwargs, threads, queue))
    proc.start()

    try:
        result = queue.get(timeout=time_limit)
    except Exception:
        proc.terminate()
        proc.join()
        return "timeout", None, None, None, None, f"exceeded {time_limit:.0f}s"

    proc.join()
    return result


def build_representations(X: np.ndarray, dims: list[int]):
    """Yield (D, explained variance, matrix) for each requested dimension."""
    X_std = StandardScaler().fit_transform(X)

    pca_dims = sorted(d for d in dims if d > 0)
    if pca_dims:
        max_dim = min(max(pca_dims), *X_std.shape)
        pca = PCA(n_components=max_dim, svd_solver="full", random_state=0)
        Z = pca.fit_transform(X_std)
        cumulative = np.cumsum(pca.explained_variance_ratio_)

    for d in dims:
        if d == 0:
            yield X_std.shape[1], 1.0, X_std
        else:
            d = min(d, Z.shape[1])
            yield d, float(cumulative[d - 1]), np.ascontiguousarray(Z[:, :d])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.csv"

    X = np.load(args.data_dir / f"{args.representation}.npy")
    y_true = np.load(args.data_dir / "labels.npy")
    assert X.shape[0] == len(y_true)

    print(f"Representation: {args.representation} {X.shape}")
    print(f"Classes: {len(np.unique(y_true))}, GMM components: {N_CLUSTERS}")
    print(f"Preprocessing: StandardScaler -> PCA; reg_covar={args.reg_covar}")
    print(f"Threads per fit: {args.threads}, time limit: {args.time_limit:.0f}s")
    print()

    # Increasing D, so a covariance type that hits the time limit is skipped
    # at every larger D instead of being re-run until it times out again.
    dims = sorted(args.dims, key=lambda d: d or float("inf"))
    timed_out: set[str] = set()

    rows = []
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for d, explained, Xd in build_representations(X, dims):
            for cov in args.cov_types:
                for seed in args.seeds:
                    row = {
                        "representation": args.representation,
                        "dimensions": d,
                        "pca_explained_variance": round(explained, 4),
                        "covariance_type": cov,
                        "seed": seed,
                        "n_parameters": n_gmm_parameters(N_CLUSTERS, d, cov),
                    }

                    if cov in timed_out:
                        row.update(status="skipped", error="timed out at this or a smaller D")
                    else:
                        kwargs = dict(
                            n_clusters=N_CLUSTERS,
                            covariance_type=cov,
                            n_init=args.n_init,
                            reg_covar=args.reg_covar,
                            max_iter=args.max_iter,
                            random_state=seed,
                        )
                        status, y_pred, info, fit_time, peak_mb, error = fit_with_limit(
                            Xd, kwargs, args.threads, args.time_limit
                        )
                        row.update(status=status, error=error)

                        if status == "ok":
                            row.update(
                                ari=adjusted_rand_score(y_true, y_pred),
                                nmi=normalized_mutual_info_score(y_true, y_pred),
                                ami=adjusted_mutual_info_score(y_true, y_pred),
                                n_clusters_used=info["n_clusters_used"],
                                bic=info["bic"],
                                mean_log_likelihood=info["mean_log_likelihood"],
                                converged=info["converged"],
                                n_iter=info["n_iter"],
                                fit_time_s=fit_time,
                                peak_mem_mb=peak_mb,
                            )
                        elif status == "timeout":
                            timed_out.add(cov)

                    rows.append(row)
                    writer.writerow(row)
                    f.flush()
                    print(format_row(row))

    print()
    print_summary(rows)
    print()
    print(f"Per-run results written to {results_path}")


def format_row(r: dict) -> str:
    prefix = f"D={r['dimensions']:<5} {r['covariance_type']:<9} seed={r['seed']}"
    if r["status"] != "ok":
        return f"{prefix}  {r['status'].upper()} ({r['error']})"
    return (
        f"{prefix}  ARI={r['ari']:.4f} NMI={r['nmi']:.4f} AMI={r['ami']:.4f} "
        f"K_used={r['n_clusters_used']} iter={r['n_iter']} "
        f"conv={r['converged']} time={r['fit_time_s']:.1f}s mem={r['peak_mem_mb']:.0f}MB"
    )


def print_summary(rows: list[dict]) -> None:
    print("Summary (mean ± std over successful seeds)")
    print(
        f"{'D':>6} {'Cov':<10}{'Var.':>7}{'ARI':>16}{'NMI':>16}{'AMI':>16}"
        f"{'Time (s)':>11}{'Mem (MB)':>10}{'OK':>5}"
    )

    keys = dict.fromkeys((r["dimensions"], r["covariance_type"]) for r in rows)
    for d, cov in keys:
        group = [r for r in rows if r["dimensions"] == d and r["covariance_type"] == cov]
        ok = [r for r in group if r["status"] == "ok"]
        explained = group[0]["pca_explained_variance"]
        line = f"{d:>6} {cov:<10}{explained:>7.3f}"

        if not ok:
            line += f"{'— ' + group[0]['status']:>16}"
        else:
            for metric in ("ari", "nmi", "ami"):
                values = np.array([r[metric] for r in ok])
                line += f"{values.mean():>9.4f} ± {values.std():.3f}"
            line += f"{np.mean([r['fit_time_s'] for r in ok]):>11.1f}"
            line += f"{np.mean([r['peak_mem_mb'] for r in ok]):>10.0f}"
        line += f"{len(ok):>3}/{len(group)}"
        print(line)


if __name__ == "__main__":
    main()

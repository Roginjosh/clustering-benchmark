from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture

COVARIANCE_TYPES = ("spherical", "diag", "tied", "full")


def n_gmm_parameters(n_clusters: int, n_features: int, covariance_type: str) -> int:
    """
    Number of free parameters of a GMM, matching scikit-learn's BIC/AIC count.

    Useful for checking feasibility before fitting: a full-covariance model
    grows as K * D^2 / 2 and quickly exceeds the number of observations.
    """
    k, d = n_clusters, n_features

    if covariance_type == "spherical":
        cov_params = k
    elif covariance_type == "diag":
        cov_params = k * d
    elif covariance_type == "tied":
        cov_params = d * (d + 1) // 2
    elif covariance_type == "full":
        cov_params = k * d * (d + 1) // 2
    else:
        raise ValueError(f"Unknown covariance_type: {covariance_type!r}")

    mean_params = k * d
    weight_params = k - 1
    return cov_params + mean_params + weight_params


def run_gmm(
    X: np.ndarray,
    n_clusters: int = 47,
    covariance_type: str = "diag",
    n_init: int = 1,
    init_params: str = "kmeans",
    reg_covar: float = 1e-6,
    max_iter: int = 200,
    tol: float = 1e-3,
    random_state: int | None = 42,
    verbose: int = 0,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """
    Fit a Gaussian Mixture Model with EM and return hard cluster assignments.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        n_clusters: Number of mixture components.
        covariance_type: One of "spherical", "diag", "tied", or "full".
        n_init: Number of EM restarts; the run with the best lower bound is kept.
        init_params: Responsibility initialization ("kmeans", "k-means++",
            "random", or "random_from_data").
        reg_covar: Non-negative value added to covariance diagonals for
            numerical stability. High-dimensional data typically needs more
            than the scikit-learn default.
        max_iter: Maximum number of EM iterations per initialization.
        tol: Convergence threshold on the lower-bound improvement.
        random_state: Random state for deterministic initialization.
        verbose: Verbosity mode passed to scikit-learn.
        return_info: If True, return (labels, info) where info contains model
            selection and convergence diagnostics.

    Returns:
        labels: 1D array of shape (n_samples,) with cluster assignments
            (0..k-1), or (labels, info) if return_info is True.
    """
    if covariance_type not in COVARIANCE_TYPES:
        raise ValueError(
            f"covariance_type must be one of {COVARIANCE_TYPES}, got {covariance_type!r}"
        )

    X = np.asarray(X, dtype=np.float64)

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        n_init=n_init,
        init_params=init_params,
        reg_covar=reg_covar,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        verbose=verbose,
    )
    gmm.fit(X)
    labels = gmm.predict(X)

    if not return_info:
        return labels

    info = {
        "bic": float(gmm.bic(X)),
        "aic": float(gmm.aic(X)),
        "mean_log_likelihood": float(gmm.score(X)),
        "lower_bound": float(gmm.lower_bound_),
        "converged": bool(gmm.converged_),
        "n_iter": int(gmm.n_iter_),
        "n_parameters": n_gmm_parameters(n_clusters, X.shape[1], covariance_type),
        "n_clusters_used": int(np.unique(labels).size),
    }
    return labels, info


def cluster(X: np.ndarray, n_clusters: int, **kwargs) -> np.ndarray:
    """
    Standard clustering interface for benchmark algorithms.

    Ground-truth labels must not be passed to this function.
    """
    kwargs.pop("return_info", None)
    return run_gmm(X, n_clusters=n_clusters, **kwargs)

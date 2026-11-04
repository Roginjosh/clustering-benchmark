"""Self-Organizing Map clustering adapter."""

from __future__ import annotations

import numpy as np
from minisom import MiniSom
from sklearn.cluster import KMeans


def run_som(
    X: np.ndarray,
    *,
    n_clusters: int,
    random_state: int = 42,
    grid_x: int = 10,
    grid_y: int = 10,
    sigma: float = 1.0,
    learning_rate: float = 0.5,
    iterations: int = 10000,
) -> np.ndarray:
    """
    Train a SOM and convert SOM units into exactly n_clusters clusters.

    The SOM learns grid_x * grid_y prototype vectors. The learned prototypes
    are then clustered with K-means into n_clusters groups. Each observation
    receives the cluster label assigned to its best-matching SOM unit.
    """

    X = np.asarray(X, dtype=np.float64)

    if grid_x * grid_y < n_clusters:
        raise ValueError(
            "SOM must contain at least as many units as requested clusters"
        )

    som = MiniSom(
        x=grid_x,
        y=grid_y,
        input_len=X.shape[1],
        sigma=sigma,
        learning_rate=learning_rate,
        neighborhood_function="gaussian",
        random_seed=random_state,
    )

    # Initialize prototypes from the observed feature distribution.
    som.random_weights_init(X)

    # Random-order training makes the benchmark seed meaningful.
    som.train_random(
        X,
        num_iteration=iterations,
        verbose=False,
    )

    # Flatten the SOM grid into one prototype vector per unit.
    prototypes = som.get_weights().reshape(
        grid_x * grid_y,
        X.shape[1],
    )

    # Convert SOM units into the requested number of final clusters.
    prototype_kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=random_state,
    )

    unit_clusters = prototype_kmeans.fit_predict(prototypes)

    # Assign each observation to its best-matching SOM unit, then map that
    # unit to one of the final clusters.
    pred = np.empty(X.shape[0], dtype=np.int64)

    for i, sample in enumerate(X):
        row, col = som.winner(sample)
        unit_index = row * grid_y + col
        pred[i] = unit_clusters[unit_index]

    return pred

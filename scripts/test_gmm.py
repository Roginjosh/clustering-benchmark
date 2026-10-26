"""Sanity check for the GMM clusterer on synthetic Gaussian blobs."""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from src.clustering.gmm import COVARIANCE_TYPES, cluster, run_gmm


def main():
    X, y = make_blobs(n_samples=2000, centers=5, n_features=10, random_state=0)

    for cov in COVARIANCE_TYPES:
        labels, info = run_gmm(X, n_clusters=5, covariance_type=cov, return_info=True)
        ari = adjusted_rand_score(y, labels)
        assert labels.shape == (len(X),)
        assert info["converged"], f"{cov} did not converge"
        assert ari > 0.95, f"{cov} ARI too low: {ari:.3f}"
        print(f"{cov:<10} ARI={ari:.4f} BIC={info['bic']:.1f} iter={info['n_iter']}")

    labels = cluster(X, n_clusters=5)
    assert np.array_equal(labels, run_gmm(X, n_clusters=5))
    print("GMM sanity check passed.")


if __name__ == "__main__":
    main()

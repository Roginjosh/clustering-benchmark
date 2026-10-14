import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)

DATA_DIR = Path("data/dtd")

REPRESENTATIONS = {
    "stem": 64,
    "layer1": 256,
    "layer2": 512,
    "layer3": 1024,
    "layer4": 2048,
}

N_CLUSTERS = 47
RANDOM_STATE = 42


def main():
    y_true = np.load(DATA_DIR / "labels.npy")

    print(f"Samples: {len(y_true)}")
    print(f"Classes: {len(np.unique(y_true))}")
    print(f"K-means clusters: {N_CLUSTERS}")
    print()

    results = []

    for name, expected_dim in REPRESENTATIONS.items():
        print(f"Running {name} ({expected_dim}D)...")

        X = np.load(DATA_DIR / f"{name}.npy")

        assert X.shape[0] == len(y_true)
        assert X.shape[1] == expected_dim

        model = KMeans(
            n_clusters=N_CLUSTERS,
            random_state=RANDOM_STATE,
            n_init=10,
        )

        start = time.perf_counter()
        y_pred = model.fit_predict(X)
        runtime = time.perf_counter() - start

        result = {
            "representation": name,
            "dimensions": X.shape[1],
            "ari": adjusted_rand_score(y_true, y_pred),
            "nmi": normalized_mutual_info_score(y_true, y_pred),
            "homogeneity": homogeneity_score(y_true, y_pred),
            "completeness": completeness_score(y_true, y_pred),
            "v_measure": v_measure_score(y_true, y_pred),
            "runtime": runtime,
        }

        results.append(result)

        print(
            f"  ARI={result['ari']:.4f} "
            f"NMI={result['nmi']:.4f} "
            f"Runtime={runtime:.2f}s"
        )

    print()
    print("Final results")
    print(
        f"{'Representation':<15}"
        f"{'D':>8}"
        f"{'ARI':>10}"
        f"{'NMI':>10}"
        f"{'Homog.':>10}"
        f"{'Compl.':>10}"
        f"{'V-meas.':>10}"
        f"{'Time (s)':>12}"
    )

    for r in results:
        print(
            f"{r['representation']:<15}"
            f"{r['dimensions']:>8}"
            f"{r['ari']:>10.4f}"
            f"{r['nmi']:>10.4f}"
            f"{r['homogeneity']:>10.4f}"
            f"{r['completeness']:>10.4f}"
            f"{r['v_measure']:>10.4f}"
            f"{r['runtime']:>12.2f}"
        )


if __name__ == "__main__":
    main()

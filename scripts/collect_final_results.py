import csv
import glob
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path("outputs/benchmark")
OUT = Path("outputs/final_results")
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = [
    "dtd_multiseed.csv",
    "dtd_a3_som_final.csv",
    "dtd_a3_gceals_stem.csv",
    "dtd_a3_gceals_layer4.csv",
    "dtd_ae_kmeans_layer4_seed*.csv",
    "camelyon16_kmeans_ninit10.csv",
    "camelyon16_custom.csv",
    "minc_classical_5seed.csv",
    "minc_gceals_5k.csv",
    "runs/minc_stem_*.csv",
]

METRICS = [
    "ari", "nmi", "ami",
    "homogeneity", "completeness", "v_measure",
    "fit_seconds", "peak_rss_mb",
]

FIELDS = [
    "dataset", "representation", "algorithm",
    "seed", "samples", "dimensions", "latent_dim",
    "n_clusters", "n_init", "pretrain_epochs",
    "preprocessing", "covariance_type", "reg_covar",
    "max_iter", "threads", "iterations",
    "clusters_used", "converged",
    *METRICS,
    "status", "error", "source_file",
]

# Keep experimental configurations separate.
CONFIG_FIELDS = [
    "dataset", "representation", "algorithm",
    "samples", "dimensions", "latent_dim",
    "n_clusters", "n_init", "pretrain_epochs",
    "preprocessing", "covariance_type", "reg_covar",
    "max_iter", "threads",
]

rows = []
seen = set()
duplicates = []

for pattern in SOURCES:
    for filename in sorted(glob.glob(str(ROOT / pattern))):
        path = Path(filename)

        with path.open(newline="") as f:
            reader = csv.DictReader(f)

            for original in reader:
                row = {field: original.get(field, "") for field in FIELDS}
                row["source_file"] = str(path)

                # Deduplicate only within the same configuration and seed.
                # Include iterations to avoid merging different SOM settings.
                key = tuple(
                    row.get(field, "")
                    for field in CONFIG_FIELDS + ["seed", "iterations"]
                )

                if key in seen:
                    duplicates.append(row)
                    continue

                seen.add(key)
                rows.append(row)

def write_csv(path, data, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)

write_csv(OUT / "all_results.csv", rows, FIELDS)

for dataset in ["dtd", "camelyon16", "minc"]:
    subset = [r for r in rows if r["dataset"] == dataset]
    write_csv(OUT / f"{dataset}_results.csv", subset, FIELDS)

# Calculate summary statistics from successful runs.
groups = defaultdict(list)

for row in rows:
    if row["status"] != "ok":
        continue

    key = tuple(row.get(field, "") for field in CONFIG_FIELDS)

    # SOM training iterations are a configuration parameter.
    if row["algorithm"] == "som":
        key += (row["iterations"],)
    else:
        key += ("",)

    groups[key].append(row)

summary = []

for key, group in sorted(groups.items()):
    record = dict(zip(CONFIG_FIELDS + ["som_iterations"], key))

    record["runs"] = len(group)
    record["seeds"] = ";".join(
        sorted({r["seed"] for r in group}, key=lambda x: int(x))
    )

    for metric in METRICS:
        values = []

        for row in group:
            try:
                values.append(float(row[metric]))
            except (ValueError, TypeError):
                pass

        record[f"{metric}_n"] = len(values)
        record[f"{metric}_mean"] = mean(values) if values else ""
        record[f"{metric}_sd"] = (
            stdev(values) if len(values) > 1 else ""
        )

    summary.append(record)

SUMMARY_FIELDS = (
    CONFIG_FIELDS
    + ["som_iterations", "runs", "seeds"]
    + [
        field
        for metric in METRICS
        for field in (
            f"{metric}_n",
            f"{metric}_mean",
            f"{metric}_sd",
        )
    ]
)

write_csv(OUT / "summary_results.csv", summary, SUMMARY_FIELDS)

print("\n=== CONSOLIDATION COMPLETE ===")
print(f"Total unique rows: {len(rows)}")
print(f"Duplicate rows skipped: {len(duplicates)}")
print(f"Summary groups: {len(summary)}")

print("\nDataset breakdown:")
for dataset in ["dtd", "camelyon16", "minc"]:
    subset = [r for r in rows if r["dataset"] == dataset]
    print(f"  {dataset:<12}: {len(subset)} runs")

print("\nFiles created:")
for path in sorted(OUT.glob("*.csv")):
    print(f"  {path}")

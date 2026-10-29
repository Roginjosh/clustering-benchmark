
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / "data" / "dtd"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

CSV_PATH = RESULTS / "dtd_gceals_10d.csv"

LAYERS = {
    "stem": 64,
    "layer1": 256,
    "layer2": 512,
    "layer3": 1024,
    "layer4": 2048,
}

FIELDS = [
    "layer", "n_samples", "n_features", "latent_dim",
    "n_clusters", "pretrain_epochs", "finetune_epochs",
    "gamma", "ACC", "NMI", "ARI", "Time"
]


def extract_metric(output, key):
    pattern = rf"(?m)^\s*{re.escape(key)}:\s*([-+0-9.eE]+)\s*$"
    matches = re.findall(pattern, output)
    return float(matches[-1]) if matches else None


# Resume support: skip completed layers.
completed = set()

if CSV_PATH.exists():
    with CSV_PATH.open(newline="") as f:
        completed = {
            row["layer"]
            for row in csv.DictReader(f)
        }

with CSV_PATH.open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)

    if f.tell() == 0:
        writer.writeheader()

    for layer, dimensions in LAYERS.items():

        if layer in completed:
            print(f"Skipping {layer}: already in CSV")
            continue

        print(f"\n{'=' * 60}", flush=True)
        print(f"Training {layer}: {dimensions} features", flush=True)
        print(f"{'=' * 60}\n", flush=True)

        command = [
            sys.executable, "-u", "gceals.py",
            "--device", "cuda:0",
            "--features", str(DATA),
            "--layers", layer,
            "--latent_dim", "10",
            "--pretrain_epochs", "1000",
            "--finetune_epochs", "1000",
            "--l_rate", "0.001",
            "--gamma", "0.1",
            "--stop_w_factor", "0.1",
            "--name", f"dtd_{layer}_full",
            "--redo_pretrain",
            "--timed",
        ]

        log_path = RESULTS / f"{layer}_training.log"

        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )

            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)

            returncode = process.wait()

        if returncode != 0:
            raise RuntimeError(
                f"{layer} failed. Check {log_path}"
            )

        output = log_path.read_text(
            encoding="utf-8", errors="replace"
        )

        metrics = {
            key: extract_metric(output, key)
            for key in ["ACC", "NMI", "ARI", "Time"]
        }

        if any(value is None for value in metrics.values()):
            raise RuntimeError(
                f"Missing metrics for {layer}. Check {log_path}"
            )

        writer.writerow({
            "layer": layer,
            "n_samples": 5640,
            "n_features": dimensions,
            "latent_dim": 10,
            "n_clusters": 47,
            "pretrain_epochs": 1000,
            "finetune_epochs": 1000,
            "gamma": 0.1,
            **metrics,
        })

        f.flush()

        print(
            f"\nCompleted {layer}: "
            f"ACC={metrics['ACC']:.4f}, "
            f"NMI={metrics['NMI']:.4f}, "
            f"ARI={metrics['ARI']:.4f}"
        )

print(f"\nAll experiments completed. Results: {CSV_PATH}")

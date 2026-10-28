from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import DataLoader

from src.datasets.minc import MINCDataset
from src.extraction.runner import extract_features


DATASET_ROOT = Path.home() / "datasets/MINC/minc-3m-224"
MANIFEST_PATH = Path("data/minc/minc_2m_manifest.csv")
OUTPUT_DIR = Path("outputs/minc")

LAYERS = [
    "stem",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
]

BATCH_SIZE = 16


def main() -> None:
    print("Loading MINC manifest...")

    dataset = MINCDataset(
        DATASET_ROOT,
        MANIFEST_PATH,
    )

    print(f"MINC samples: {len(dataset):,}")
    print(f"MINC classes: {len(dataset.classes)}")
    print(f"Batch size:   {BATCH_SIZE}")
    print()

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    extract_features(
        dataloader=dataloader,
        output_dir=OUTPUT_DIR,
        layers=LAYERS,
    )

    class_metadata = {
        str(index): class_name
        for class_name, index in dataset.class_to_idx.items()
    }

    with open(OUTPUT_DIR / "classes.json", "w") as file:
        json.dump(class_metadata, file, indent=2)

    print(
        f"Saved class mapping: "
        f"{OUTPUT_DIR / 'classes.json'}"
    )


if __name__ == "__main__":
    main()

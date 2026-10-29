from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader, Subset

from src.datasets.camelyon16 import CAMELYON16Dataset
from src.extraction.runner import extract_features


DATASET_ROOT = Path.home() / "datasets/CAMELYON16/patches_v2"
MANIFEST_PATH = Path("data/camelyon16/manifest_001_050.csv")
OUTPUT_DIR = Path("outputs/camelyon16")

LAYERS = [
    "stem",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
]

BATCH_SIZE = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ResNet-50 features from CAMELYON16."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Extract only the first N patches for testing.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading CAMELYON16 manifest...")

    full_dataset = CAMELYON16Dataset(
        root=DATASET_ROOT,
        manifest=MANIFEST_PATH,
    )

    if args.limit is not None:
        if args.limit <= 0 or args.limit > len(full_dataset):
            raise ValueError(
                f"--limit must be between 1 and {len(full_dataset)}."
            )

        dataset = Subset(full_dataset, range(args.limit))
        output_dir = Path("outputs/camelyon16_smoke")
    else:
        dataset = full_dataset
        output_dir = OUTPUT_DIR

    print()
    print("CAMELYON16 extraction configuration")
    print("-----------------------------------")
    print(f"Samples:      {len(dataset):,}")
    print(f"Classes:      {len(full_dataset.classes)}")
    print(f"Batch size:   {BATCH_SIZE}")
    print(f"Layers:       {', '.join(LAYERS)}")
    print(f"Output:       {output_dir}")
    print()

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    extract_features(
        dataloader=dataloader,
        output_dir=output_dir,
        layers=LAYERS,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    class_metadata = {
        str(index): name
        for name, index in full_dataset.class_to_idx.items()
    }

    classes_path = output_dir / "classes.json"

    with classes_path.open("w") as file:
        json.dump(class_metadata, file, indent=2)

    print()
    print("CAMELYON16 feature extraction complete.")
    print(f"Features:  {output_dir}")
    print(f"Class map: {classes_path}")


if __name__ == "__main__":
    main()

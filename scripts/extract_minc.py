from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader, Subset

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
]

BATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ResNet-50 features from MINC."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Extract only the first N samples for testing.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading MINC manifest...")

    full_dataset = MINCDataset(
        root=DATASET_ROOT,
        manifest=MANIFEST_PATH,
    )

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero.")

        if args.limit > len(full_dataset):
            raise ValueError(
                f"--limit cannot exceed dataset size "
                f"({len(full_dataset):,})."
            )

        dataset = Subset(
            full_dataset,
            range(args.limit),
        )
        output_dir = Path("outputs/minc_smoke")
    else:
        dataset = full_dataset
        output_dir = OUTPUT_DIR

    print()
    print("MINC extraction configuration")
    print("-----------------------------")
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
        str(index): class_name
        for class_name, index in full_dataset.class_to_idx.items()
    }

    classes_path = output_dir / "classes.json"

    with classes_path.open("w") as file:
        json.dump(class_metadata, file, indent=2)

    print()
    print("MINC feature extraction complete.")
    print(f"Features:     {output_dir}")
    print(f"Class map:    {classes_path}")


if __name__ == "__main__":
    main()

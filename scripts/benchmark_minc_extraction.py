from __future__ import annotations

import time
from pathlib import Path

from torch.utils.data import DataLoader, Subset

from src.datasets.minc import MINCDataset
from src.extraction.resnet50 import ResNet50Extractor


DATASET_ROOT = Path.home() / "datasets/MINC/minc-3m-224"
MANIFEST_PATH = Path("data/minc/minc_2m_manifest.csv")

SAMPLE_SIZE = 5_000
BATCH_SIZE = 16

LAYERS = [
    "stem",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
]


def main() -> None:
    print("Loading MINC manifest...")

    dataset = MINCDataset(
        DATASET_ROOT,
        MANIFEST_PATH,
    )

    subset = Subset(
        dataset,
        range(SAMPLE_SIZE),
    )

    dataloader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    extractor = ResNet50Extractor(
        layers=LAYERS,
    )

    print(f"Device:       {extractor.device}")
    print(f"Samples:      {SAMPLE_SIZE:,}")
    print(f"Batch size:   {BATCH_SIZE}")
    print()

    processed = 0

    # Synchronize MPS before timing.
    if extractor.device.type == "mps":
        import torch
        torch.mps.synchronize()

    start = time.perf_counter()

    for images, _, _ in dataloader:
        extractor(images)

        processed += len(images)

        print(
            f"\rProcessed {processed:,} / {SAMPLE_SIZE:,}",
            end="",
            flush=True,
        )

    if extractor.device.type == "mps":
        import torch
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start

    images_per_second = processed / elapsed
    estimated_seconds = 2_000_000 / images_per_second

    print()
    print()
    print(f"Elapsed:       {elapsed:.2f} s")
    print(f"Throughput:    {images_per_second:.2f} images/s")
    print(
        f"Estimated 2M:  "
        f"{estimated_seconds / 3600:.2f} hours"
    )


if __name__ == "__main__":
    main()

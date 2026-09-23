from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader

from src.extraction.resnet50 import (
    FEATURE_DIMS,
    ResNet50Extractor,
)


def extract_features(
    dataloader: DataLoader,
    output_dir: str | Path,
    layers: Iterable[str],
) -> None:
    """
    Extract ResNet-50 embeddings from an entire dataset.

    The dataset used by the DataLoader must return:

        image, sample_id, label

    Feature matrices are written directly to disk using NumPy
    memory-mapped .npy files so the complete dataset does not
    need to fit in RAM.

    Output:
        <layer>.npy
        labels.npy
        sample_ids.npy
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = tuple(layers)

    if not layers:
        raise ValueError("At least one extraction layer must be specified.")

    invalid_layers = set(layers) - set(FEATURE_DIMS)

    if invalid_layers:
        raise ValueError(
            f"Unknown layers: {sorted(invalid_layers)}. "
            f"Valid layers are: {list(FEATURE_DIMS)}"
        )

    dataset_size = len(dataloader.dataset)

    if dataset_size == 0:
        raise RuntimeError("The dataset contains no samples.")

    extractor = ResNet50Extractor(layers=layers)

    print(f"Device: {extractor.device}")
    print(f"Dataset size: {dataset_size:,}")
    print(f"Layers: {', '.join(layers)}")
    print()

    #
    # Allocate the final feature matrices directly on disk.
    #
    feature_arrays: dict[str, np.memmap] = {}

    for layer in layers:
        output_path = output_dir / f"{layer}.npy"

        feature_arrays[layer] = open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=(dataset_size, FEATURE_DIMS[layer]),
        )

    sample_ids: list[str] = []
    labels = np.empty(dataset_size, dtype=np.int64)

    write_index = 0

    for images, batch_ids, batch_labels in dataloader:
        batch_size = len(images)

        features = extractor(images)

        start = write_index
        end = write_index + batch_size

        if end > dataset_size:
            raise RuntimeError(
                "DataLoader produced more samples than the dataset size."
            )

        for layer in layers:
            batch_features = (
                features[layer]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )

            feature_arrays[layer][start:end] = batch_features

        labels[start:end] = batch_labels.cpu().numpy()

        sample_ids.extend(str(sample_id) for sample_id in batch_ids)

        write_index = end

        print(
            f"\rProcessed {write_index:,} / {dataset_size:,} samples",
            end="",
            flush=True,
        )

    print()
    print()

    if write_index != dataset_size:
        raise RuntimeError(
            f"Expected {dataset_size:,} samples but processed "
            f"{write_index:,}."
        )

    if len(sample_ids) != dataset_size:
        raise RuntimeError(
            f"Expected {dataset_size:,} sample IDs but received "
            f"{len(sample_ids):,}."
        )

    #
    # Explicitly flush feature data to disk.
    #
    for layer, array in feature_arrays.items():
        array.flush()

        print(
            f"Saved {layer:6s}: "
            f"{array.shape} -> {output_dir / f'{layer}.npy'}"
        )

    np.save(
        output_dir / "labels.npy",
        labels,
    )

    np.save(
        output_dir / "sample_ids.npy",
        np.asarray(sample_ids),
    )

    print(f"Saved labels:     {len(labels):,}")
    print(f"Saved sample IDs: {len(sample_ids):,}")
    print()
    print(f"Finished extracting {dataset_size:,} samples.")
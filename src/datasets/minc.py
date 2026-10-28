from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.extraction.resnet50 import get_resnet50_preprocessing


class MINCDataset(Dataset):
    """
    MINC dataset backed by a fixed CSV manifest.

    Each sample is returned as:

        image, sample_id, label

    The manifest defines the exact observations and their ordering,
    allowing feature extraction and downstream experiments to use
    the same reproducible dataset.
    """

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
    ) -> None:
        self.root = Path(root)
        self.manifest = Path(manifest)

        if not self.root.is_dir():
            raise FileNotFoundError(
                f"MINC dataset root not found: {self.root}"
            )

        if not self.manifest.is_file():
            raise FileNotFoundError(
                f"MINC manifest not found: {self.manifest}"
            )

        self.transform = get_resnet50_preprocessing()

        self.samples: list[tuple[str, int]] = []
        class_mapping: dict[int, str] = {}

        with open(self.manifest, newline="") as file:
            reader = csv.DictReader(file)

            required_columns = {
                "index",
                "path",
                "label",
                "class_name",
            }

            if reader.fieldnames is None:
                raise RuntimeError(
                    f"MINC manifest has no header: {self.manifest}"
                )

            missing_columns = (
                required_columns - set(reader.fieldnames)
            )

            if missing_columns:
                raise RuntimeError(
                    "MINC manifest is missing required columns: "
                    f"{sorted(missing_columns)}"
                )

            for expected_index, row in enumerate(reader):
                index = int(row["index"])
                relative_path = row["path"]
                label = int(row["label"])
                class_name = row["class_name"]

                if index != expected_index:
                    raise RuntimeError(
                        "MINC manifest index mismatch: "
                        f"expected {expected_index}, found {index}"
                    )

                existing_name = class_mapping.get(label)

                if (
                    existing_name is not None
                    and existing_name != class_name
                ):
                    raise RuntimeError(
                        f"MINC label {label} maps to both "
                        f"{existing_name!r} and {class_name!r}"
                    )

                class_mapping[label] = class_name

                self.samples.append(
                    (
                        relative_path,
                        label,
                    )
                )

        if not self.samples:
            raise RuntimeError(
                f"MINC manifest contains no samples: {self.manifest}"
            )

        self.class_to_idx = {
            class_name: label
            for label, class_name in sorted(class_mapping.items())
        }

        self.classes = [
            class_name
            for label, class_name in sorted(class_mapping.items())
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ):
        relative_path, label = self.samples[index]

        image_path = self.root / relative_path

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, relative_path, label

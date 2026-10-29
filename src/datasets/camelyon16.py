from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.extraction.resnet50 import get_resnet50_preprocessing


class CAMELYON16Dataset(Dataset):
    """
    CAMELYON16 V2 patch dataset backed by a fixed CSV manifest.

    Returns:
        image, sample_id, label

    Labels:
        exterior = 0
        boundary = 1
        interior = 2
    """

    GROUP_TO_LABEL = {
        "exterior": 0,
        "boundary": 1,
        "interior": 2,
    }

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
    ) -> None:
        self.root = Path(root)
        self.manifest = Path(manifest)

        if not self.root.is_dir():
            raise FileNotFoundError(
                f"CAMELYON16 dataset root not found: {self.root}"
            )

        if not self.manifest.is_file():
            raise FileNotFoundError(
                f"CAMELYON16 manifest not found: {self.manifest}"
            )

        self.transform = get_resnet50_preprocessing()
        self.samples: list[tuple[str, int]] = []

        with self.manifest.open(newline="") as file:
            reader = csv.DictReader(file)

            required_columns = {
                "patch_id",
                "slide_id",
                "group",
                "relative_path",
                "tumor_fraction",
                "annotation_incomplete",
            }

            if reader.fieldnames is None:
                raise RuntimeError("Manifest has no header.")

            missing = required_columns - set(reader.fieldnames)

            if missing:
                raise RuntimeError(
                    f"Manifest missing columns: {sorted(missing)}"
                )

            for row in reader:
                group = row["group"]

                if group not in self.GROUP_TO_LABEL:
                    raise ValueError(
                        f"Unknown CAMELYON16 group: {group}"
                    )

                relative_path = (
                    Path(row["slide_id"]) / row["relative_path"]
                )

                label = self.GROUP_TO_LABEL[group]

                self.samples.append((str(relative_path), label))

        if not self.samples:
            raise RuntimeError("CAMELYON16 manifest is empty.")

        self.classes = ["exterior", "boundary", "interior"]
        self.class_to_idx = self.GROUP_TO_LABEL.copy()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        relative_path, label = self.samples[index]
        image_path = self.root / relative_path

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, relative_path, label

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.extraction.resnet50 import get_resnet50_preprocessing


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class DTDDataset(Dataset):
    """
    Describable Textures Dataset (DTD).

    Each sample is returned as:

        image, sample_id, label

    where:

        image:
            Preprocessed tensor suitable for the pretrained ResNet-50.

        sample_id:
            Stable identifier based on the DTD relative image path.

        label:
            Integer class label.

    Class labels are assigned alphabetically by DTD class name.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"

        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"DTD images directory not found: {self.images_dir}"
            )

        self.transform = get_resnet50_preprocessing()

        self.classes = sorted(
            path.name
            for path in self.images_dir.iterdir()
            if path.is_dir()
        )

        if not self.classes:
            raise RuntimeError(
                f"No class directories found in {self.images_dir}"
            )

        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.classes)
        }

        self.samples: list[tuple[Path, str, int]] = []

        for class_name in self.classes:
            class_dir = self.images_dir / class_name
            label = self.class_to_idx[class_name]

            image_paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            for image_path in image_paths:
                relative_path = image_path.relative_to(self.images_dir)

                # Example:
                #     banded/banded_0001.jpg
                #
                # This remains unique even if filenames are reused
                # between different DTD classes.
                sample_id = relative_path.as_posix()

                self.samples.append(
                    (image_path, sample_id, label)
                )

        if not self.samples:
            raise RuntimeError(
                f"No images found in {self.images_dir}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ):
        image_path, sample_id, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, sample_id, label
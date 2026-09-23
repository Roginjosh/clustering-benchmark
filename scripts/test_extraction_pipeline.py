from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.extraction.resnet50 import get_resnet50_preprocessing
from src.extraction.runner import extract_features


class TestImageDataset(Dataset):
    """
    Small dataset used only to validate the extraction pipeline.
    """

    def __init__(self, image_dir: str | Path) -> None:
        self.image_dir = Path(image_dir)

        self.image_paths = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        if not self.image_paths:
            raise RuntimeError(
                f"No test images found in {self.image_dir}"
            )

        # Use the exact same preprocessing as the extractor.
        self.preprocessing = get_resnet50_preprocessing()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]

        image = Image.open(path).convert("RGB")
        image = self.preprocessing(image)

        sample_id = path.stem

        # These test images don't have real classes.
        # Give each one a dummy label of 0.
        label = 0

        return image, sample_id, label


def main() -> None:
    dataset = TestImageDataset("test_images")

    dataloader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    print()

    output_dir = Path("outputs/test_extraction")

    extract_features(
        dataloader=dataloader,
        output_dir=output_dir,
        layers=["stem", "layer2"],
    )

    print()
    print("Reloading saved files...")
    print()

    stem = np.load(output_dir / "stem.npy")
    layer2 = np.load(output_dir / "layer2.npy")
    labels = np.load(output_dir / "labels.npy")
    sample_ids = np.load(output_dir / "sample_ids.npy")

    print(f"stem:       {stem.shape}")
    print(f"layer2:     {layer2.shape}")
    print(f"labels:     {labels.shape}")
    print(f"sample_ids: {sample_ids.shape}")

    print()
    print("Sample IDs:")

    for i, sample_id in enumerate(sample_ids):
        print(f"  {i}: {sample_id}")

    assert stem.shape == (len(dataset), 64)
    assert layer2.shape == (len(dataset), 512)
    assert labels.shape == (len(dataset),)
    assert sample_ids.shape == (len(dataset),)

    print()
    print("All extraction pipeline checks passed.")


if __name__ == "__main__":
    main()
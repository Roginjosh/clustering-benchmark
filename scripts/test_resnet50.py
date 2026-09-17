from pathlib import Path

from PIL import Image
import torch

from src.features.resnet50_extractor import (
    FEATURE_DIMS,
    ResNet50Extractor,
)


def main() -> None:
    image_dir = Path("test_images")

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_paths:
        raise RuntimeError(
            "No test images found. Put a JPG or PNG in test_images/."
        )

    print(f"Found {len(image_paths)} test image(s).")

    extractor = ResNet50Extractor()

    print(f"Device: {extractor.device}")

    tensors = []

    for path in image_paths:
        image = Image.open(path).convert("RGB")
        tensor = extractor.preprocessing(image)
        tensors.append(tensor)

    batch = torch.stack(tensors)

    print(f"Input batch: {tuple(batch.shape)}")

    features = extractor(batch)

    for layer, tensor in features.items():
        expected_dim = FEATURE_DIMS[layer]

        print(
            f"{layer:6s} "
            f"shape={tuple(tensor.shape)} "
            f"expected_dim={expected_dim}"
        )


if __name__ == "__main__":
    main()
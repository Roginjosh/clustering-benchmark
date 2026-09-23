from pathlib import Path

from PIL import Image
import torch

from src.extraction.resnet50 import (
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
            "No test images found. Put JPG or PNG images in test_images/."
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
    print()

    features = extractor(batch)

    for layer, tensor in features.items():
        expected_shape = (
            len(image_paths),
            FEATURE_DIMS[layer],
        )

        actual_shape = tuple(tensor.shape)

        status = "PASS" if actual_shape == expected_shape else "FAIL"

        print(
            f"{layer:6s} "
            f"shape={actual_shape} "
            f"expected={expected_shape} "
            f"[{status}]"
        )

    print()
    print("Images processed:")

    for i, path in enumerate(image_paths):
        print(f"  {i}: {path.name}")


if __name__ == "__main__":
    main()
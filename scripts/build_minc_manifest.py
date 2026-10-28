from __future__ import annotations

import csv
import random
from pathlib import Path

from tqdm import tqdm


DATASET_ROOT = Path.home() / "datasets/MINC/minc-3m-224"
SPLIT = "train"
OUTPUT_PATH = Path("data/minc/minc_2m_manifest.csv")

SAMPLE_SIZE = 2_000_000
EXPECTED_TRAIN_SAMPLES = 2_517_205
SEED = 42

CLASS_TO_IDX = {
    "brick": 0,
    "carpet": 1,
    "ceramic": 2,
    "fabric": 3,
    "foliage": 4,
    "food": 5,
    "glass": 6,
    "hair": 7,
    "leather": 8,
    "metal": 9,
    "mirror": 10,
    "other": 11,
    "painted": 12,
    "paper": 13,
    "plastic": 14,
    "polishedstone": 15,
    "skin": 16,
    "sky": 17,
    "stone": 18,
    "tile": 19,
    "wallpaper": 20,
    "water": 21,
    "wood": 22,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    split_dir = DATASET_ROOT / SPLIT

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"MINC split directory not found: {split_dir}"
        )

    print(f"Scanning: {split_dir}")
    print(f"Target sample size: {SAMPLE_SIZE:,}")
    print(f"Random seed: {SEED}")
    print()

    samples: list[tuple[str, int, str]] = []
    class_counts: dict[str, int] = {}

    progress = tqdm(
        total=EXPECTED_TRAIN_SAMPLES,
        desc="Scanning MINC",
        unit="images",
    )

    for class_name, label in CLASS_TO_IDX.items():
        class_dir = split_dir / class_name

        if not class_dir.is_dir():
            progress.close()
            raise FileNotFoundError(
                f"MINC class directory not found: {class_dir}"
            )

        class_count = 0

        for path in class_dir.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                relative_path = path.relative_to(
                    DATASET_ROOT
                ).as_posix()

                samples.append(
                    (
                        relative_path,
                        label,
                        class_name,
                    )
                )

                class_count += 1
                progress.update(1)

        class_counts[class_name] = class_count

    progress.close()

    print()
    print("Class counts:")

    for class_name, label in CLASS_TO_IDX.items():
        print(
            f"  {label:2d} {class_name:15s}: "
            f"{class_counts[class_name]:,}"
        )

    print()
    print(f"Total available: {len(samples):,}")

    if len(samples) != EXPECTED_TRAIN_SAMPLES:
        print(
            f"WARNING: Expected {EXPECTED_TRAIN_SAMPLES:,} "
            f"training images but found {len(samples):,}."
        )

    if len(samples) < SAMPLE_SIZE:
        raise RuntimeError(
            f"Requested {SAMPLE_SIZE:,} samples but only "
            f"{len(samples):,} are available."
        )

    #
    # Sort before shuffling so filesystem traversal order cannot
    # affect the deterministic sample produced by the random seed.
    #
    print()
    print("Sorting sample paths...")
    samples.sort(key=lambda sample: sample[0])

    print("Shuffling samples...")
    rng = random.Random(SEED)
    rng.shuffle(samples)

    selected = samples[:SAMPLE_SIZE]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Writing {SAMPLE_SIZE:,}-sample manifest...")

    with open(OUTPUT_PATH, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "index",
                "path",
                "label",
                "class_name",
            ]
        )

        for index, (path, label, class_name) in tqdm(
            enumerate(selected),
            total=len(selected),
            desc="Writing manifest",
            unit="rows",
        ):
            writer.writerow(
                [
                    index,
                    path,
                    label,
                    class_name,
                ]
            )

    print()
    print("MINC manifest complete.")
    print(f"Available: {len(samples):,}")
    print(f"Selected:  {len(selected):,}")
    print(f"Seed:      {SEED}")
    print(f"Manifest:  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

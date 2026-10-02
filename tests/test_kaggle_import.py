import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from class_labels import CLASS_NAMES
from scripts.import_kaggle_dataset import import_training_split


def _make_image(path: Path, seed: int):
    rng = np.random.RandomState(seed)
    Image.fromarray(rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)


def test_import_keeps_existing_data_deduplicates_and_preserves_validation(tmp_path):
    dataset = tmp_path / "dataset"
    manifest = dataset / "manifest.csv"
    existing_healthy = dataset / "healthy"
    existing_healthy.mkdir(parents=True)
    for class_name in CLASS_NAMES:
        (dataset / class_name).mkdir(exist_ok=True)

    existing_image = existing_healthy / "existing.jpg"
    _make_image(existing_image, 101)
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "file",
                "class",
                "source_dataset",
                "source_folder",
                "original_path",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "file": "healthy/existing.jpg",
            "class": "healthy",
            "source_dataset": "existing",
            "source_folder": "Healthy",
            "original_path": str(existing_image),
        })

    source = tmp_path / "kaggle"
    training_leaf_scald = None
    folder_aliases = {
        "healthy": "healthy",
        "blast": "leaf_blast",
        "brown_spot": "brown_spot",
        "bacterial_leaf_blight": "bacterial_leaf_blight",
        "leaf_scald": "leaf_scald",
        "narrow_brown_spot": "narrow_brown_spot",
    }
    for split in ("train", "validation"):
        for index, class_name in enumerate(CLASS_NAMES):
            image_dir = source / split / folder_aliases[class_name]
            image_dir.mkdir(parents=True)
            image_path = image_dir / f"{split}_{class_name}.jpg"
            _make_image(image_path, 300 + index + (1000 if split == "validation" else 0))
            if split == "train" and class_name == "healthy":
                shutil.copy2(existing_image, image_path)
            if split == "train" and class_name == "leaf_scald":
                training_leaf_scald = image_path
            if split == "validation" and class_name == "leaf_scald":
                shutil.copy2(training_leaf_scald, image_path)

    brown_validation = source / "validation" / "brown_spot" / "validation_brown_spot.jpg"
    shutil.copy2(
        brown_validation,
        source / "validation" / "brown_spot" / "brown_duplicate.jpg",
    )

    result = import_training_split(source, dataset)
    assert result["skipped_duplicates"]["healthy"] == 1
    assert result["added"]["healthy"] == 0
    assert result["added"]["leaf_scald"] == 0
    assert result["excluded_validation_overlap"]["leaf_scald"] == 1
    assert result["added"]["narrow_brown_spot"] == 1
    assert result["validation_duplicates"]["brown_spot"] == 1
    assert result["validation_counts"] == {name: 1 for name in CLASS_NAMES}
    assert len(list((source / "validation").rglob("*.jpg"))) == len(CLASS_NAMES) + 1

    result_again = import_training_split(source, dataset)
    assert sum(result_again["added"].values()) == 0
    assert len(list(csv.DictReader(manifest.open(encoding="utf-8")))) == (
        1 + len(CLASS_NAMES) - 2
    )

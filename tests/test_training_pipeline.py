import shutil

import numpy as np
import pytest
from PIL import Image

from rice_cnn import split_data
from rice_cnn import validate_external_split


def test_split_is_reproducible_and_keeps_duplicate_images_together(tmp_path):
    rng = np.random.RandomState(24)
    paths = []
    labels = []
    duplicates = []

    for label in range(6):
        for index in range(8):
            pixels = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            path = tmp_path / f"class_{label}_{index}.jpg"
            Image.fromarray(pixels).save(path)
            paths.append(str(path))
            labels.append(label)
            if index == 0:
                duplicate = tmp_path / f"class_{label}_{index}_copy.jpg"
                shutil.copy2(path, duplicate)
                paths.append(str(duplicate))
                labels.append(label)
                duplicates.append((str(path), str(duplicate)))

    first = split_data(np.array(paths), np.array(labels), 0.15, 0.15)
    second = split_data(np.array(paths), np.array(labels), 0.15, 0.15)

    first_paths = [set(split[0].tolist()) for split in first]
    second_paths = [set(split[0].tolist()) for split in second]
    assert first_paths == second_paths
    assert not any(
        first_paths[i] & first_paths[j]
        for i in range(3)
        for j in range(i + 1, 3)
    )
    for original, duplicate in duplicates:
        assert sum(original in group for group in first_paths) == 1
        assert sum(duplicate in group for group in first_paths) == 1
        assert next(group for group in first_paths if original in group) == next(
            group for group in first_paths if duplicate in group
        )


def test_split_rejects_exact_duplicates_with_conflicting_labels(tmp_path):
    rng = np.random.RandomState(42)
    paths = []
    labels = []
    first_image = None

    for label in range(6):
        for index in range(4):
            pixels = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            path = tmp_path / f"class_{label}_{index}.jpg"
            Image.fromarray(pixels).save(path)
            paths.append(str(path))
            labels.append(label)
            if label == 0 and index == 0:
                first_image = path

    conflicting_copy = tmp_path / "class_1_conflicting_copy.jpg"
    shutil.copy2(first_image, conflicting_copy)
    paths.append(str(conflicting_copy))
    labels.append(1)

    with pytest.raises(ValueError, match="Byte-identical images have conflicting class labels"):
        split_data(np.array(paths), np.array(labels), 0.15, 0.15)


def test_external_validation_rejects_images_also_in_training(tmp_path):
    training_paths = []
    training_labels = []
    validation_paths = []
    validation_labels = []
    rng = np.random.RandomState(87)

    for label in range(6):
        pixels = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        train_path = tmp_path / f"train_{label}.jpg"
        Image.fromarray(pixels).save(train_path)
        training_paths.append(str(train_path))
        training_labels.append(label)

        validation_path = tmp_path / f"validation_{label}.jpg"
        Image.fromarray(rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)).save(
            validation_path
        )
        validation_paths.append(str(validation_path))
        validation_labels.append(label)

    shutil.copy2(training_paths[0], tmp_path / "duplicate_validation.jpg")
    validation_paths[0] = str(tmp_path / "duplicate_validation.jpg")

    with pytest.raises(ValueError, match="byte-identical"):
        validate_external_split(
            np.array(training_paths),
            np.array(training_labels),
            np.array(validation_paths),
            np.array(validation_labels),
        )

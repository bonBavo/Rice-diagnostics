"""Import only the training split of a labeled Kaggle rice-leaf dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import pathlib
import tempfile
from collections import Counter

from PIL import Image

from class_labels import CLASS_NAMES, canonical_class_name

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_DIR.parent / "RiceLeafsDisease"
DEFAULT_DATASET = PROJECT_DIR / "dataset"
MANIFEST_FIELDS = [
    "file",
    "class",
    "source_dataset",
    "source_folder",
    "original_path",
]
SOURCE_NAME = "kaggle_riceleafsdisease"


def file_hash(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_is_readable(path: pathlib.Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError, Image.DecompressionBombError):
        return False


def class_directories(split_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    mapped: dict[str, pathlib.Path] = {}
    for folder in split_dir.iterdir():
        if not folder.is_dir():
            continue
        class_name = canonical_class_name(folder.name)
        if class_name is None:
            raise ValueError(f"Unsupported class folder in {split_dir}: {folder.name}")
        if class_name in mapped:
            raise ValueError(
                f"Multiple folders map to {class_name!r} in {split_dir}"
            )
        mapped[class_name] = folder
    missing = sorted(set(CLASS_NAMES) - set(mapped))
    if missing:
        raise ValueError(
            f"Missing class folders in {split_dir}: {', '.join(missing)}"
        )
    return mapped


def _read_manifest(manifest_path: pathlib.Path) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Dataset manifest is required to preserve provenance: {manifest_path}"
        )
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != MANIFEST_FIELDS:
            raise ValueError(
                f"Unexpected manifest columns in {manifest_path}: {reader.fieldnames}"
            )
        return list(reader)


def import_training_split(source_root: pathlib.Path, dataset_root: pathlib.Path) -> dict:
    """Append nonduplicate training images, retaining existing manifest rows."""
    source_root = source_root.resolve()
    dataset_root = dataset_root.resolve()
    train_root = source_root / "train"
    validation_root = source_root / "validation"
    if not train_root.is_dir() or not validation_root.is_dir():
        raise FileNotFoundError(
            "Expected separate train and validation folders under " + str(source_root)
        )

    train_dirs = class_directories(train_root)
    validation_dirs = class_directories(validation_root)
    manifest_path = dataset_root / "manifest.csv"
    rows = _read_manifest(manifest_path)

    known_hashes: dict[str, str] = {}
    for class_name in CLASS_NAMES:
        class_dir = dataset_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for image_path in class_dir.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                digest = file_hash(image_path)
                previous_class = known_hashes.get(digest)
                if previous_class is not None and previous_class != class_name:
                    raise ValueError(
                        f"An existing image appears under conflicting classes: "
                        f"{previous_class} and {class_name}"
                    )
                known_hashes[digest] = class_name

    training_images: dict[str, list[tuple[pathlib.Path, pathlib.Path, str]]] = {}
    validation_hashes: dict[str, str] = {}
    validation_duplicates = Counter()
    for class_name in CLASS_NAMES:
        training_images[class_name] = []
        for image_path in sorted(train_dirs[class_name].rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if not image_is_readable(image_path):
                continue
            training_images[class_name].append(
                (image_path, train_root, file_hash(image_path))
            )

        for image_path in sorted(validation_dirs[class_name].rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if not image_is_readable(image_path):
                continue
            digest = file_hash(image_path)
            prior = validation_hashes.get(digest)
            if prior is not None:
                if prior != class_name:
                    raise ValueError(
                        "Kaggle validation contains a byte-identical cross-class duplicate."
                    )
                validation_duplicates[class_name] += 1
                continue
            validation_hashes[digest] = class_name
            if known_hashes.get(digest) is not None:
                raise ValueError(
                    f"Kaggle validation image duplicates an existing training image: {image_path}"
                )

    source_train_hashes: dict[str, str] = {}
    excluded_validation_overlap = Counter()
    for first_class, images in training_images.items():
        for image_path, _split_root, digest in images:
            source_class = source_train_hashes.get(digest)
            if source_class is not None and source_class != first_class:
                raise ValueError(
                    f"Training image {image_path} is duplicated across conflicting classes."
                )
            source_train_hashes[digest] = first_class
            prior_class = known_hashes.get(digest)
            if prior_class is not None and prior_class != first_class:
                raise ValueError(
                    f"Training image {image_path} duplicates an image labeled {prior_class}, "
                    f"but is labeled {first_class}."
                )
            if digest in validation_hashes:
                excluded_validation_overlap[first_class] += 1

    added = Counter()
    skipped_duplicates = Counter()
    new_rows = []
    for class_name in CLASS_NAMES:
        for image_path, split_root, digest in training_images[class_name]:
            if digest in validation_hashes:
                continue
            prior_class = known_hashes.get(digest)
            if prior_class == class_name:
                skipped_duplicates[class_name] += 1
                continue

            target_dir = dataset_root / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            target_name = (
                f"{SOURCE_NAME}__train__{image_path.stem}__{digest[:12]}"
                f"{image_path.suffix.lower()}"
            )
            destination = target_dir / target_name
            if destination.exists():
                if file_hash(destination) != digest:
                    raise FileExistsError(
                        f"Refusing to overwrite a different file: {destination}"
                    )
            else:
                with image_path.open("rb") as source, destination.open("xb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)

            new_rows.append({
                "file": destination.relative_to(dataset_root).as_posix(),
                "class": class_name,
                "source_dataset": SOURCE_NAME,
                "source_folder": image_path.parent.relative_to(source_root).as_posix(),
                "original_path": str(image_path.resolve()),
            })
            known_hashes[digest] = class_name
            added[class_name] += 1

    rows.extend(new_rows)
    dataset_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=dataset_root,
        delete=False,
    ) as stream:
        temporary_path = pathlib.Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, manifest_path)

    return {
        "added": added,
        "skipped_duplicates": skipped_duplicates,
        "excluded_validation_overlap": excluded_validation_overlap,
        "validation_duplicates": validation_duplicates,
        "validation_counts": Counter({
            class_name: sum(
                1 for image_path in validation_dirs[class_name].rglob("*")
                if image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
                and image_is_readable(image_path)
            )
            - validation_duplicates[class_name]
            for class_name in CLASS_NAMES
        }),
        "manifest_rows": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dataset-dir", type=pathlib.Path, default=DEFAULT_DATASET)
    args = parser.parse_args()

    summary = import_training_split(args.source, args.dataset_dir)
    print("Imported Kaggle training images:")
    for class_name in CLASS_NAMES:
        print(
            f"  {class_name}: added {summary['added'][class_name]}, "
            f"skipped existing duplicates {summary['skipped_duplicates'][class_name]}, "
            f"excluded train/validation overlaps "
            f"{summary['excluded_validation_overlap'][class_name]}"
        )
    print("Kaggle validation images preserved separately:")
    for class_name in CLASS_NAMES:
        print(
            f"  {class_name}: {summary['validation_counts'][class_name]} unique images "
            f"({summary['validation_duplicates'][class_name]} repeated file(s) ignored)"
        )
    print(f"Manifest records: {summary['manifest_rows']}")


if __name__ == "__main__":
    main()

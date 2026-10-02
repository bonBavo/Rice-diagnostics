#!/usr/bin/env python3
"""
Organize the Paddy Doctor dataset for rice disease classification.

Paddy Doctor contains 16,225 annotated rice leaf images across 13 classes.

This script copies only supported class folders found in the selected source.
The supported six-class model labels are:

    healthy
    blast
    brown_spot
    bacterial_leaf_blight
    leaf_scald
    narrow_brown_spot

The additional Kaggle source is imported with scripts/import_kaggle_dataset.py
so its pre-existing validation split stays separate. RYMV remains unsupported
until labelled RYMV data is provided.

All other Paddy Doctor classes are ignored.

Usage:

    Preview:
        python organize_dataset.py

    Actually copy the images:
        python organize_dataset.py --copy

    Specify a different Paddy Doctor location:
        python organize_dataset.py --source "C:\\path\\to\\Paddy Doctor" --copy
"""

import argparse
import csv
import filecmp
import os
import pathlib
import re
import shutil
import tempfile
from collections import Counter, defaultdict

from class_labels import CLASS_NAMES, canonical_class_name

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png"}

CLASSES = list(CLASS_NAMES)
PROJECT_DIR = pathlib.Path(__file__).resolve().parent
LOCAL_SOURCE = (
    PROJECT_DIR / "paddy_doctor"
    if (PROJECT_DIR / "paddy_doctor").is_dir()
    else PROJECT_DIR / "Rice Leaf Disease Images"
)


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def normalise(name):
    """Convert a folder name into a predictable format."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def map_label(folder_name):
    """
    Convert a Paddy Doctor folder name into one of our
    four required classes.

    Return None for classes we do not need.
    """

    return canonical_class_name(folder_name)


# ---------------------------------------------------------
# Scan Paddy Doctor
# ---------------------------------------------------------

def scan(root):
    """
    Find image-containing folders inside Paddy Doctor.

    Returns:

        image_path
        source_folder
        target_class
    """

    root = pathlib.Path(root)

    if not root.is_dir():
        raise SystemExit(
            f"\nERROR: Paddy Doctor folder was not found:\n{root}\n"
        )

    folders = sorted({
        p.parent
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    })

    for folder in folders:

        label = folder.name
        target = map_label(label)

        for img in sorted(folder.iterdir()):

            if (
                img.is_file()
                and img.suffix.lower() in IMG_EXTS
            ):
                yield (
                    img,
                    folder.relative_to(root).as_posix(),
                    target,
                )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--source",
        default=str(LOCAL_SOURCE),
        help=(
            "Paddy Doctor dataset folder. "
            f"Default: {LOCAL_SOURCE}"
        ),
    )

    parser.add_argument(
        "--healthy-source",
        type=pathlib.Path,
        help="Optional folder containing supplemental healthy images only.",
    )

    parser.add_argument(
        "--healthy-source-dataset",
        default="paddy_doctor",
        help="Source name recorded for images supplied through --healthy-source.",
    )

    parser.add_argument(
        "--out",
        default="dataset",
        help="Output folder. Default: dataset",
    )

    parser.add_argument(
        "--copy",
        action="store_true",
        help="Actually copy the images. Default is preview only.",
    )

    args = parser.parse_args()

    source = pathlib.Path(args.source)

    print("\n==============================================")
    print("        PADDY DOCTOR DATASET ORGANIZER")
    print("==============================================\n")

    print(f"Source : {source}")
    print(f"Output : {args.out}")

    if not args.copy:
        print("Mode   : PREVIEW")
    else:
        print("Mode   : COPY")

    print()

    # -----------------------------------------------------
    # Scan dataset
    # -----------------------------------------------------

    plan = []

    table = defaultdict(Counter)
    healthy_source = args.healthy_source.resolve() if args.healthy_source else None

    for img, folder_label, target in scan(source):
        if healthy_source and img.parent.resolve() == healthy_source:
            continue

        plan.append(
            (
                img,
                folder_label,
                target,
                "paddy_doctor",
            )
        )

        table[
            (
                folder_label,
                target,
            )
        ]["n"] += 1

    if args.healthy_source:
        healthy_source = args.healthy_source.resolve()
        if not healthy_source.is_dir():
            raise SystemExit(
                f"\nERROR: Healthy source folder was not found:\n{healthy_source}\n"
            )
        healthy_files = sorted(
            p for p in healthy_source.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
        healthy_source_name = normalise(args.healthy_source_dataset)
        for img in healthy_files:
            plan.append((img, "Healthy", "healthy", healthy_source_name))
            table[("Healthy", "healthy")]["n"] += 1

    if not plan:
        raise SystemExit(
            "\nERROR: No JPG, JPEG or PNG images were found."
        )

    # -----------------------------------------------------
    # Display mapping
    # -----------------------------------------------------

    print(
        f"{'SOURCE FOLDER':<45}"
        f"{'OUR CLASS':<30}"
        f"IMAGES"
    )

    print("-" * 85)

    for (folder, target), count in sorted(table.items()):

        print(
            f"{folder[:44]:<45}"
            f"{(target or 'IGNORED'):<30}"
            f"{count['n']}"
        )

    # -----------------------------------------------------
    # Calculate totals
    # -----------------------------------------------------

    totals = Counter(
        target
        for _, _, target, _ in plan
        if target is not None
    )

    ignored = sum(
        1
        for _, _, target, _ in plan
        if target is None
    )

    print("\n==============================================")
    print("             DATASET SUMMARY")
    print("==============================================\n")

    for cls in CLASSES:

        print(
            f"{cls:<30}"
            f"{totals.get(cls, 0):>6}"
        )

    print(
        f"{'Ignored':<30}"
        f"{ignored:>6}"
    )

    print(
        f"{'Total scanned':<30}"
        f"{sum(count['n'] for count in table.values()):>6}"
    )

    # -----------------------------------------------------
    # Check missing classes
    # -----------------------------------------------------

    missing = [
        cls
        for cls in CLASSES
        if totals.get(cls, 0) == 0
    ]

    if missing:

        print("\nWARNING:")
        print(
            "The following classes have no Paddy Doctor images:"
        )

        for cls in missing:
            print(f"  - {cls}")

    # -----------------------------------------------------
    # Preview mode
    # -----------------------------------------------------

    if not args.copy:

        print(
            "\nPreview only - nothing has been copied."
        )

        print(
            "\nRun:"
        )

        print(
            "    python organize_dataset.py --copy"
        )

        return

    # -----------------------------------------------------
    # Copy images
    # -----------------------------------------------------

    out = pathlib.Path(args.out)

    manifest_rows = []

    used = set()
    manifest_path = out / "manifest.csv"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            expected = [
                "file",
                "class",
                "source_dataset",
                "source_folder",
                "original_path",
            ]
            if reader.fieldnames != expected:
                raise SystemExit(f"Unexpected manifest columns in {manifest_path}")
            manifest_rows = list(reader)
    manifest_by_file = {row["file"]: row for row in manifest_rows}

    copied = 0

    print(
        f"\nCopying selected images into:\n{out}\n"
    )

    for img, folder_label, target, source_dataset in plan:

        # Ignore unwanted Paddy Doctor classes
        if target is None:
            continue

        dest_dir = out / target

        dest_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Make filenames unique and traceable
        stem = f"{source_dataset}__{normalise(folder_label)}__{img.stem}"

        dest = (
            dest_dir
            / f"{stem}{img.suffix.lower()}"
        )

        counter = 1

        if dest.exists() and filecmp.cmp(img, dest, shallow=False):
            used.add(dest)
        else:
            while dest in used or dest.exists():
                dest = (
                    dest_dir
                    / f"{stem}_{counter}"
                    f"{img.suffix.lower()}"
                )
                counter += 1

            used.add(dest)
            shutil.copy2(img, dest)

        relative_destination = dest.relative_to(out).as_posix()
        existing_row = manifest_by_file.get(relative_destination)
        if existing_row:
            if existing_row["class"] != target:
                raise ValueError(
                    f"Manifest class mismatch for existing image: {relative_destination}"
                )
        else:
            manifest_by_file[relative_destination] = {
                "file": relative_destination,
                "class": target,
                "source_dataset": source_dataset,
                "source_folder": folder_label,
                "original_path": str(img.resolve()),
            }
            manifest_rows.append(manifest_by_file[relative_destination])

        copied += 1

    # -----------------------------------------------------
    # Manifest
    # -----------------------------------------------------

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=out,
        delete=False,
    ) as fh:
        temporary_manifest = pathlib.Path(fh.name)
        writer = csv.writer(fh)

        writer.writerow(
            [
                "file",
                "class",
                "source_dataset",
                "source_folder",
                "original_path",
            ]
        )

        writer.writerows(
            [
                [
                    row["file"],
                    row["class"],
                    row["source_dataset"],
                    row["source_folder"],
                    row["original_path"],
                ]
                for row in manifest_rows
            ]
        )
    os.replace(temporary_manifest, manifest_path)

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    print("\n==============================================")
    print("                 COMPLETE")
    print("==============================================\n")

    print(
        f"Selected images : {copied}"
    )

    print(
        f"Output folder : {out}"
    )

    print(
        f"Manifest      : {manifest_path}"
    )

    print("\nClasses created:")

    for cls in CLASSES:

        count = totals.get(cls, 0)

        print(
            f"  {cls:<30}{count:>6}"
        )

    print("\nNext step:")
    print(
        "Train and evaluate the four supported classes; add RYMV only when labelled data is available."
    )


if __name__ == "__main__":
    main()
"""Download and prepare the BSDS300 source-disjoint natural-image split."""

from __future__ import annotations

import argparse
import csv
import random
import tarfile
import urllib.request
from pathlib import Path

from .config import (
    BSDS300_ARCHIVE,
    BSDS300_ROOT,
    BSDS300_URL,
    MANIFEST_PATH,
    RANDOM_STATE,
    create_project_directories,
)


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract a tar archive only if every member remains inside destination."""
    destination_resolved = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if not target.is_relative_to(destination_resolved):
            raise RuntimeError(f"Unsafe archive member: {member.name}")
    try:
        archive.extractall(destination, filter="data")
    except TypeError:
        archive.extractall(destination)


def download_dataset(archive_path: Path = BSDS300_ARCHIVE) -> Path:
    """Download the official BSDS300 image archive when it is not already present."""
    create_project_directories()
    if not archive_path.exists():
        print(f"Downloading BSDS300 images from: {BSDS300_URL}")
        urllib.request.urlretrieve(BSDS300_URL, archive_path)
    return archive_path


def extract_dataset(archive_path: Path = BSDS300_ARCHIVE) -> Path:
    """Extract BSDS300 images and return the dataset root."""
    if not BSDS300_ROOT.exists():
        with tarfile.open(archive_path, "r:gz") as archive:
            _safe_extract(archive, archive_path.parent)
    return BSDS300_ROOT


def _records_for_split(image_paths: list[Path], split: str) -> list[dict[str, str]]:
    return [
        {
            "source_id": path.stem,
            "split": split,
            "relative_path": path.relative_to(MANIFEST_PATH.parents[2]).as_posix(),
            "origin": "BSDS300",
        }
        for path in image_paths
    ]


def create_manifest(seed: int = RANDOM_STATE) -> Path:
    """Create a deterministic source-level train/validation/test manifest."""
    dataset_root = extract_dataset(download_dataset())
    train_images = sorted((dataset_root / "images" / "train").glob("*.jpg"))
    test_images = sorted((dataset_root / "images" / "test").glob("*.jpg"))
    if len(train_images) != 200 or len(test_images) != 100:
        raise RuntimeError("Unexpected BSDS300 image counts; dataset extraction may be incomplete.")

    shuffled = train_images.copy()
    random.Random(seed).shuffle(shuffled)
    train_split = shuffled[:160]
    validation_split = shuffled[160:]
    records = (
        _records_for_split(train_split, "train")
        + _records_for_split(validation_split, "validation")
        + _records_for_split(test_images, "test")
    )
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source_id", "split", "relative_path", "origin"))
        writer.writeheader()
        writer.writerows(records)
    return MANIFEST_PATH


def prepare_dataset() -> Path:
    """Download, extract, and split the training image corpus."""
    manifest = create_manifest()
    print(f"Prepared source-disjoint manifest: {manifest}")
    return manifest


def dataset_images_available() -> bool:
    """Return whether the extracted official image folders are complete."""
    train_images = list((BSDS300_ROOT / "images" / "train").glob("*.jpg"))
    test_images = list((BSDS300_ROOT / "images" / "test").glob("*.jpg"))
    return len(train_images) == 200 and len(test_images) == 100


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the BSDS300 image corpus.")
    parser.add_argument(
        "--force-manifest",
        action="store_true",
        help="Recreate the deterministic train/validation/test manifest.",
    )
    arguments = parser.parse_args()
    if arguments.force_manifest or not MANIFEST_PATH.exists() or not dataset_images_available():
        prepare_dataset()
    else:
        print(f"Manifest already exists: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

"""Paired low-light reconstruction dataset built from clean natural images."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as transform_functional

from .config import PATCH_SIZE, PROJECT_ROOT, RANDOM_STATE


def load_manifest(manifest_path: Path, split: str) -> list[dict[str, str]]:
    """Read one source-disjoint split from the prepared manifest."""
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        records = [row for row in csv.DictReader(handle) if row["split"] == split]
    if not records:
        raise ValueError(f"No records found for split: {split}")
    return records


def validate_manifest(manifest_path: Path, check_files: bool = True) -> dict[str, int]:
    """Check split separation and source-image availability."""
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    sources: dict[str, str] = {}
    counts = {"train": 0, "validation": 0, "test": 0}
    for record in records:
        source_id = record["source_id"]
        split = record["split"]
        if split not in counts:
            raise ValueError(f"Unknown split in manifest: {split}")
        if source_id in sources and sources[source_id] != split:
            raise ValueError(f"Source leakage detected for image: {source_id}")
        sources[source_id] = split
        if check_files:
            path = PROJECT_ROOT / record["relative_path"]
            if not path.is_file():
                raise FileNotFoundError(path)
        counts[split] += 1
    if counts != {"train": 160, "validation": 40, "test": 100}:
        raise ValueError(f"Unexpected split counts: {counts}")
    return counts


def _random_float(generator: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * float(torch.rand(1, generator=generator).item())


def degrade_low_light(clean: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Create a camera-inspired low-light/noisy image from a clean RGB tensor."""
    _, height, width = clean.shape
    gamma = _random_float(generator, 1.35, 2.20)
    exposure = _random_float(generator, 0.40, 0.68)
    shade_x_start = _random_float(generator, 0.72, 0.98)
    shade_x_end = _random_float(generator, 0.72, 1.04)
    shade_y_start = _random_float(generator, 0.78, 1.00)
    shade_y_end = _random_float(generator, 0.78, 1.02)
    color_cast = torch.tensor(
        [
            _random_float(generator, 0.90, 1.07),
            _random_float(generator, 0.94, 1.05),
            _random_float(generator, 0.90, 1.10),
        ],
        dtype=clean.dtype,
    ).view(3, 1, 1)
    horizontal = torch.linspace(shade_x_start, shade_x_end, width, dtype=clean.dtype)
    vertical = torch.linspace(shade_y_start, shade_y_end, height, dtype=clean.dtype)
    illumination = (torch.outer(vertical, horizontal) * exposure).unsqueeze(0)

    low_light = clean.clamp(0, 1).pow(gamma) * illumination * color_cast
    noise_sigma = _random_float(generator, 0.004, 0.018)
    signal_noise = torch.sqrt(low_light.clamp_min(0.01)) * noise_sigma
    noise = torch.randn(low_light.shape, generator=generator, dtype=clean.dtype)
    low_light = low_light + noise * signal_noise
    low_light = torch.round(low_light.clamp(0, 1) * 255.0) / 255.0
    return low_light.clamp(0, 1)


def _load_rgb(path: Path, patch_size: int) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = image.size
    if min(width, height) < patch_size:
        scale = patch_size / min(width, height)
        image = image.resize(
            (round(width * scale), round(height * scale)),
            Image.Resampling.BICUBIC,
        )
    return image


class LowLightPatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Generate low-light/clean pairs from deterministic source image crops."""

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        patches_per_image: int,
        patch_size: int = PATCH_SIZE,
        training: bool = False,
        seed: int = RANDOM_STATE,
    ) -> None:
        self.records = load_manifest(manifest_path, split)
        self.patches_per_image = patches_per_image
        self.patch_size = patch_size
        self.training = training
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change deterministic crop/degradation seeds between training epochs."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records) * self.patches_per_image

    def _generator(self, index: int) -> torch.Generator:
        generator = torch.Generator()
        epoch_offset = self.epoch * 1_000_003 if self.training else 0
        generator.manual_seed(self.seed + epoch_offset + index * 97)
        return generator

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record_index = index % len(self.records)
        record = self.records[record_index]
        generator = self._generator(index)
        image = _load_rgb(PROJECT_ROOT / record["relative_path"], self.patch_size)
        width, height = image.size
        x = int(torch.randint(0, width - self.patch_size + 1, (1,), generator=generator).item())
        y = int(torch.randint(0, height - self.patch_size + 1, (1,), generator=generator).item())
        image = image.crop((x, y, x + self.patch_size, y + self.patch_size))
        clean = transform_functional.pil_to_tensor(image).float() / 255.0
        if self.training and bool(torch.randint(0, 2, (1,), generator=generator).item()):
            clean = torch.flip(clean, dims=(2,))
        low_light = degrade_low_light(clean, generator)
        return low_light, clean

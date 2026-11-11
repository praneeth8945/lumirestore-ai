"""Portable CPU inference utilities for LumiRestore AI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as transform_functional

from .config import MODEL_PATH
from .model import ResidualUNet


@dataclass
class ModelBundle:
    """Checkpoint metadata and a CPU-ready Residual U-Net."""

    model: ResidualUNet
    architecture: str
    base_channels: int
    patch_size: int
    best_validation_psnr: float


@dataclass
class RestorationResult:
    """Output image and execution details for one reconstruction."""

    image: Image.Image
    elapsed_ms: float
    tiled: bool
    tile_count: int
    input_size: tuple[int, int]


def load_model_bundle(model_path: Path = MODEL_PATH) -> ModelBundle:
    """Load the portable state-dict checkpoint safely for CPU inference."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {model_path}. Run: python -m src.train_model"
        )
    checkpoint: dict[str, Any] = torch.load(model_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "ResidualUNet":
        raise ValueError("The supplied checkpoint is not a LumiRestore Residual U-Net.")
    model = ResidualUNet(base_channels=int(checkpoint["base_channels"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return ModelBundle(
        model=model,
        architecture=str(checkpoint["architecture"]),
        base_channels=int(checkpoint["base_channels"]),
        patch_size=int(checkpoint["patch_size"]),
        best_validation_psnr=float(checkpoint.get("best_validation_psnr", 0.0)),
    )


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    step = tile_size - 2 * overlap
    if step <= 0:
        raise ValueError("Tile overlap must be smaller than half of the tile size.")
    starts = list(range(0, max(1, length - tile_size), step))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _feather_window(height: int, width: int, device: torch.device) -> torch.Tensor:
    vertical = torch.hann_window(height, periodic=False, device=device).clamp_min(0.08)
    horizontal = torch.hann_window(width, periodic=False, device=device).clamp_min(0.08)
    return torch.outer(vertical, horizontal).unsqueeze(0).unsqueeze(0)


def restore_tensor(
    image: torch.Tensor,
    model_bundle: ModelBundle,
    blend: float = 1.0,
    tile_size: int = 384,
    overlap: int = 48,
) -> tuple[torch.Tensor, bool, int]:
    """Restore an RGB image tensor, using feathered tiles when necessary."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected an RGB tensor shaped [3, height, width].")
    blend = float(np.clip(blend, 0.0, 1.0))
    model = model_bundle.model
    _, height, width = image.shape
    if height <= tile_size and width <= tile_size:
        with torch.inference_mode():
            restored = model(image.unsqueeze(0))[0]
        return (blend * restored + (1.0 - blend) * image).clamp(0, 1), False, 1

    output = torch.zeros_like(image)
    weights = torch.zeros_like(image[:1])
    row_starts = _tile_starts(height, tile_size, overlap)
    column_starts = _tile_starts(width, tile_size, overlap)
    tile_count = 0
    for top in row_starts:
        for left in column_starts:
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            tile = image[:, top:bottom, left:right]
            with torch.inference_mode():
                restored_tile = model(tile.unsqueeze(0))[0]
            window = _feather_window(tile.shape[1], tile.shape[2], tile.device)[0]
            output[:, top:bottom, left:right] += restored_tile * window
            weights[:, top:bottom, left:right] += window
            tile_count += 1
    restored = output / weights.clamp_min(1e-6)
    return (blend * restored + (1.0 - blend) * image).clamp(0, 1), True, tile_count


def restore_image(
    image: Image.Image,
    model_bundle: ModelBundle | None = None,
    blend: float = 1.0,
) -> RestorationResult:
    """Restore a user image while preserving its original pixel dimensions."""
    bundle = model_bundle or load_model_bundle()
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    tensor = transform_functional.pil_to_tensor(normalized).float() / 255.0
    started = perf_counter()
    restored, tiled, tile_count = restore_tensor(tensor, bundle, blend=blend)
    elapsed_ms = (perf_counter() - started) * 1000.0
    array = restored.mul(255).round().byte().permute(1, 2, 0).numpy()
    return RestorationResult(
        image=Image.fromarray(array),
        elapsed_ms=elapsed_ms,
        tiled=tiled,
        tile_count=tile_count,
        input_size=normalized.size,
    )


def luminance_mean(image: Image.Image) -> float:
    """Calculate a readable 0-100 perceived luminance statistic."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luminance = 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
    return float(luminance.mean() * 100.0)

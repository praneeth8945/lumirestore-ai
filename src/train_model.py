"""Train and evaluate the LumiRestore Residual U-Net on paired low-light data."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import (
    ASSETS_DIR,
    BASE_CHANNELS,
    EVAL_PATCHES_PER_IMAGE,
    HISTORY_PATH,
    MANIFEST_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PATCH_SIZE,
    RANDOM_STATE,
    TRAIN_PATCHES_PER_IMAGE,
    create_project_directories,
)
from .dataset import LowLightPatchDataset, validate_manifest
from .metrics import psnr_per_image, reconstruction_loss, ssim_per_image
from .model import ResidualUNet, adaptive_exposure
from .prepare_data import prepare_dataset


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Set deterministic seeds for repeatable CPU training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _metric_summary(
    low_light: torch.Tensor,
    reconstructed: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Return paired image-reconstruction metrics for a complete evaluation split."""
    baseline = adaptive_exposure(low_light)
    return {
        "input_psnr": float(psnr_per_image(low_light, target).mean().item()),
        "input_ssim": float(ssim_per_image(low_light, target).mean().item()),
        "adaptive_gamma_psnr": float(psnr_per_image(baseline, target).mean().item()),
        "adaptive_gamma_ssim": float(ssim_per_image(baseline, target).mean().item()),
        "reconstruction_psnr": float(psnr_per_image(reconstructed, target).mean().item()),
        "reconstruction_ssim": float(ssim_per_image(reconstructed, target).mean().item()),
        "reconstruction_mae": float((reconstructed - target).abs().mean().item()),
        "reconstruction_mse": float((reconstructed - target).pow(2).mean().item()),
    }


def evaluate(
    model: ResidualUNet,
    loader: DataLoader,
    device: torch.device,
    capture_first_batch: bool = False,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    """Evaluate a model and, optionally, capture a visual demo batch."""
    model.eval()
    low_batches: list[torch.Tensor] = []
    output_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    visual_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    with torch.inference_mode():
        for low_light, target in loader:
            low_light = low_light.to(device)
            target = target.to(device)
            reconstructed = model(low_light)
            low_batches.append(low_light.cpu())
            output_batches.append(reconstructed.cpu())
            target_batches.append(target.cpu())
            if visual_batch is None and capture_first_batch:
                visual_batch = (low_light.cpu(), reconstructed.cpu(), target.cpu())
    return (
        _metric_summary(
            torch.cat(low_batches, dim=0),
            torch.cat(output_batches, dim=0),
            torch.cat(target_batches, dim=0),
        ),
        visual_batch,
    )


def _save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    """Write a [3,H,W] tensor as a portable PNG."""
    array = (
        tensor.detach()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    Image.fromarray(array).save(path)


def _save_demo_assets(
    visual_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    metrics: dict[str, float],
) -> None:
    """Save a real low-light/reconstruction/reference triplet for the app."""
    low_light, reconstructed, target = visual_batch
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    _save_tensor_image(low_light[0], ASSETS_DIR / "demo_low_light.png")
    _save_tensor_image(reconstructed[0], ASSETS_DIR / "demo_reconstruction.png")
    _save_tensor_image(target[0], ASSETS_DIR / "demo_reference.png")
    demo_metrics = _metric_summary(low_light[0:1], reconstructed[0:1], target[0:1])
    with (ASSETS_DIR / "demo_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "description": (
                    "Paired metrics for the packaged demonstration triplet, plus the "
                    "aggregate source-disjoint test-split metrics from the same run."
                ),
                "demo_example": demo_metrics,
                "test_split_aggregate": metrics,
            },
            handle,
            indent=2,
        )


def _write_history(rows: list[dict[str, float | int]]) -> None:
    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_and_evaluate(
    epochs: int = 12,
    batch_size: int = 6,
    learning_rate: float = 2e-3,
    train_patches_per_image: int = TRAIN_PATCHES_PER_IMAGE,
    eval_patches_per_image: int = EVAL_PATCHES_PER_IMAGE,
) -> dict[str, Any]:
    """Train the residual U-Net and save a checkpoint plus measured reports."""
    create_project_directories()
    set_seed()
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    try:
        split_counts = validate_manifest(MANIFEST_PATH)
    except (FileNotFoundError, ValueError):
        prepare_dataset()
        split_counts = validate_manifest(MANIFEST_PATH)
    device = torch.device("cpu")

    train_data = LowLightPatchDataset(
        MANIFEST_PATH,
        "train",
        patches_per_image=train_patches_per_image,
        training=True,
    )
    validation_data = LowLightPatchDataset(
        MANIFEST_PATH,
        "validation",
        patches_per_image=eval_patches_per_image,
        training=False,
    )
    test_data = LowLightPatchDataset(
        MANIFEST_PATH,
        "test",
        patches_per_image=eval_patches_per_image,
        training=False,
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ResidualUNet(base_channels=BASE_CHANNELS).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_state: dict[str, torch.Tensor] | None = None
    best_validation_psnr = float("-inf")
    history: list[dict[str, float | int]] = []
    no_improvement = 0
    patience = 4
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_data.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_images = 0
        for low_light, target in train_loader:
            low_light = low_light.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(low_light)
            loss = reconstruction_loss(reconstructed, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * low_light.size(0)
            total_images += low_light.size(0)
        scheduler.step()

        validation_metrics, _ = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(total_images, 1), 6),
            "validation_psnr": round(validation_metrics["reconstruction_psnr"], 6),
            "validation_ssim": round(validation_metrics["reconstruction_ssim"], 6),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{epochs} | loss {row['train_loss']:.4f} | "
            f"val PSNR {row['validation_psnr']:.2f} | val SSIM {row['validation_ssim']:.4f}"
        )

        if validation_metrics["reconstruction_psnr"] > best_validation_psnr:
            best_validation_psnr = validation_metrics["reconstruction_psnr"]
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stopping: validation PSNR did not improve.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    test_metrics, visual_batch = evaluate(model, test_loader, device, capture_first_batch=True)
    if visual_batch is None:
        raise RuntimeError("No test images available for demo asset generation.")

    checkpoint = {
        "schema_version": 1,
        "architecture": "ResidualUNet",
        "base_channels": BASE_CHANNELS,
        "patch_size": PATCH_SIZE,
        "model_state_dict": model.state_dict(),
        "training_seed": RANDOM_STATE,
        "degradation": "camera_inspired_synthetic_low_light_v1",
        "best_validation_psnr": best_validation_psnr,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(checkpoint, MODEL_PATH)
    _write_history(history)
    _save_demo_assets(visual_batch, test_metrics)

    # Measure CPU latency after one warm-up inference, using one real image patch.
    low_light_example = visual_batch[0][0:1].to(device)
    with torch.inference_mode():
        _ = model(low_light_example)
    timings: list[float] = []
    with torch.inference_mode():
        for _ in range(12):
            started = time.perf_counter()
            _ = model(low_light_example)
            timings.append((time.perf_counter() - started) * 1000.0)

    metrics = {
        "project": "LumiRestore AI",
        "architecture": "ResidualUNet",
        "base_channels": BASE_CHANNELS,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": "CPU",
        "dataset": {
            "name": "BSDS300 clean images with synthetic low-light degradation",
            "split_counts": split_counts,
            "source_disjoint": True,
        },
        "training": {
            "seed": RANDOM_STATE,
            "configured_epochs": epochs,
            "completed_epochs": len(history),
            "batch_size": batch_size,
            "patch_size": PATCH_SIZE,
            "train_patches_per_image": train_patches_per_image,
            "loss": "0.80 Charbonnier + 0.20 (1 - SSIM)",
            "best_validation_psnr": best_validation_psnr,
            "training_seconds": round(time.perf_counter() - start_time, 2),
        },
        "evaluation": {
            **{key: round(value, 6) for key, value in test_metrics.items()},
            "psnr_gain_over_input": round(
                test_metrics["reconstruction_psnr"] - test_metrics["input_psnr"],
                6,
            ),
            "ssim_gain_over_input": round(
                test_metrics["reconstruction_ssim"] - test_metrics["input_ssim"],
                6,
            ),
            "cpu_inference_ms_median": round(float(np.median(timings)), 3),
            "cpu_inference_ms_p95": round(float(np.percentile(timings, 95)), 3),
        },
        "limitation": (
            "Metrics are measured on source-disjoint BSDS300 images with this project's "
            "synthetic degradation process. They are not a claim of universal performance "
            "on all real low-light cameras or scenes."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LumiRestore Residual U-Net.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--quick", action="store_true", help="Run a short CPU smoke training.")
    arguments = parser.parse_args()
    if arguments.quick:
        arguments.epochs = 1
        arguments.batch_size = 4
    metrics = train_and_evaluate(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        train_patches_per_image=1 if arguments.quick else TRAIN_PATCHES_PER_IMAGE,
        eval_patches_per_image=1,
    )
    print(f"Checkpoint saved to: {MODEL_PATH}")
    print(f"Test reconstruction PSNR: {metrics['evaluation']['reconstruction_psnr']:.2f}")
    print(f"Test reconstruction SSIM: {metrics['evaluation']['reconstruction_ssim']:.4f}")


if __name__ == "__main__":
    main()

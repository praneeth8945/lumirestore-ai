"""Differentiable reconstruction loss and PSNR/SSIM metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def ssim_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute a compact differentiable SSIM approximation for RGB batches."""
    prediction = prediction.clamp(0, 1)
    target = target.clamp(0, 1)
    mu_prediction = functional.avg_pool2d(prediction, kernel_size=7, stride=1, padding=3)
    mu_target = functional.avg_pool2d(target, kernel_size=7, stride=1, padding=3)
    sigma_prediction = (
        functional.avg_pool2d(prediction * prediction, 7, 1, 3) - mu_prediction.pow(2)
    )
    sigma_target = functional.avg_pool2d(target * target, 7, 1, 3) - mu_target.pow(2)
    covariance = (
        functional.avg_pool2d(prediction * target, 7, 1, 3) - mu_prediction * mu_target
    )
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_prediction * mu_target + c1) * (2 * covariance + c2)
    denominator = (
        (mu_prediction.pow(2) + mu_target.pow(2) + c1)
        * (sigma_prediction + sigma_target + c2)
    )
    return (numerator / denominator.clamp_min(1e-8)).mean(dim=(1, 2, 3))


def psnr_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute PSNR values for image batches in the [0, 1] range."""
    mse = (prediction.clamp(0, 1) - target.clamp(0, 1)).pow(2).mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    """Robust pixel-reconstruction loss less sensitive to outliers than MSE."""
    return torch.sqrt((prediction - target).pow(2) + epsilon**2).mean()


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Hybrid Charbonnier plus structural-similarity reconstruction objective."""
    pixel_loss = charbonnier_loss(prediction, target)
    structure_loss = 1.0 - ssim_per_image(prediction, target).mean()
    return 0.80 * pixel_loss + 0.20 * structure_loss

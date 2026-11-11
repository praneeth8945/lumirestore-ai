"""Residual U-Net used for low-light image reconstruction."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from .config import BASE_CHANNELS


def _group_count(channels: int) -> int:
    for groups in (8, 6, 4, 3, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvAct(nn.Module):
    """Reflection-padded convolution, group normalization, and LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class ResidualBlock(nn.Module):
    """Two convolutional layers with a projected residual connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.first = ConvAct(in_channels, out_channels)
        self.second = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.second(self.first(inputs)) + self.skip(inputs))


class UpBlock(nn.Module):
    """Bilinear upsampling followed by a residual fusion block."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = ConvAct(in_channels, out_channels)
        self.fusion = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = functional.interpolate(
            inputs,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.fusion(torch.cat((self.projection(upsampled), skip), dim=1))


def adaptive_exposure(inputs: torch.Tensor) -> torch.Tensor:
    """Create a classical global-exposure estimate for residual reconstruction."""
    luminance = (
        0.2126 * inputs[:, 0:1]
        + 0.7152 * inputs[:, 1:2]
        + 0.0722 * inputs[:, 2:3]
    )
    mean_luminance = luminance.mean(dim=(2, 3), keepdim=True).clamp(0.02, 0.95)
    gamma = (torch.log(torch.tensor(0.52, device=inputs.device)) / torch.log(mean_luminance)).clamp(
        0.35,
        0.90,
    )
    return inputs.clamp(0, 1).pow(gamma).clamp(0, 1)


class ResidualUNet(nn.Module):
    """Compact three-level U-Net that predicts a correction over adaptive exposure."""

    def __init__(self, base_channels: int = BASE_CHANNELS) -> None:
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        self.encoder1 = ResidualBlock(3, c1)
        self.down1 = ConvAct(c1, c2, stride=2)
        self.encoder2 = ResidualBlock(c2, c2)
        self.down2 = ConvAct(c2, c3, stride=2)
        self.encoder3 = ResidualBlock(c3, c3)
        self.down3 = ConvAct(c3, c4, stride=2)
        self.bottleneck = nn.Sequential(ResidualBlock(c4, c4), ResidualBlock(c4, c4))
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c1, 3, kernel_size=3),
        )
        nn.init.zeros_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_height, original_width = inputs.shape[-2:]
        padding_height = (8 - original_height % 8) % 8
        padding_width = (8 - original_width % 8) % 8
        padded = functional.pad(inputs, (0, padding_width, 0, padding_height), mode="replicate")
        baseline = adaptive_exposure(padded)

        encoded1 = self.encoder1(padded)
        encoded2 = self.encoder2(self.down1(encoded1))
        encoded3 = self.encoder3(self.down2(encoded2))
        bottleneck = self.bottleneck(self.down3(encoded3))
        decoded3 = self.up3(bottleneck, encoded3)
        decoded2 = self.up2(decoded3, encoded2)
        decoded1 = self.up1(decoded2, encoded1)
        correction = 0.55 * torch.tanh(self.head(decoded1))
        reconstructed = (baseline + correction).clamp(0, 1)
        return reconstructed[:, :, :original_height, :original_width]

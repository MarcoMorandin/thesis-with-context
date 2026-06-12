"""Per-sensor channel projection: C_sensor → 3 (RGB-compatible)."""
from __future__ import annotations
import torch
import torch.nn as nn


class SensorProjection(nn.Module):
    """Maps native sensor channels to 3-channel RGB for VisualEncoder input.

    Args:
        in_channels: Number of native sensor channels (e.g. 1 thermal, 2 SAR, 13 Sentinel-2).

    Init strategy: identity for first 3 channels (C>=3), channel replication for C<3.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
        self._init_identity(in_channels)

    def _init_identity(self, in_channels: int) -> None:
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        if in_channels >= 3:
            for i in range(3):
                self.proj.weight.data[i, i, 0, 0] = 1.0
        else:
            for i in range(3):
                self.proj.weight.data[i, i % in_channels, 0, 0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, T_v, C_sensor, H, W]``
        Returns:
            ``[B, T_v, 3, H, W]``
        """
        B, T_v, C, H, W = x.shape
        x = x.reshape(B * T_v, C, H, W)
        x = self.proj(x)
        return x.reshape(B, T_v, 3, H, W)

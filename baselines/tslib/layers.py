"""Shared building blocks for the tier-2 supervised models.

Compact re-implementations of the standard components used by the
Time-Series-Library family (RevIN instance normalization, series
decomposition, patch embedding, gated residual network).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RevIN(nn.Module):
    """Reversible instance normalization (Kim et al., ICLR 2022).

    Statistics are computed over valid history steps only (mask-aware),
    so missing steps never contaminate the normalizer.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def normalize(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """x (B, T), mask (B, T) → normalized x, mean (B, 1), std (B, 1)."""
        if mask is None:
            mask = torch.ones_like(x)
        count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (x * mask).sum(dim=1, keepdim=True) / count
        var = ((x - mean) ** 2 * mask).sum(dim=1, keepdim=True) / count
        std = (var + self.eps).sqrt()
        return (x - mean) / std * mask, mean, std

    @staticmethod
    def denormalize(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        return x * std + mean


class MovingAverage(nn.Module):
    """Trend extraction by edge-padded moving average (DLinear decomposition)."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:  # (B, T)
        front = x[:, :1].repeat(1, (self.kernel_size - 1) // 2)
        back = x[:, -1:].repeat(1, self.kernel_size // 2)
        return self.avg(torch.cat([front, x, back], dim=1).unsqueeze(1)).squeeze(1)


class PatchEmbedding(nn.Module):
    """Split a univariate series into overlapping patches and embed them."""

    def __init__(self, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)

    def n_patches(self, seq_len: int) -> int:
        return (seq_len - self.patch_len) // self.stride + 1

    def forward(self, x: Tensor) -> Tensor:  # (B, T) → (B, P, d_model)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        return self.proj(patches)


class GRN(nn.Module):
    """Gated residual network (TFT, Lim et al. 2021), without context input."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, 2 * d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc2(torch.nn.functional.elu(self.fc1(x)))
        h = self.dropout(h)
        value, gate = self.gate(h).chunk(2, dim=-1)
        return self.norm(x + value * torch.sigmoid(gate))


def encoder(d_model: int, n_heads: int, n_layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=4 * d_model,
        dropout=dropout,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)

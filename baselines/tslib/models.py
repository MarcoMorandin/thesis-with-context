"""Tier-2 supervised deep TS models (BASELINE_COMPARISON.md §1, Tier 2).

Compact ports in the spirit of the Time-Series-Library reference
implementations, sharing one forward signature:

    forward(y_hist (B,T), cov (B,T+H,C), mask_hist (B,T)) -> (B,H) or (B,H,Q)

* MLP          — simplest learned baseline (flattened inputs)
* DLinear      — decomposition + two linear maps, Y only (the
                 "embarrassingly simple" check)
* PatchTST     — channel-independent patch transformer with RevIN
* iTransformer — variates-as-tokens transformer
* TFTLite      — compact Temporal Fusion Transformer: LSTM encoder/decoder
                 with future-known covariates, gated residual blocks,
                 cross-attention, native quantile output. Slimmed (no
                 per-variable selection networks); deviations documented
                 in baselines/README.md.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from common import config

from .layers import GRN, MovingAverage, PatchEmbedding, RevIN, encoder


class MLP(nn.Module):
    output_quantiles = False

    def __init__(self, history: int, horizon: int, n_cov: int,
                 hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        in_dim = 2 * history + (history + horizon) * n_cov
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, horizon),
        )

    def forward(self, y_hist: Tensor, cov: Tensor, mask_hist: Tensor) -> Tensor:
        flat = torch.cat([y_hist, mask_hist, cov.flatten(1)], dim=1)
        return self.net(flat)


class DLinear(nn.Module):
    """Zeng et al. 2023; univariate (inputs `Y` only, per the comparison matrix)."""

    output_quantiles = False

    def __init__(self, history: int, horizon: int, n_cov: int = 0,
                 kernel_size: int = 9):
        super().__init__()
        self.decomp = MovingAverage(kernel_size)
        self.linear_seasonal = nn.Linear(history, horizon)
        self.linear_trend = nn.Linear(history, horizon)

    def forward(self, y_hist: Tensor, cov: Tensor, mask_hist: Tensor) -> Tensor:
        trend = self.decomp(y_hist)
        seasonal = y_hist - trend
        return self.linear_seasonal(seasonal) + self.linear_trend(trend)


class PatchTST(nn.Module):
    """Nie et al., ICLR 2023 — channel-independent, RevIN on every channel."""

    output_quantiles = False

    def __init__(self, history: int, horizon: int, n_cov: int,
                 patch_len: int = 8, stride: int = 4, d_model: int = 128,
                 n_heads: int = 8, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.horizon = horizon
        self.n_channels = 1 + n_cov
        self.revin = RevIN()
        self.patch = PatchEmbedding(patch_len, stride, d_model)
        n_patches = self.patch.n_patches(history)
        self.pos = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        self.encoder = encoder(d_model, n_heads, n_layers, dropout)
        self.head = nn.Linear(n_patches * d_model, horizon)

    def forward(self, y_hist: Tensor, cov: Tensor, mask_hist: Tensor) -> Tensor:
        b, t = y_hist.shape
        # target channel gets the mask-aware RevIN normalization
        y_norm, mean, std = self.revin.normalize(y_hist, mask_hist)
        channels = torch.cat(
            [y_norm.unsqueeze(1), cov[:, :t, :].permute(0, 2, 1)], dim=1
        )  # (B, 1+C, T)
        flat = channels.reshape(b * self.n_channels, t)
        z = self.patch(flat) + self.pos
        z = self.encoder(z)
        out = self.head(z.flatten(1)).view(b, self.n_channels, self.horizon)
        return self.revin.denormalize(out[:, 0], mean, std)


class ITransformer(nn.Module):
    """Liu et al., ICLR 2024 — each variate's history is one token."""

    output_quantiles = False

    def __init__(self, history: int, horizon: int, n_cov: int,
                 d_model: int = 128, n_heads: int = 8, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.revin = RevIN()
        self.embed = nn.Linear(history, d_model)
        self.encoder = encoder(d_model, n_heads, n_layers, dropout)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, y_hist: Tensor, cov: Tensor, mask_hist: Tensor) -> Tensor:
        t = y_hist.shape[1]
        y_norm, mean, std = self.revin.normalize(y_hist, mask_hist)
        variates = torch.cat(
            [y_norm.unsqueeze(1), cov[:, :t, :].permute(0, 2, 1)], dim=1
        )  # (B, 1+C, T)
        tokens = self.encoder(self.embed(variates))
        return self.revin.denormalize(self.head(tokens[:, 0]), mean, std)


class TFTLite(nn.Module):
    """Compact TFT: quantile-native, uses future-known covariates."""

    output_quantiles = True

    def __init__(self, history: int, horizon: int, n_cov: int,
                 d_model: int = 64, n_heads: int = 4, dropout: float = 0.1,
                 quantiles: tuple[float, ...] = config.QUANTILE_LEVELS):
        super().__init__()
        self.horizon = horizon
        self.quantiles = quantiles
        n_known = len(config.DETERMINISTIC_COV_IDX)
        self.known_idx = list(config.DETERMINISTIC_COV_IDX)
        self.revin = RevIN()
        self.embed_hist = nn.Linear(1 + n_cov + 1, d_model)   # y + cov + mask
        self.embed_future = nn.Linear(n_known, d_model)
        self.grn_hist = GRN(d_model, dropout)
        self.grn_future = GRN(d_model, dropout)
        self.enc_lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.dec_lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.grn_out = GRN(d_model, dropout)
        self.head = nn.Linear(d_model, len(quantiles))

    def forward(self, y_hist: Tensor, cov: Tensor, mask_hist: Tensor) -> Tensor:
        t = y_hist.shape[1]
        y_norm, mean, std = self.revin.normalize(y_hist, mask_hist)
        hist = torch.cat(
            [y_norm.unsqueeze(-1), cov[:, :t, :], mask_hist.unsqueeze(-1)], dim=-1
        )
        future = cov[:, t:, self.known_idx]

        h = self.grn_hist(self.embed_hist(hist))
        enc_out, state = self.enc_lstm(h)
        f = self.grn_future(self.embed_future(future))
        dec_out, _ = self.dec_lstm(f, state)
        attended, _ = self.attn(dec_out, enc_out, enc_out)
        z = self.grn_out(dec_out + attended)
        q = self.head(z)                                   # (B, H, Q)
        q, _ = torch.sort(q, dim=-1)                       # enforce monotone quantiles
        return self.revin.denormalize(q, mean.unsqueeze(-1), std.unsqueeze(-1))


MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "mlp": MLP,
    "dlinear": DLinear,
    "patchtst": PatchTST,
    "itransformer": ITransformer,
    "tft": TFTLite,
}

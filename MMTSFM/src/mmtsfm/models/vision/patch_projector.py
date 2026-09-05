"""Resampler-free visual token path for the s2d arm (A30).

V-JEPA patch field -> pixel shuffle -> MLP projector -> spatial cell embedding
-> EVS pruning. No learned-query compressor anywhere: this is the
``encoder -> MLP projector -> decoder`` shape of NVIDIA's Nemotron 3 Nano Omni
(arXiv 2604.24954v2, §2), adopted because every arm that pools the patch field
through ``LatentSummarizer`` sits at the ramp seed floor.

See ``knowledge/specs/2026-09-05-A30-s2d-design.md`` §3.1.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def pixel_shuffle_tokens(x: torch.Tensor, r: int) -> torch.Tensor:
    """``[B, T, g0*g0, D]`` -> ``[B, T, (g0/r)^2, D*r*r]`` (space-to-depth).

    Trades spatial resolution for channel depth, so the r*r merged patches are
    *concatenated* rather than averaged. That is the whole point: averaging is
    the pooling operation this arm exists to remove.
    """
    B, T, P, D = x.shape
    g0 = int(round(P**0.5))
    if g0 * g0 != P:
        raise ValueError(f"patch count {P} is not a square grid")
    if g0 % r != 0:
        raise ValueError(f"grid {g0} not divisible by pixel-shuffle factor r={r}")
    g = g0 // r
    x = x.reshape(B, T, g, r, g, r, D)
    x = x.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, T, g * g, D * r * r)
    return x


def evs_select(
    tokens: torch.Tensor, keep: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Efficient Video Sampling — keep the ``keep`` most temporally novel tokens.

    Per spatial cell, score each frame by cosine *dissimilarity* to the same cell
    in the previous frame; frame 0 is pinned to ``+inf`` so the anchor field is
    always retained whole. Scores are ranked globally over ``T*n_cells`` and the
    surviving indices are re-sorted ascending, so the kept sequence stays in
    (frame, cell) order and the sequence length is a fixed ``keep``.

    Nemotron applies EVS at runtime only (§2.3); here it is inside training too —
    a deliberate deviation, flagged in the design doc §5.2.

    Args:
        tokens: ``[B, T, n_cells, d]``
        keep: number of tokens to retain; ``>= T*n_cells`` is a no-op.

    Returns:
        kept:      ``[B, keep, d]``
        frame_idx: ``[B, keep]`` long — source frame of each kept token
        cell_idx:  ``[B, keep]`` long — source spatial cell of each kept token
    """
    B, T, C, d = tokens.shape
    N = T * C
    flat = tokens.reshape(B, N, d)
    if keep >= N:
        idx = torch.arange(N, device=tokens.device).expand(B, N)
        return flat, idx // C, idx % C

    prev = tokens[:, :-1]
    cur = tokens[:, 1:]
    dissim = 1.0 - torch.nn.functional.cosine_similarity(cur, prev, dim=-1, eps=1e-6)
    anchor = torch.full(
        (B, 1, C), float("inf"), device=tokens.device, dtype=dissim.dtype
    )
    scores = torch.cat([anchor, dissim], dim=1).reshape(B, N)

    idx = scores.topk(keep, dim=1).indices.sort(dim=1).values  # [B, keep]
    kept = flat.gather(1, idx.unsqueeze(-1).expand(B, keep, d))
    return kept, idx // C, idx % C


class VisualPatchProjector(nn.Module):
    """V-JEPA latents ``[B, T, P, D_v]`` -> visual tokens ``[B, K, d_model]``.

    Args:
        d_v: V-JEPA embedding dim (ViT-L/16 -> 1024).
        d_model: Chronos-2 model dim.
        shuffle_r: pixel-shuffle factor. 2 gives Nemotron's 4x token reduction
            (14x14 -> 7x7).
        n_cells: spatial cells after shuffle; sized for the cell embedding table
            and asserted against the actual grid at forward time.
        evs_keep: tokens surviving EVS. ``<= 0`` disables pruning (the q=0 eval).
        dropout: applied after the projector.
    """

    def __init__(
        self,
        d_v: int,
        d_model: int,
        shuffle_r: int = 2,
        n_cells: int = 49,
        evs_keep: int = 98,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shuffle_r = int(shuffle_r)
        self.n_cells = int(n_cells)
        self.evs_keep = int(evs_keep)
        d_in = d_v * self.shuffle_r * self.shuffle_r
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Spatial identity only. There is deliberately NO temporal-slice
        # embedding: fractional RoPE positions already encode frame time, and a
        # second encoding of that axis would hand A09 (frame shuffle) an escape
        # hatch — shuffling permutes positions, a slice embedding could re-leak
        # the order. Design doc §3.3.
        self.cell_embed = nn.Parameter(
            torch.randn(self.n_cells, d_model) * (d_model**-0.5)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, video_latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(tokens [B, K, d], frame_idx [B, K], cell_idx [B, K])``."""
        x = pixel_shuffle_tokens(video_latents, self.shuffle_r)  # [B, T, C, d_in]
        if x.shape[2] != self.n_cells:
            raise ValueError(
                f"pixel shuffle produced {x.shape[2]} cells but n_cells={self.n_cells}; "
                f"check shuffle_r={self.shuffle_r} against the cache patch grid."
            )
        x = self.dropout(self.proj(x))
        x = x + self.cell_embed[None, None, :, :]
        keep = self.evs_keep if self.evs_keep > 0 else x.shape[1] * self.n_cells
        return evs_select(x, keep)

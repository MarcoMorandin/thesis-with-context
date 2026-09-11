"""Resampler-free visual token path for the s2d arm (A30).

V-JEPA patch field -> pixel shuffle -> MLP projector -> spatial cell embedding
-> EVS pruning. No learned-query compressor anywhere: this is the
``encoder -> MLP projector -> decoder`` shape of NVIDIA's Nemotron 3 Nano Omni
(arXiv 2604.24954v2, §2), adopted because every arm that pools the patch field
through ``LatentSummarizer`` sits at the ramp seed floor.

Two component ablations live here as mode switches rather than forks, so the
foils share this file's code path exactly:

* ``pool_mode="avg"``   — A37 / ticket 41. Mean-pool the r*r block instead of
  concatenating it. Same token count, same layout, sub-cell detail destroyed.
* ``evs_mode="random"`` — A38 / ticket 42. Keep the same NUMBER of tokens but
  choose them uniformly at random instead of by temporal novelty.

See ``knowledge/specs/2026-09-05-A30-s2d-design.md`` §3.1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

POOL_MODES = ("shuffle", "avg")
EVS_MODES = ("novelty", "random")


def _split_blocks(x: torch.Tensor, r: int, what: str) -> tuple[torch.Tensor, int]:
    """``[B, T, g0*g0, D]`` -> ``[B, T, g*g, r*r, D]`` with the r*r block last."""
    B, T, P, D = x.shape
    g0 = int(round(P**0.5))
    if g0 * g0 != P:
        raise ValueError(f"patch count {P} is not a square grid")
    if g0 % r != 0:
        raise ValueError(f"grid {g0} not divisible by {what} factor r={r}")
    g = g0 // r
    x = x.reshape(B, T, g, r, g, r, D)
    return x.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, T, g * g, r * r, D), g * g


def pixel_shuffle_tokens(x: torch.Tensor, r: int) -> torch.Tensor:
    """``[B, T, g0*g0, D]`` -> ``[B, T, (g0/r)^2, D*r*r]`` (space-to-depth).

    Trades spatial resolution for channel depth, so the r*r merged patches are
    *concatenated* rather than averaged. That is the whole point: averaging is
    the pooling operation this arm exists to remove.
    """
    blocks, n_cells = _split_blocks(x, r, "pixel-shuffle")
    B, T = blocks.shape[0], blocks.shape[1]
    return blocks.reshape(B, T, n_cells, -1)


def avg_pool_tokens(x: torch.Tensor, r: int) -> torch.Tensor:
    """``[B, T, g0*g0, D]`` -> ``[B, T, (g0/r)^2, D]`` (mean over the r*r block).

    A37's foil for :func:`pixel_shuffle_tokens`. Identical token count and
    spatial layout; the only thing that changes is that the r*r sub-cells are
    averaged instead of concatenated, so sub-cell structure is destroyed while
    sequence length — the other candidate explanation for s2d's ramp gain — is
    held fixed.
    """
    blocks, _ = _split_blocks(x, r, "pooling")
    return blocks.mean(dim=3)


def evs_select(
    tokens: torch.Tensor, keep: int, mode: str = "novelty", n_groups: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Efficient Video Sampling — keep ``keep`` tokens out of ``T*n_cells``.

    ``mode="novelty"`` (default, the shipped s2d rule): per spatial cell, score
    each frame by cosine *dissimilarity* to the same cell in the previous frame;
    frame 0 is pinned to ``+inf`` so the anchor field is always retained whole.

    ``mode="random"`` (A38): uniform random scores. Same ``keep``, same output
    shapes, same downstream path — so any metric difference is attributable to
    the *selection rule* rather than to token count. This is the foil for "EVS
    is doing nothing but setting the sequence length", which the length-confounded
    A30-d q-sweep could not separate.

    Scores are ranked globally over ``T*n_cells`` and the surviving indices are
    re-sorted ascending, so the kept sequence stays in (frame, cell) order and
    the sequence length is a fixed ``keep``.

    ``n_groups > 1`` (ticket 45, canonical multi-anchor interleaving) splits the
    frame axis into ``n_groups`` contiguous temporal anchors and gives each its
    own budget of ``keep // n_groups``, ranked only against its own frames. A
    global top-K would slice one ranked list arbitrarily, so the caller's later
    ``reshape(B, n_vis, N_vis_tok, d)`` would not put anchor *i*'s tokens in
    block *i* — the blocks would be temporally scrambled. Folding groups into the
    batch axis and recursing keeps every rule identical per anchor: novelty's
    pinned frame-0 ``+inf`` becomes a per-anchor pin for free, and ``"random"``
    stays uniform within each anchor.

    Nemotron applies EVS at runtime only (§2.3); here it is inside training too —
    a deliberate deviation, flagged in the design doc §5.2.

    Args:
        tokens: ``[B, T, n_cells, d]``
        keep: number of tokens to retain; ``>= T*n_cells`` is a no-op.
        mode: ``"novelty"`` | ``"random"``.
        n_groups: temporal anchors to budget separately. ``1`` (default) is the
            shipped s2d behaviour — one global ranking over the whole window.

    Returns:
        kept:      ``[B, keep, d]``
        frame_idx: ``[B, keep]`` long — source frame of each kept token, in the
                   ORIGINAL ``[0, T)`` numbering even when grouped
        cell_idx:  ``[B, keep]`` long — source spatial cell of each kept token
    """
    if mode not in EVS_MODES:
        raise ValueError(f"unknown EVS mode: {mode!r}; expected one of {EVS_MODES}")
    B, T, C, d = tokens.shape

    if n_groups > 1:
        if T % n_groups != 0:
            raise ValueError(
                f"cannot split {T} latent frames into {n_groups} anchors evenly"
            )
        if keep % n_groups != 0:
            raise ValueError(
                f"EVS keep={keep} is not divisible by n_groups={n_groups}; every "
                "anchor must carry the same token count or the interleaved blocks "
                "are ragged"
            )
        T_g, keep_g = T // n_groups, keep // n_groups
        grouped = tokens.reshape(B * n_groups, T_g, C, d)
        kept, frame_idx, cell_idx = evs_select(grouped, keep_g, mode=mode)
        # keep_g >= T_g*C is a no-op inside the recursion and returns the whole
        # anchor, so read the realised count back rather than trusting keep_g.
        keep_g = kept.shape[1]
        # Frame indices come back local to each anchor; lift them back onto the
        # original [0, T) clock so the caller's Δt gather still lines up.
        offset = (
            torch.arange(n_groups, device=tokens.device).repeat_interleave(keep_g) * T_g
        )
        return (
            kept.reshape(B, n_groups * keep_g, d),
            frame_idx.reshape(B, n_groups * keep_g) + offset[None, :],
            cell_idx.reshape(B, n_groups * keep_g),
        )

    N = T * C
    flat = tokens.reshape(B, N, d)
    if keep >= N:
        idx = torch.arange(N, device=tokens.device).expand(B, N)
        return flat, idx // C, idx % C

    if mode == "random":
        # float32 regardless of autocast dtype: bf16 ties would make the "random"
        # subset partly deterministic through topk's tie-breaking order.
        scores = torch.rand(B, N, device=tokens.device, dtype=torch.float32)
    else:
        prev = tokens[:, :-1]
        cur = tokens[:, 1:]
        dissim = 1.0 - torch.nn.functional.cosine_similarity(
            cur, prev, dim=-1, eps=1e-6
        )
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
        shuffle_r: pixel-shuffle / pooling factor. 2 gives Nemotron's 4x token
            reduction (14x14 -> 7x7).
        n_cells: spatial cells after the merge; sized for the cell embedding
            table and asserted against the actual grid at forward time.
        evs_keep: tokens surviving EVS. ``<= 0`` disables pruning (the q=0 eval).
        dropout: applied after the projector.
        pool_mode: ``"shuffle"`` (s2d) | ``"avg"`` (A37). NOTE this changes
            ``proj[0].in_features`` (``d_v*r*r`` vs ``d_v``), so an ``avg`` arm
            cannot warm-start from an s2d checkpoint — train it from s1.
        evs_mode: ``"novelty"`` (s2d) | ``"random"`` (A38). Parameter-free, so
            an ``evs_mode`` foil CAN be evaluated on an existing s2d checkpoint.
        evs_groups: temporal anchors to budget EVS over separately — set this to
            ``n_visual_context_steps``. ``1`` (s2d) ranks the whole window at
            once; ``>1`` (ticket 45) gives each interleaved anchor its own
            ``evs_keep // evs_groups`` tokens drawn from its own frames, which is
            what makes the caller's per-anchor regrouping temporally coherent.
    """

    def __init__(
        self,
        d_v: int,
        d_model: int,
        shuffle_r: int = 2,
        n_cells: int = 49,
        evs_keep: int = 98,
        dropout: float = 0.1,
        pool_mode: str = "shuffle",
        evs_mode: str = "novelty",
        evs_groups: int = 1,
    ):
        super().__init__()
        if pool_mode not in POOL_MODES:
            raise ValueError(
                f"unknown pool_mode: {pool_mode!r}; expected one of {POOL_MODES}"
            )
        if evs_mode not in EVS_MODES:
            raise ValueError(
                f"unknown evs_mode: {evs_mode!r}; expected one of {EVS_MODES}"
            )
        self.shuffle_r = int(shuffle_r)
        self.n_cells = int(n_cells)
        self.evs_keep = int(evs_keep)
        self.evs_groups = max(1, int(evs_groups))
        self.pool_mode = pool_mode
        self.evs_mode = evs_mode
        d_in = d_v * self.shuffle_r * self.shuffle_r if pool_mode == "shuffle" else d_v
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
        if self.pool_mode == "shuffle":
            x = pixel_shuffle_tokens(video_latents, self.shuffle_r)  # [B,T,C,d_v*r*r]
        else:
            x = avg_pool_tokens(video_latents, self.shuffle_r)  # [B, T, C, d_v]
        if x.shape[2] != self.n_cells:
            raise ValueError(
                f"{self.pool_mode} merge produced {x.shape[2]} cells but "
                f"n_cells={self.n_cells}; check shuffle_r={self.shuffle_r} against "
                "the cache patch grid."
            )
        x = self.dropout(self.proj(x))
        x = x + self.cell_embed[None, None, :, :]
        keep = self.evs_keep if self.evs_keep > 0 else x.shape[1] * self.n_cells
        return evs_select(x, keep, mode=self.evs_mode, n_groups=self.evs_groups)

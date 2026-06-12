"""Latent Summarization module for Vision-Time FM.

Resolves the frequency mismatch between the dense video latent stream
(T_lat × P spatial-temporal tokens from VidTok) and the forecasting
cadence (T_ts TS timesteps).

Architecture: Causal Perceiver-per-timestep cross-attention compressor.
  - Learned queries: one per visual context TS step  [n_vis_steps, d_model]
  - Keys / Values:   all flattened video latent tokens  [T_lat * P, D_v_proj]
  - Causal mask:     query t_vis sees only frames in its causal sub-interval
  - Output:          Visual Summary Tokens  [B, n_vis_steps, d_model]
  - Padding:         null_visual_token (learned parameter) for TS steps outside the visual context window

The output is aligned 1:1 with the TS token sequence so it can be directly
injected into Group Attention in VisionChronos2.

Input shapes (one entity / one sample)
---------------------------------------
video_tokens  : [B, T_lat, P, D_v]   from VidTokEncoder
T_ts          : int                   number of TS context patches (encoder input)
n_vis_steps   : int                   how many recent TS steps have visual coverage

Output
------
visual_summary : [B, T_ts, d_model]
  Filled with ``null_visual_token`` (a learned parameter) for macro positions
  outside the visual window; visual summary tokens occupy the last
  ``n_vis_steps`` positions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentSummarizer(nn.Module):
    """Causal Perceiver-per-timestep cross-attention compressor.

    C3 fix: each query at TS position t_vis (0-indexed within the visual
    window) may only attend to video frames whose temporal index falls within
    the causal sub-interval [0, ceil((t_vis+1)*T_lat/n_vis_steps) - 1].
    Future frames are blocked via an additive -inf attn_mask built once per
    forward call, keeping the implementation as a single fused MHA call with
    no Python loop over timesteps.

    Parameters
    ----------
    d_v:
        Dimension of VidTok latent tokens (D_v, e.g. 4 for KL-4ch).
    d_model:
        Chronos-2 hidden dimension.
    n_vis_steps:
        Number of recent TS context steps covered by the visual window.
        These are the *last* n_vis_steps positions of the TS sequence.
    n_heads:
        Number of attention heads for cross-attention.
    dropout:
        Dropout applied to cross-attention output.
    """

    def __init__(
        self,
        d_v: int,
        d_model: int,
        n_vis_steps: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_v = d_v
        self.d_model = d_model
        self.n_vis_steps = n_vis_steps
        self.n_heads = n_heads

        # Project VidTok latent dim → d_model (K, V projection)
        self.kv_proj = nn.Linear(d_v, d_model, bias=False)

        # Learned latent queries — one per visual context step
        self.latent_queries = nn.Parameter(
            torch.randn(1, n_vis_steps, d_model) * (d_model ** -0.5)
        )

        # Manual cross-attention projections.
        # nn.MultiheadAttention is replaced because its Flash/MemEff backends on
        # A100 produce NaN in the backward even when the forward is numerically
        # stable — the tiled softmax in FlashAttn computes logsumexp in a way
        # that can hit NaN under bf16 autocast during backprop.  A manual eager
        # implementation with explicit nan_to_num guards at every step avoids
        # every attention backend entirely.
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads
        self.q_proj  = nn.Linear(d_model, d_model, bias=True)
        self.k_proj  = nn.Linear(d_model, d_model, bias=True)
        self.v_proj  = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(dropout)   # applied to attention weights
        self.dropout   = nn.Dropout(dropout)   # applied to attention output

        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_kv = nn.LayerNorm(d_model)

        # Learned null token for macro positions (outside visual window).
        # Prevents degenerate Plücker subspaces at macro/refinement boundary.
        # Init N(0, d^{-1/2}) to keep scale consistent with d_model embedding norms.
        self.null_visual_token = nn.Parameter(
            torch.randn(1, 1, d_model) * (d_model ** -0.5)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_causal_attn_mask(
        self, n_vis: int, T_lat: int, P: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build additive causal mask ``[n_vis, T_lat * P]``.

        Query at position t_vis (0-indexed) is allowed to attend to frames
        whose temporal index is strictly less than
        ``ceil((t_vis + 1) * T_lat / n_vis)``.
        All spatial patches (P) of frames beyond that boundary receive -1e4.

        Returns
        -------
        mask : ``[n_vis, T_lat * P]``  float tensor — 0 = attend, -1e4 = block.
        """
        t_vis_idx = torch.arange(n_vis, device=device)                 # [n_vis]
        # ceil division: number of frames visible to query t_vis
        frame_limit = ((t_vis_idx + 1) * T_lat + n_vis - 1) // n_vis  # [n_vis]
        frame_limit = frame_limit.clamp(max=T_lat)

        # frame index repeated P times for spatial patches: [T_lat * P]
        frame_idx = torch.arange(T_lat, device=device).repeat_interleave(P)

        # Use -1e4 instead of -inf: large enough to suppress attention but avoids
        # the -inf - (-inf) = NaN that FlashAttention/MemEffAttn backward can
        # trigger when all keys for a query row are masked. The numerically stable
        # softmax handles -1e4 identically to -inf for attended positions.
        mask = torch.where(
            frame_idx.unsqueeze(0) < frame_limit.unsqueeze(1),  # [n_vis, T_lat*P]
            torch.zeros(1, device=device, dtype=torch.float32),
            torch.full((1,), -1e4, device=device, dtype=torch.float32),
        )
        return mask  # [n_vis, T_lat * P]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        video_tokens: torch.Tensor,
        T_ts: int,
        visual_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compress video latents to causal visual summary tokens.

        Parameters
        ----------
        video_tokens:
            ``[B, T_lat, P, D_v]`` — output of VidTokEncoder.
        T_ts:
            Number of TS context patches (encoder sequence length).
        visual_mask:
            ``[B, T_lat]`` — 1 = frame available, 0 = missing/corrupt.
            If None, all frames are treated as available.

        Returns
        -------
        visual_summary : ``[B, T_ts, d_model]``
            Zero-padded for TS steps outside the visual window.
            Query at TS position t only attends to frames in its causal
            sub-interval — no future-frame leakage.
        """
        B, T_lat, P, D_v = video_tokens.shape
        device = video_tokens.device
        dtype = video_tokens.dtype

        n_vis = min(self.n_vis_steps, T_ts)

        kv_len = T_lat * P

        # Flatten spatial & temporal → [B, kv_len, D_v]
        kv_flat = video_tokens.reshape(B, kv_len, D_v)
        kv_flat = torch.nan_to_num(kv_flat, nan=0.0)  # guard: NaN from video encoder

        # Project → [B, kv_len, d_model]
        kv = self.kv_proj(kv_flat)
        kv = self.layer_norm_kv(kv)
        kv = torch.nan_to_num(kv, nan=0.0)

        # Causal temporal-window mask: [n_vis, kv_len], values 0 or -1e4
        causal_mask = self._build_causal_attn_mask(n_vis, T_lat, P, device, torch.float32)
        # [1, 1, n_vis, kv_len] — broadcast over batch and heads in scores
        mask = causal_mask[None, None, :, :]  # [1, 1, n_vis, kv_len]
        if visual_mask is not None:
            frame_exp = visual_mask.unsqueeze(-1).expand(B, T_lat, P).reshape(B, kv_len)
            pad_penalty = (1.0 - frame_exp.float()) * -1e4   # [B, kv_len]
            mask = mask + pad_penalty[:, None, None, :]       # [B, 1, n_vis, kv_len]

        # Learned queries: [B, n_vis, d_model]
        queries = self.latent_queries[:, :n_vis, :].expand(B, -1, -1)
        queries = self.layer_norm_q(queries)

        # --- Manual eager multi-head cross-attention -------------------------
        # Using manual attention instead of nn.MultiheadAttention to avoid
        # Flash Attention / MemEffAttn backends which produce NaN in backward
        # under bf16 autocast on A100 even with finite (-1e4) mask values.
        h, d_h = self.n_heads, self.d_head
        Q = self.q_proj(queries).view(B, n_vis, h, d_h).transpose(1, 2)   # [B,h,n_vis,d_h]
        K = self.k_proj(kv).view(B, kv_len, h, d_h).transpose(1, 2)       # [B,h,kv_len,d_h]
        V = self.v_proj(kv).view(B, kv_len, h, d_h).transpose(1, 2)       # [B,h,kv_len,d_h]

        scores = (Q @ K.transpose(-2, -1)) * (d_h ** -0.5)  # [B,h,n_vis,kv_len]
        scores = scores + mask                                # add causal + pad mask
        # Guard: replace any NaN/inf in scores before softmax (handles all-masked rows)
        scores = torch.nan_to_num(scores.float(), nan=0.0, neginf=-1e4)
        attn_w = F.softmax(scores, dim=-1).to(Q.dtype)       # [B,h,n_vis,kv_len]
        attn_w = torch.nan_to_num(attn_w, nan=0.0)           # zero out NaN rows
        attn_w = self.attn_drop(attn_w)
        attn_out = attn_w @ V                                 # [B,h,n_vis,d_h]
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, n_vis, self.d_model)
        attn_out = self.out_proj(attn_out)
        # Final guard: catch any remaining NaN before returning
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        attn_out = self.dropout(attn_out)

        # Macro positions → learned null token; refinement window → cross-attn output.
        # Using null_visual_token (not zeros) prevents degenerate Plücker subspaces.
        T_macro = T_ts - n_vis
        null = self.null_visual_token.expand(B, T_macro, self.d_model).to(dtype)
        if n_vis > 0:
            visual_summary = torch.cat([null, attn_out], dim=1)  # [B, T_ts, d_model]
        else:
            visual_summary = null  # [B, T_ts, d_model]

        return visual_summary

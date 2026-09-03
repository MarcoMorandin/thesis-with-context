"""Latent Summarization module for Vision-Time FM.

Resolves the frequency mismatch between the dense video latent stream
(T_lat × P spatial-temporal tokens from V-JEPA 2.1) and the forecasting
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
video_tokens  : [B, T_lat, P, D_v]   from the V-JEPA video encoder
T_ts          : int                   number of TS context patches (encoder input)
n_vis_steps   : int                   how many recent TS steps have visual coverage

Output
------
visual_summary : [B, T_ts, d_model]
  Filled with ``null_visual_token`` (a learned parameter) for macro positions
  outside the visual window; visual summary tokens occupy the last
  ``n_vis_steps`` positions.

  A29: with ``n_time_slices`` or ``spatial_grid`` above 1 the summarizer emits
  ``n_sub = n_time_slices * spatial_grid**2`` tokens per visual step instead of
  one, each masked to its own (temporal slice, spatial block), and the return
  becomes ``[B, T_ts, n_sub, d_model]``. Both 1 → the 3-D output above, byte
  for byte.
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
        Dimension of V-JEPA latent tokens (D_v, e.g. 4 for KL-4ch).
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
        n_time_slices: int = 1,
        spatial_grid: int = 1,
    ):
        super().__init__()
        self.d_v = d_v
        self.d_model = d_model
        self.n_vis_steps = n_vis_steps
        self.n_heads = n_heads
        # A29: sub-resolution inside each visual TS step. n_sub == 1 reproduces the
        # historical single-query-per-step behaviour EXACTLY — same query shape, same
        # mask, same 3-D return — so existing checkpoints load and the four already
        # published arms are bit-identical. See the class docstring.
        self.n_time_slices = max(1, int(n_time_slices))
        self.spatial_grid = max(1, int(spatial_grid))
        self.n_spatial = self.spatial_grid**2
        self.n_sub = self.n_time_slices * self.n_spatial

        # Project V-JEPA latent dim → d_model (K, V projection)
        self.kv_proj = nn.Linear(d_v, d_model, bias=False)

        # Learned latent queries — n_sub per visual context step, laid out
        # (t_vis, sub) with sub varying fastest. At n_sub == 1 this is
        # [1, n_vis_steps, d_model], byte-identical to the pre-A29 parameter, so
        # state_dict shape is unchanged and any existing checkpoint warm-starts.
        self.latent_queries = nn.Parameter(
            torch.randn(1, n_vis_steps * self.n_sub, d_model) * (d_model**-0.5)
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
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(dropout)  # applied to attention weights
        self.dropout = nn.Dropout(dropout)  # applied to attention output

        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_kv = nn.LayerNorm(d_model)

        # Learned null token for macro positions (outside visual window).
        # Prevents degenerate Plücker subspaces at macro/refinement boundary.
        # Init N(0, d^{-1/2}) to keep scale consistent with d_model embedding norms.
        self.null_visual_token = nn.Parameter(
            torch.randn(1, 1, d_model) * (d_model**-0.5)
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
        t_vis_idx = torch.arange(n_vis, device=device)  # [n_vis]
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

    def _build_time_attn_mask(
        self,
        frame_delta_t: torch.Tensor,
        n_vis: int,
        T_lat: int,
        P: int,
    ) -> torch.Tensor:
        """Build a per-sample causal mask ``[B, n_vis, T_lat * P]`` from true Δt.

        Unlike :meth:`_build_causal_attn_mask` (which assumes uniformly spaced
        frames), this maps each frame to a normalized time fraction in ``[0, 1]``
        (oldest → 0, newest → 1) using ``frame_delta_t`` (seconds before the
        forecast origin). Query ``t_vis`` may attend to a frame whose time
        fraction is ``<= (t_vis + 1) / n_vis`` — i.e. progressively wider causal
        windows that reflect the actual temporal spacing of the frames.
        """
        device = frame_delta_t.device
        B = frame_delta_t.shape[0]
        dt = frame_delta_t.float()  # [B, T_lat]
        # Normalize to recency fraction per sample; oldest→0, newest→1.
        dt_max = dt.amax(dim=1, keepdim=True)
        dt_min = dt.amin(dim=1, keepdim=True)
        span = (dt_max - dt_min).clamp(min=1e-6)
        time_frac = (dt_max - dt) / span  # [B, T_lat]
        # Per-spatial-patch expansion → [B, kv_len]
        time_frac = time_frac.unsqueeze(-1).expand(B, T_lat, P).reshape(B, T_lat * P)

        # Per-query thresholds (t_vis+1)/n_vis → [n_vis]
        q_idx = torch.arange(n_vis, device=device, dtype=torch.float32)
        thresholds = (q_idx + 1.0) / max(n_vis, 1)  # [n_vis]

        visible = time_frac[:, None, :] <= (thresholds[None, :, None] + 1e-6)
        mask = torch.where(
            visible,
            torch.zeros(1, device=device, dtype=torch.float32),
            torch.full((1,), -1e4, device=device, dtype=torch.float32),
        )
        return mask  # [B, n_vis, T_lat * P]

    def _build_sub_attn_mask(
        self, T_lat: int, P: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the sub-resolution masks, both ``[n_sub, T_lat * P]`` (A29).

        Splits each visual TS step into ``n_time_slices`` contiguous temporal
        slices and ``spatial_grid**2`` spatial blocks, so query ``(tau, s)`` sees
        only the patches of its own block within its own slice. This is what makes
        ``n_sub`` genuine bandwidth rather than a fan-out: without it, extra
        queries are free to collapse onto the same global average, which is
        exactly what the pooled-token widening (A13) demonstrated.

        Sub-index layout is ``tau * n_spatial + s``, matching ``latent_queries``.
        Returns all-zero masks when ``n_sub == 1`` so the pre-A29 behaviour is
        recovered exactly.

        Returns
        -------
        sub_mask : ``[n_sub, T_lat * P]``
            Temporal slice AND spatial block — the intended restriction.
        spatial_mask : ``[n_sub, T_lat * P]``
            Spatial block only. Used as the fallback for a sub-query whose
            temporal slice lies entirely outside its step's causal window: the
            time restriction is dropped, the spatial identity is kept, so the
            query never degenerates into a copy of its neighbours.
        """
        kv_len = T_lat * P
        if self.n_sub == 1:
            z = torch.zeros(1, kv_len, device=device, dtype=torch.float32)
            return z, z

        frame_idx = torch.arange(T_lat, device=device).repeat_interleave(P)
        patch_idx = torch.arange(P, device=device).repeat(T_lat)

        # --- temporal slices over the full latent-frame axis ---
        n_t = self.n_time_slices
        tau = torch.arange(n_t, device=device)
        lo = (tau * T_lat) // n_t
        hi = ((tau + 1) * T_lat) // n_t
        hi = torch.maximum(hi, lo + 1).clamp(max=T_lat)  # never an empty slice
        t_ok = (frame_idx[None, :] >= lo[:, None]) & (frame_idx[None, :] < hi[:, None])

        # --- spatial blocks over the square patch grid ---
        g = self.spatial_grid
        if g == 1:
            s_ok = torch.ones(1, kv_len, device=device, dtype=torch.bool)
        else:
            side = int(math.isqrt(P))
            if side * side != P:
                raise ValueError(
                    f"spatial_grid={g} requires a square patch grid, but the visual "
                    f"encoder emitted P={P} patches per frame, which is not a perfect "
                    f"square. Set spatial_grid=1 or supply a square-grid encoder."
                )
            if g > side:
                raise ValueError(
                    f"spatial_grid={g} exceeds the native patch grid side {side}. "
                    f"Choose spatial_grid <= {side}."
                )
            blk = ((patch_idx // side) * g // side) * g + (
                (patch_idx % side) * g // side
            )
            s_ok = blk[None, :] == torch.arange(g * g, device=device)[:, None]

        # [n_t, 1, kv] & [1, n_spatial, kv] -> [n_t, n_spatial, kv] -> [n_sub, kv]
        ok = (t_ok[:, None, :] & s_ok[None, :, :]).reshape(self.n_sub, kv_len)
        s_only = s_ok[None, :, :].expand(n_t, self.n_spatial, kv_len)
        s_only = s_only.reshape(self.n_sub, kv_len)

        zero = torch.zeros(1, device=device, dtype=torch.float32)
        blocked = torch.full((1,), -1e4, device=device, dtype=torch.float32)
        return torch.where(ok, zero, blocked), torch.where(s_only, zero, blocked)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        video_tokens: torch.Tensor,
        T_ts: int,
        visual_mask: torch.Tensor | None = None,
        frame_delta_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compress video latents to causal visual summary tokens.

        Parameters
        ----------
        video_tokens:
            ``[B, T_lat, P, D_v]`` — output of the V-JEPA video encoder.
        T_ts:
            Number of TS context patches (encoder sequence length).
        visual_mask:
            ``[B, T_lat]`` — 1 = frame available, 0 = missing/corrupt.
            If None, all frames are treated as available.
        frame_delta_t:
            ``[B, T_lat]`` — seconds before the forecast origin for each latent
            frame (0 = now, larger = older). When provided, the causal
            attention window is built from true temporal spacing rather than
            assuming uniformly spaced frames (W5). Accepts per-frame Δt of
            length ``k · T_lat`` (pooled to latent resolution via per-group
            max) or length ``T_lat``; anything else is ignored and uniform
            spacing is used.

        Returns
        -------
        visual_summary : ``[B, T_ts, d_model]`` when ``n_sub == 1``,
            otherwise ``[B, T_ts, n_sub, d_model]`` (A29).
            Null-padded for TS steps outside the visual window.
            Query at TS position t only attends to frames in its causal
            sub-interval — no future-frame leakage. Sub-tokens within a step are
            ordered ``time_slice * n_spatial + spatial_block``.
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

        # Causal temporal-window mask.
        # Default: uniform-spacing boundary [n_vis, kv_len] → broadcast over batch.
        # W5: when frame_delta_t is supplied (and matches T_lat), build the
        # boundary from true temporal spacing instead → [B, n_vis, kv_len].
        # The dataloader emits per-FRAME Δt (length T_v) while V-JEPA's temporal
        # stride folds frames into T_lat latent steps (T_v = stride · T_lat), so
        # pool Δt to latent resolution first — otherwise the shape check below
        # never passes and W5 silently degrades to uniform spacing on every run.
        # amax picks the OLDER frame of each stride group, which also ignores the
        # Δt=0 left-padding of missing frames (those slots are masked anyway).
        if frame_delta_t is not None and frame_delta_t.shape[-1] != T_lat:
            L = frame_delta_t.shape[-1]
            if L % T_lat == 0:
                frame_delta_t = frame_delta_t.reshape(B, T_lat, L // T_lat).amax(dim=-1)
        use_dt = frame_delta_t is not None and frame_delta_t.shape[-1] == T_lat
        if use_dt:
            time_mask = self._build_time_attn_mask(frame_delta_t, n_vis, T_lat, P)
            mask = time_mask[:, None, :, :]  # [B, 1, n_vis, kv_len]
        else:
            causal_mask = self._build_causal_attn_mask(
                n_vis, T_lat, P, device, torch.float32
            )
            # [1, 1, n_vis, kv_len] — broadcast over batch and heads in scores
            mask = causal_mask[None, None, :, :]  # [1, 1, n_vis, kv_len]

        # A29: split each visual step into n_sub = (temporal slice x spatial block)
        # queries. At n_sub == 1 this whole block is skipped and `mask` is untouched.
        n_sub = self.n_sub
        n_q = n_vis * n_sub
        if n_sub > 1:
            sub, spat = self._build_sub_attn_mask(T_lat, P, device)  # [n_sub, kv_len]
            base = mask.unsqueeze(-2)  # [., ., n_vis, 1, kv_len]
            wide = base + sub.view(1, 1, 1, n_sub, kv_len)
            # A sub-query whose temporal slice falls entirely outside its step's
            # causal window would have every key blocked, and a uniformly-blocked
            # softmax row attends to EVERYTHING — future frames included. Fall back
            # to causal AND spatial there: the causal guarantee then holds for every
            # (t_vis, sub) pair, and the query keeps its own spatial block instead
            # of degenerating into a copy of its neighbours.
            fallback = base + spat.view(1, 1, 1, n_sub, kv_len)
            blocked = (wide <= -9e3).all(dim=-1, keepdim=True)
            wide = torch.where(blocked, fallback.expand_as(wide), wide)
            mask = wide.reshape(wide.shape[0], wide.shape[1], n_q, kv_len)

        if visual_mask is not None:
            frame_exp = visual_mask.unsqueeze(-1).expand(B, T_lat, P).reshape(B, kv_len)
            pad_penalty = (1.0 - frame_exp.float()) * -1e4  # [B, kv_len]
            mask = mask + pad_penalty[:, None, None, :]  # [B, 1, n_q, kv_len]

        # Learned queries: [B, n_q, d_model], laid out (t_vis, sub) with sub fastest
        queries = self.latent_queries[:, :n_q, :].expand(B, -1, -1)
        queries = self.layer_norm_q(queries)

        # --- Manual eager multi-head cross-attention -------------------------
        # Using manual attention instead of nn.MultiheadAttention to avoid
        # Flash Attention / MemEffAttn backends which produce NaN in backward
        # under bf16 autocast on A100 even with finite (-1e4) mask values.
        h, d_h = self.n_heads, self.d_head
        Q = self.q_proj(queries).view(B, n_q, h, d_h).transpose(1, 2)  # [B,h,n_q,d_h]
        K = self.k_proj(kv).view(B, kv_len, h, d_h).transpose(1, 2)  # [B,h,kv_len,d_h]
        V = self.v_proj(kv).view(B, kv_len, h, d_h).transpose(1, 2)  # [B,h,kv_len,d_h]

        scores = (Q @ K.transpose(-2, -1)) * (d_h**-0.5)  # [B,h,n_q,kv_len]
        scores = scores + mask  # add causal + sub + pad mask
        # Guard: replace any NaN/inf in scores before softmax (handles all-masked rows)
        scores = torch.nan_to_num(scores.float(), nan=0.0, neginf=-1e4)
        attn_w = F.softmax(scores, dim=-1).to(Q.dtype)  # [B,h,n_q,kv_len]
        attn_w = torch.nan_to_num(attn_w, nan=0.0)  # zero out NaN rows
        attn_w = self.attn_drop(attn_w)
        attn_out = attn_w @ V  # [B,h,n_q,d_h]
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, n_q, self.d_model)
        attn_out = self.out_proj(attn_out)
        # Final guard: catch any remaining NaN before returning
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        attn_out = self.dropout(attn_out)

        # Macro positions → learned null token; refinement window → cross-attn output.
        # Using null_visual_token (not zeros) prevents degenerate Plücker subspaces.
        T_macro = T_ts - n_vis
        if n_sub == 1:
            # Pre-A29 path, preserved exactly: [B, T_ts, d_model].
            null = self.null_visual_token.expand(B, T_macro, self.d_model).to(dtype)
            if n_vis > 0:
                return torch.cat([null, attn_out], dim=1)  # [B, T_ts, d_model]
            return null

        # A29 path: [B, T_ts, n_sub, d_model]. Every macro position carries n_sub
        # copies of the null token so the shape is uniform across the sequence and
        # the caller can treat sub-tokens as ordinary visual tokens.
        sub_out = attn_out.view(B, n_vis, n_sub, self.d_model)
        null = self.null_visual_token.view(1, 1, 1, self.d_model)
        null = null.expand(B, T_macro, n_sub, self.d_model).to(dtype)
        if n_vis > 0:
            return torch.cat([null, sub_out], dim=1)  # [B, T_ts, n_sub, d_model]
        return null

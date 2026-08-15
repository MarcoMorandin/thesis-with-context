"""CausalGrassmannMixing — novel O(L) temporal mixing layer.

Implements the Grassmann flow algorithm (Section 3.2): causal multi-scale
Plücker pairing as an O(L) replacement for temporal self-attention.
"""

import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import Chronos2CoreConfig
from .layers import AttentionOutput, Chronos2LayerNorm, RoPE

# Forward-side NaN tracer. Off unless MMTSFM_GRASSMANN_DEBUG=1.
#
# The per-block guards in model.py:75-100 scrub each block's output AND its
# gradient, so a NaN born inside this layer never reaches the loss, the trunk,
# or a neighbouring block — it shows up only as that block's parameter
# gradients going non-finite (dL/dW = dL/dz · hᵀ with a NaN in the saved
# activation h). By the time on_before_optimizer_step sees it, every trace of
# WHERE it started is gone. This reports the first non-finite intermediate
# inside the forward, before any guard runs.
_DEBUG = os.environ.get("MMTSFM_GRASSMANN_DEBUG", "") not in ("", "0")
_DEBUG_MAX = int(os.environ.get("MMTSFM_GRASSMANN_DEBUG_MAX", "40"))
_dbg_emitted = 0


def _probe(layer, tag: str, t: torch.Tensor, **ctx) -> torch.Tensor:
    """Report `t` if it holds any non-finite value. Returns `t` untouched.

    Prints at most MMTSFM_GRASSMANN_DEBUG_MAX lines per process so a
    100%-of-steps failure does not produce a gigabyte of log.
    """
    global _dbg_emitted
    if not _DEBUG or _dbg_emitted >= _DEBUG_MAX:
        return t
    finite = torch.isfinite(t)
    if finite.all():
        return t
    _dbg_emitted += 1
    bad = ~finite
    n_bad = int(bad.sum())
    # Which sequence positions are affected, and which batch rows.
    pos = rows = "n/a"
    if t.dim() >= 2:
        pos_bad = bad.any(dim=0)
        while pos_bad.dim() > 1:
            pos_bad = pos_bad.any(dim=-1)
        idx = torch.nonzero(pos_bad).flatten().tolist()
        pos = f"{idx[:12]}{'...' if len(idx) > 12 else ''} ({len(idx)}/{t.shape[1]})"
        row_bad = bad.flatten(1).any(dim=1)
        ridx = torch.nonzero(row_bad).flatten().tolist()
        rows = (
            f"{ridx[:12]}{'...' if len(ridx) > 12 else ''} ({len(ridx)}/{t.shape[0]})"
        )
    finite_vals = t.detach()[finite]
    peak = float(finite_vals.abs().max()) if finite_vals.numel() else float("nan")
    extra = " ".join(f"{k}={v}" for k, v in ctx.items())
    print(
        f"[grassmann-nan] blk={getattr(layer, '_dbg_idx', '?')} tag={tag} "
        f"shape={tuple(t.shape)} dtype={t.dtype} nonfinite={n_bad}/{t.numel()} "
        f"nan={int(torch.isnan(t).sum())} posinf={int((t == float('inf')).sum())} "
        f"neginf={int((t == float('-inf')).sum())} max_finite_abs={peak:.4e} "
        f"seq_pos={pos} batch_rows={rows} {extra}",
        flush=True,
    )
    return t


class CausalGrassmannMixing(nn.Module):
    """Causal Grassmann Mixing layer — O(L) replacement for self-attention.

    Implements the algorithm from Section 3.2 of the Grassmann Flow paper:
    causal (forward-only) pairing with multi-scale offsets and Plücker encoding.
    """

    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()

        d_model = config.d_model
        r = config.grassmann_reduced_dim
        assert r % 2 == 0, (
            f"grassmann_reduced_dim must be even for RoPE (sin/cos pairs); got {r}. "
            "Set grassmann_reduced_dim to an even value in your config."
        )
        self.plucker_eps = config.grassmann_plucker_eps
        self.r = r
        self.window_offsets = config.grassmann_window_offsets
        self.num_offsets = len(self.window_offsets)

        # Plücker dimension: C(r, 2) = r*(r-1)/2
        self.plucker_dim = r * (r - 1) // 2

        self.layer_norm = Chronos2LayerNorm(d_model, eps=config.layer_norm_epsilon)

        self.W_red = nn.Linear(d_model, r, bias=True)
        self.rope_embed = RoPE(dim=r, base=config.rope_theta)

        self.W_plu = nn.Linear(self.plucker_dim, d_model, bias=True)
        self.W_gate = nn.Linear(2 * d_model, d_model, bias=True)

        # Learned weights for each window offset
        self.offset_weights = nn.Parameter(torch.ones(self.num_offsets))

        self.dropout = nn.Dropout(config.dropout_rate)

        self.use_modality_pair_bias = getattr(
            config, "grassmann_modality_pair_bias", False
        )
        if self.use_modality_pair_bias:
            # 4 scalar biases: TT=0, TV=1, VT=2, VV=3
            # Added to offset logit before softmax for position-dependent weighting.
            # Init zeros → no initial bias toward any pair type.
            self.modality_pair_bias = nn.Parameter(torch.zeros(4))

        # Plücker pair indices (upper-triangular, offset=1) are NOT stored as
        # buffers. When the model is built on `meta` and materialized with
        # `to_empty()` (Chronos-2 pretrained load path), every buffer — persistent
        # or not — is allocated with uninitialized storage; index buffers absent
        # from the checkpoint are then left as garbage, causing out-of-bounds
        # advanced-indexing (CUDA device-side assert / IndexError). Deriving them
        # on-device inside _compute_plucker is O(r²), negligible next to the linear
        # layers, and immune to materialization order.

    def _compute_plucker(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute normalized Plücker vectors for pairs (u, v)."""
        # Advanced indexing (u[..., idx]) instead of torch.gather + expand.
        # gather on stride-0 expanded views triggers CUDA OOB asserts on A100+newer
        # drivers even after .contiguous(). Advanced indexing uses a safe kernel path.
        # Indices derived on-the-fly on u's device (see __init__ note).
        idx_i, idx_j = torch.triu_indices(self.r, self.r, offset=1, device=u.device)

        u_i = u[..., idx_i]  # [B, L, plucker_dim]
        v_j = v[..., idx_j]
        u_j = u[..., idx_j]
        v_i = v[..., idx_i]

        p = u_i * v_j - u_j * v_i
        _probe(self, "plucker_raw", p)

        p_norm = torch.sqrt((p * p).sum(dim=-1, keepdim=True) + self.plucker_eps)
        _probe(self, "plucker_norm", p_norm)
        p = p / p_norm
        _probe(self, "plucker_normalised", p)
        return p

    def _pair_validity(
        self,
        valid_mask: torch.Tensor,  # [B, L] bool
        valid_offsets: list,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """[B, L, n_valid] — 1 where BOTH endpoints of the (i-δ, i) pair are valid."""
        B, L = valid_mask.shape
        out = torch.zeros(
            B, L, len(valid_offsets), device=valid_mask.device, dtype=dtype
        )
        for k, delta in enumerate(valid_offsets):
            out[:, delta:, k] = (valid_mask[:, : L - delta] & valid_mask[:, delta:]).to(
                dtype
            )
        return out

    def _process_offset(
        self,
        z: torch.Tensor,
        delta: int,
        weight: torch.Tensor,  # [B, L, 1] — already masked AND renormalised
        g_sum: torch.Tensor,
    ) -> None:
        """Process a single offset: Plücker + project + accumulate."""
        L_eff = z.shape[1] - delta

        # C2 fix: causal pairing — position i receives Plücker(z[i-δ], z[i]).
        # Old code used (z_curr=z[:,:L_eff], z_future=z[:,delta:]) written to
        # positions 0..L-δ-1, which made position i ingest z[i+δ] (future leak).
        # Correct: z_past covers positions 0..L-δ-1, z_curr covers δ..L-1;
        # result is written to g_sum[:,delta:,:] (positions δ..L-1).
        z_past = z[:, :L_eff, :]  # [B, L-δ, r]  — the earlier tokens
        z_curr = z[:, delta:, :]  # [B, L-δ, r]  — the later tokens (present)

        # Plücker encoding + projection
        plucker = self._compute_plucker(z_past, z_curr)
        g = self.W_plu(plucker)

        # Weighted accumulation into positions δ..L-1 (causal write target).
        # `weight` already carries the pair-validity mask, so no separate v here.
        g_sum[:, delta:, :] += g * weight[:, delta:, :].to(g.dtype)

    def _compute_modality_biases(
        self,
        modality_mask: torch.Tensor,  # [B, L]  0=TS, 1=visual
        valid_offsets: list,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute per-position modality-pair bias [B, L, n_valid_offsets]."""
        B, L = modality_mask.shape
        n_valid = len(valid_offsets)
        device = modality_mask.device
        pair_biases = torch.zeros(B, L, n_valid, device=device, dtype=dtype)
        curr = modality_mask.long()  # [B, L]
        for k, delta in enumerate(valid_offsets):
            past = torch.cat(
                [
                    torch.zeros(B, delta, device=device, dtype=torch.long),
                    modality_mask[:, :-delta].long(),
                ],
                dim=1,
            )  # [B, L]
            pair_type = past * 2 + curr  # {0=TT, 1=TV, 2=VT, 3=VV}  [B, L]
            pair_biases[:, :, k] = self.modality_pair_bias[pair_type]
        return pair_biases  # [B, L, n_valid]

    def _offset_weights_for(
        self,
        valid_offsets: list,
        valid_mask: torch.Tensor,  # [B, L] bool
        modality_mask: Optional[torch.Tensor],
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-position offset mixing weights, renormalised over VALID pairs.

        Returns (mixing_weights [B, L, n_valid], softmax_weights) — the first is
        what multiplies each offset's Plücker term, the second is the raw softmax
        kept for the `output_attentions` diagnostic.

        Computed in fp32 on purpose. This branch is a gradient SINK: nothing
        reads it except `offset_weights` / `modality_pair_bias`, and its backward
        is a reduction over d_model — the largest in the layer. A single +inf in
        it turns *every* offset weight into NaN, because softmax's backward is
        y*(ĝ - Σ y·ĝ) and inf - inf = NaN. That is exactly the failure observed
        on Leonardo job 50574535 (`3:layer.0.offset_weights(6/6)` non-finite on
        every step while every other parameter stayed finite), which silently
        zeroed all gradients and froze training for 6.5 GPU-hours.
        """
        # Index the in-range offsets directly instead of torch.where(-inf) +
        # prefix slice: the old form only produced finite logits because
        # window_offsets happens to be sorted ascending.
        valid_idx = torch.tensor(
            [i for i, d in enumerate(self.window_offsets) if d < seq_len],
            device=device,
        )
        logits = self.offset_weights[valid_idx].float().view(1, 1, -1)  # [1,1,n]

        if self.use_modality_pair_bias and modality_mask is not None:
            logits = logits + self._compute_modality_biases(
                modality_mask, valid_offsets, torch.float32
            )  # [B, L, n]

        softmax_weights = torch.softmax(logits, dim=-1)

        # Renormalise over the offsets that actually have a valid pair at this
        # position, BEFORE accumulation. Algebraically identical to the old
        # accumulate-then-divide, Σ(v·w·g) / Σ(v·w), but keeps 1/Σ(v·w) out of
        # the g_sum backward. The old form clamped that denominator at 1e-6, so
        # positions with no valid pair at all amplified dL/dg_sum by up to 1e6 —
        # five orders of magnitude of headroom handed to the sink above.
        pair_valid = self._pair_validity(valid_mask, valid_offsets, torch.float32)
        wv = softmax_weights * pair_valid
        denom = wv.sum(dim=-1, keepdim=True)
        # torch.where (not a bare clamp) so the all-invalid positions contribute
        # exactly zero gradient rather than a 1e6-scaled one.
        mixing = torch.where(
            denom > 0, wv / denom.clamp(min=1e-6), torch.zeros_like(wv)
        )
        return mixing, softmax_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        output_attentions: bool = False,
        modality_mask: Optional[torch.Tensor] = None,  # [B, L] 0=TS, 1=visual
    ) -> AttentionOutput:
        residual = hidden_states
        _probe(self, "input", hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        _probe(self, "post_layer_norm", hidden_states)

        batch_size, seq_len, d_model = hidden_states.shape

        # Step 1: Linear reduction  [B, L, d] → [B, L, r]
        z = self.W_red(hidden_states)
        _probe(self, "post_W_red", z)

        # Inject RoPE phase information into reduced features
        cos, sin = self.rope_embed(z.unsqueeze(1), position_ids)
        z_rope, _ = RoPE.apply_rotary_pos_emb(
            z.unsqueeze(1), z.unsqueeze(1), cos, sin, unsqueeze_dim=1
        )
        z = z_rope.squeeze(1)
        _probe(self, "post_rope", z)

        # Mask: [batch, 1, q_len, kv_len] → [batch, seq_len]
        valid_mask = attention_mask[:, 0, 0, :] > -1.0  # [B, L]

        valid_offsets = [d for d in self.window_offsets if d < seq_len]
        num_offsets = len(valid_offsets)

        if num_offsets == 0:
            g = torch.zeros_like(hidden_states)
            normalized_weights = None
        else:
            mixing, normalized_weights = self._offset_weights_for(
                valid_offsets, valid_mask, modality_mask, seq_len, hidden_states.device
            )
            _probe(
                self,
                "mixing_weights",
                mixing,
                n_valid_offsets=num_offsets,
                valid_rows=f"{int(valid_mask.any(dim=1).sum())}/{valid_mask.shape[0]}",
                modality=(
                    "none"
                    if modality_mask is None
                    else f"vis={int((modality_mask == 1).sum())}"
                    f" uniq={sorted(set(modality_mask.flatten().tolist()))[:6]}"
                ),
            )

            g_sum = torch.zeros(
                batch_size, seq_len, d_model, device=z.device, dtype=hidden_states.dtype
            )
            for i, delta in enumerate(valid_offsets):
                self._process_offset(z, delta, mixing[:, :, i : i + 1], g_sum)

            # No trailing division: `mixing` is already normalised per position.
            g = g_sum
            _probe(self, "g_sum", g)

        # Step 5: Gated fusion
        u = torch.cat([hidden_states, g], dim=-1)  # [B, L, 2*d]
        alpha = torch.sigmoid(self.W_gate(u))  # [B, L, d]
        h_mix = alpha * hidden_states + (1 - alpha) * g

        # Residual connection + dropout
        output = residual + self.dropout(h_mix)

        # Optional: expose offset weights & entropy for regularization
        attn_weights = None
        if output_attentions and num_offsets > 0:
            # normalized_weights may be [n_valid] or [B, L, n_valid]; take scalar summary
            w = normalized_weights
            if w.dim() > 1:
                w = w.mean(dim=(0, 1))  # collapse batch/position dims → [n_valid]
            entropy = -(w * torch.log(w + 1e-8)).sum()
            attn_weights = {"offset_weights": normalized_weights, "entropy": entropy}

        return AttentionOutput(hidden_states=output, attn_weights=attn_weights)

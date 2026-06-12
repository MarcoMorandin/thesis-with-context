"""CausalGrassmannMixing — novel O(L) temporal mixing layer.

Implements the Grassmann flow algorithm (Section 3.2): causal multi-scale
Plücker pairing as an O(L) replacement for temporal self-attention.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import Chronos2CoreConfig
from .layers import AttentionOutput, Chronos2LayerNorm, RoPE


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

        self.use_modality_pair_bias = getattr(config, "grassmann_modality_pair_bias", False)
        if self.use_modality_pair_bias:
            # 4 scalar biases: TT=0, TV=1, VT=2, VV=3
            # Added to offset logit before softmax for position-dependent weighting.
            # Init zeros → no initial bias toward any pair type.
            self.modality_pair_bias = nn.Parameter(torch.zeros(4))

        # Cache Plücker indices as buffers (avoids recomputing every forward).
        # persistent=True ensures correct DDP broadcast and checkpoint round-trips.
        idx_i, idx_j = torch.triu_indices(r, r, offset=1)
        self.register_buffer("_plucker_idx_i", idx_i, persistent=True)
        self.register_buffer("_plucker_idx_j", idx_j, persistent=True)

    def _compute_plucker(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute normalized Plücker vectors for pairs (u, v) using cached index buffers."""
        # Advanced indexing (u[..., idx]) instead of torch.gather + expand.
        # gather on stride-0 expanded views triggers CUDA OOB asserts on A100+newer
        # drivers even after .contiguous(). Advanced indexing uses a safe kernel path.
        idx_i = self._plucker_idx_i  # [plucker_dim]
        idx_j = self._plucker_idx_j  # [plucker_dim]

        u_i = u[..., idx_i]   # [B, L, plucker_dim]
        v_j = v[..., idx_j]
        u_j = u[..., idx_j]
        v_i = v[..., idx_i]

        p = u_i * v_j - u_j * v_i

        p_norm = torch.sqrt((p * p).sum(dim=-1, keepdim=True) + self.plucker_eps)
        p = p / p_norm
        return p

    def _process_offset(
        self, z: torch.Tensor, valid_mask: torch.Tensor,
        delta: int, weight: torch.Tensor,
        g_sum: torch.Tensor, weight_sum: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> None:
        """Process a single offset: Plücker + project + accumulate."""
        L_eff = z.shape[1] - delta

        # C2 fix: causal pairing — position i receives Plücker(z[i-δ], z[i]).
        # Old code used (z_curr=z[:,:L_eff], z_future=z[:,delta:]) written to
        # positions 0..L-δ-1, which made position i ingest z[i+δ] (future leak).
        # Correct: z_past covers positions 0..L-δ-1, z_curr covers δ..L-1;
        # result is written to g_sum[:,delta:,:] (positions δ..L-1).
        z_past = z[:, :L_eff, :]   # [B, L-δ, r]  — the earlier tokens
        z_curr = z[:, delta:, :]   # [B, L-δ, r]  — the later tokens (present)

        # Validity mask: both z_past[i-δ] and z_curr[i] must be valid
        v = (valid_mask[:, :L_eff] & valid_mask[:, delta:]).unsqueeze(-1).to(target_dtype)

        # Plücker encoding + projection
        plucker = self._compute_plucker(z_past, z_curr)
        g = self.W_plu(plucker)

        # Weighted accumulation into positions δ..L-1 (causal write target)
        g_sum[:, delta:, :] += g * v * weight
        weight_sum[:, delta:, :] += v * weight

    def _compute_modality_biases(
        self,
        modality_mask: torch.Tensor,   # [B, L]  0=TS, 1=visual
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
            past = torch.cat([
                torch.zeros(B, delta, device=device, dtype=torch.long),
                modality_mask[:, :-delta].long(),
            ], dim=1)  # [B, L]
            pair_type = past * 2 + curr  # {0=TT, 1=TV, 2=VT, 3=VV}  [B, L]
            pair_biases[:, :, k] = self.modality_pair_bias[pair_type]
        return pair_biases  # [B, L, n_valid]

    def _process_offset_positional(
        self,
        z: torch.Tensor,            # [B, L, r]
        valid_mask: torch.Tensor,   # [B, L]
        delta: int,
        weight: torch.Tensor,       # [B, L]  position-wise weight
        g_sum: torch.Tensor,
        weight_sum: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> None:
        L_eff = z.shape[1] - delta
        z_past = z[:, :L_eff, :]
        z_curr = z[:, delta:, :]
        v = (valid_mask[:, :L_eff] & valid_mask[:, delta:]).unsqueeze(-1).to(target_dtype)
        plucker = self._compute_plucker(z_past, z_curr)
        g = self.W_plu(plucker)
        w = weight[:, delta:].unsqueeze(-1).to(target_dtype)  # [B, L_eff, 1]
        g_sum[:, delta:, :] += g * v * w
        weight_sum[:, delta:, :] += v * w

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        output_attentions: bool = False,
        modality_mask: Optional[torch.Tensor] = None,  # [B, L] 0=TS, 1=visual
    ) -> AttentionOutput:
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)

        batch_size, seq_len, d_model = hidden_states.shape

        # Step 1: Linear reduction  [B, L, d] → [B, L, r]
        z = self.W_red(hidden_states)

        # Inject RoPE phase information into reduced features
        cos, sin = self.rope_embed(z.unsqueeze(1), position_ids)
        z_rope, _ = RoPE.apply_rotary_pos_emb(
            z.unsqueeze(1), z.unsqueeze(1), cos, sin, unsqueeze_dim=1
        )
        z = z_rope.squeeze(1)

        # Mask: [batch, 1, q_len, kv_len] → [batch, seq_len]
        valid_mask = attention_mask[:, 0, 0, :] > -1.0  # [B, L]

        valid_offsets = [d for d in self.window_offsets if d < seq_len]
        num_offsets = len(valid_offsets)

        if num_offsets == 0:
            g = torch.zeros_like(hidden_states)
        else:
            # Build base logits for valid offsets (mask out-of-range offsets)
            offset_mask = torch.tensor(
                [d < seq_len for d in self.window_offsets],
                device=hidden_states.device
            )
            masked_logits = torch.where(
                offset_mask,
                self.offset_weights,
                torch.full_like(self.offset_weights, float('-inf')),
            )[:num_offsets]  # [n_valid]

            position_wise = (
                self.use_modality_pair_bias
                and modality_mask is not None
            )

            if position_wise:
                # [B, L, n_valid] = base[n_valid] + modality_bias[B, L, n_valid]
                pair_biases = self._compute_modality_biases(
                    modality_mask, valid_offsets, hidden_states.dtype
                )
                logits = masked_logits.unsqueeze(0).unsqueeze(0) + pair_biases
                normalized_weights = torch.softmax(logits, dim=-1)  # [B, L, n_valid]
            else:
                normalized_weights = torch.softmax(masked_logits, dim=0)  # [n_valid]

            g_sum = torch.zeros(
                batch_size, seq_len, d_model,
                device=z.device, dtype=hidden_states.dtype
            )
            weight_sum = torch.zeros(
                batch_size, seq_len, 1,
                device=z.device, dtype=hidden_states.dtype
            )

            for i, delta in enumerate(valid_offsets):
                if position_wise:
                    w = normalized_weights[:, :, i]  # [B, L]
                    self._process_offset_positional(
                        z, valid_mask, delta, w, g_sum, weight_sum, hidden_states.dtype
                    )
                else:
                    self._process_offset(
                        z, valid_mask, delta, normalized_weights[i],
                        g_sum, weight_sum, hidden_states.dtype,
                    )

            weight_sum = torch.clamp(weight_sum, min=1e-6)
            g = g_sum / weight_sum

        # Step 5: Gated fusion
        u = torch.cat([hidden_states, g], dim=-1)   # [B, L, 2*d]
        alpha = torch.sigmoid(self.W_gate(u))        # [B, L, d]
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

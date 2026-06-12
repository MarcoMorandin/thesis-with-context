"""Cross-Modal Alignment Adapter for Vision-Time FM.

Projects visual summary tokens (d_model → N_soft × d_model) before
injection into Chronos-2 Group Attention.

Three adapter variants (controlled by ``adapter_type``):
  - ``"linear"``         : single linear layer  (start here)
  - ``"mlp"``            : 2-layer MLP with GELU  (ablation)
  - ``"cross_attention"`` : learned queries × context cross-attention  (ablation)

Input  : [B, T, d_model]  visual summary tokens from LatentSummarizer
Output : [B, T, N_soft, d_model]  projected soft tokens for Group Attention
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalAdapter(nn.Module):
    """Projects visual embeddings into Chronos-2 latent token space.

    Parameters
    ----------
    d_model:
        Chronos-2 hidden dimension.  Input and output dim.
    n_soft_tokens:
        Number of soft covariate tokens to produce per timestep.
        N_soft ∈ {1, 4, 16, 64} — ablation parameter.
    adapter_type:
        ``"linear"`` | ``"mlp"`` | ``"cross_attention"``.
    n_layers:
        Number of hidden layers for ``"mlp"`` variant (ignored otherwise).
    dropout:
        Dropout rate applied to the adapter output.
    """

    def __init__(
        self,
        d_model: int,
        n_soft_tokens: int = 1,
        adapter_type: str = "linear",
        n_layers: int = 2,
        dropout: float = 0.1,
        attn_head_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_soft_tokens = n_soft_tokens
        self.adapter_type = adapter_type
        self.attn_head_dim = attn_head_dim

        out_dim = d_model * n_soft_tokens

        if adapter_type == "linear":
            self.proj = nn.Linear(d_model, out_dim, bias=True)

        elif adapter_type == "mlp":
            layers: list[nn.Module] = []
            in_d = d_model
            for i in range(n_layers):
                layers.append(nn.Linear(in_d, out_dim if i == n_layers - 1 else d_model, bias=True))
                if i < n_layers - 1:
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(dropout))
            self.proj = nn.Sequential(*layers)

        elif adapter_type == "cross_attention":
            # Learned queries attend to the single visual summary token per step
            self.learned_queries = nn.Parameter(
                torch.randn(1, n_soft_tokens, d_model) * (d_model ** -0.5)
            )
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=max(1, d_model // attn_head_dim),
                dropout=dropout,
                batch_first=True,
            )
            self.layer_norm = nn.LayerNorm(d_model)
            self.proj = None  # type: ignore[assignment]

        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type!r}. "
                             "Choose 'linear', 'mlp', or 'cross_attention'.")

        self.dropout = nn.Dropout(dropout)
        self.layer_norm_in = nn.LayerNorm(d_model)

    def forward(self, visual_summary: torch.Tensor) -> torch.Tensor:
        """Project visual summary tokens to N_soft soft tokens.

        Parameters
        ----------
        visual_summary:
            ``[B, T, d_model]`` — output of LatentSummarizer.

        Returns
        -------
        soft_tokens : ``[B, T, N_soft, d_model]``
        """
        B, T, D = visual_summary.shape
        x = self.layer_norm_in(visual_summary)   # pre-norm

        if self.adapter_type in ("linear", "mlp"):
            # [B, T, d_model] → [B, T, N_soft * d_model]
            out = self.proj(x)
            out = self.dropout(out)
            # [B, T, N_soft, d_model]
            out = out.view(B, T, self.n_soft_tokens, self.d_model)

        else:  # cross_attention
            # Process each time step: treat visual_summary as KV, learned queries as Q
            # Reshape for cross-attention: [B*T, 1, d_model] as KV
            kv = x.reshape(B * T, 1, D)
            q = self.learned_queries.expand(B * T, -1, -1)  # [B*T, N_soft, d_model]
            q = self.layer_norm(q)

            attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
            attn_out = self.dropout(attn_out)
            # Reshape back: [B, T, N_soft, d_model]
            out = attn_out.reshape(B, T, self.n_soft_tokens, self.d_model)

        return out

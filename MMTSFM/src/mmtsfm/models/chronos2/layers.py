# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import ModelOutput

from .config import Chronos2CoreConfig


class RoPE(nn.Module):
    """Applies rotary position embeddings (RoPE) to input tensors.

    Implementation adapted from:
    https://github.com/huggingface/transformers/blob/965cf677695dd363285831afca8cf479cf0c600c/src/transformers/models/llama/modeling_llama.py#L95
    """

    def __init__(self, dim: int, base: float = 10000):
        super().__init__()

        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.inv_freq: torch.Tensor  # type hint for type checker
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies Rotary Position Embedding to the query and key tensors."""
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (RoPE.rotate_half(q) * sin)
        k_embed = (k * cos) + (RoPE.rotate_half(k) * sin)
        return q_embed, k_embed


class Chronos2LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


# This is how transformers keeps track of LayerNorm classes ¯\_(ツ)_/¯
ALL_LAYERNORM_LAYERS.append(Chronos2LayerNorm)  # type: ignore


class MLP(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.wi(hidden_states)
        if hidden_states.requires_grad:
            hidden_states.register_hook(lambda g: torch.clamp(torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0))
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        hidden_states = self.wo(hidden_states)
        if hidden_states.requires_grad:
            hidden_states.register_hook(lambda g: torch.clamp(torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0))
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()

        assert not config.is_gated_act, "gated activations are unsupported"
        self.mlp: nn.Module = MLP(config)
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states)
        # Guard: MLP can produce NaN (large wi activations × pretrained weights
        # at unusual input scales). nan_to_num before residual add prevents NaN
        # from contaminating the residual stream and the backward for wi/wo.
        forwarded_states = torch.nan_to_num(forwarded_states, nan=0.0)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


@dataclass
class AttentionOutput(ModelOutput):
    hidden_states: torch.Tensor | None = None
    attn_weights: torch.Tensor | None = None


class MHA(nn.Module):
    """Multi-head Attention Layer"""

    def __init__(self, config: Chronos2CoreConfig, use_rope: bool = True):
        super().__init__()
        self.d_model: int = config.d_model
        self.kv_proj_dim: int = config.d_kv
        self.n_heads: int = config.num_heads
        self.dropout: float = config.dropout_rate
        self.inner_dim: int = self.n_heads * self.kv_proj_dim
        self.config = config

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        self.use_rope = use_rope
        if use_rope:
            self.rope_embed = RoPE(dim=self.kv_proj_dim, base=config.rope_theta)

    def _eager_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute attention weights with 1/sqrt(dk) scaling for numerical stability.
        # Original Chronos-2 implementation omitted this, which works for small
        # models but leads to gradient explosion during multimodal fine-tuning.
        scale = self.kv_proj_dim**-0.5
        scores = torch.matmul(query_states, key_states.transpose(3, 2)) * scale

        # Masking: handle both additive huge-neg sentinels and boolean masks.
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            else:
                scores = scores + mask

        scores = torch.nan_to_num(
            scores.float(),
            nan=0.0,
            posinf=torch.finfo(torch.float32).max,
            neginf=torch.finfo(torch.float32).min,
        )
        attn_weights = nn.functional.softmax(scores, dim=-1).type_as(query_states)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=0.0, neginf=0.0)

        return attn_output, attn_weights

    def _sdpa_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        # Upcast to float32 to match the eager path (which calls scores.float() before softmax).
        # Use a boolean SDPA mask instead of the eager path's huge additive sentinel:
        # CUDA SDPA can produce non-finite values when fed float32.min masks.
        orig_dtype = query_states.dtype
        sdpa_mask = mask >= 0 if mask is not None else None

        # We must disable autocast here, otherwise PyTorch AMP will automatically 
        # downcast the float32 inputs back to float16 inside SDPA!
        with torch.autocast(device_type=query_states.device.type, enabled=False):
            attn_output = nn.functional.scaled_dot_product_attention(
                query_states.float(),
                key_states.float(),
                value_states.float(),
                attn_mask=sdpa_mask,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.kv_proj_dim ** -0.5,
            )
        attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=0.0, neginf=0.0)
        return attn_output.to(orig_dtype), None

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        if self.use_rope:
            assert position_ids is not None, "position_ids must be provided when self.use_rope=True"

        # Force eager attention if output_attentions is True (only eager returns weights)
        attn_implementation = self.config._attn_implementation
        if output_attentions or self.training:
            attn_implementation = "eager"

        seq_length = hidden_states.shape[1]

        def shape(states: torch.Tensor) -> torch.Tensor:
            return rearrange(states, "b s (h d) -> b h s d", h=self.n_heads, s=seq_length, d=self.kv_proj_dim)

        def unshape(states: torch.Tensor) -> torch.Tensor:
            return rearrange(states, "b h s d -> b s (h d)", h=self.n_heads, s=seq_length, d=self.kv_proj_dim)

        q_out = self.q(hidden_states)
        if q_out.requires_grad:
            q_out.register_hook(lambda g: torch.clamp(torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0))
        query_states = shape(q_out)
        
        is_cross_attention = encoder_states is not None

        if is_cross_attention:
            k_out = self.k(encoder_states)
            v_out = self.v(encoder_states)
        else:
            k_out = self.k(hidden_states)
            v_out = self.v(hidden_states)
            
        if k_out.requires_grad:
            k_out.register_hook(lambda g: torch.clamp(torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0))
        if v_out.requires_grad:
            v_out.register_hook(lambda g: torch.clamp(torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0))
            
        key_states = shape(k_out)
        value_states = shape(v_out)
        
        if not is_cross_attention and self.use_rope:
            cos, sin = self.rope_embed(value_states, position_ids)
            query_states, key_states = RoPE.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if attn_implementation == "sdpa":
            attn_output, attn_weights = self._sdpa_attention(query_states, key_states, value_states, mask)
        else:  # eager
            attn_output, attn_weights = self._eager_attention(query_states, key_states, value_states, mask)

        # Project attention output
        attn_output = unshape(attn_output)
        attn_output = self.o(attn_output)
        
        if attn_output.requires_grad:
            attn_output.register_hook(lambda g: torch.clamp(torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0))

        return AttentionOutput(hidden_states=attn_output, attn_weights=attn_weights if output_attentions else None)


class TimeSelfAttention(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        self.self_attention = MHA(config, use_rope=True)
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        output_attentions: bool = False,
        modality_mask: Optional[torch.Tensor] = None,  # accepted, ignored
    ) -> AttentionOutput:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output: AttentionOutput = self.self_attention(
            normed_hidden_states, position_ids=position_ids, mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])

        return AttentionOutput(hidden_states=hidden_states, attn_weights=attention_output.attn_weights)


class TimeCrossAttention(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        self.cross_attention = MHA(config, use_rope=False)
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output: AttentionOutput = self.cross_attention(
            normed_hidden_states,
            mask=attention_mask,
            encoder_states=encoder_states,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])

        return AttentionOutput(hidden_states=hidden_states, attn_weights=attention_output.attn_weights)


class GroupSelfAttention(nn.Module):
    """Self-attention applied along the batch axis masked by the group attention mask"""

    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        # we don't use RoPE here because there's no natural ordering along the batch axis
        self.self_attention = MHA(config, use_rope=False)
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, output_attentions: bool = False
    ) -> AttentionOutput:
        # flip time and batch axes because attention operates along dim=-2
        hidden_states = rearrange(hidden_states, "batch time d -> time batch d")
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output: AttentionOutput = self.self_attention(
            normed_hidden_states, mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # flip time and batch axes back to their original position
        hidden_states = rearrange(hidden_states, "time batch d -> batch time d")

        return AttentionOutput(hidden_states=hidden_states, attn_weights=attention_output.attn_weights)


# Local variant — differs from chronos_bolt.ResidualBlock. See chronos_bolt.py for upstream version.
class ResidualBlock(nn.Module):
    """A generic residual block which can be used for input and output embedding layers"""

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = Chronos2LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)

        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(out)
        return out

"""VisionChronos2 — Chronos-2 extended with multimodal video soft covariates.

Architecture (Phase 3 implementation)
--------------------------------------
1. TS context → Chronos-2 tokenization → input_embeds [B, T_ctx, d_model]
2. Video frames → V-JEPA 2.1 VisualEncoder → latent tokens [B, T_lat, P, D_v]
3. LatentSummarizer → visual summary [B, T_ctx, d_model]  (Perceiver)
4. CrossModalAdapter → soft tokens [B, T_ctx, N_soft, d_model]
5. Batch-dim concat: encoder sees [B + B*N_soft, T_full, d_model]
   — Group Attention fuses numeric + visual at each timestep
6. Slice first B rows, decode last num_output_patches → quantile forecasts

Zero-shot regression guarantee
-------------------------------
When ``video=None`` the encoder receives exactly the same inputs as
vanilla Chronos-2 (group_ids default to ``arange(B)`` matching Chronos-2
default). Output is provably identical up to fp32 precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from .model import Chronos2Model
from ..vision.latent_summarizer import LatentSummarizer
from ..vision.cross_modal_adapter import CrossModalAdapter
from ..vision.patch_projector import VisualPatchProjector


# ---------------------------------------------------------------------------
# Visual-context-window derivation (W7)
# ---------------------------------------------------------------------------


def t_ctx_from_context(context_length: int, input_patch_size: int) -> int:
    """Number of TS context patches for a given history length and patch size.

    Chronos-2 patchifies the ``context_length``-step history into
    ``ceil(context_length / input_patch_size)`` non-overlapping input patches —
    the upper bound on how many of those patches can receive a visual summary
    token (``n_visual_context_steps``).
    """
    return math.ceil(int(context_length) / int(input_patch_size))


def validate_n_visual_context_steps(
    n_visual_context_steps: int, context_length: int, input_patch_size: int
) -> int:
    """Assert the visual window fits inside the TS context; return ``T_ctx``.

    ``n_visual_context_steps`` is how many of the most-recent context patches
    are given visual tokens, so it must not exceed ``T_ctx`` (W7). Fires loudly
    on an impossible config instead of silently clamping.
    """
    t_ctx = t_ctx_from_context(context_length, input_patch_size)
    if n_visual_context_steps > t_ctx:
        raise ValueError(
            f"n_visual_context_steps={n_visual_context_steps} exceeds the number "
            f"of TS context patches T_ctx={t_ctx} "
            f"(context_length={context_length}, input_patch_size={input_patch_size}). "
            f"Reduce n_visual_context_steps to <= {t_ctx}."
        )
    return t_ctx


# ---------------------------------------------------------------------------
# Interleaving helpers
# ---------------------------------------------------------------------------


def interleave_sequences(
    ts_tokens: torch.Tensor,  # [B, T_ctx, d]
    vis_tokens: torch.Tensor,  # [B, n_vis, d] or [B, n_vis, N_soft, d]
    n_vis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Selectively interleave visual summary tokens into the refinement window.

    Builds, with N = N_soft visual tokens per refined step:

        [ts_0..ts_{T_M-1}] || [ts_{T_M}, v¹_{T_M}..v^N_{T_M}, ts_{T_M+1}, ...]

    A 3-D ``vis_tokens`` is treated as N=1, which is the historical shape and
    reproduces the original pairwise interleave exactly.

    Returns:
        interleaved: ``[B, T_ctx + n_vis*N, d]``
        modality_mask: ``[B, T_ctx + n_vis*N]`` long tensor — 0=TS, 1=visual
    """
    B, T_ctx, d = ts_tokens.shape
    T_M = T_ctx - n_vis
    if vis_tokens.dim() == 3:
        vis_tokens = vis_tokens.unsqueeze(2)  # [B, n_vis, 1, d]
    n_soft = vis_tokens.shape[2]

    macro = ts_tokens[:, :T_M, :]
    ts_refine = ts_tokens[:, T_M:, :].unsqueeze(2)  # [B, n_vis, 1, d]
    blocks = torch.cat([ts_refine, vis_tokens], dim=2)  # [B, n_vis, 1+N, d]
    refinement = blocks.reshape(B, n_vis * (1 + n_soft), d)
    interleaved = torch.cat([macro, refinement], dim=1)  # [B, T_ctx+n_vis*N, d]

    device = ts_tokens.device
    seq_len = T_ctx + n_vis * n_soft
    modality_mask = torch.zeros(B, seq_len, dtype=torch.long, device=device)
    # Offsets 1..N inside each (1+N)-token block are the visual ones.
    block_start = T_M + torch.arange(n_vis, device=device) * (1 + n_soft)
    vis_positions = (
        block_start[:, None] + 1 + torch.arange(n_soft, device=device)[None, :]
    ).reshape(-1)
    modality_mask[:, vis_positions] = 1

    return interleaved, modality_mask


def build_interleaved_position_ids(
    T_M: int,
    n_vis: int,
    T_fut: int,
    device: torch.device,
    n_soft: int = 1,
) -> torch.Tensor:
    """Build temporal position IDs for the interleaved sequence.

    A refined step and ALL of its visual tokens share one position ID, so RoPE
    treats the whole (1+N_soft)-token block as co-temporal and the N visual
    tokens are order-free within it.

    Returns:
        ``[1, T_M + n_vis*(1+n_soft) + T_fut]`` long tensor
    """
    macro_ids = torch.arange(T_M, device=device)
    refine_ids = torch.arange(T_M, T_M + n_vis, device=device)
    refine_blocks = refine_ids[:, None].expand(n_vis, 1 + n_soft).reshape(-1)
    future_ids = torch.arange(T_M + n_vis, T_M + n_vis + T_fut, device=device)
    return torch.cat([macro_ids, refine_blocks, future_ids]).unsqueeze(0)


def reduce_delta_t_to_latents(frame_delta_t: torch.Tensor, T_lat: int) -> torch.Tensor:
    """``[B, T_v]`` raw-frame Δt -> ``[B, T_lat]``, one value per latent frame.

    V-JEPA tubelets pool 2 raw frames, so T_v is a multiple of T_lat. Takes the
    ``amax`` (the OLDEST frame in each tubelet), matching
    ``LatentSummarizer.forward``'s reduction so s2b and s2d read the same clock.
    """
    B, L = frame_delta_t.shape
    if L == T_lat:
        return frame_delta_t
    if L % T_lat != 0:
        raise ValueError(f"video_delta_t length {L} is not a multiple of T_lat={T_lat}")
    return frame_delta_t.reshape(B, T_lat, L // T_lat).amax(dim=-1)


def build_subpatch_position_ids(
    T_M: int,
    T_fut: int,
    frame_idx: torch.Tensor,  # [B, K] long — source latent frame of each visual token
    frame_delta_t: Optional[torch.Tensor],  # [B, T_lat] seconds before the origin
    span_seconds: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """s2d — FRACTIONAL positions placing visual tokens *inside* the last TS patch.

    At ``input_patch_size=16`` on 30-minute data one TS token spans 8 hours and the
    whole 6-hour visual window falls inside the final context patch, so there is no
    integer position left to interleave at (design doc §3.3). Instead every visual
    token lands at

        pos = T_M + clamp(1 - Δt/span, 0, 1) * 0.99

    i.e. in ``[T_M, T_M + 0.99]`` — after the co-temporal TS token at ``T_M``, before
    the first future token at ``T_M + 1``, ordered oldest-to-newest. ``layers.py:90``
    casts ``position_ids`` to float before the RoPE frequency outer product, so
    fractional positions need no kernel change.

    ``0.99`` rather than ``1.0`` keeps the newest frame strictly below the future
    token; without it a Δt=0 frame would collide with the forecast position.

    Returns ``[B, T_M + 1 + K + T_fut]`` float — note this is a per-sample tensor,
    unlike the shared row ``build_interleaved_position_ids`` returns, because Δt
    varies across the batch.
    """
    B, K = frame_idx.shape
    device = frame_idx.device
    ctx = torch.arange(T_M + 1, device=device, dtype=dtype).expand(B, T_M + 1)
    if frame_delta_t is None:
        # No clock: every visual token sits exactly on its TS partner, which
        # degenerates to s2b's co-temporal block.
        vis = torch.full((B, K), float(T_M), device=device, dtype=dtype)
    else:
        dt = frame_delta_t.to(device=device, dtype=dtype).gather(1, frame_idx)
        vis = float(T_M) + (1.0 - dt / float(span_seconds)).clamp(0.0, 1.0) * 0.99
    fut = torch.arange(T_M + 1, T_M + 1 + T_fut, device=device, dtype=dtype).expand(
        B, T_fut
    )
    return torch.cat([ctx, vis, fut], dim=1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class VisionChronos2Config:
    """Vision hyper-parameters added on top of Chronos2CoreConfig.

    Attributes
    ----------
    n_visual_context_steps:
        TS context patch-steps covered by the visual window.
        These are the *last* n positions of the context sequence.
    n_soft_tokens:
        N_soft — visual tokens per TS step per entity. Ablation: {1,4,16,64}.
    adapter_type:
        ``"linear"`` | ``"mlp"`` | ``"cross_attention"``.
    adapter_n_layers:
        Hidden layers for MLP adapter (ignored otherwise).
    summarizer_n_heads:
        Attention heads in LatentSummarizer cross-attention.
    summarizer_time_slices, summarizer_spatial_grid:
        A29 — sub-resolution inside each visual TS step. The summarizer emits
        ``n_sub = summarizer_time_slices * summarizer_spatial_grid**2`` tokens per
        step instead of one, each masked to its own (temporal slice, spatial block).
        Both 1 → the historical single-token-per-step behaviour, unchanged.
    visual_dropout_prob:
        Probability of zeroing the entire visual stream per sample during
        training (Asymmetric Bernoulli modality dropout — visual rate).
    numeric_dropout_prob:
        M3 fix — Probability of zeroing the entire numeric stream per sample
        during training (Asymmetric Bernoulli modality dropout — numeric rate).
        Must be < 1 to prevent information collapse. Default 0.1.
        Effective runtime rate = numeric_dropout_prob × (1 − visual_dropout_prob)
        because the guard prevents dropping both streams simultaneously (N2).
        With defaults 0.1 and 0.5 the effective numeric drop rate is 0.05.
    dropout:
        Dropout for adapter and summarizer.
    """

    n_visual_context_steps: int = 24
    n_soft_tokens: int = 1
    # s2c (fusion_mode='future_query'): side of the spatial grid retained from the
    # V-JEPA patch field. 4 -> a 4x4 grid per temporal slice. 4 is not arbitrary: it
    # is exactly the arm the latent probe measured (ramp R^2 0.0512 at t+30, 0.0815
    # at t+60), so a null here is interpretable rather than confounded by an
    # unmeasured resolution.
    visual_grid: int = 4
    adapter_type: str = "linear"
    adapter_n_layers: int = 2
    summarizer_n_heads: int = 4
    # A29: sub-resolution INSIDE the LatentSummarizer. Each visual TS step is split
    # into summarizer_time_slices temporal slices x summarizer_spatial_grid**2 spatial
    # blocks, and each of those n_sub queries is masked to its own block. This is the
    # only way to widen the interleaved payload with real information: the
    # CrossModalAdapter sits DOWNSTREAM of the summarizer bottleneck and can only
    # fan one d_model vector out into N_soft copies, which is why A13 (n_soft 1 -> 16)
    # was a null on reliance by construction. Defaults 1/1 reproduce every published
    # arm bit-identically.
    summarizer_time_slices: int = 1
    summarizer_spatial_grid: int = 1
    visual_dropout_prob: float = 0.5
    numeric_dropout_prob: float = (
        0.1  # M3 fix: asymmetric Bernoulli — numeric stream dropout rate
    )
    dropout: float = 0.1
    n_entities: int = (
        0  # >0 enables entity-identity embedding; set to num_entities in data config
    )

    # --- NEW fields for proposal ---
    fusion_mode: str = "late"
    # "late"            → existing CrossModalAdapter path (batch-dim concat)
    # "interleaved"     → selective temporal interleaving (refinement window only)
    # "future_query"    → s2c, forecast positions cross-attend a retained field
    # "interleaved_raw" → s2d / A30, the same interleaved sequence assembly but the
    #                     visual tokens come from VisualPatchProjector (pixel shuffle
    #                     + MLP + EVS) instead of the LatentSummarizer, and they carry
    #                     FRACTIONAL positions. No resampler anywhere in the path.

    # --- s2d / A30 (fusion_mode="interleaved_raw") ---
    # Pixel-shuffle factor: 2 gives Nemotron's 4x token reduction, 14x14 -> 7x7.
    visual_shuffle_r: int = 2
    # Spatial cells after the shuffle; must equal (grid/visual_shuffle_r)**2 and is
    # checked against the real patch field at forward time.
    visual_n_cells: int = 49
    # Tokens surviving EVS out of T_lat*visual_n_cells (4*49=196). 98 is q=0.5.
    # <= 0 disables pruning entirely — that is the q=0 arm of the §3.5 sweep, and it
    # is a pure eval-time knob: EVS has no parameters.
    visual_evs_keep: int = 98
    # Seconds spanned by one TS input patch: input_patch_size * step_seconds. On
    # uk_pv, 16 * 1800 = 28800 (8 h). Only used to normalise Δt into the fractional
    # slot [T_M, T_M+0.99]; a wrong value rescales the visual positions but keeps
    # their order.
    visual_position_span_seconds: float = 28800.0

    visual_encoder_ckpt_path: str = ""
    freeze_visual_encoder: bool = True
    skip_vision_stack: bool = False

    # --- Ablation switches (see knowledge/ablations.md §2.5) ---
    # A19: the learned per-lead-time query offset on the s2c future queries. When
    # False the three future positions are distinguished by sequence position
    # alone, which is exactly the degeneracy ticket 15 exists to detect. Only
    # meaningful for fusion_mode="future_query".
    use_lead_time_embed: bool = True
    # M1a/M1b/M1c: leave-one-out over the additive MultimodalEmbedding channels.
    # The embedding tables are still built, so state_dict shape — and therefore
    # warm-starting from any existing checkpoint — is unchanged.
    disable_modality_embed: bool = False
    disable_segment_embed: bool = False
    disable_token_type_embed: bool = False
    disable_entity_embed: bool = False


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class VisionChronos2Output:
    loss: Optional[torch.Tensor] = None
    quantile_preds: Optional[torch.Tensor] = None
    enc_time_self_attn_weights: Optional[Tuple] = None
    enc_group_self_attn_weights: Optional[Tuple] = None
    visual_active: Optional[torch.Tensor] = (
        None  # [B] bool — which samples kept visual stream
    )
    numeric_active: Optional[torch.Tensor] = (
        None  # [B] bool — M3: which samples kept numeric stream
    )


# ---------------------------------------------------------------------------
# Multimodal embedding
# ---------------------------------------------------------------------------


class MultimodalEmbedding(nn.Module):
    """Modality-type, segment-type, token-type, and entity-identity bias embeddings.

    Modality IDs : 0 = numeric, 1 = visual.
    Segment IDs  : 0 = context (past), 1 = future.
    Token-type IDs (M1 fix):
        0 = target      — patched target series tokens (context + forecast horizon).
        1 = covariate   — future covariate tokens injected via _prepare_patched_future.
        2 = visual      — visual soft-context tokens from CrossModalAdapter.
    """

    def __init__(
        self,
        d_model: int,
        n_entities: int = 0,
        disable_modality_embed: bool = False,
        disable_segment_embed: bool = False,
        disable_token_type_embed: bool = False,
        disable_entity_embed: bool = False,
    ):
        super().__init__()
        self.modality_embed = nn.Embedding(2, d_model)
        self.segment_embed = nn.Embedding(2, d_model)
        self.token_type_embed = nn.Embedding(
            3, d_model
        )  # M1 fix: target/covariate/visual
        self.entity_embed = (
            nn.Embedding(n_entities, d_model) if n_entities > 0 else None
        )
        # Leave-one-out ablation switches (registry M1a/M1b/M1c). The tables are
        # still CONSTRUCTED when disabled, so the state_dict shape is unchanged and
        # a disabled run loads from — and warm-starts — any existing checkpoint.
        # Only the additive contribution is suppressed, at the single place each
        # channel is applied, rather than at the ~30 call sites in forward().
        self.disable_modality_embed = disable_modality_embed
        self.disable_segment_embed = disable_segment_embed
        self.disable_token_type_embed = disable_token_type_embed
        self.disable_entity_embed = disable_entity_embed
        # Default N(0,1) init gives magnitude ≈ sqrt(d_model) ≈ 22 which swamps the
        # pretrained Chronos-2 activations and causes bf16 overflow → NaN loss.
        # Use small std (0.02, same as BERT/T5 token embeddings) to keep scale compatible.
        for emb in [self.modality_embed, self.segment_embed, self.token_type_embed]:
            nn.init.normal_(emb.weight, std=0.02)
        if self.entity_embed is not None:
            nn.init.normal_(self.entity_embed.weight, std=0.02)

    def add_modality(self, tokens: torch.Tensor, modality_id: int) -> torch.Tensor:
        """``tokens [B, T, d]`` + scalar modality embedding."""
        if self.disable_modality_embed:
            return tokens
        idx = torch.tensor(modality_id, device=tokens.device, dtype=torch.long)
        return tokens + self.modality_embed(idx)

    def add_segment(self, tokens: torch.Tensor, segment_id: int) -> torch.Tensor:
        """``tokens [B, T, d]`` + scalar segment embedding (0=context, 1=future)."""
        if self.disable_segment_embed:
            return tokens
        idx = torch.tensor(segment_id, device=tokens.device, dtype=torch.long)
        return tokens + self.segment_embed(idx)

    def add_token_type(self, tokens: torch.Tensor, token_type_id: int) -> torch.Tensor:
        """M1 fix — ``tokens [B, T, d]`` + scalar token-type embedding.

        Token-type IDs
        --------------
        0 : target    — patched numeric target series (context + forecast).
        1 : covariate — future covariate tokens.
        2 : visual    — soft visual context tokens.
        """
        if self.disable_token_type_embed:
            return tokens
        idx = torch.tensor(token_type_id, device=tokens.device, dtype=torch.long)
        return tokens + self.token_type_embed(idx)

    def add_entity(
        self, tokens: torch.Tensor, entity_ids: torch.Tensor
    ) -> torch.Tensor:
        """``tokens [B, T, d]`` + entity embedding ``[B, d]`` from position indices."""
        if self.entity_embed is None or self.disable_entity_embed:
            return tokens
        assert entity_ids.max() < self.entity_embed.num_embeddings, (
            f"entity_ids max={entity_ids.max().item()} >= "
            f"n_entities={self.entity_embed.num_embeddings}. "
            "Set VisionChronos2Config.n_entities to at least max(entity_ids)+1."
        )
        return tokens + self.entity_embed(entity_ids).unsqueeze(1)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class VisionChronos2Model(nn.Module):
    """Chronos-2 with V-JEPA 2.1 video soft covariates (Phase 3).

    Parameters
    ----------
    chronos_model:
        Pre-loaded ``Chronos2Model``. Its parameters are included in
        training unless manually frozen.
    vision_config:
        Vision hyper-parameters.
    video_encoder:
        Optional pre-built video encoder injected for testing (avoids loading
        real V-JEPA 2.1 weights). Must expose ``.d_v`` and a ``forward(video)``
        returning ``[B, T_lat, P, D_v]``. When ``None`` a frozen V-JEPA 2.1
        ``VisualEncoder`` is constructed.
    """

    def __init__(
        self,
        chronos_model: Chronos2Model,
        vision_config: VisionChronos2Config,
        video_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.chronos = chronos_model
        self.vcfg = vision_config
        d_model: int = chronos_model.model_dim

        # W7: the visual window (n_visual_context_steps) must fit inside the TS
        # context patch grid; assert loudly rather than silently clamping later.
        if not vision_config.skip_vision_stack:
            validate_n_visual_context_steps(
                vision_config.n_visual_context_steps,
                chronos_model.chronos_config.context_length,
                chronos_model.chronos_config.input_patch_size,
            )

        if not vision_config.skip_vision_stack:
            if video_encoder is not None:
                # Injected encoder (tests) — bypass real V-JEPA 2.1 weight load.
                self.video_encoder: Optional[nn.Module] = video_encoder
            else:
                from ..vision.visual_encoder import VisualEncoder

                self.video_encoder = VisualEncoder(
                    arch="vit_large",
                    freeze=vision_config.freeze_visual_encoder,
                )
            _d_v = self.video_encoder.d_v

            # s2d removes the resampler outright, so it must not be constructed:
            # a summarizer left in the module would put unused `latent_queries` in
            # the state_dict and, worse, make "did the summarizer cause it?" ask a
            # question about a tensor that is still there.
            self.raw_visual: bool = vision_config.fusion_mode == "interleaved_raw"
            if self.raw_visual:
                if vision_config.n_soft_tokens > 1:
                    raise ValueError(
                        "fusion_mode='interleaved_raw' with n_soft_tokens="
                        f"{vision_config.n_soft_tokens}: the adapter fan-out copies "
                        "one vector N times and adds no information (A13 was null by "
                        "construction). Widen with visual_evs_keep instead."
                    )
                self.latent_summarizer: Optional[nn.Module] = None
                self.patch_projector: Optional[nn.Module] = VisualPatchProjector(
                    d_v=_d_v,
                    d_model=d_model,
                    shuffle_r=vision_config.visual_shuffle_r,
                    n_cells=vision_config.visual_n_cells,
                    evs_keep=vision_config.visual_evs_keep,
                    dropout=vision_config.dropout,
                )
                n_sub = 1
            else:
                self.patch_projector = None
                self.latent_summarizer = LatentSummarizer(
                    d_v=_d_v,
                    d_model=d_model,
                    n_vis_steps=vision_config.n_visual_context_steps,
                    n_heads=vision_config.summarizer_n_heads,
                    dropout=vision_config.dropout,
                    n_time_slices=vision_config.summarizer_time_slices,
                    spatial_grid=vision_config.summarizer_spatial_grid,
                )
                # n_sub > 1 and N_soft > 1 would widen the payload twice, once with
                # real information and once with a fan-out, and no post-hoc analysis
                # could separate the two. Refuse rather than run a confound.
                n_sub = self.latent_summarizer.n_sub
            if n_sub > 1 and vision_config.n_soft_tokens > 1:
                raise ValueError(
                    f"summarizer sub-resolution (n_sub={n_sub}) and n_soft_tokens="
                    f"{vision_config.n_soft_tokens} are both > 1. The adapter fan-out "
                    "would confound the summarizer bandwidth. Set n_soft_tokens=1."
                )
            if n_sub > 1 and vision_config.fusion_mode != "interleaved":
                raise ValueError(
                    f"summarizer sub-resolution (n_sub={n_sub}) is only consumed by "
                    f"fusion_mode='interleaved', got '{vision_config.fusion_mode}'."
                )

            # Late fusion always expands through the adapter. Interleaved fusion
            # only needs it to widen the bottleneck: at N_soft == 1 it would be an
            # identity-shaped extra projection, and building it would add
            # parameters that no existing curriculum checkpoint carries. So it is
            # constructed for interleaved ONLY when it does something.
            # s2c bypasses the summarizer and the adapter entirely: the whole point is
            # that no stage pools the patch grid before fusion.
            if vision_config.fusion_mode == "future_query":
                self.visual_kv_proj: Optional[nn.Module] = nn.Linear(_d_v, d_model)
                # Learned lead-time embedding, one per future patch position. This is
                # explicit and learnable ON PURPOSE: three queries distinguished only
                # by sequence position can collapse to near-duplicates, and the model
                # then learns three generic visual summaries. That failure produces
                # the same flat metric as a genuinely falsified hypothesis, so it must
                # be made detectable (ticket 15) rather than merely hoped against.
                n_lead = int(
                    getattr(chronos_model.chronos_config, "max_output_patches", 4)
                )
                # A19 turns this off: the parameter is then absent from the
                # state_dict and forward() falls through its `is not None` guard,
                # leaving the future queries separated by position alone.
                self.lead_time_embed: Optional[nn.Parameter] = (
                    nn.Parameter(torch.randn(n_lead, d_model) * (d_model**-0.5))
                    if vision_config.use_lead_time_embed
                    else None
                )
            else:
                self.visual_kv_proj = None
                self.lead_time_embed = None

            needs_adapter = (
                vision_config.fusion_mode == "late" or vision_config.n_soft_tokens > 1
            )
            if needs_adapter:
                self.cross_modal_adapter: Optional[nn.Module] = CrossModalAdapter(
                    d_model=d_model,
                    n_soft_tokens=vision_config.n_soft_tokens,
                    adapter_type=vision_config.adapter_type,
                    n_layers=vision_config.adapter_n_layers,
                    dropout=vision_config.dropout,
                )
            else:
                self.cross_modal_adapter = None
        else:
            self.video_encoder = None
            self.visual_kv_proj = None
            self.lead_time_embed = None
            self.latent_summarizer = None
            self.patch_projector = None
            self.raw_visual = False
            self.cross_modal_adapter = None

        self.multimodal_embed = MultimodalEmbedding(
            d_model=d_model,
            n_entities=vision_config.n_entities,
            disable_modality_embed=vision_config.disable_modality_embed,
            disable_segment_embed=vision_config.disable_segment_embed,
            disable_token_type_embed=vision_config.disable_token_type_embed,
            disable_entity_embed=vision_config.disable_entity_embed,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _modality_dropout(
        self,
        visual_embeds: torch.Tensor,
        input_embeds_mm: torch.Tensor,
        future_embeds_mm: torch.Tensor,
        force_vision_off: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """M3 fix — Asymmetric Bernoulli modality dropout.

        Independently zeros out the visual and/or numeric streams per sample
        with separate configurable probabilities during training only.

        Parameters
        ----------
        visual_embeds     : ``[B, n_vis, d_model]``
        input_embeds_mm   : ``[B, T_ctx, d_model]`` — numeric context tokens
        future_embeds_mm  : ``[B, T_fut, d_model]`` — numeric future tokens

        Returns
        -------
        visual_embeds    : zeroed for dropped samples
        input_embeds_mm  : zeroed for dropped samples
        future_embeds_mm : zeroed for dropped samples
        visual_active    : ``[B]`` bool — True if sample kept visual stream
        numeric_active   : ``[B]`` bool — True if sample kept numeric stream
        """
        B = visual_embeds.shape[0]
        device = visual_embeds.device
        visual_active = torch.ones(B, dtype=torch.bool, device=device)
        numeric_active = torch.ones(B, dtype=torch.bool, device=device)

        # W6: forced vision-off (eval-time visual-marginal-gain pass). Zero the
        # visual stream and short-circuit so the result matches a manually
        # visual-masked forward, independent of training-time dropout rates.
        if force_vision_off:
            visual_active = torch.zeros(B, dtype=torch.bool, device=device)
            visual_embeds = visual_embeds * 0.0
            return (
                visual_embeds,
                input_embeds_mm,
                future_embeds_mm,
                visual_active,
                numeric_active,
            )

        if self.training:
            # Visual dropout
            if self.vcfg.visual_dropout_prob > 0:
                drop_vis = torch.rand(B, device=device) < self.vcfg.visual_dropout_prob
                visual_active = ~drop_vis
                visual_embeds = visual_embeds * (~drop_vis).float().view(B, 1, 1)

            # Numeric dropout (M3 fix — asymmetric rate)
            if self.vcfg.numeric_dropout_prob > 0:
                drop_num = torch.rand(B, device=device) < self.vcfg.numeric_dropout_prob
                # Guard: never drop numeric if visual is also dropped for the same sample
                # (that would give a zero-information row and destabilise training).
                drop_num = (
                    drop_num & visual_active
                )  # only drop numeric when visual is present
                numeric_active = ~drop_num
                num_mask = (~drop_num).float().view(B, 1, 1)
                input_embeds_mm = input_embeds_mm * num_mask
                future_embeds_mm = future_embeds_mm * num_mask

        return (
            visual_embeds,
            input_embeds_mm,
            future_embeds_mm,
            visual_active,
            numeric_active,
        )

    def _build_visual_embeds(
        self,
        video: Optional[torch.Tensor],
        visual_mask: Optional[torch.Tensor],
        T_ctx: int,
        input_embeds_mm: torch.Tensor,
        future_embeds_mm: torch.Tensor,
        video_latents: Optional[torch.Tensor] = None,
        force_vision_off: bool = False,
        video_delta_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Video encoder → Summarizer → Adapter → soft tokens.

        Parameters
        ----------
        video :
            ``[B, C, T_v, H, W]`` raw video frames. Mutually exclusive with ``video_latents``.
        visual_mask :
            ``[B, T_v]`` frame availability mask (1=available). Optional.
        video_latents:
            Pre-computed V-JEPA latents ``[B, T_lat, P, D_v]``.
            When provided, the video encoder is bypassed.
        input_embeds_mm:
            Numeric context token embeddings ``[B, T_ctx, d_model]``.
        future_embeds_mm:
            Numeric future token embeddings ``[B, T_fut, d_model]``.

        Returns
        -------
        soft_flat        : ``[B * N_soft, T_ctx, d_model]``
        input_embeds_mm  : ``[B, T_ctx, d_model]`` (possibly zeroed by numeric dropout)
        future_embeds_mm : ``[B, T_fut, d_model]`` (possibly zeroed by numeric dropout)
        visual_active    : ``[B]`` bool
        numeric_active   : ``[B]`` bool
        """
        if video_latents is not None:
            # Use pre-computed latents — skip the encoder (saves ~90% of visual compute)
            video_tokens = video_latents
        else:
            # V-JEPA encode — [B, T_lat, P, D_v]
            video_tokens = self.video_encoder(video)
        # Guard: video encoder (random-init ViT or bf16 encoder) may produce NaN
        # which enters LatentSummarizer KV and makes backward NaN even after the
        # forward is cleaned. Catch at source before reaching cross-attention.
        video_tokens = torch.nan_to_num(video_tokens, nan=0.0, posinf=0.0, neginf=0.0)
        B, T_lat, P, D_v = video_tokens.shape

        # Temporal stride mask for latent space
        lat_mask: Optional[torch.Tensor] = None
        if visual_mask is not None:
            T_v = visual_mask.shape[1]
            stride = max(1, T_v // T_lat)
            # O7 fix: Add assert to prevent reshape issues when T_v < T_lat
            # Add padding when T_v < T_lat to prevent reshape issues
            if T_v < T_lat:
                # Pad visual_mask to ensure we have enough elements for proper reshaping
                pad_length = T_lat * stride - T_v
                if pad_length > 0:
                    # Pad with zeros at the end to match expected length
                    visual_mask = torch.cat(
                        [
                            visual_mask,
                            torch.zeros(
                                B,
                                pad_length,
                                device=visual_mask.device,
                                dtype=visual_mask.dtype,
                            ),
                        ],
                        dim=1,
                    )
                    T_v = visual_mask.shape[1]  # Update T_v to new length
                lat_mask = (
                    visual_mask[:, : T_lat * stride]
                    .reshape(B, T_lat, stride)
                    .max(dim=-1)
                    .values
                )
            else:
                lat_mask = (
                    visual_mask[:, : T_lat * stride]
                    .reshape(B, T_lat, stride)
                    .max(dim=-1)
                    .values
                )

        # Perceiver compressor — [B, n_vis_steps, d_model] for the visual window only
        n_vis = min(self.vcfg.n_visual_context_steps, T_ctx)
        vis_window = self.latent_summarizer(
            video_tokens=video_tokens,
            T_ts=n_vis,
            visual_mask=lat_mask,
            frame_delta_t=video_delta_t,
        )
        # vis_window: [B, n_vis, d_model] — non-zero only here

        # Modality type embedding applied only to visual window (avoids contaminating zero-pad)
        vis_window = self.multimodal_embed.add_modality(vis_window, modality_id=1)

        # Modality dropout on the visual window (M3 fix: pass numeric embeds through)
        vis_window, input_embeds_mm, future_embeds_mm, visual_active, numeric_active = (
            self._modality_dropout(
                vis_window,
                input_embeds_mm,
                future_embeds_mm,
                force_vision_off=force_vision_off,
            )
        )

        # Cross-modal adapter on the visual window — [B, n_vis, N_soft, d_model]
        soft_win = self.cross_modal_adapter(vis_window)

        # Zero-pad to full T_ctx: early positions (long TS history) receive no visual tokens
        B_ = soft_win.shape[0]
        N_s = soft_win.shape[2]
        D_ = soft_win.shape[3]
        soft = torch.zeros(
            B_, T_ctx, N_s, D_, device=soft_win.device, dtype=soft_win.dtype
        )
        soft[:, T_ctx - n_vis :, :, :] = soft_win

        # Flatten N_soft into batch — [B * N_soft, T_ctx, d_model], row b*N_soft+n.
        # The permute is load-bearing: `soft` is [B, T_ctx, N_soft, d], so a bare
        # reshape walks t before n and interleaves the two axes. It happens to be
        # correct at N_soft == 1 (the only value ever run), which is why this never
        # fired. Consumers index the flat batch as b*N_soft+n — see the
        # repeat_interleave(N_soft) calls on entity/group ids below.
        return (
            soft.permute(0, 2, 1, 3).reshape(B_ * N_s, T_ctx, D_),
            input_embeds_mm,
            future_embeds_mm,
            visual_active,
            numeric_active,
        )

    def _build_visual_kv(self, video_tokens: torch.Tensor) -> torch.Tensor:
        """V-JEPA latents ``[B, T_lat, P, D_v]`` -> visual KV ``[B, T_lat*g*g, d]``.

        Spatial layout is RETAINED, only coarsened: the patch grid is block-pooled to
        g x g and every temporal slice is kept as its own set of key/value entries.
        Keeping all T_lat slices is what makes motion inferable at all -- a single
        instant cannot express displacement.
        """
        B, T_lat, Pn, D_v = video_tokens.shape
        g0 = int(round(Pn**0.5))
        if g0 * g0 != Pn:
            raise ValueError(
                f"V-JEPA patch count {Pn} is not a square grid; cannot retain spatial "
                "layout for future_query fusion."
            )
        g = min(int(self.vcfg.visual_grid), g0)
        b = g0 // g
        z = video_tokens.reshape(B, T_lat, g0, g0, D_v)[:, :, : g * b, : g * b, :]
        z = z.reshape(B, T_lat, g, b, g, b, D_v).mean(dim=(3, 5))
        return self.visual_kv_proj(z.reshape(B, T_lat * g * g, D_v))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
        future_covariates: Optional[torch.Tensor] = None,
        future_covariates_mask: Optional[torch.Tensor] = None,
        covariate_channels: Optional[List[torch.Tensor]] = None,
        num_output_patches: int = 1,
        future_target: Optional[torch.Tensor] = None,
        future_target_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        # Vision inputs
        video: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        video_latents: Optional[torch.Tensor] = None,
        # Entity position indices [B] in [0, n_entities)
        entity_ids: Optional[torch.Tensor] = None,
        # W6: force the visual stream off for the visual-marginal-gain eval pass.
        force_vision_off: bool = False,
        # W5: seconds-before-origin per latent frame [B, T_lat]; when its length
        # matches the latent temporal dim the summarizer builds its causal window
        # from true spacing instead of assuming uniform frame spacing.
        video_delta_t: Optional[torch.Tensor] = None,
    ) -> VisionChronos2Output:
        """Forward pass.

        Parameters
        ----------
        context : ``[B, context_length]``
        context_mask : ``[B, context_length]`` 1=observed.
        group_ids : ``[B]``.  None → ``arange(B)`` (independent, matches Chronos-2).
        future_covariates : ``[B, horizon]``  — primary covariate channel (loss path).
        future_covariates_mask : ``[B, horizon]``
        covariate_channels : list of ``[B, horizon]`` tensors, one per C_cov channel.
            M2 fix: each channel is independently tokenized in the encoder as a
            separate batch row sharing the target's Group ID (token-type=covariate).
            When None or empty falls back to single-channel behaviour.
        num_output_patches : number of output prediction patches.
        future_target : ``[B, horizon]`` for loss.
        future_target_mask : ``[B, horizon]``
        output_attentions : return attention weights.
        video : ``[B, C, T_v, H, W]`` in [0, 1]. None → pure TS (identical to Chronos-2).
        visual_mask : ``[B, T_v]`` 1=available.
        video_latents : ``[B, T_lat, P, D_v]`` pre-computed V-JEPA latents.
            Mutually exclusive with ``video``. When provided the video encoder is bypassed.
        """
        B = context.shape[0]
        device = context.device
        dtype = self.chronos.dtype

        # Default group_ids: match Chronos-2 (each series independent)
        if group_ids is None:
            group_ids = torch.arange(B, dtype=torch.long, device=device)

        # ---- TS preprocessing (Chronos-2 path) --------------------------
        patched_context, attention_mask, loc_scale = (
            self.chronos._prepare_patched_context(
                context=context, context_mask=context_mask
            )
        )
        patched_future, patched_future_cov_mask = self.chronos._prepare_patched_future(
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            loc_scale=loc_scale,
            num_output_patches=num_output_patches,
            batch_size=B,
        )

        # Context patch embeddings [B, T_ctx, d_model]
        input_embeds: torch.Tensor = self.chronos.input_patch_embedding(patched_context)
        # NOTE: modality-type embedding applied ONLY when video is present,
        # so that video=None produces output identical to vanilla Chronos-2.

        # REG token (optional)
        if self.chronos.chronos_config.use_reg_token:
            reg_ids = torch.full(
                (B, 1),
                self.chronos.config.reg_token_id,  # type: ignore[attr-defined]
                device=device,
                dtype=torch.long,
            )
            reg_embeds = self.chronos.shared(reg_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask.to(dtype),
                    torch.ones(B, 1, device=device, dtype=dtype),
                ],
                dim=1,
            )

        T_ctx = input_embeds.shape[1]  # context + optional reg token

        # Future patch embeddings [B, num_output_patches, d_model]. NOT
        # `input_patch_embedding`: a future patch is output_patch_size wide, and
        # this arm is the only one that makes the two sizes differ.
        future_embeds: torch.Tensor = self.chronos.embed_future(patched_future)
        future_attn_mask = torch.ones(B, num_output_patches, device=device, dtype=dtype)

        # ---- Visual stream (optional) ------------------------------------
        visual_active: Optional[torch.Tensor] = None
        numeric_active: Optional[torch.Tensor] = None
        use_video = (
            (video is not None or video_latents is not None)
            and not (visual_mask is not None and visual_mask.sum() == 0)
            and self.video_encoder is not None  # guard for skip_vision_stack
        )

        N_soft = self.vcfg.n_soft_tokens

        # N1 fix: build covariate rows unconditionally (runs regardless of use_video)
        cov_embed_rows: list[torch.Tensor] = []
        cov_mask_rows: list[torch.Tensor] = []
        cov_group_rows: list[torch.Tensor] = []
        # Future-only covariate embeddings [B, T_fut, d] (context is zeros for all
        # branches). The interleaved path pads the context to T_ctx+n_vis; the
        # late/numeric paths pad to T_ctx. Kept separately so both can build rows.
        cov_fut_rows: list[torch.Tensor] = []

        if covariate_channels:
            H_cov = covariate_channels[0].shape[-1]
            # Known future covariates: mask=1 (observed) so _prepare_patched_future
            # KEEPS the values. Passing mask=0 (the old code) routed every value
            # through `torch.where(mask>0, value, 0.0)` in model.py and zeroed them
            # before embedding — the covariate token-rows carried only the
            # embedding bias, so future weather never influenced the forecast.
            cov_mask_ones = torch.ones(B, H_cov, device=device)
            for cov_ch in covariate_channels:
                patched_cov, _ = self.chronos._prepare_patched_future(
                    future_covariates=cov_ch,
                    future_covariates_mask=cov_mask_ones,
                    loc_scale=loc_scale,
                    num_output_patches=num_output_patches,
                    batch_size=B,
                )
                patched_cov = torch.nan_to_num(
                    patched_cov, nan=0.0, posinf=0.0, neginf=0.0
                )
                cov_embeds = self.chronos.embed_future(patched_cov)
                cov_ctx = torch.zeros(
                    B, T_ctx, self.chronos.model_dim, device=device, dtype=dtype
                )
                cov_embeds = self.multimodal_embed.add_modality(
                    cov_embeds, modality_id=0
                )
                cov_embeds = self.multimodal_embed.add_segment(cov_embeds, segment_id=1)
                cov_embeds = self.multimodal_embed.add_token_type(
                    cov_embeds, token_type_id=1
                )
                if entity_ids is not None:
                    cov_embeds = self.multimodal_embed.add_entity(
                        cov_embeds, entity_ids
                    )
                cov_fut_rows.append(cov_embeds)
                cov_full = torch.cat([cov_ctx, cov_embeds], dim=1)
                cov_embed_rows.append(cov_full)
                cov_mask_rows.append(
                    torch.cat(
                        [
                            torch.zeros(B, T_ctx, device=device, dtype=dtype),
                            torch.ones(
                                B, num_output_patches, device=device, dtype=dtype
                            ),
                        ],
                        dim=1,
                    )
                )
                cov_group_rows.append(group_ids)

        visual_kv = None
        visual_query_mask = None
        if self.vcfg.fusion_mode == "future_query" and self.lead_time_embed is not None:
            # The lead-time embedding is part of how the forecast positions are
            # PARAMETERISED, not part of the visual pathway, so it is applied
            # whether or not this batch carries vision. Gating it on `use_video`
            # would make a vision-free batch a different model, and the
            # marginal-gain counterfactual would then be measuring tau as well as
            # vision.
            T_fut_q = future_embeds.shape[1]
            future_embeds = future_embeds + self.lead_time_embed[:T_fut_q].unsqueeze(0)

        if use_video and self.vcfg.fusion_mode == "future_query":
            # --- s2c: forecast positions cross-attend a retained spatial field ---
            # The token sequence is the plain [context, future] one; the visual field
            # never enters it. Vision reaches the model only through cross-attention
            # at the future positions, inside the last k encoder blocks.
            if video_latents is not None:
                video_tokens = video_latents
            else:
                video_tokens = self.video_encoder(video)
            video_tokens = torch.nan_to_num(
                video_tokens, nan=0.0, posinf=0.0, neginf=0.0
            )
            visual_kv = self._build_visual_kv(video_tokens)

            # Forced vision-off (marginal-gain pass) and per-sample modality dropout
            # both act by withholding the QUERY, not by zeroing the field: with the
            # mask off, every future position keeps its self-attention result and the
            # forward is numerically identical to the vision-free path.
            vis_on = torch.ones(B, device=device, dtype=torch.bool)
            if force_vision_off:
                vis_on = torch.zeros_like(vis_on)
            elif self.training and self.vcfg.visual_dropout_prob > 0:
                vis_on = torch.rand(B, device=device) >= self.vcfg.visual_dropout_prob
            if visual_mask is not None:
                vis_on = vis_on & (visual_mask.sum(dim=1) > 0)
            self._last_visual_active = vis_on
            # Reported out so the marginal-gain pass and the modality-dropout logging
            # see s2c the same way they see every other arm.
            visual_active = vis_on

        if use_video and self.vcfg.fusion_mode in ("interleaved", "interleaved_raw"):
            # --- Interleaved fusion path (s2b, and s2d via `raw_visual`) ---
            # Works for both Variant A (use_grassmann=True) and Variant B (use_grassmann=False)
            #
            # s2d shares this block deliberately. At n_vis == 1 the sequence
            # `interleave_sequences` emits is already s2d's layout —
            # [macro(0..T_M-1)] || [ts_{T_M}, v_1..v_K] — so the two arms differ in
            # exactly two places, both flagged `raw_visual` below: which module turns
            # the patch field into tokens, and whether those tokens get integer or
            # fractional positions. Everything else (masks, modality embeds, dropout,
            # FIX F, covariate rows, decode) is bit-identical between them, which is
            # what makes the s2b vs s2d contrast attributable.
            n_vis = min(self.vcfg.n_visual_context_steps, T_ctx)
            if self.raw_visual and n_vis != 1:
                raise ValueError(
                    f"fusion_mode='interleaved_raw' needs n_visual_context_steps=1, "
                    f"got {self.vcfg.n_visual_context_steps}. The whole visual window "
                    "falls inside ONE 8-hour TS patch, so there is exactly one "
                    "co-temporal TS token to interleave against (design doc §3.3); "
                    "sub-patch position is carried by the fractional position IDs."
                )

            # Encode video
            if video_latents is not None:
                video_tokens = video_latents
            else:
                # video is [B, 3, T_v, H, W]; frames arrive as RGB from the .h5 loader
                video_tokens = self.video_encoder(video)  # [B, T_lat, P, D_v]
            video_tokens = torch.nan_to_num(
                video_tokens, nan=0.0, posinf=0.0, neginf=0.0
            )

            B_, T_lat, P, D_v = video_tokens.shape

            # Frame availability mask for latent space
            lat_mask = None
            if visual_mask is not None:
                T_v = visual_mask.shape[1]
                stride = max(1, T_v // T_lat)
                if T_v < T_lat:
                    # Pad visual_mask to ensure we have enough elements for proper reshaping
                    pad_length = T_lat * stride - T_v
                    if pad_length > 0:
                        visual_mask = torch.cat(
                            [
                                visual_mask,
                                torch.zeros(
                                    B_,
                                    pad_length,
                                    device=visual_mask.device,
                                    dtype=visual_mask.dtype,
                                ),
                            ],
                            dim=1,
                        )
                        T_v = visual_mask.shape[1]
                lat_mask = (
                    visual_mask[:, : T_lat * stride]
                    .reshape(B_, T_lat, stride)
                    .max(-1)
                    .values
                )

            # --- Visual tokens: the ONE place s2b and s2d diverge ---
            vis_frame_idx = None
            lat_delta_t = None
            if video_delta_t is not None:
                lat_delta_t = reduce_delta_t_to_latents(video_delta_t, T_lat)

            if self.raw_visual:
                # s2d: pixel shuffle -> MLP projector -> cell embedding -> EVS.
                # No pooling over the patch field at any point.
                vis_summary, vis_frame_idx, _cell_idx = self.patch_projector(
                    video_tokens
                )  # [B, K, d]
                N_vis_tok = vis_summary.shape[1]  # n_vis == 1, so K tokens per step
                if lat_mask is not None:
                    # Zero the tokens whose source frame was unavailable. EVS already
                    # deprioritises blank frames (they are near-identical to each
                    # other, so their dissimilarity is ~0), but frame 0 is a pinned
                    # anchor and would survive even when absent.
                    keep_mask = lat_mask.to(vis_summary.dtype).gather(1, vis_frame_idx)
                    vis_summary = vis_summary * keep_mask.unsqueeze(-1)
            else:
                # LatentSummarizer — [B, n_vis, d_model], or [B, n_vis, n_sub, d_model]
                # when A29 sub-resolution is on  (T_ts=n_vis → no null tokens)
                vis_summary = self.latent_summarizer(
                    video_tokens=video_tokens,
                    T_ts=n_vis,
                    visual_mask=lat_mask,
                    frame_delta_t=video_delta_t,
                )

                # Visual tokens per refined step. Two mutually exclusive ways to get
                # more than one, and __init__ refuses the combination:
                #   n_sub  — A29, real bandwidth: each token attends to its own
                #            (temporal slice, spatial block) of the patch field.
                #   N_soft — A13, a fan-out of the single pooled vector by the adapter
                #            downstream of the bottleneck; adds capacity, not signal.
                N_vis_tok = N_soft
                if vis_summary.dim() == 4:
                    N_vis_tok = vis_summary.shape[2]
                    vis_summary = vis_summary.reshape(B_, n_vis * N_vis_tok, -1)

                # Widen the bottleneck: one summary token per step becomes N_soft.
                # Everything downstream (modality embeds, dropout, interleave, masks)
                # treats them as ordinary visual tokens, so this is the only place
                # N_soft enters the interleaved path. At N_soft == 1 the adapter is
                # not built and vis_summary passes through untouched.
                elif self.cross_modal_adapter is not None:
                    vis_summary = self.cross_modal_adapter(
                        vis_summary
                    )  # [B, n_vis, N_soft, d]
                    vis_summary = vis_summary.reshape(B_, n_vis * N_soft, -1)

            # Multimodal embeddings for TS tokens
            input_embeds_mm = self.multimodal_embed.add_modality(
                input_embeds, modality_id=0
            )
            input_embeds_mm = self.multimodal_embed.add_segment(
                input_embeds_mm, segment_id=0
            )
            input_embeds_mm = self.multimodal_embed.add_token_type(
                input_embeds_mm, token_type_id=0
            )
            future_embeds_mm = self.multimodal_embed.add_modality(
                future_embeds, modality_id=0
            )
            future_embeds_mm = self.multimodal_embed.add_segment(
                future_embeds_mm, segment_id=1
            )
            future_embeds_mm = self.multimodal_embed.add_token_type(
                future_embeds_mm, token_type_id=0
            )
            if entity_ids is not None:
                input_embeds_mm = self.multimodal_embed.add_entity(
                    input_embeds_mm, entity_ids
                )
                future_embeds_mm = self.multimodal_embed.add_entity(
                    future_embeds_mm, entity_ids
                )

            # Multimodal embeddings for visual summary tokens
            vis_summary = self.multimodal_embed.add_modality(vis_summary, modality_id=1)
            vis_summary = self.multimodal_embed.add_segment(vis_summary, segment_id=0)
            vis_summary = self.multimodal_embed.add_token_type(
                vis_summary, token_type_id=2
            )
            if entity_ids is not None:
                vis_summary = self.multimodal_embed.add_entity(vis_summary, entity_ids)

            # Modality dropout on visual summary
            (
                vis_summary,
                input_embeds_mm,
                future_embeds_mm,
                visual_active,
                numeric_active,
            ) = self._modality_dropout(
                vis_summary,
                input_embeds_mm,
                future_embeds_mm,
                force_vision_off=force_vision_off,
            )

            # FIX F: same as late-fusion — restore vanilla embeddings for dropped samples
            if visual_active is not None:
                vis_active_3d = visual_active.view(B, 1, 1)
                input_embeds_mm = torch.where(
                    vis_active_3d, input_embeds_mm, input_embeds
                )
                future_embeds_mm = torch.where(
                    vis_active_3d, future_embeds_mm, future_embeds
                )

            # Interleave refinement window. Regroup the flat visual tokens back to
            # [B, n_vis, N_vis_tok, d] so each refined step gets its own block.
            interleaved_ctx, modality_mask_ctx = interleave_sequences(
                input_embeds_mm,
                vis_summary.reshape(B, n_vis, N_vis_tok, -1),
                n_vis,
            )

            # Full sequence: [B, T_ctx + n_vis*N_vis_tok + T_fut, d]
            T_fut = future_embeds_mm.shape[1]
            n_vis_tok = n_vis * N_vis_tok  # visual tokens inserted into the context
            T_M = T_ctx - n_vis  # macro region: context patches with no visual partner
            all_embeds = torch.cat([interleaved_ctx, future_embeds_mm], dim=1)
            modality_mask_fut = torch.zeros(B, T_fut, dtype=torch.long, device=device)
            modality_mask = torch.cat([modality_mask_ctx, modality_mask_fut], dim=1)

            # Interleave the CONTEXT attention mask the same way the tokens were
            # interleaved. This path used to hand the encoder all-ones, discarding
            # `attention_mask` from _prepare_patched_context (the late and numeric
            # paths never did). `build_site_series` reindexes onto a regular grid, so
            # night steps are NaN and — at input_patch_size 16, an 8-hour patch —
            # whole patches can be unobserved; they were being presented as valid
            # tokens for temporal mixing. Note the mask is ALSO a patch feature, so
            # the embedding always reflected it; only the mixing/attention side was
            # wrong, which is why the symptom was subtle rather than catastrophic.
            ctx_mask = attention_mask.to(dtype)  # [B, T_ctx]
            macro_mask = ctx_mask[:, :T_M]
            refine_mask = torch.cat(
                [
                    ctx_mask[:, T_M:, None],  # the TS token opening each block
                    torch.ones(  # its N_vis_tok visual partners
                        B, n_vis, N_vis_tok, device=device, dtype=dtype
                    ),
                ],
                dim=2,
            ).reshape(B, n_vis * (1 + N_vis_tok))
            all_mask = torch.cat(
                [
                    macro_mask,
                    refine_mask,
                    torch.ones(B, T_fut, device=device, dtype=dtype),
                ],
                dim=1,
            )
            all_group_ids = group_ids

            # Position IDs: TS and vis tokens at same step share position
            if self.raw_visual:
                # s2d: the K visual tokens spread across [T_M, T_M+0.99] by Δt, so a
                # frame permutation (A09) changes the sequence — `_apply_eval_control`
                # permutes video_latents and visual_mask but NOT video_delta_t, so the
                # clock stays put while the content moves. On s2b every visual token
                # shares one integer position and A09 is inert by construction.
                position_ids = build_subpatch_position_ids(
                    T_M=T_M,
                    T_fut=T_fut,
                    frame_idx=vis_frame_idx,
                    frame_delta_t=lat_delta_t,
                    span_seconds=self.vcfg.visual_position_span_seconds,
                )
            else:
                position_ids = build_interleaved_position_ids(
                    T_M, n_vis, T_fut, device, n_soft=N_vis_tok
                ).expand(B, -1)

            # Covariate rows (batch-axis) — the interleaved path previously dropped
            # them, so known future weather never reached the encoder in the final
            # (interleaved) model. Context is zeros (length T_ctx+n_vis*N_soft to match
            # interleaved target seq); only the future positions carry the covariate
            # embedding and are valid for GroupSelfAttention. Rows are all-numeric,
            # so their modality_mask is 0 everywhere.
            if cov_fut_rows:
                seq_len = T_ctx + n_vis_tok + T_fut
                cov_ctx_il = torch.zeros(
                    B,
                    T_ctx + n_vis_tok,
                    self.chronos.model_dim,
                    device=device,
                    dtype=dtype,
                )
                cov_valid = torch.cat(
                    [
                        torch.zeros(B, T_ctx + n_vis_tok, device=device, dtype=dtype),
                        torch.ones(B, T_fut, device=device, dtype=dtype),
                    ],
                    dim=1,
                )
                cov_modality = torch.zeros(B, seq_len, dtype=torch.long, device=device)
                embed_parts = [all_embeds]
                mask_parts = [all_mask]
                group_parts = [all_group_ids]
                modality_parts = [modality_mask]
                pos_parts = [position_ids]
                for cov_fut in cov_fut_rows:
                    embed_parts.append(torch.cat([cov_ctx_il, cov_fut], dim=1))
                    mask_parts.append(cov_valid)
                    group_parts.append(group_ids)
                    modality_parts.append(cov_modality)
                    pos_parts.append(position_ids)
                all_embeds = torch.cat(embed_parts, dim=0)
                all_mask = torch.cat(mask_parts, dim=0)
                all_group_ids = torch.cat(group_parts, dim=0)
                modality_mask = torch.cat(modality_parts, dim=0)
                position_ids = torch.cat(pos_parts, dim=0)

            all_embeds = torch.nan_to_num(all_embeds, nan=0.0)
            encoder_out = self.chronos.encoder(
                inputs_embeds=all_embeds,
                group_ids=all_group_ids,
                attention_mask=all_mask,
                position_ids=position_ids,
                modality_mask=modality_mask,
                output_attentions=output_attentions,
            )
            hidden_states_raw: torch.Tensor = encoder_out.last_hidden_state
            hidden_states_raw = torch.nan_to_num(hidden_states_raw, nan=0.0)

            # Decode: last T_fut positions are the future predictions
            hidden_states = hidden_states_raw[:B, -T_fut:]

            # --- Skip the shared encoder block + decode block ---
            # Build loss and return directly
            forecast_embeds = hidden_states
            quantile_preds = self.chronos.output_patch_embedding(forecast_embeds)
            quantile_preds = rearrange(
                quantile_preds,
                "b n (q p) -> b q (n p)",
                n=T_fut,
                q=self.chronos.num_quantiles,
                p=self.chronos.chronos_config.output_patch_size,
            )
            quantile_preds = torch.nan_to_num(quantile_preds, nan=0.0)

            loss = None
            if future_target is not None:
                loss = self.chronos._compute_loss(
                    quantile_preds=quantile_preds,
                    future_target=future_target,
                    future_target_mask=future_target_mask,
                    patched_future_covariates_mask=patched_future_cov_mask,
                    loc_scale=loc_scale,
                    num_output_patches=T_fut,
                )

            quantile_preds = rearrange(
                quantile_preds,
                "b q h -> b (q h)",
                b=B,
                q=self.chronos.num_quantiles,
                h=T_fut * self.chronos.chronos_config.output_patch_size,
            )
            quantile_preds = self.chronos.instance_norm.inverse(
                quantile_preds, loc_scale
            )
            quantile_preds = rearrange(
                quantile_preds,
                "b (q h) -> b q h",
                q=self.chronos.num_quantiles,
                h=T_fut * self.chronos.chronos_config.output_patch_size,
            )
            return VisionChronos2Output(
                loss=loss,
                quantile_preds=quantile_preds,
                enc_time_self_attn_weights=encoder_out.all_time_self_attn_weights,
                enc_group_self_attn_weights=encoder_out.all_group_self_attn_weights,
                visual_active=visual_active,
                numeric_active=numeric_active,
            )

        elif use_video and self.vcfg.fusion_mode == "late":
            # Modality-type (numeric=0), segment (context=0, future=1),
            # token-type (target=0, covariate=1), entity embeddings
            input_embeds_mm = self.multimodal_embed.add_modality(
                input_embeds, modality_id=0
            )
            input_embeds_mm = self.multimodal_embed.add_segment(
                input_embeds_mm, segment_id=0
            )
            input_embeds_mm = self.multimodal_embed.add_token_type(
                input_embeds_mm, token_type_id=0
            )  # target
            future_embeds_mm = self.multimodal_embed.add_modality(
                future_embeds, modality_id=0
            )
            future_embeds_mm = self.multimodal_embed.add_segment(
                future_embeds_mm, segment_id=1
            )
            future_embeds_mm = self.multimodal_embed.add_token_type(
                future_embeds_mm, token_type_id=0
            )  # target (forecast horizon)
            if entity_ids is not None:
                input_embeds_mm = self.multimodal_embed.add_entity(
                    input_embeds_mm, entity_ids
                )
                future_embeds_mm = self.multimodal_embed.add_entity(
                    future_embeds_mm, entity_ids
                )

            # soft_ctx: [B*N_soft, T_ctx, d_model]
            (
                soft_ctx,
                input_embeds_mm,
                future_embeds_mm,
                visual_active,
                numeric_active,
            ) = self._build_visual_embeds(
                video=video,
                visual_mask=visual_mask,
                T_ctx=T_ctx,
                input_embeds_mm=input_embeds_mm,
                future_embeds_mm=future_embeds_mm,
                video_latents=video_latents,
                force_vision_off=force_vision_off,
                video_delta_t=video_delta_t,
            )
            # FIX F (late-fusion): restore vanilla Chronos-2 embeddings for samples
            # where the visual stream was dropped by modality dropout inside
            # _build_visual_embeds. Without this, numeric tokens carry
            # modality/segment/token-type noise even with no visual signal.
            if visual_active is not None:
                vis_active_3d = visual_active.view(B, 1, 1)
                input_embeds_mm = torch.where(
                    vis_active_3d, input_embeds_mm, input_embeds
                )
                future_embeds_mm = torch.where(
                    vis_active_3d, future_embeds_mm, future_embeds
                )

            # Token-type: visual soft tokens
            soft_ctx = self.multimodal_embed.add_token_type(
                soft_ctx, token_type_id=2
            )  # visual
            # Entity embedding on visual soft tokens (context segment, visual modality already applied)
            if entity_ids is not None:
                vis_entity_ids = entity_ids.repeat_interleave(N_soft)
                soft_ctx = self.multimodal_embed.add_entity(soft_ctx, vis_entity_ids)
            # Segment embedding: visual tokens are context-aligned
            soft_ctx = self.multimodal_embed.add_segment(soft_ctx, segment_id=0)

            # Future visual tokens: zero (no visual in forecast window)
            soft_fut = torch.zeros(
                B * N_soft,
                num_output_patches,
                self.chronos.model_dim,
                device=device,
                dtype=dtype,
            )

            # Full sequence: [T_ctx + T_fut] tokens
            ts_full = torch.cat(
                [input_embeds_mm, future_embeds_mm], dim=1
            )  # [B, T_full, d]
            vis_full = torch.cat([soft_ctx, soft_fut], dim=1)  # [B*N_soft, T_full, d]

            ts_mask_full = torch.cat([attention_mask, future_attn_mask], dim=1)
            # N9 fix: mask zero-padded early context positions and future-window visual tokens
            n_vis = min(self.vcfg.n_visual_context_steps, T_ctx)
            vis_ctx_mask = torch.zeros(B, T_ctx, device=device, dtype=dtype)
            vis_ctx_mask[:, T_ctx - n_vis :] = 1.0
            vis_mask_full = torch.cat(
                [
                    vis_ctx_mask,
                    torch.zeros(B, num_output_patches, device=device, dtype=dtype),
                ],
                dim=1,
            ).repeat_interleave(N_soft, dim=0)

            # Stack target + covariate-channel rows + visual rows
            all_embed_parts = [ts_full] + cov_embed_rows + [vis_full]
            all_mask_parts = [ts_mask_full] + cov_mask_rows + [vis_mask_full]
            vis_group_ids = group_ids.repeat_interleave(N_soft)
            all_group_parts = [group_ids] + cov_group_rows + [vis_group_ids]

            all_embeds = torch.cat(all_embed_parts, dim=0)
            all_mask = torch.cat(all_mask_parts, dim=0)
            all_group_ids = torch.cat(all_group_parts, dim=0)
        else:
            if cov_embed_rows:
                # Numeric+covariate: apply target embeddings to distinguish from covariate rows
                input_embeds_nm = self.multimodal_embed.add_modality(
                    input_embeds, modality_id=0
                )
                input_embeds_nm = self.multimodal_embed.add_segment(
                    input_embeds_nm, segment_id=0
                )
                input_embeds_nm = self.multimodal_embed.add_token_type(
                    input_embeds_nm, token_type_id=0
                )
                future_embeds_nm = self.multimodal_embed.add_modality(
                    future_embeds, modality_id=0
                )
                future_embeds_nm = self.multimodal_embed.add_segment(
                    future_embeds_nm, segment_id=1
                )
                future_embeds_nm = self.multimodal_embed.add_token_type(
                    future_embeds_nm, token_type_id=0
                )
                if entity_ids is not None:
                    input_embeds_nm = self.multimodal_embed.add_entity(
                        input_embeds_nm, entity_ids
                    )
                    future_embeds_nm = self.multimodal_embed.add_entity(
                        future_embeds_nm, entity_ids
                    )
                ts_full = torch.cat([input_embeds_nm, future_embeds_nm], dim=1)
                ts_mask_full = torch.cat([attention_mask, future_attn_mask], dim=1)
                all_embeds = torch.cat([ts_full] + cov_embed_rows, dim=0)
                all_mask = torch.cat([ts_mask_full] + cov_mask_rows, dim=0)
                all_group_ids = torch.cat([group_ids] + cov_group_rows, dim=0)
            else:
                # Pure TS: no modality embeddings → identical to vanilla Chronos-2
                all_embeds = torch.cat([input_embeds, future_embeds], dim=1)
                all_mask = torch.cat([attention_mask, future_attn_mask], dim=1)
                all_group_ids = group_ids

        # ---- Encoder ----------------------------------------------------
        if visual_kv is not None:
            # Query mask over the FULL batch: true only at the future positions of the
            # target rows. Covariate rows appended along the batch axis stay false, so
            # they never attend the visual field.
            n_rows, seq_len_full = all_embeds.shape[0], all_embeds.shape[1]
            T_fut_q = future_embeds.shape[1]
            visual_query_mask = torch.zeros(
                n_rows, seq_len_full, dtype=torch.bool, device=all_embeds.device
            )
            visual_query_mask[:B, seq_len_full - T_fut_q :] = (
                self._last_visual_active.view(B, 1)
            )
            if n_rows > B:
                visual_kv = visual_kv.repeat((n_rows + B - 1) // B, 1, 1)[:n_rows]

        all_embeds = torch.nan_to_num(all_embeds, nan=0.0, posinf=0.0, neginf=0.0)
        # Backward hook: the encoder backward runs under Lightning's bf16 autocast
        # (loss.backward() fires outside our autocast(enabled=False) context).
        # Under bf16, accumulated Q/K/V matmul gradients inside unfrozen blocks
        # can overflow → NaN, which then flows backward to input_patch_embedding
        # and the visual adapter.  Intercepting d_all_embeds here and zeroing NaN
        # blocks that cascade without affecting learning for finite-gradient steps.
        if self.training and all_embeds.requires_grad:
            all_embeds.register_hook(
                lambda g: torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            )
        encoder_out = self.chronos.encoder(
            inputs_embeds=all_embeds,
            group_ids=all_group_ids,
            attention_mask=all_mask,
            output_attentions=output_attentions,
            visual_kv=visual_kv,
            visual_query_mask=visual_query_mask,
        )
        hidden_states: torch.Tensor = (
            encoder_out.last_hidden_state
        )  # [B_ext, T_full, d]
        hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=0.0, neginf=0.0)

        # Extract TS entity rows
        hidden_states = hidden_states[:B]  # [B, T_full, d]

        # ---- Decode (matches Chronos-2 exactly) -------------------------
        forecast_embeds = hidden_states[:, -num_output_patches:]  # [B, P_out, d]
        quantile_preds = self.chronos.output_patch_embedding(forecast_embeds)
        quantile_preds = rearrange(
            quantile_preds,
            "b n (q p) -> b q (n p)",
            n=num_output_patches,
            q=self.chronos.num_quantiles,
            p=self.chronos.chronos_config.output_patch_size,
        )
        quantile_preds = torch.nan_to_num(
            quantile_preds, nan=0.0, posinf=0.0, neginf=0.0
        )

        loss: Optional[torch.Tensor] = None
        if future_target is not None:
            loss = self.chronos._compute_loss(
                quantile_preds=quantile_preds,
                future_target=future_target,
                future_target_mask=future_target_mask,
                patched_future_covariates_mask=patched_future_cov_mask,
                loc_scale=loc_scale,
                num_output_patches=num_output_patches,
            )

        # Unscale (matches Chronos-2)
        quantile_preds = rearrange(
            quantile_preds,
            "b q h -> b (q h)",
            b=B,
            q=self.chronos.num_quantiles,
            h=num_output_patches * self.chronos.chronos_config.output_patch_size,
        )
        quantile_preds = self.chronos.instance_norm.inverse(quantile_preds, loc_scale)
        quantile_preds = rearrange(
            quantile_preds,
            "b (q h) -> b q h",
            q=self.chronos.num_quantiles,
            h=num_output_patches * self.chronos.chronos_config.output_patch_size,
        )

        return VisionChronos2Output(
            loss=loss,
            quantile_preds=quantile_preds,
            enc_time_self_attn_weights=encoder_out.all_time_self_attn_weights,
            enc_group_self_attn_weights=encoder_out.all_group_self_attn_weights,
            visual_active=visual_active,
            numeric_active=numeric_active,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def forward_numeric_only(self, *args, **kwargs) -> VisionChronos2Output:
        """Pure numeric path — output identical to vanilla Chronos-2."""
        kwargs.pop("video", None)
        kwargs.pop("visual_mask", None)
        return self.forward(*args, **kwargs, video=None, visual_mask=None)

    def trainable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p

    def vision_parameters(self):
        # Each of these is None on some arm: the summarizer and adapter are absent
        # on the raw path (s2d) and on skip_vision_stack, the projector is absent
        # everywhere else. Yield whatever this arm actually built.
        for mod in (
            self.latent_summarizer,
            self.cross_modal_adapter,
            getattr(self, "patch_projector", None),
        ):
            if mod is not None:
                yield from mod.parameters()
        yield from self.multimodal_embed.parameters()

    def chronos_parameters(self):
        yield from self.chronos.parameters()

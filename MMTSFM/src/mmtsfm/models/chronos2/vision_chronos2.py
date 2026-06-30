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
    vis_tokens: torch.Tensor,  # [B, n_vis, d]
    n_vis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Selectively interleave visual summary tokens into the refinement window.

    Builds: [ts_0..ts_{T_M-1}] || [ts_{T_M}, v_{T_M}, ..., ts_{T_ctx-1}, v_{T_ctx-1}]

    Returns:
        interleaved: ``[B, T_ctx + n_vis, d]``
        modality_mask: ``[B, T_ctx + n_vis]`` long tensor — 0=TS, 1=visual
    """
    B, T_ctx, d = ts_tokens.shape
    T_M = T_ctx - n_vis

    macro = ts_tokens[:, :T_M, :]
    ts_refine = ts_tokens[:, T_M:, :]
    pairs = torch.stack([ts_refine, vis_tokens], dim=2)  # [B, n_vis, 2, d]
    refinement = pairs.reshape(B, 2 * n_vis, d)
    interleaved = torch.cat([macro, refinement], dim=1)  # [B, T_ctx+n_vis, d]

    device = ts_tokens.device
    modality_mask = torch.zeros(B, T_ctx + n_vis, dtype=torch.long, device=device)
    vis_positions = T_M + 1 + torch.arange(n_vis, device=device) * 2
    modality_mask[:, vis_positions] = 1

    return interleaved, modality_mask


def build_interleaved_position_ids(
    T_M: int,
    n_vis: int,
    T_fut: int,
    device: torch.device,
) -> torch.Tensor:
    """Build temporal position IDs for the interleaved sequence.

    TS and vis tokens at the same step share the same position ID so that
    RoPE treats them as co-temporal.

    Returns:
        ``[1, T_M + 2*n_vis + T_fut]`` long tensor
    """
    macro_ids = torch.arange(T_M, device=device)
    refine_ids = torch.arange(T_M, T_M + n_vis, device=device)
    refine_pairs = torch.stack([refine_ids, refine_ids], dim=1).reshape(2 * n_vis)
    future_ids = torch.arange(T_M + n_vis, T_M + n_vis + T_fut, device=device)
    return torch.cat([macro_ids, refine_pairs, future_ids]).unsqueeze(0)


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
    adapter_type: str = "linear"
    adapter_n_layers: int = 2
    summarizer_n_heads: int = 4
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
    # "late"        → existing CrossModalAdapter path (batch-dim concat)
    # "interleaved" → selective temporal interleaving (refinement window only)

    visual_encoder_ckpt_path: str = ""
    freeze_visual_encoder: bool = True
    skip_vision_stack: bool = False


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

    def __init__(self, d_model: int, n_entities: int = 0):
        super().__init__()
        self.modality_embed = nn.Embedding(2, d_model)
        self.segment_embed = nn.Embedding(2, d_model)
        self.token_type_embed = nn.Embedding(
            3, d_model
        )  # M1 fix: target/covariate/visual
        self.entity_embed = (
            nn.Embedding(n_entities, d_model) if n_entities > 0 else None
        )
        # Default N(0,1) init gives magnitude ≈ sqrt(d_model) ≈ 22 which swamps the
        # pretrained Chronos-2 activations and causes bf16 overflow → NaN loss.
        # Use small std (0.02, same as BERT/T5 token embeddings) to keep scale compatible.
        for emb in [self.modality_embed, self.segment_embed, self.token_type_embed]:
            nn.init.normal_(emb.weight, std=0.02)
        if self.entity_embed is not None:
            nn.init.normal_(self.entity_embed.weight, std=0.02)

    def add_modality(self, tokens: torch.Tensor, modality_id: int) -> torch.Tensor:
        """``tokens [B, T, d]`` + scalar modality embedding."""
        idx = torch.tensor(modality_id, device=tokens.device, dtype=torch.long)
        return tokens + self.modality_embed(idx)

    def add_segment(self, tokens: torch.Tensor, segment_id: int) -> torch.Tensor:
        """``tokens [B, T, d]`` + scalar segment embedding (0=context, 1=future)."""
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
        idx = torch.tensor(token_type_id, device=tokens.device, dtype=torch.long)
        return tokens + self.token_type_embed(idx)

    def add_entity(
        self, tokens: torch.Tensor, entity_ids: torch.Tensor
    ) -> torch.Tensor:
        """``tokens [B, T, d]`` + entity embedding ``[B, d]`` from position indices."""
        if self.entity_embed is None:
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

            self.latent_summarizer: Optional[nn.Module] = LatentSummarizer(
                d_v=_d_v,
                d_model=d_model,
                n_vis_steps=vision_config.n_visual_context_steps,
                n_heads=vision_config.summarizer_n_heads,
                dropout=vision_config.dropout,
            )

            if vision_config.fusion_mode == "late":
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
            self.latent_summarizer = None
            self.cross_modal_adapter = None

        self.multimodal_embed = MultimodalEmbedding(
            d_model=d_model, n_entities=vision_config.n_entities
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

        # Flatten N_soft into batch — [B * N_soft, T_ctx, d_model]
        return (
            soft.reshape(B_ * N_s, T_ctx, D_),
            input_embeds_mm,
            future_embeds_mm,
            visual_active,
            numeric_active,
        )

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

        # Future patch embeddings [B, num_output_patches, d_model]
        future_embeds: torch.Tensor = self.chronos.input_patch_embedding(patched_future)
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

        if covariate_channels:
            H_cov = covariate_channels[0].shape[-1]
            cov_mask_zeros = torch.zeros(B, H_cov, device=device)
            for cov_ch in covariate_channels:
                patched_cov, _ = self.chronos._prepare_patched_future(
                    future_covariates=cov_ch,
                    future_covariates_mask=cov_mask_zeros,
                    loc_scale=loc_scale,
                    num_output_patches=num_output_patches,
                    batch_size=B,
                )
                patched_cov = torch.nan_to_num(
                    patched_cov, nan=0.0, posinf=0.0, neginf=0.0
                )
                cov_embeds = self.chronos.input_patch_embedding(patched_cov)
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

        if use_video and self.vcfg.fusion_mode == "interleaved":
            # --- Interleaved fusion path ---
            # Works for both Variant A (use_grassmann=True) and Variant B (use_grassmann=False)
            n_vis = min(self.vcfg.n_visual_context_steps, T_ctx)

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

            # LatentSummarizer — [B, n_vis, d_model]  (T_ts=n_vis → no null tokens)
            vis_summary = self.latent_summarizer(
                video_tokens=video_tokens,
                T_ts=n_vis,
                visual_mask=lat_mask,
                frame_delta_t=video_delta_t,
            )

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

            # Interleave refinement window
            interleaved_ctx, modality_mask_ctx = interleave_sequences(
                input_embeds_mm, vis_summary, n_vis
            )

            # Full sequence: [B, T_ctx + n_vis + T_fut, d]
            T_fut = future_embeds_mm.shape[1]
            all_embeds = torch.cat([interleaved_ctx, future_embeds_mm], dim=1)
            modality_mask_fut = torch.zeros(B, T_fut, dtype=torch.long, device=device)
            modality_mask = torch.cat([modality_mask_ctx, modality_mask_fut], dim=1)

            # All interleaved positions are valid
            all_mask = torch.ones(B, T_ctx + n_vis + T_fut, device=device, dtype=dtype)
            all_group_ids = group_ids

            # Position IDs: TS and vis tokens at same step share position
            T_M = T_ctx - n_vis
            position_ids = build_interleaved_position_ids(
                T_M, n_vis, T_fut, device
            ).expand(B, -1)

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

        elif use_video:
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
        yield from self.latent_summarizer.parameters()
        yield from self.cross_modal_adapter.parameters()
        yield from self.multimodal_embed.parameters()

    def chronos_parameters(self):
        yield from self.chronos.parameters()

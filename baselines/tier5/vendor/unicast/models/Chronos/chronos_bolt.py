# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>, Caner Turkmen <atturkm@amazon.com>, Lorenzo Stella <stellalo@amazon.com>
# Original source:
# https://github.com/autogluon/autogluon/blob/f57beb26cb769c6e0d484a6af2b89eab8aee73a8/timeseries/src/autogluon/timeseries/models/chronos/pipeline/chronos_bolt.py

import copy
import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.t5.modeling_t5 import (
    ACT2FN,
    T5Config,
    T5LayerNorm,
    T5PreTrainedModel,
)
from transformers.utils import ModelOutput

from .base import BaseChronosPipeline, ForecastType

logger = logging.getLogger(__file__)

############################### CustomT5Stack ###############################
from transformers import add_start_docstrings
from transformers.utils.import_utils import is_torchdynamo_compiling
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.models.t5.modeling_t5 import T5Block
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (`Dict[int, list]`, *optional*):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the t5 models have the
            following number of attention modules:

                - google-t5/t5-small: 6
                - google-t5/t5-base: 12
                - google-t5/t5-large: 24
                - google-t5/t5-3b: 24
                - google-t5/t5-11b: 24

    Example:

    ```python
    # Here is an example of a device map on a machine with 4 GPUs using google-t5/t5-3b, which has a total of 24 attention modules:
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```python
    # On a 4 GPU machine with google-t5/t5-3b:
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""

class CustomT5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None, PT_len: int = 0):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.prompts = []
        self.prompts_token_len = PT_len
        if self.prompts_token_len > 0:
            for i in range(self.config.num_hidden_layers):
                self.prompts.append(nn.init.xavier_uniform_(nn.Parameter(torch.randn(1, PT_len, config.hidden_size))))
        self.prompts = nn.ParameterList(self.prompts)

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0), layer_idx=i) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5Stack.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        vision_embedding=None,
        text_embedding=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        if vision_embedding is not None:
            seq_length += 1 # add vision_embedding
        if text_embedding is not None:
            seq_length += 1 # add text_embedding
        if self.prompts_token_len > 0: # add prompt_token_len
            seq_length += self.prompts_token_len

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values
        return_legacy_cache = False
        return_self_attention_cache = False
        if self.is_decoder and (use_cache or past_key_values is not None):
            if isinstance(past_key_values, Cache) and not isinstance(past_key_values, EncoderDecoderCache):
                return_self_attention_cache = True
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            elif not isinstance(past_key_values, EncoderDecoderCache):
                return_legacy_cache = True
                logger.warning_once(
                    "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.48.0. "
                    "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                    "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
                )
                past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
            elif past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
        elif not self.is_decoder:
            # do not pass cache object down the line for encoder stack
            # it messes indexing later in decoder-stack because cache object is modified in-place
            past_key_values = None

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )

        if attention_mask is None and not is_torchdynamo_compiling():
            # required mask seq length can be calculated via length of past cache
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.config.is_decoder:
            causal_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values.self_attention_cache if past_key_values is not None else None,
                output_attentions,
            )
        elif attention_mask is not None:
            causal_mask = attention_mask[:, None, None, :]
            causal_mask = causal_mask.to(dtype=inputs_embeds.dtype)
            causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min
        else:
            causal_mask = None

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        # add text_embedding
        if text_embedding is not None:
            if self.config.vocab_size == 2: # use_reg_token == True
                inputs_embeds = torch.cat((inputs_embeds[:, :1], text_embedding.unsqueeze(dim=1), inputs_embeds[:, 1:]), dim=1)
            else:
                inputs_embeds = torch.cat((inputs_PT, text_embedding.unsqueeze(dim=1), inputs_embeds), dim=1)
        
        # add vision_embedding
        if vision_embedding is not None:
            if self.config.vocab_size == 2: # use_reg_token == True
                inputs_embeds = torch.cat((inputs_embeds[:, :1], vision_embedding.unsqueeze(dim=1), inputs_embeds[:, 1:]), dim=1)
            else:
                inputs_embeds = torch.cat((inputs_PT, vision_embedding.unsqueeze(dim=1), inputs_embeds), dim=1)
        
        # add prompt_token
        if self.prompts_token_len > 0:
            inputs_PT = self.prompts[0].repeat(inputs_embeds.size(0), 1, 1).to(inputs_embeds.device).to(inputs_embeds.dtype)
            if self.config.vocab_size == 2: # use_reg_token == True
                inputs_embeds = torch.cat((inputs_embeds[:, :1], inputs_PT, inputs_embeds[:, 1:]), dim=1)
            else:
                inputs_embeds = torch.cat((inputs_PT,inputs_embeds, inputs_embeds), dim=1)

        hidden_states = self.dropout(inputs_embeds)

        for i, layer_module in enumerate(self.block):
            if self.prompts_token_len > 0:
                if self.config.vocab_size == 2: # use_reg_token == True
                    hidden_states[:, 1:1+self.prompts_token_len, :] = self.prompts[i].repeat(inputs_embeds.size(0),1, 1).to(hidden_states.device).to(hidden_states.dtype)
                else:
                    hidden_states[:, :self.prompts_token_len, :] = self.prompts[i].repeat(inputs_embeds.size(0),1, 1).to(hidden_states.device).to(hidden_states.dtype)
            
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if causal_mask is not None:
                    causal_mask = causal_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    causal_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
                    return_dict,
                    cache_position,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, next_decoder_cache = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_self_attention_cache:
            next_cache = past_key_values.self_attention_cache
        if return_legacy_cache:
            next_cache = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.llama.modeling_llama.LlamaPreTrainedModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

#############################################################################

@dataclass
class ChronosBoltConfig:
    context_length: int
    prediction_length: int
    input_patch_size: int
    input_patch_stride: int
    quantiles: List[float]
    use_reg_token: bool = False


@dataclass
class ChronosBoltOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    quantile_preds: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    cross_attentions: Optional[torch.Tensor] = None


class Patch(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]

        if length % self.patch_size != 0:
            padding_size = (
                *x.shape[:-1],
                self.patch_size - (length % self.patch_size),
            )
            padding = torch.full(
                size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device
            )
            x = torch.concat((padding, x), dim=-1)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        return x


class InstanceNorm(nn.Module):
    """
    See, also, RevIN. Apply standardization along the last dimension.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num(
                torch.nanmean((x - loc).square(), dim=-1, keepdim=True).sqrt(), nan=1.0
            )
            scale = torch.where(scale == 0, self.eps, scale)
        else:
            loc, scale = loc_scale

        return (x - loc) / scale, (loc, scale)

    def inverse(
        self, x: torch.Tensor, loc_scale: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        loc, scale = loc_scale
        return x * scale + loc


class ResidualBlock(nn.Module):
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
            self.layer_norm = T5LayerNorm(out_dim)

    def forward(self, x: torch.Tensor):
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)

        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(out)
        return out


class ChronosBoltModelForForecasting(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [  # type: ignore
        r"input_patch_embedding\.",
        r"output_patch_embedding\.",
    ]
    _keys_to_ignore_on_load_unexpected = [r"lm_head.weight"]  # type: ignore
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]  # type: ignore

    def __init__(self, config: T5Config, PT_len: int = 0):
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        super().__init__(config)
        self.model_dim = config.d_model

        self.chronos_config = ChronosBoltConfig(**config.chronos_config)

        # Only decoder_start_id (and optionally REG token)
        if self.chronos_config.use_reg_token:
            config.reg_token_id = 1

        config.vocab_size = 2 if self.chronos_config.use_reg_token else 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        self.prompt_token_length = PT_len

        # Input patch embedding layer
        self.input_patch_embedding = ResidualBlock(
            in_dim=self.chronos_config.input_patch_size * 2,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        # patching layer
        self.patch = Patch(
            patch_size=self.chronos_config.input_patch_size,
            patch_stride=self.chronos_config.input_patch_stride,
        )

        # instance normalization, also referred to as "scaling" in Chronos and GluonTS
        self.instance_norm = InstanceNorm()

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = CustomT5Stack(encoder_config, self.shared, PT_len)

        self._init_decoder(config)

        self.num_quantiles = len(self.chronos_config.quantiles)
        quantiles = torch.tensor(self.chronos_config.quantiles, dtype=self.dtype)
        self.register_buffer("quantiles", quantiles, persistent=False)

        self.output_patch_embedding = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=self.num_quantiles * self.chronos_config.prediction_length,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def _init_weights(self, module):
        super()._init_weights(module)
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, (self.__class__)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, ResidualBlock):
            module.hidden_layer.weight.data.normal_(
                mean=0.0,
                std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5),
            )
            if (
                hasattr(module.hidden_layer, "bias")
                and module.hidden_layer.bias is not None
            ):
                module.hidden_layer.bias.data.zero_()

            module.residual_layer.weight.data.normal_(
                mean=0.0,
                std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5),
            )
            if (
                hasattr(module.residual_layer, "bias")
                and module.residual_layer.bias is not None
            ):
                module.residual_layer.bias.data.zero_()

            module.output_layer.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_ff) ** -0.5)
            )
            if (
                hasattr(module.output_layer, "bias")
                and module.output_layer.bias is not None
            ):
                module.output_layer.bias.data.zero_()

    def encode(
        self, context: torch.Tensor, vision_embedding: Optional[torch.Tensor] = None, text_embedding: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor
    ]:
        mask = (
            mask.to(context.dtype)
            if mask is not None
            else torch.isnan(context).logical_not().to(context.dtype)
        )

        batch_size, _ = context.shape
        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length :]
            mask = mask[..., -self.chronos_config.context_length :]

        # scaling
        context, loc_scale = self.instance_norm(context)

        # the scaling op above is done in 32-bit precision,
        # then the context is moved to model's dtype
        context = context.to(self.dtype)
        mask = mask.to(self.dtype)

        # patching
        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)
        # concat context and mask along patch dim
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)

        # attention_mask = 1 if at least one item in the patch is observed
        attention_mask = (
            patched_mask.sum(dim=-1) > 0
        )  # (batch_size, patched_seq_length)

        input_embeds = self.input_patch_embedding(patched_context)

        if self.chronos_config.use_reg_token:
            # Append [REG]
            reg_input_ids = torch.full(
                (batch_size, 1),
                self.config.reg_token_id,
                device=input_embeds.device,
            )
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [
                    attention_mask.to(self.dtype),
                    torch.ones_like(reg_input_ids).to(self.dtype),
                ],
                dim=-1,
            )
        
        if vision_embedding is not None:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones([batch_size, 1]).to(self.dtype).to(self.device)
                ],
                dim=-1
            )
        
        if text_embedding is not None:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones([batch_size, 1]).to(self.dtype).to(self.device)
                ],
                dim=-1
            )
        
        if self.prompt_token_length > 0:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones([batch_size, self.prompt_token_length]).to(self.dtype).to(self.device)
                ],
                dim=-1
            )

        encoder_outputs = self.encoder(
            attention_mask=attention_mask,
            vision_embedding=vision_embedding,
            text_embedding=text_embedding,
            inputs_embeds=input_embeds,
        )

        return encoder_outputs[0], loc_scale, input_embeds, attention_mask

    def forward(
        self,
        context: torch.Tensor,
        vision_embedding: Optional[torch.Tensor] = None,
        text_embedding: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> ChronosBoltOutput:
        batch_size = context.size(0)
        
        hidden_states, loc_scale, input_embeds, attention_mask = self.encode(
            context=context, vision_embedding=vision_embedding, text_embedding=text_embedding, mask=mask
        )

        sequence_output = self.decode(input_embeds, attention_mask, hidden_states)

        quantile_preds_shape = (
            batch_size,
            self.num_quantiles,
            self.chronos_config.prediction_length,
        )
        quantile_preds = self.output_patch_embedding(sequence_output).view(
            *quantile_preds_shape
        )

        loss = None
        if target is not None:
            # normalize target
            target, _ = self.instance_norm(target, loc_scale)
            target = target.unsqueeze(1)  # type: ignore
            assert self.chronos_config.prediction_length >= target.shape[-1]

            target = target.to(quantile_preds.device)
            target_mask = (
                target_mask.unsqueeze(1).to(quantile_preds.device)
                if target_mask is not None
                else ~torch.isnan(target)
            )
            target[~target_mask] = 0.0

            # pad target and target_mask if they are shorter than model's prediction_length
            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (
                    *target.shape[:-1],
                    self.chronos_config.prediction_length - target.shape[-1],
                )
                target = torch.cat(
                    [target, torch.zeros(padding_shape).to(target)], dim=-1
                )
                target_mask = torch.cat(
                    [target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1
                )

            loss = (
                2
                * torch.abs(
                    (target - quantile_preds)
                    * (
                        (target <= quantile_preds).float()
                        - self.quantiles.view(1, self.num_quantiles, 1)  # type: ignore
                    )
                )
                * target_mask.float()
            )
            loss = loss.mean(dim=-2)  # Mean over prediction horizon
            loss = loss.sum(dim=-1)  # Sum over quantile levels
            loss = loss.mean()  # Mean over batch

        # Unscale predictions
        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(batch_size, -1),
            loc_scale,
        ).view(*quantile_preds_shape)

        return ChronosBoltOutput(
            loss=loss,
            quantile_preds=quantile_preds,
        )

    def _init_decoder(self, config):
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = CustomT5Stack(decoder_config, self.shared)

    def decode(
        self,
        input_embeds,
        attention_mask,
        hidden_states,
        output_attentions=False,
    ):
        """
        Parameters
        ----------
        input_embeds: torch.Tensor
            Patched and embedded inputs. Shape (batch_size, patched_context_length, d_model)
        attention_mask: torch.Tensor
            Attention mask for the patched context. Shape (batch_size, patched_context_length), type: torch.int64
        hidden_states: torch.Tensor
            Hidden states returned by the encoder. Shape (batch_size, patched_context_length, d_model)

        Returns
        -------
        last_hidden_state
            Last hidden state returned by the decoder, of shape (batch_size, 1, d_model)
        """
        batch_size = input_embeds.shape[0]
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.config.decoder_start_token_id,
            device=input_embeds.device,
        )
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )

        return decoder_outputs.last_hidden_state  # sequence_outputs, b x 1 x d_model


class ChronosBoltPipeline(BaseChronosPipeline):
    forecast_type: ForecastType = ForecastType.QUANTILES
    default_context_length: int = 2048

    def __init__(self, model: ChronosBoltModelForForecasting):
        super().__init__(inner_model=model)  # type: ignore
        self.model = model

    @property
    def quantiles(self) -> List[float]:
        return self.model.config.chronos_config["quantiles"]

    @torch.no_grad()
    def embed(
        self, context: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get encoder embeddings for the given time series.

        Parameters
        ----------
        context
            Input series. This is either a 1D tensor, or a list
            of 1D tensors, or a 2D tensor whose first dimension
            is batch. In the latter case, use left-padding with
            ``torch.nan`` to align series of different lengths.

        Returns
        -------
        embeddings, loc_scale
            A tuple of two items: the encoder embeddings and the loc_scale,
            i.e., the mean and std of the original time series.
            The encoder embeddings are shaped (batch_size, num_patches + 1, d_model),
            where num_patches is the number of patches in the time series
            and the extra 1 is for the [REG] token (if used by the model).
        """
        context_tensor = self._prepare_and_validate_context(context=context)
        model_context_length = self.model.config.chronos_config["context_length"]

        if context_tensor.shape[-1] > model_context_length:
            context_tensor = context_tensor[..., -model_context_length:]

        context_tensor = context_tensor.to(
            device=self.model.device,
            dtype=torch.float32,
        )
        embeddings, loc_scale, *_ = self.model.encode(context=context_tensor)
        return embeddings.cpu(), (
            loc_scale[0].squeeze(-1).cpu(),
            loc_scale[1].squeeze(-1).cpu(),
        )

    def predict(  # type: ignore[override]
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        limit_prediction_length: bool = False,
    ) -> torch.Tensor:
        """
        Get forecasts for the given time series.

        Refer to the base method (``BaseChronosPipeline.predict``)
        for details on shared parameters.
        Additional parameters
        ---------------------
        limit_prediction_length
            Force prediction length smaller or equal than the
            built-in prediction length from the model. False by
            default. When true, fail loudly if longer predictions
            are requested, otherwise longer predictions are allowed.

        Returns
        -------
        torch.Tensor
            Forecasts of shape (batch_size, num_quantiles, prediction_length)
            where num_quantiles is the number of quantiles the model has been
            trained to output. For official Chronos-Bolt models, the value of
            num_quantiles is 9 for [0.1, 0.2, ..., 0.9]-quantiles.

        Raises
        ------
        ValueError
            When limit_prediction_length is True and the prediction_length is
            greater than model's trainig prediction_length.
        """
        context_tensor = self._prepare_and_validate_context(context=context)

        model_context_length = self.model.config.chronos_config["context_length"]
        model_prediction_length = self.model.config.chronos_config["prediction_length"]
        if prediction_length is None:
            prediction_length = model_prediction_length

        if prediction_length > model_prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {model_prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg)

        predictions = []
        remaining = prediction_length

        # We truncate the context here because otherwise batches with very long
        # context could take up large amounts of GPU memory unnecessarily.
        if context_tensor.shape[-1] > model_context_length:
            context_tensor = context_tensor[..., -model_context_length:]

        # TODO: We unroll the forecast of Chronos Bolt greedily with the full forecast
        # horizon that the model was trained with (i.e., 64). This results in variance collapsing
        # every 64 steps.
        context_tensor = context_tensor.to(
            device=self.model.device,
            dtype=torch.float32,
        )
        while remaining > 0:
            with torch.no_grad():
                prediction = self.model(
                    context=context_tensor,
                ).quantile_preds.to(context_tensor)

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            central_idx = torch.abs(torch.tensor(self.quantiles) - 0.5).argmin()
            central_prediction = prediction[:, central_idx]

            context_tensor = torch.cat([context_tensor, central_prediction], dim=-1)

        return torch.cat(predictions, dim=-1)[..., :prediction_length].to(
            dtype=torch.float32, device="cpu"
        )

    def predict_quantiles(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **predict_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Refer to the base method (``BaseChronosPipeline.predict_quantiles``).
        """
        # shape (batch_size, prediction_length, len(training_quantile_levels))
        predictions = (
            self.predict(context, prediction_length=prediction_length, **predict_kwargs)
            .detach()
            .swapaxes(1, 2)
        )

        training_quantile_levels = self.quantiles

        if set(quantile_levels).issubset(set(training_quantile_levels)):
            # no need to perform intra/extrapolation
            quantiles = predictions[
                ..., [training_quantile_levels.index(q) for q in quantile_levels]
            ]
        else:
            # we rely on torch for interpolating quantiles if quantiles that
            # Chronos Bolt was trained on were not provided
            if min(quantile_levels) < min(training_quantile_levels) or max(
                quantile_levels
            ) > max(training_quantile_levels):
                logger.warning(
                    f"\tQuantiles to be predicted ({quantile_levels}) are not within the range of "
                    f"quantiles that Chronos-Bolt was trained on ({training_quantile_levels}). "
                    "Quantile predictions will be set to the minimum/maximum levels at which Chronos-Bolt "
                    "was trained on. This may significantly affect the quality of the predictions."
                )

            # TODO: this is a hack that assumes the model's quantiles during training (training_quantile_levels)
            # made up an equidistant grid along the quantile dimension. i.e., they were (0.1, 0.2, ..., 0.9).
            # While this holds for official Chronos-Bolt models, this may not be true in the future, and this
            # function may have to be revised.
            augmented_predictions = torch.cat(
                [predictions[..., [0]], predictions, predictions[..., [-1]]],
                dim=-1,
            )
            quantiles = torch.quantile(
                augmented_predictions,
                q=torch.tensor(quantile_levels, dtype=augmented_predictions.dtype),
                dim=-1,
            ).permute(1, 2, 0)
        # NOTE: the median is returned as the mean here
        mean = predictions[:, :, training_quantile_levels.index(0.5)]
        return quantiles, mean

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Load the model, either from a local path or from the HuggingFace Hub.
        Supports the same arguments as ``AutoConfig`` and ``AutoModel``
        from ``transformers``.
        """

        config = AutoConfig.from_pretrained(*args, **kwargs)
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        architecture = config.architectures[0]
        class_ = globals().get(architecture)

        if class_ is None:
            logger.warning(
                f"Unknown architecture: {architecture}, defaulting to ChronosBoltModelForForecasting"
            )
            class_ = ChronosBoltModelForForecasting

        model = class_.from_pretrained(*args, **kwargs)
        return cls(model=model)

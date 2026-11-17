"""Runtime compatibility shims for mt3-infer against transformers >= 4.44 and 5.x."""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from typing import Any

import torch
import torch.nn.functional as F


def _ensure_model_parallel_utils() -> None:
    """Provide transformers.utils.model_parallel_utils stub when removed (transformers v5+)."""
    mod_name = "transformers.utils.model_parallel_utils"
    if mod_name in sys.modules:
        return
    try:
        importlib.import_module(mod_name)
        return
    except ModuleNotFoundError:
        pass

    import transformers.utils

    stub = types.ModuleType(mod_name)

    def assert_device_map(device_map: dict[Any, Any], num_blocks: int) -> None:
        return None

    def get_device_map(n_layers: int, devices: Any) -> dict[Any, Any]:
        return {}

    stub.assert_device_map = assert_device_map
    stub.get_device_map = get_device_map
    sys.modules[mod_name] = stub
    transformers.utils.model_parallel_utils = stub


def _ensure_pytorch_utils() -> None:
    """Restore find_pruneable_heads_and_indices on transformers.pytorch_utils for v5+."""
    import transformers.pytorch_utils as pt_utils

    if hasattr(pt_utils, "find_pruneable_heads_and_indices"):
        return

    def find_pruneable_heads_and_indices(
        heads: list[int],
        n_heads: int,
        head_size: int,
        already_pruned_heads: set[int],
    ) -> tuple[set[int], torch.LongTensor]:
        mask = torch.ones(n_heads, head_size)
        new_heads = set(heads) - already_pruned_heads
        for head in new_heads:
            adjusted = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[adjusted] = 0
        flat = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(flat))[flat].long()
        return new_heads, index

    pt_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


def _create_extended_attention_mask_for_decoder(
    input_shape: tuple[int, ...],
    attention_mask: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    dev = attention_mask.device if device is None else device
    batch_size, seq_length = input_shape
    seq_ids = torch.arange(seq_length, device=dev)
    causal_mask = (seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]).to(
        attention_mask.dtype
    )
    if causal_mask.shape[1] < attention_mask.shape[1]:
        prefix_len = attention_mask.shape[1] - causal_mask.shape[1]
        prefix = torch.ones((batch_size, seq_length, prefix_len), device=dev, dtype=causal_mask.dtype)
        causal_mask = torch.cat([prefix, causal_mask], dim=-1)
    return causal_mask[:, None, :, :] * attention_mask[:, None, None, :]


def _ensure_pretrained_model_masks() -> None:
    """Restore ModuleUtilsMixin mask helpers removed from PreTrainedModel in transformers v5+."""
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "invert_attention_mask"):

        def invert_attention_mask(self: PreTrainedModel, encoder_attention_mask: torch.Tensor) -> torch.Tensor:
            if encoder_attention_mask.dim() == 3:
                ext = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                ext = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(f"Unsupported encoder_attention_mask rank: {encoder_attention_mask.dim()}")
            ext = ext.to(dtype=self.dtype)
            return (1.0 - ext) * torch.finfo(self.dtype).min

        PreTrainedModel.invert_attention_mask = invert_attention_mask

    if not hasattr(PreTrainedModel, "create_extended_attention_mask_for_decoder"):
        PreTrainedModel.create_extended_attention_mask_for_decoder = staticmethod(
            _create_extended_attention_mask_for_decoder
        )

    if not hasattr(PreTrainedModel, "get_extended_attention_mask"):

        def get_extended_attention_mask(
            self: PreTrainedModel,
            attention_mask: torch.Tensor,
            input_shape: tuple[int, ...],
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> torch.Tensor:
            target_dtype = self.dtype if dtype is None else dtype
            if attention_mask.dim() == 3:
                ext = attention_mask[:, None, :, :]
            elif attention_mask.dim() == 2:
                if self.config.is_decoder:
                    ext = _create_extended_attention_mask_for_decoder(input_shape, attention_mask, device)
                else:
                    ext = attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
                )
            ext = ext.to(dtype=target_dtype)
            return (1.0 - ext) * torch.finfo(target_dtype).min

        PreTrainedModel.get_extended_attention_mask = get_extended_attention_mask

    if not hasattr(PreTrainedModel, "get_head_mask"):

        def _convert_head_mask_to_5d(
            self: PreTrainedModel,
            head_mask: torch.Tensor,
            num_hidden_layers: int,
        ) -> torch.Tensor:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
            return head_mask.to(dtype=self.dtype)

        def get_head_mask(
            self: PreTrainedModel,
            head_mask: torch.Tensor | None,
            num_hidden_layers: int,
            is_attention_chunked: bool = False,
        ) -> torch.Tensor | list[None]:
            if head_mask is None:
                return [None] * num_hidden_layers
            converted = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            return converted.unsqueeze(-1) if is_attention_chunked else converted

        PreTrainedModel._convert_head_mask_to_5d = _convert_head_mask_to_5d
        PreTrainedModel.get_head_mask = get_head_mask


def _project_kv(
    hidden_states: torch.Tensor,
    proj_layer: torch.nn.Linear,
    key_value_states: torch.Tensor | None,
    past_kv: torch.Tensor | None,
    batch_size: int,
    n_heads: int,
    head_dim: int,
) -> torch.Tensor:
    def _shape(states: torch.Tensor) -> torch.Tensor:
        return states.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)

    if key_value_states is None:
        projected = _shape(proj_layer(hidden_states))
        return torch.cat([past_kv, projected], dim=2) if past_kv is not None else projected
    if past_kv is not None:
        return past_kv
    return _shape(proj_layer(key_value_states))


def _legacy_t5_attention_forward(
    attn: Any,
    hidden_states: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    key_value_states: torch.Tensor | None = None,
    position_bias: torch.Tensor | None = None,
    past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    layer_head_mask: torch.Tensor | None = None,
    query_length: int | None = None,
    use_cache: bool = False,
    output_attentions: bool = False,
) -> tuple[Any, ...]:
    """Execute pre-4.44 tuple-KV T5 attention expected by YourMT3's T5BlockYMT3."""
    batch_size, seq_length = hidden_states.shape[:2]
    real_seq_length = seq_length
    if past_key_value is not None:
        real_seq_length = query_length if query_length is not None else seq_length + past_key_value[0].shape[2]

    query_states = attn.q(hidden_states).view(batch_size, -1, attn.n_heads, attn.key_value_proj_dim).transpose(1, 2)
    key_states = _project_kv(
        hidden_states,
        attn.k,
        key_value_states,
        past_key_value[0] if past_key_value is not None else None,
        batch_size,
        attn.n_heads,
        attn.key_value_proj_dim,
    )
    value_states = _project_kv(
        hidden_states,
        attn.v,
        key_value_states,
        past_key_value[1] if past_key_value is not None else None,
        batch_size,
        attn.n_heads,
        attn.key_value_proj_dim,
    )

    present_key_value_state = (key_states, value_states) if (attn.is_decoder and use_cache) else None
    scores = torch.matmul(query_states, key_states.transpose(3, 2))
    key_length = key_states.shape[-2]

    if position_bias is None:
        if not attn.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, attn.n_heads, seq_length, key_length),
                device=scores.device,
                dtype=scores.dtype,
            )
        else:
            bias_query_len = real_seq_length if key_value_states is None else seq_length
            position_bias = attn.compute_bias(bias_query_len, key_length, device=scores.device)
            if past_key_value is not None:
                position_bias = position_bias[:, :, -seq_length:, :]
        if mask is not None:
            position_bias = position_bias + mask[:, :, :, :key_length]

    pruned_heads = getattr(attn, "pruned_heads", None)
    if pruned_heads:
        head_keep = torch.ones(position_bias.shape[1], device=position_bias.device, dtype=torch.bool)
        head_keep[list(pruned_heads)] = False
        position_bias_masked = position_bias[:, head_keep]
    else:
        position_bias_masked = position_bias

    scores = scores + position_bias_masked
    attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn_weights = F.dropout(attn_weights, p=attn.dropout, training=attn.training)
    if layer_head_mask is not None:
        attn_weights = attn_weights * layer_head_mask

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, attn.inner_dim)
    attn_output = attn.o(attn_output)

    outputs: tuple[Any, ...] = (attn_output, present_key_value_state, position_bias)
    if output_attentions:
        outputs = outputs + (attn_weights,)
    return outputs


def _ensure_yourmt3_t5_attention() -> None:
    """Adapt YourMT3's T5 attention layers to transformers >= 4.44 without affecting MR-MT3."""
    from transformers.models.t5.modeling_t5 import (
        T5LayerCrossAttention as HFT5LayerCrossAttention,
    )
    from transformers.models.t5.modeling_t5 import (
        T5LayerSelfAttention as HFT5LayerSelfAttention,
    )

    params = inspect.signature(HFT5LayerSelfAttention.forward).parameters
    if "past_key_values" not in params and "cache_position" not in params:
        return

    from mt3_infer.models.yourmt3.model import t5mod

    if getattr(t5mod.T5LayerSelfAttention, "_ymt3_compat", False):
        return

    class _CompatT5LayerSelfAttention(HFT5LayerSelfAttention):
        _ymt3_compat = True

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_bias: torch.Tensor | None = None,
            layer_head_mask: torch.Tensor | None = None,
            past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
            use_cache: bool = False,
            output_attentions: bool = False,
        ) -> tuple[Any, ...]:
            normed = self.layer_norm(hidden_states)
            attn_out = _legacy_t5_attention_forward(
                self.SelfAttention,
                normed,
                mask=attention_mask,
                position_bias=position_bias,
                past_key_value=past_key_value,
                layer_head_mask=layer_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            return (hidden_states + self.dropout(attn_out[0]),) + attn_out[1:]

    class _CompatT5LayerCrossAttention(HFT5LayerCrossAttention):
        _ymt3_compat = True

        def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_bias: torch.Tensor | None = None,
            layer_head_mask: torch.Tensor | None = None,
            past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
            use_cache: bool = False,
            query_length: int | None = None,
            output_attentions: bool = False,
        ) -> tuple[Any, ...]:
            normed = self.layer_norm(hidden_states)
            attn_out = _legacy_t5_attention_forward(
                self.EncDecAttention,
                normed,
                mask=attention_mask,
                key_value_states=key_value_states,
                position_bias=position_bias,
                past_key_value=past_key_value,
                layer_head_mask=layer_head_mask,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            return (hidden_states + self.dropout(attn_out[0]),) + attn_out[1:]

    t5mod.T5LayerSelfAttention = _CompatT5LayerSelfAttention
    t5mod.T5LayerCrossAttention = _CompatT5LayerCrossAttention


def ensure_mt3_transformers_compat() -> None:
    """Install idempotent compatibility shims so mt3-infer works with modern transformers."""
    _ensure_model_parallel_utils()
    _ensure_pytorch_utils()
    _ensure_pretrained_model_masks()
    _ensure_yourmt3_t5_attention()

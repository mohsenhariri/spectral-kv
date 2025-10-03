from typing import Any, Dict, List, Optional, Tuple, Union

import hqq
import torch
from hqq.core.quantize import Quantizer
from transformers import DynamicCache

from .KVQConfig import KVQConfig


class KVQ(DynamicCache):

    def __init__(self, config: KVQConfig) -> None:
        super().__init__()

        self._quantized_key_cache: List[torch.Tensor] = []
        self._quantized_value_cache: List[torch.Tensor] = []

        self.config = config

        self.group_size_k = config.group_size_k
        self.group_size_v = config.group_size_v
        self.residual_length_k = config.residual_length_k
        self.residual_length_v = config.residual_length_v
        self.axis_k = config.axis_k
        self.axis_v = config.axis_v
        self.compute_dtype = config.compute_dtype
        self.device = config.device

        self.quantizer = Quantizer

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Check layer index validity
        if layer_idx > len(self.key_cache):
            raise ValueError(
                "QuantizedCache does not support model usage where layers are skipped. Use DynamicCache."
            )

        # Determine bits for this layer (key)
        nbits_k = self.config.nbits_k
        if isinstance(nbits_k, list):
            nbits_k_layer = (
                nbits_k[layer_idx] if layer_idx < len(nbits_k) else nbits_k[-1]
            )
        else:
            nbits_k_layer = nbits_k

        # Determine bits for this layer (value)
        nbits_v = self.config.nbits_v
        if isinstance(nbits_v, list):
            nbits_v_layer = (
                nbits_v[layer_idx] if layer_idx < len(nbits_v) else nbits_v[-1]
            )
        else:
            nbits_v_layer = nbits_v

        # Determine group size for this layer (key)
        # group_size_k can be either int (fixed) or List[int] (dynamic per layer)
        group_size_k = self.config.group_size_k
        if isinstance(group_size_k, list):
            group_size_k_layer = (
                group_size_k[layer_idx]
                if layer_idx < len(group_size_k)
                else group_size_k[-1]
            )
        else:
            group_size_k_layer = group_size_k

        # Determine group size for this layer (value)
        # group_size_v can be either int (fixed) or List[int] (dynamic per layer)
        group_size_v = self.config.group_size_v
        if isinstance(group_size_v, list):
            group_size_v_layer = (
                group_size_v[layer_idx]
                if layer_idx < len(group_size_v)
                else group_size_v[-1]
            )
        else:
            group_size_v_layer = group_size_v

        # Update key/value caches with layer-specific bits and group sizes
        keys_to_return = self._update(
            key_states,
            self._quantized_key_cache,
            self.key_cache,
            layer_idx,
            axis=self.axis_k,
            nbits=nbits_k_layer,
            group_size=group_size_k_layer,
            residual_length=self.residual_length_k,
        )

        values_to_return = self._update(
            value_states,
            self._quantized_value_cache,
            self.value_cache,
            layer_idx,
            axis=self.axis_v,
            nbits=nbits_v_layer,
            group_size=group_size_v_layer,
            residual_length=self.residual_length_v,
        )

        return keys_to_return, values_to_return

    def _update(
        self,
        k_or_v_states: torch.Tensor,
        quantized_k_or_v_cache,
        k_or_v_cache,
        layer_idx: int,
        axis: int,
        nbits: int,
        group_size: int,
        residual_length: int,
    ) -> torch.Tensor:

        if len(k_or_v_cache) == layer_idx:
            quantized_k_or_v_cache.append(
                self._quantize(
                    k_or_v_states.contiguous(),
                    axis=axis,
                    nbits=nbits,
                    group_size=group_size,
                )
            )
            k_or_v_cache.append(
                torch.zeros(0, dtype=k_or_v_states.dtype, device=k_or_v_states.device)
            )
            k_or_v_to_return = k_or_v_states
        else:
            dequant_key = self._dequantize(quantized_k_or_v_cache[layer_idx])
            k_or_v_to_return = torch.cat(
                [dequant_key, k_or_v_cache[layer_idx], k_or_v_states], dim=-2
            )

            if (
                k_or_v_cache[layer_idx].dim() == 4
                and k_or_v_cache[layer_idx].shape[-2] + 1 >= residual_length
            ):
                quantized_k_or_v_cache[layer_idx] = self._quantize(
                    k_or_v_to_return.contiguous(),
                    axis=axis,
                    nbits=nbits,
                    group_size=group_size,
                )
                k_or_v_cache[layer_idx] = torch.zeros(
                    0, dtype=k_or_v_states.dtype, device=k_or_v_states.device
                )
            else:
                k_or_v_cache[layer_idx] = torch.cat(
                    [k_or_v_cache[layer_idx], k_or_v_states], dim=-2
                )

        return k_or_v_to_return

    def _quantize(self, tensor, axis, nbits, group_size):
        qtensor, meta = self.quantizer.quantize(
            tensor,
            axis=axis,
            device=self.device,
            compute_dtype=self.compute_dtype,
            nbits=nbits,
            group_size=group_size,
        )
        meta["compute_dtype"] = self.compute_dtype
        self.quantizer.cuda(qtensor, meta=meta, device=self.device)
        meta["scale"] = meta["scale"].to(qtensor.device)
        meta["zero"] = meta["zero"].to(qtensor.device)
        return qtensor, meta

    def _dequantize(self, qtensor):
        quant_tensor, meta = qtensor
        tensor = self.quantizer.dequantize(quant_tensor, meta)
        return tensor

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

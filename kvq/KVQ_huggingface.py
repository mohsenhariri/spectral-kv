"""
KVQ implementation using the latest Hugging Face Transformers QuantizedCache API.

HuggingFace with HQQQuantizedCache and QuantoQuantizedCache backends.
Compatible with transformers >= 4.46.0
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import Cache
from transformers.cache_utils import QuantizedCacheConfig

from .KVQConfig import KVQConfig


class KVQHuggingFace(Cache):
    """
    KVQ Cache using HuggingFace's official QuantizedCache implementation.
    - Uses CacheProcessor for quantization operations
    - Supports both HQQ and Quanto backends via QuantizedCacheConfig
    - Compatible with torch.compile() when used with StaticCache
    """

    def __init__(self, config: KVQConfig) -> None:
        # Don't call super().__init__() as Cache API has changed in newer transformers
        # Instead, manually initialize the required attributes

        # Initialize seen tokens counter
        self._seen_tokens = 0

        self.config = config
        self.compute_dtype = config.compute_dtype
        self.device = config.device

        # Initialize layers attribute required by HQQQuantizedCacheProcessor
        # We'll populate this dynamically as layers are added
        from transformers.cache_utils import DynamicLayer
        self.layers: List[DynamicLayer] = []

        # Initialize cache storage (use different names to avoid property conflicts)
        self._quantized_key_cache: List[Any] = []
        self._quantized_value_cache: List[Any] = []
        # Use _residual prefix since key_cache/value_cache are properties from Cache
        self._residual_key_cache: List[torch.Tensor] = []
        self._residual_value_cache: List[torch.Tensor] = []

        # Configuration
        self.nbits_k = config.nbits_k
        self.nbits_v = config.nbits_v
        self.group_size_k = config.group_size_k
        self.group_size_v = config.group_size_v
        self.residual_length_k = config.residual_length_k
        self.residual_length_v = config.residual_length_v

        # Don't initialize the processor - we'll handle quantization manually using HQQ
        self.processor = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new key and value states.
        1. Updates the base cache (pre_update)
        2. Applies quantization via processor (post_update)
        3. Returns full key/value tensors for attention

        Args:
            key_states: New key states [batch_size, num_heads, seq_len, head_dim]
            value_states: New value states [batch_size, num_heads, seq_len, head_dim]
            layer_idx: Index of the current layer
            cache_kwargs: Additional cache configuration (optional)

        Returns:
            Tuple of (keys_to_return, values_to_return) for attention computation
        """

        # SAME AS OLD CODE
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Add a new DynamicLayer if this is a new layer (do this once per layer, not per key/value)
        # Note: layers may already be created by get_mask_sizes() during generation
        if len(self.layers) == layer_idx:
            from transformers.cache_utils import DynamicLayer
            self.layers.append(DynamicLayer())

        # Get layer-specific configs
        nbits_k = self._get_layer_config(self.nbits_k, layer_idx)
        nbits_v = self._get_layer_config(self.nbits_v, layer_idx)
        group_size_k = self._get_layer_config(self.group_size_k, layer_idx)
        group_size_v = self._get_layer_config(self.group_size_v, layer_idx)

        # Update keys
        keys_to_return = self._update_with_processor(
            key_states,
            self._quantized_key_cache,
            self._residual_key_cache,
            layer_idx,
            nbits=nbits_k,
            group_size=group_size_k,
            axis=self.config.axis_k,
            residual_length=self.residual_length_k,
        )

        # Update values
        values_to_return = self._update_with_processor(
            value_states,
            self._quantized_value_cache,
            self._residual_value_cache,
            layer_idx,
            nbits=nbits_v,
            group_size=group_size_v,
            axis=self.config.axis_v,
            residual_length=self.residual_length_v,
        )

        return keys_to_return, values_to_return

    def _update_with_processor(
        self,
        states: torch.Tensor,
        quantized_cache: List[Any],
        residual_cache: List[torch.Tensor],
        layer_idx: int,
        nbits: int,
        group_size: int,
        axis: int,
        residual_length: int,
    ) -> torch.Tensor:
        """
        Update cache using the processor pattern from HuggingFace.
        Args:
            states: New key or value states
            quantized_cache: Storage for quantized tensors
            residual_cache: Storage for recent unquantized tensors
            layer_idx: Current layer index
            nbits: Quantization bits for this layer
            group_size: Group size for this layer
            axis: Quantization axis
            residual_length: Maximum residual cache size

        Returns:
            Complete key or value tensor for attention
        """

        # Kinda same as old code
        if len(quantized_cache) == layer_idx:
            # Initialize with empty quantized cache and states in residual
            # Don't quantize yet - let the residual buffer fill up first
            quantized_cache.append(None)  # Placeholder for quantized data
            residual_cache.append(states)  # Store initial states in residual

            return states
        else:
            # Kinda same as old code as well
            # Dequantize if we have quantized data, otherwise empty tensor
            if quantized_cache[layer_idx] is not None:
                dequantized = self._dequantize_states(quantized_cache[layer_idx])
                full_cache = torch.cat(
                    [dequantized, residual_cache[layer_idx], states], dim=-2
                )
            else:
                # No quantized data yet, just residual + new states
                full_cache = torch.cat(
                    [residual_cache[layer_idx], states], dim=-2
                )
            if (
                residual_cache[layer_idx].dim() == 4
                and residual_cache[layer_idx].shape[-2] + states.shape[-2] >= residual_length
            ):
                quantized_cache[layer_idx] = self._quantize_states(
                    full_cache.contiguous(),
                    nbits=nbits,
                    group_size=group_size,
                    axis=axis,
                )
                # Reset residual to empty 4D tensor with correct shape
                batch, heads, _, dim = states.shape
                residual_cache[layer_idx] = torch.zeros(
                    batch, heads, 0, dim, dtype=states.dtype, device=states.device
                )
            else:
                residual_cache[layer_idx] = torch.cat(
                    [residual_cache[layer_idx], states], dim=-2
                )

            return full_cache

    def _quantize_states(
        self,
        tensor: torch.Tensor,
        nbits: int,
        group_size: int,
        axis: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        try:
            from hqq.core.quantize import Quantizer

            # SAME AS OLD CODE
            qtensor, meta = Quantizer.quantize(
                tensor,
                axis=axis,
                device=self.device,
                compute_dtype=self.compute_dtype,
                nbits=nbits,
                group_size=group_size,
            )
            meta["compute_dtype"] = self.compute_dtype
            Quantizer.cuda(qtensor, meta=meta, device=self.device)
            meta["scale"] = meta["scale"].to(qtensor.device)
            meta["zero"] = meta["zero"].to(qtensor.device)

            return qtensor, meta

        except ImportError:
            raise ImportError(
                "HQQ library not found. Please install: pip install hqq"
            )

    def _dequantize_states(
        self,
        quantized_data: Tuple[torch.Tensor, Dict[str, Any]],
    ) -> torch.Tensor:
        try:
            from hqq.core.quantize import Quantizer

            # Same as old code
            quant_tensor, meta = quantized_data
            return Quantizer.dequantize(quant_tensor, meta)

        except ImportError:
            raise ImportError(
                "HQQ library not found. Please install: pip install hqq"
            )

    def _get_layer_config(
        self,
        config_value,
        layer_idx: int,
    ) -> Any:
        """
        Get layer-specific configuration value.
        Supports both fixed (int) and dynamic (List[int]) configurations.
        """
        # QoL method to get per-layer config
        if isinstance(config_value, list):
            return (
                config_value[layer_idx]
                if layer_idx < len(config_value)
                else config_value[-1]
            )
        return config_value

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        # Return the sequence length for the specified layer
        if len(self._residual_key_cache) <= layer_idx:
            return 0
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

    def get_max_length(self) -> Optional[int]:
        # Placeholder for compatibility
        # NOT IMPLEMENTED
        return None

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> Tuple[int, int]:
        """
        Override to ensure layers are created dynamically before being accessed.
        This is called during generation before update() is called.
        """
        # Ensure we have enough layers
        from transformers.cache_utils import DynamicLayer
        while len(self.layers) <= layer_idx:
            self.layers.append(DynamicLayer())

        # Call parent implementation
        return super().get_mask_sizes(cache_position, layer_idx)

    def reset(self):
        # QoL method to reset the cache
        self._seen_tokens = 0
        self._quantized_key_cache = []
        self._quantized_value_cache = []
        self._residual_key_cache = []
        self._residual_value_cache = []
        self.layers = []

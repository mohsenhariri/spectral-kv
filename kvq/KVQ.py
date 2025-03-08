import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from optimum.quanto import MaxOptimizer, qint2, qint4, quantize_weight
from transformers import CacheConfig, DynamicCache


@dataclass
class KVQCacheConfig(CacheConfig):
    """
    Configuration class for KVQ cache quantization.

    Parameters:
        backend (str, optional): Quantization backend to use. Currently, only "quanto" is supported.
        nbits_k (int, optional): Bit-width for quantizing key states. Must be 2 or 4. Defaults to 4.
        nbits_v (int, optional): Bit-width for quantizing value states. Must be 2 or 4. Defaults to 2.
        axis_key (int, optional): Axis along which to quantize key states (0, 1, or -1). Defaults to 0.
        axis_value (int, optional): Axis along which to quantize value states (0, 1, or -1). Defaults to 0.
        q_group_size (int, optional): Number of elements in each quantization group. Defaults to 64.
        residual_length (int, optional): Number of tokens to accumulate before re-quantizing. Defaults to 128.
        compute_dtype (torch.dtype, optional): Data type for intermediate computations
            (e.g., torch.bfloat16). Defaults to torch.bfloat16.
        device (str, optional): Device for storing and performing quantization (e.g., "cpu" or "cuda").
            Defaults to "cpu".

    Raises:
        ValueError: If any of the provided parameters have invalid values (e.g., unsupported bit-widths).
        Warning: If `nbits_k < nbits_v`, indicating potential suboptimal performance.

    Example:
        >>> config = KVQCacheConfig(
        ...     backend="quanto",
        ...     nbits_k=4,
        ...     nbits_v=2,
        ...     axis_key=0,
        ...     axis_value=0,
        ...     q_group_size=64,
        ...     residual_length=128,
        ...     compute_dtype=torch.bfloat16,
        ...     device="cuda",
        ... )
        >>> # Pass 'config' to KVQ or directly instantiate KVQ with a dict.
    """

    def __init__(
        self,
        backend: str = "quanto",
        nbits_k: Optional[int] = 4,
        nbits_v: Optional[int] = 2,
        axis_key: Optional[int] = 0,
        axis_value: Optional[int] = 0,
        q_group_size: Optional[int] = 64,
        residual_length: Optional[int] = 128,
        compute_dtype: Optional[torch.dtype] = torch.bfloat16,
        device: Optional[str] = "cpu",
    ):
        """
        Set the

        """
        self.backend = backend
        self.nbits_k = nbits_k
        self.nbits_v = nbits_v
        self.axis_key = axis_key
        self.axis_value = axis_value
        self.q_group_size = q_group_size
        self.residual_length = residual_length
        self.compute_dtype = compute_dtype
        self.device = device

        self.validate()

    def validate(self):

        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )

        if self.backend not in ["quanto"]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="backend",
                    correct_value="`quanto`",
                    found_value=self.backend,
                ),
            )

        if self.nbits_k not in [2, 4]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="nbits_k",
                    correct_value="2 or 4",
                    found_value=self.nbits_k,
                ),
            )

        if self.nbits_v not in [2, 4]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="nbits_v",
                    correct_value="2 or 4",
                    found_value=self.nbits_v,
                ),
            )

        if self.q_group_size <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="q_group_size",
                    correct_value="a positive integer",
                    found_value=self.q_group_size,
                ),
            )

        if self.residual_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="residual_length",
                    correct_value="a positive integer",
                    found_value=self.residual_length,
                ),
            )

        if self.axis_key not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_key",
                    correct_value="`1` or `0`, `-1`",
                    found_value=self.axis_key,
                ),
            )

        if self.axis_value not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_value",
                    correct_value="`1` or `0` or `-1`",
                    found_value=self.axis_value,
                ),
            )

        if self.nbits_k < self.nbits_v:
            warnings.warn(
                f"You've set `nbits_k`={self.nbits_k} and `nbits_v`={self.nbits_v}. "
                "For optimal performance, consider using more bits for keys and fewer bits for values."
            )


class KVQ(DynamicCache):
    """
    Attributes:
        nbits_k (int): Bit-width for quantizing key states, must be 2 or 4.
            Recommended to be 4.
        nbits_v (int): Bit-width for quantizing value states, must be 2 or 4.
            Recommended to be 2.
        axis_key (int): Axis along which key states are quantized (0, 1, or -1).
            Derived from the configuration.
        axis_value (int): Axis along which value states are quantized (0, 1, or -1).
            Derived from the configuration.
        q_group_size (int): Number of elements in each quantization group.
        residual_length (int): Number of new tokens to collect before re-quantizing.
            Derived from the configuration.
        compute_dtype (torch.dtype): Data type used for intermediate computations
            (e.g., ``torch.bfloat16``).
        backend (str): Quantization backend. Currently only ``"quanto"`` is supported.
        device (str): Device on which the cache is stored and quantization is performed
            (e.g., ``"cpu"`` or ``"cuda"``).

    Example:
        >>> import torch
        >>> from kvq import KVQ, KVQCacheConfig
        >>>
        >>> # Option 1: Create a KVQ object using a configuration object
        >>> config = KVQCacheConfig(
        ...     nbits_k=4,
        ...     nbits_v=2,
        ...     axis_key=0,
        ...     axis_value=0,
        ...     q_group_size=64,
        ...     residual_length=128,
        ...     compute_dtype=torch.bfloat16,
        ...     backend="quanto",
        ...     device="cuda",
        ... )
        >>> kvq = KVQ(config)
        >>>
        >>> # Option 2: Create a KVQ object directly from a dictionary
        >>> kvq_dict = {
        ...     "nbits_k": 2,
        ...     "nbits_v": 2,
        ...     "axis_key": 0,
        ...     "axis_value": 0,
        ...     "q_group_size": 64,
        ...     "residual_length": 128,
        ...     "compute_dtype": torch.float16,
        ...     "backend": "quanto",
        ...     "device": "cuda",
        ... }
        >>> kvq = KVQ(kvq_dict)
        >>>
        >>> # Example usage in text generation with a transformer model
        >>> # Assume 'model' is a transformer-like model supporting cache usage.
        >>> outputs = model.generate(
        ...     **inputs,
        ...     max_new_tokens=1024,
        ...     use_cache=True,
        ...     past_key_values=kvq,
        ... )
        >>> print(outputs)
    """

    def __init__(self, kvq_config: Union[KVQCacheConfig, Dict[str, Any]]) -> None:

        super().__init__()

        if isinstance(kvq_config, dict):
            cache_config = KVQCacheConfig(**kvq_config)
        elif isinstance(kvq_config, KVQCacheConfig):
            cache_config = kvq_config
        else:
            raise ValueError(
                "`kvq_config` must be a dictionary or a KVQCacheConfig instance."
            )

        self._quantized_key_cache: List[torch.Tensor] = []
        self._quantized_value_cache: List[torch.Tensor] = []

        self.nbits_k = cache_config.nbits_k
        self.nbits_v = cache_config.nbits_v
        self.axis_key = cache_config.axis_key
        self.axis_value = cache_config.axis_value
        self.q_group_size = cache_config.q_group_size
        self.residual_length = cache_config.residual_length
        self.compute_dtype = cache_config.compute_dtype
        self.backend = cache_config.backend
        self.device = cache_config.device

        self.qtype_k = qint4 if self.nbits_k == 4 else qint2
        self.qtype_v = qint4 if self.nbits_v == 4 else qint2

        self.optimizer = MaxOptimizer()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens (for layer 0)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) < layer_idx:
            raise ValueError(
                "KVQ does not support model usage where layers are skipped. Use DynamicCache."
            )
        elif len(self.key_cache) == layer_idx:
            # First time: quantize both key and value with their respective bit-widths
            self._quantized_key_cache.append(
                self._quantize(
                    key_states.contiguous(),
                    axis=self.axis_key,
                    qtype=self.qtype_k,
                )
            )
            self._quantized_value_cache.append(
                self._quantize(
                    value_states.contiguous(),
                    axis=self.axis_value,
                    qtype=self.qtype_v,
                )
            )
            self.key_cache.append(
                torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
            )
            self.value_cache.append(
                torch.zeros(0, dtype=value_states.dtype, device=key_states.device)
            )
            keys_to_return, values_to_return = key_states, value_states
        else:
            # Retrieve dequantized keys/values for previous cache state
            dequant_key = self._dequantize(self._quantized_key_cache[layer_idx])
            dequant_value = self._dequantize(self._quantized_value_cache[layer_idx])
            keys_to_return = torch.cat(
                [dequant_key, self.key_cache[layer_idx], key_states], dim=-2
            )
            values_to_return = torch.cat(
                [dequant_value, self.value_cache[layer_idx], value_states], dim=-2
            )
            if (
                self.key_cache[layer_idx].dim() == 4
                and self.key_cache[layer_idx].shape[-2] + 1 >= self.residual_length
            ):
                self._quantized_key_cache[layer_idx] = self._quantize(
                    keys_to_return.contiguous(),
                    axis=self.axis_key,
                    qtype=self.qtype_k,
                )
                self._quantized_value_cache[layer_idx] = self._quantize(
                    values_to_return.contiguous(),
                    axis=self.axis_value,
                    qtype=self.qtype_v,
                )
                self.key_cache[layer_idx] = torch.zeros(
                    0, dtype=key_states.dtype, device=key_states.device
                )
                self.value_cache[layer_idx] = torch.zeros(
                    0, dtype=key_states.dtype, device=key_states.device
                )
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )

        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

    def _quantize(self, tensor: torch.Tensor, axis: int, qtype) -> torch.Tensor:
        scale, zeropoint = self.optimizer(tensor, qtype, axis, self.q_group_size)

        qtensor = quantize_weight(
            tensor,
            qtype,
            axis,
            scale,
            shift=zeropoint,
            group_size=self.q_group_size,
            optimized=True,
        )

        return qtensor

    def _dequantize(self, qtensor: torch.Tensor) -> torch.Tensor:
        return qtensor.dequantize()

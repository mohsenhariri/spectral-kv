from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch

from kvq.bit_pattern import bit_pattern
from kvq.const import _DEFAULT_GROUP_SIZES, _SUPPORTED_BITS
from kvq.group_pattern import group_pattern

NbVal = Union[int, List[int]]
GsVal = Union[int, List[int]]


@dataclass(slots=True, frozen=True)
class KVQConfig:

    # Score metric for quantization optimization
    score: int = 0  # 0 for frobenius_norm, 1 for spectral_norm

    # Quantization bit allocation
    # Option 1:
    # nbits = n, n for both keys and values
    # nbits = {"k": n_k, "v": n_v} for different quantization bits for keys and values
    nbits: Optional[Union[int, Dict[str, int]]] = None

    # Option 2:
    # budget = n, total budget for kv quantization = n * 2 * num_layers
    # Dynamically allocate bits for keys and values based on KV norms
    # bit_range: range of bits to consider for kv quantization
    budget: Optional[int] = None
    bit_range: Optional[List[int]] = None

    # Group size for quantization
    # Option 1:
    # group_size = n, n for both keys and values or
    # group_size = {"k": n_k, "v": n_v} for different group sizes for keys and values
    group_size: Optional[Union[int, Dict[str, int]]] = None

    # Option 2:
    # group_budget = n, total budget for group size quantization = n * 2 * num_layers
    # Dynamically allocate group sizes for keys and values based on KV norms
    # group_range: range of group sizes to consider for kv quantization
    group_budget: Optional[int] = None
    group_range: Optional[List[int]] = None

    model: str = None
    # Residual length for keys and values
    # residual_length = n, n for both keys and values
    # residual_length = {"k": n_k, "v": n_v} -> this is not recommended
    residual_length: Union[int, Dict[str, int]] = 64

    axis: Union[int, Dict[str, int]] = 1
    compute_dtype: torch.dtype = torch.bfloat16
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    def __post_init__(self) -> None:
        set_attr = object.__setattr__

        # nbits and budget are mutually exclusive.
        if (self.nbits is None) == (self.budget is None):

            raise ValueError("Exactly one of `nbits` or `budget` must be provided ")

        # group_size and group_budget are mutually exclusive.
        if (self.group_size is None) == (self.group_budget is None):
            # If both are None, default to group_size = 64
            if self.group_size is None and self.group_budget is None:
                set_attr(self, "group_size", 64)
            else:
                raise ValueError(
                    "Exactly one of `group_size` or `group_budget` must be provided "
                )

        if self.nbits is not None:
            nbits_dict = (
                {"k": self.nbits, "v": self.nbits}
                if isinstance(self.nbits, int)
                else dict(self.nbits)  # defensive copy
            )
            if not {"k", "v"}.issubset(nbits_dict):
                raise ValueError("`nbits` dict must contain both 'k' and 'v'.")
            for which, bits in nbits_dict.items():
                if bits not in _SUPPORTED_BITS:
                    raise ValueError(
                        f"`nbits['{which}']` must be in {_SUPPORTED_BITS}, got {bits}."
                    )

            set_attr(self, "nbits", nbits_dict)

        if self.budget is not None:
            if self.model is None:
                raise ValueError("`model` must be provided when `budget` is set.")

            if self.budget not in _SUPPORTED_BITS:
                raise ValueError(
                    f"`budget` must be in {_SUPPORTED_BITS}, got {self.budget}."
                )
            if self.bit_range is not None:
                if not isinstance(self.bit_range, list):
                    raise ValueError("`bit_range` must be a list of integers.")
                for bits in self.bit_range:
                    if bits not in _SUPPORTED_BITS:
                        raise ValueError(
                            f"All elements of `bit_range` must be in {_SUPPORTED_BITS}, got {bits}."
                        )
                bit_range = self.bit_range
            else:
                bit_range = _SUPPORTED_BITS

            kv_bits = bit_pattern(
                budget=self.budget,
                bit_range=bit_range,
                model_name_or_path=self.model,
                score=self.score,
            )

            set_attr(
                self,
                "nbits",
                {
                    "k": kv_bits["nbits_k"],
                    "v": kv_bits["nbits_v"],
                },
            )

        # Handle group_size vs group_budget
        if self.group_size is not None:

            def _canon(x: Union[int, Dict[str, int]], name: str) -> Dict[str, int]:
                if isinstance(x, int):
                    return {"k": x, "v": x}
                if not {"k", "v"}.issubset(x):
                    raise ValueError(f"`{name}` dict must contain 'k' and 'v'.")
                return dict(x)

            set_attr(self, "group_size", _canon(self.group_size, "group_size"))

        if self.group_budget is not None:
            if self.model is None:
                raise ValueError("`model` must be provided when `group_budget` is set.")

            if self.group_range is not None:
                if not isinstance(self.group_range, list):
                    raise ValueError("`group_range` must be a list of integers.")
                for gsize in self.group_range:
                    if not isinstance(gsize, int) or gsize <= 0:
                        raise ValueError(
                            f"All elements of `group_range` must be positive integers, got {gsize}."
                        )
                group_range = self.group_range
            else:
                group_range = _DEFAULT_GROUP_SIZES

            kv_groups = group_pattern(
                model_name_or_path=self.model,
                group_size_budget=self.group_budget,
                group_sizes=group_range,
                score=self.score,
            )

            set_attr(
                self,
                "group_size",
                {
                    "k": kv_groups["g_k"],
                    "v": kv_groups["g_v"],
                },
            )

        def _canon(x: Union[int, Dict[str, int]], name: str) -> Dict[str, int]:
            if isinstance(x, int):
                return {"k": x, "v": x}
            if not {"k", "v"}.issubset(x):
                raise ValueError(f"`{name}` dict must contain 'k' and 'v'.")
            return dict(x)

        set_attr(
            self, "residual_length", _canon(self.residual_length, "residual_length")
        )
        axis = _canon(self.axis, "axis")
        for ax in axis.values():
            if ax not in (0, 1):
                raise ValueError("`axis` values must be 0 or 1.")
        set_attr(self, "axis", axis)

    @property
    def nbits_k(self) -> NbVal:  # int  OR  List[int]
        return None if self.nbits is None else self.nbits["k"]

    @property
    def nbits_v(self) -> NbVal:
        return None if self.nbits is None else self.nbits["v"]

    @property
    def residual_length_k(self) -> int:
        return self.residual_length["k"]

    @property
    def residual_length_v(self) -> int:
        return self.residual_length["v"]

    @property
    def group_size_k(self) -> GsVal:
        return self.group_size["k"]

    @property
    def group_size_v(self) -> GsVal:
        return self.group_size["v"]

    @property
    def axis_k(self) -> int:
        return self.axis["k"]

    @property
    def axis_v(self) -> int:
        return self.axis["v"]

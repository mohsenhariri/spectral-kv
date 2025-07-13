from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from kvq.const import _SUPPORTED_BITS
from kvq.bit_pattern import bit_pattern

NbVal = Union[int, List[int]]


@dataclass(slots=True, frozen=True)
class KVQConfig:

    # will be removed later
    score: int = 0  # 0 for frobenius_norm, 1 for spectral_norm

    nbits: Optional[Union[int, Dict[str, int]]] = None
    budget: Optional[int] = None
    model: str = None
    bit_range: Optional[List[int]] = None
    residual_length: Union[int, Dict[str, int]] = 64
    group_size: Union[int, Dict[str, int]] = 64
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
                    model=self.model,
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

        def _canon(x: Union[int, Dict[str, int]], name: str) -> Dict[str, int]:
            if isinstance(x, int):
                return {"k": x, "v": x}
            if not {"k", "v"}.issubset(x):
                raise ValueError(f"`{name}` dict must contain 'k' and 'v'.")
            return dict(x)

        set_attr(
            self, "residual_length", _canon(self.residual_length, "residual_length")
        )
        set_attr(self, "group_size", _canon(self.group_size, "group_size"))
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
    def group_size_k(self) -> int:
        return self.group_size["k"]

    @property
    def group_size_v(self) -> int:
        return self.group_size["v"]

    @property
    def axis_k(self) -> int:
        return self.axis["k"]

    @property
    def axis_v(self) -> int:
        return self.axis["v"]

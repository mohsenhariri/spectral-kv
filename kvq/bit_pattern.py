import json
import warnings
from typing import Dict, List, Sequence

from kvq.const import _SUPPORTED_BITS
from kvq.helpers import load_kv_norms

RD_EXP: int = 2  # Rate-Distortion exponent: error is proportional to 2^(−RD_EXP·b)
TOL: float = 1e-6  # Tolerance for floating point comparisons


def _build_next_bit_dict(bits: Sequence[float]) -> Dict[float, float]:
    """Build a lookup dict mapping each bit-width to the next higher one."""
    bits = sorted(bits)  # sorted if user passes unsorted bits
    return {bits[i]: bits[i + 1] for i in range(len(bits) - 1)}


def allocate_bits_greedy(
    sensitivities: Sequence[float],
    total_budget: float,
    bit_range: Sequence[float] = _SUPPORTED_BITS,
    rd_exp: int = RD_EXP,
) -> List[float]:
    """
    Greedy bit allocation algorithm (similar to water-filling).

    Allocates bit-widths to tensors based on their sensitivity scores to minimize
    total quantization error under a fixed bit budget constraint.

    This is a discrete approximation to the water-filling algorithm:
    - Water-filling: continuous bit allocation based on channel capacity
    - This algorithm: discrete bit allocation based on rate-distortion gain

    Key difference from classic water-filling:
    - Water-filling allocates more bits to channels with higher SNR/capacity
    - This allocates more bits to tensors with higher sensitivity (error coefficient)

    The algorithm is **symmetric**: it treats all tensors identically based solely
    on their sensitivity scores, regardless of whether they are keys, values, or
    any other tensor type.

    Args:
        sensitivities: Sensitivity coefficient for each tensor (e.g., norm values).
        total_budget: Total number of bits to allocate across all tensors.
        bit_range: Available quantization bit-widths (default: _SUPPORTED_BITS).
        rd_exp: Rate-distortion exponent (default: RD_EXP=2).
                Error is modeled as: sensitivity^rd_exp * 2^(-rd_exp * bits)

    Returns:
        List of allocated bit-widths for each tensor.

    Example:
        >>> sensitivities = [1.5, 2.0, 1.0]  # 3 tensors with different sensitivities
        >>> bits = allocate_bits_greedy(sensitivities, total_budget=12)
        >>> # Result might be [4, 6, 2] - more bits to higher sensitivity tensors
    """
    n_tensors = len(sensitivities)
    
    if n_tensors == 0:
        return []
    
    if not bit_range:
        raise ValueError("bit_range cannot be empty.")
    
    # Precompute error coefficients: c_i = sensitivity_i^rd_exp
    c = [s**rd_exp for s in sensitivities]

    supported_bits = sorted(bit_range)
    next_bit_dict = _build_next_bit_dict(supported_bits)

    # Initialize all tensors at minimum precision
    min_bits = supported_bits[0]
    current_bits = [min_bits] * n_tensors
    bits_used = n_tensors * min_bits

    # Greedy allocation: iteratively add bits to the tensor with best gain
    while bits_used + TOL < total_budget:
        best_gain = -1.0
        cand_idx = None
        cand_next = None
        cand_delta = None

        for idx, (c_i, b_i) in enumerate(zip(c, current_bits)):
            # Check if a higher precision is available
            next_b = next_bit_dict.get(b_i)
            if next_b is None:
                continue

            delta_b = next_b - b_i
            # Do not exceed the budget
            if bits_used + delta_b - total_budget > TOL:
                continue

            # Calculate the reduction in error per bit added (marginal gain)
            cur_err = c_i * 2 ** (-rd_exp * b_i)
            next_err = c_i * 2 ** (-rd_exp * next_b)
            gain = (cur_err - next_err) / delta_b

            # Select the tensor with the best gain
            if gain > best_gain + TOL:
                best_gain, cand_idx = gain, idx
                cand_next, cand_delta = next_b, delta_b

        if cand_idx is None:
            # No more bits can be added without exceeding the budget
            break

        # Allocate bits to the best candidate
        current_bits[cand_idx] = cand_next
        bits_used += cand_delta

    return current_bits


def bit_pattern(
    model_name_or_path: str,
    budget=4,
    layers="all",
    bit_range=_SUPPORTED_BITS,
    score: int = 0,
):
    """
    Allocate bit-widths for KV caches per layer based on sensitivity scores.

    This function uses a greedy algorithm to distribute a total bit budget across
    the key (K) and value (V) caches of a model. The goal is to minimize the
    overall quantization error, which is estimated using pre-computed norm scores
    (Frobenius or spectral) as a measure of sensitivity.

    The algorithm iteratively assigns bits to the matrix (K or V cache matrix
    for a given layer) that will yield the largest reduction in error for the
    number of bits added.

    Args:
    model_name_or_path : str
        HuggingFace repo name (must exist in kvq.const.model_dict).
    budget : int | float, default 4
        Average bits per matrix (total budget = 2 * num_layers * budget).
    layers : "all" | int, default "all"
        Currently only "all" layers are supported.
    bit_range : sequence of float, default _SUPPORTED_BITS
        Allowed quantization bit-widths.
    score : {0, 1}, default 0.
        0: Use Frobenius norm scores.
        1: Use spectral norm scores.

    Returns
    dict with keys
        "nbits_k": list[float]: per-layer bits for W_k
        "nbits_v": list[float]: per-layer bits for W_v
    """

    if budget > 8:
        raise ValueError("Budget should be less than or equal to 8 bits.")

    if layers != "all":
        raise NotImplementedError("Only layers='all' is currently supported.")

    kv_norms = load_kv_norms(model_name_or_path, score)

    num_layers = len(kv_norms["w_k"])
    total_budget = 2 * budget * num_layers

    # Interleave K and V sensitivities: [K0, V0, K1, V1, ..., Kn, Vn]
    sens = []
    for k, v in zip(kv_norms["w_k"], kv_norms["w_v"]):
        sens.extend([k, v])

    # Use the general allocation function
    current_bits = allocate_bits_greedy(
        sensitivities=sens,
        total_budget=total_budget,
        bit_range=bit_range,
        rd_exp=RD_EXP,
    )

    # De-interleave results back into K and V
    w_k_bits = current_bits[0::2]
    w_v_bits = current_bits[1::2]

    # Verify budget usage
    used = sum(w_k_bits) + sum(w_v_bits)
    if abs(used - total_budget) > TOL:
        warnings.warn(
            f"Total bits used = {used:.6f} differs from budget "
            f"{total_budget} by > {TOL}."
        )

    return {"nbits_k": w_k_bits, "nbits_v": w_v_bits}


if __name__ == "__main__":

    models = [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-32B",
    ]

    bit_range = [8, 6, 4, 2, 1.58, 1]  # used in evaluation experiments

    budgets = [2, 4]

    scores = [0, 1]  # 0 for frobenius_norm, 1 for spectral_norm

    for model in models:
        for budget in budgets:
            for score in scores:
                kv_bits = bit_pattern(
                    model_name_or_path=model,
                    bit_range=bit_range,
                    budget=budget,
                    score=score,
                )
                print(f"Model: {model}, Budget: {budget}, Score: {score}")
                print(kv_bits)
                print("\n")

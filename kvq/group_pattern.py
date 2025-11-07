import json
import math
import warnings
from typing import Dict, List, Sequence, Tuple

from kvq.const import _DEFAULT_GROUP_SIZES
from kvq.helpers import load_kv_norms

TOL: float = 1e-6  # Tolerance for floating point comparisons


def _adjacent_maps(sizes: Sequence[int]):
    """Return dicts that map size: next_smaller / next_larger size."""
    sizes = sorted(set(sizes))  # sort if user passes unsorted sizes
    next_small = {sizes[i]: sizes[i - 1] for i in range(1, len(sizes))}
    next_large = {sizes[i]: sizes[i + 1] for i in range(len(sizes) - 1)}
    return next_small, next_large


def _continuous_optimum(c: List[float], target_sum: float) -> List[float]:
    """g* ∝ 1/√c  under Σ 1/g = target_sum."""
    kappa = sum(math.sqrt(x) for x in c) / target_sum
    return [kappa / math.sqrt(x) for x in c]


def _nearest(supported: Sequence[int], g: float) -> int:
    """Supported value nearest to g, when ties: smaller."""
    return min(supported, key=lambda x: (abs(x - g), x))


def allocate_groups_greedy(
    sensitivities: Sequence[float],
    target_sum: float,
    group_sizes: Sequence[int] = _DEFAULT_GROUP_SIZES,
) -> List[int]:
    """
    Greedy group size allocation algorithm.

    Allocates group sizes to tensors based on their sensitivity scores to minimize
    total quantization error under a fixed group budget constraint.

    The algorithm is based on the rate-distortion principle where quantization
    error is proportional to group_size × sensitivity. Smaller group sizes mean
    more frequent recalibration (scale/zero-point computation), leading to lower
    quantization error but higher metadata overhead.

    The algorithm is **symmetric**: it treats all tensors identically based solely
    on their sensitivity scores, regardless of whether they are keys, values, or
    any other tensor type.

    Key relationship:
    - Higher sensitivity → Smaller group size (more recalibration)
    - Lower sensitivity → Larger group size (less overhead)

    This is opposite to bit allocation where higher sensitivity → more bits.

    Args:
        sensitivities: Sensitivity coefficient for each tensor (e.g., norm values).
        target_sum: Target sum of 1/g across all tensors (controls metadata budget).
        group_sizes: Available group sizes (default: _DEFAULT_GROUP_SIZES).

    Returns:
        List of allocated group sizes for each tensor.

    Example:
        >>> sensitivities = [1.5, 2.0, 1.0]  # 3 tensors with different sensitivities
        >>> groups = allocate_groups_greedy(sensitivities, target_sum=0.1)
        >>> # Result might be [64, 32, 128] - smaller groups for higher sensitivity
    """
    n_tensors = len(sensitivities)
    
    if n_tensors == 0:
        return []
    
    if not group_sizes:
        raise ValueError("`group_sizes` cannot be empty.")
    
    if any(g <= 0 for g in group_sizes):
        raise ValueError("`group_sizes` must all be positive.")

    # Compute continuous optimal solution: g* ∝ 1/√c
    c = sensitivities
    g_star = _continuous_optimum(c, target_sum)
    
    # Round to nearest supported sizes
    sizes = [_nearest(group_sizes, g) for g in g_star]

    # Build adjacency maps for greedy refinement
    nxt_small, nxt_large = _adjacent_maps(group_sizes)
    cur_sum = sum(1.0 / g for g in sizes)

    # Greedily adjust the group sizes to meet the target sum
    while abs(cur_sum - target_sum) > TOL:
        if cur_sum > target_sum:  # Need to decrease sum of 1/g, so grow some g
            best_i = best_ratio = None
            for i, (g_cur, c_i) in enumerate(zip(sizes, c)):
                g_next = nxt_large.get(g_cur)
                if g_next is None:
                    continue
                delta_sum = (1 / g_next) - (1 / g_cur)  # This will be negative
                new_sum = cur_sum + delta_sum
                if new_sum + TOL < target_sum:  # Would overshoot
                    continue
                # Penalize changes that cause a large increase in error
                delta_err = c_i * (g_next - g_cur)
                ratio = delta_err / (-delta_sum)  # penalty per unit of sum reduction
                if best_ratio is None or ratio < best_ratio:
                    best_ratio, best_i = ratio, i
            if best_i is None:
                break  # No legal move found
            cur = sizes[best_i]
            nxt = nxt_large[cur]
            cur_sum += (1 / nxt) - (1 / cur)
            sizes[best_i] = nxt

        else:  # cur_sum < target -> shrink some g
            best_i = best_gain = None
            for i, (g_cur, c_i) in enumerate(zip(sizes, c)):
                g_next = nxt_small.get(g_cur)
                if g_next is None:
                    continue
                delta_sum = (1 / g_next) - (1 / g_cur)  # This will be positive
                new_sum = cur_sum + delta_sum
                if new_sum - TOL > target_sum:  # Would overshoot
                    continue
                # Reward changes that cause a large decrease in error
                delta_err = c_i * (g_cur - g_next)
                gain = delta_err / delta_sum  # benefit per unit of sum increase
                if best_gain is None or gain > best_gain:
                    best_gain, best_i = gain, i
            if best_i is None:
                break  # No legal move found
            cur = sizes[best_i]
            nxt = nxt_small[cur]
            cur_sum += (1 / nxt) - (1 / cur)
            sizes[best_i] = nxt

    return sizes


def group_pattern(
    model_name_or_path: str,
    group_size_budget: int = 64,
    layers: str | int = "all",
    group_sizes: Sequence[int] = _DEFAULT_GROUP_SIZES,
    score: int = 0,  # 0 = Frobenius, 1 = spectral
) -> Dict[str, List[int]]:
    """
    Allocate group sizes for for KV caches per layer.

    This function determines the optimal group size for key (K) and value (V)
    cache quantization on a per-layer basis. The allocation is performed under
    a fixed budget for the total number of groups, which is equivalent to
    maintaining a target average group size across all layers.

    The method is based on the idea that the quantization error is proportional
    to the group size, weighted by a sensitivity score (Frobenius or spectral
    norm). The function first computes a continuous optimal solution and then
    adjusts it to the discrete set of available group sizes using a greedy
    refinement algorithm.

    Args:
    model_name_or_path : str
        HuggingFace repo name (must exist in kvq.const.model_dict).
    group_size_budget : int, default 64
        Target *average* group size. This preserves the total metadata size
        that would be used with a fixed group size of this value.
    layers : "all" | int, default "all"
        Only "all" is supported for now.
    group_sizes : sequence[int], default (32, 64, 128)
        Allowed group sizes.
    score : {0, 1}, default 0
            0: Use Frobenius norm scores.
            1: Use spectral norm scores.

    Returns
    dict
        {"g_k": [int]*n , "g_v": [int]*n}
    """
    if group_size_budget <= 0:
        raise ValueError("`group_size_budget` must be positive.")

    if layers != "all":
        raise NotImplementedError("Only layers='all' is currently supported.")

    norms = load_kv_norms(model_name_or_path, score)

    num_layers = len(norms["w_k"])
    target_sum = 2.0 * num_layers / group_size_budget

    # Interleave K and V sensitivities: [K0, V0, K1, V1, ..., Kn, Vn]
    sens = []
    for k, v in zip(norms["w_k"], norms["w_v"]):
        sens.extend([k, v])

    # Use the general allocation function
    sizes = allocate_groups_greedy(
        sensitivities=sens,
        target_sum=target_sum,
        group_sizes=group_sizes,
    )

    # De-interleave results back into K and V
    g_k = sizes[0::2]
    g_v = sizes[1::2]

    # Verify budget usage
    used_sum = sum(1.0 / g for g in sizes)
    if abs(used_sum - target_sum) > TOL:
        warnings.warn(
            f"Final sum 1/g = {used_sum:.6f} vs target {target_sum:.6f} (>|{TOL}|)."
        )

    return {"g_k": g_k, "g_v": g_v}


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

    group_fixed_budget = 64

    for model in models:

        kv_groups = group_pattern(
            model_name_or_path=model,
            group_size_budget=group_fixed_budget,
            group_sizes=[16, 32, 64, 128],
            score=1,
        )

        print("w_k group sizes:", kv_groups["g_k"])
        print("w_v group sizes:", kv_groups["g_v"])

        n_layers = len(kv_groups["g_k"])

        budget_using_fixed_group_size = 2 * n_layers / group_fixed_budget

        budget_using_dynamic_group_size = sum(1 / g for g in kv_groups["g_k"]) + sum(
            1 / g for g in kv_groups["g_v"]
        )

        print(
            f"Model: {model} | "
            f"Budget using fixed group size: {budget_using_fixed_group_size:.6f} | "
            f"Budget using dynamic group size: {budget_using_dynamic_group_size:.6f}"
        )

        assert (
            abs(budget_using_fixed_group_size - budget_using_dynamic_group_size) < TOL
        ), f"Budget mismatch for model {model!r}."
        print("\n")

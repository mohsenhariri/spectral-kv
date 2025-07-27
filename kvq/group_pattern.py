import json
import math
import warnings
from typing import Dict, List, Sequence, Tuple

import importlib.resources as pkg_resources

from kvq.const import model_dict, supported_models, _DEFAULT_GROUP_SIZES
from kvq.helpers import extract_model_name


TOL: float = 1e-6
ASSETS_PATH = "kvq.assets"


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


def group_pattern(
    model_name_or_path: str,
    group_size_budget: int = 64,
    layers: str | int = "all",
    group_sizes: Sequence[int] = _DEFAULT_GROUP_SIZES,
    score: int = 0,  # 0 = Frobenius, 1 = spectral
) -> Dict[str, List[int]]:
    """
    Allocate group sizes for for KV caches per layer.

    Args:
    model_name_or_path : str
        HuggingFace repo name (must exist in kvq.const.model_dict).
    group_size_budget : int, default 64
        Target *average* group size g in Eq. (1). (preserve total metadata size)
    layers : "all" | int, default "all"
        Only "all" is supported for now.
    group_sizes : sequence[int], default (32, 64, 128)
        Allowed group sizes.
    score : {0, 1}, default 0
            0: Frobenius norm file, 1: spectral norm file.

    Returns
    dict
        {"g_k": [int]*n , "g_v": [int]*n}
    """

    model = extract_model_name(model_name_or_path)

    if model not in supported_models:
        raise ValueError(
            f"Model {model!r} is not supported. "
            f"Supported models: {', '.join(supported_models)}"
        )

    if layers != "all":
        raise NotImplementedError("Only layers='all' is currently supported.")

    if not group_sizes:
        raise ValueError("`group_sizes` cannot be empty.")

    if any(g <= 0 for g in group_sizes):
        raise ValueError("`group_sizes` must all be positive.")

    model_name = model_dict.get(model)
    if model_name is None:
        raise ValueError(f"No entry for {model!r} in kvq.const.model_dict.")

    norm_type = "frobenius_norm" if score == 0 else "spectral_norm"

    score_file = f"{norm_type}/{model_name}.json"

    with pkg_resources.files(ASSETS_PATH).joinpath(score_file).open() as f:
        norms = json.load(f)

    n_layers = len(norms["w_k"])
    c: List[float] = [val for pair in zip(norms["w_k"], norms["w_v"]) for val in pair]

    target_sum = 2.0 * n_layers / group_size_budget
    g_star = _continuous_optimum(c, target_sum)
    sizes = [_nearest(group_sizes, g) for g in g_star]

    nxt_small, nxt_large = _adjacent_maps(group_sizes)
    cur_sum = sum(1.0 / g for g in sizes)

    while abs(cur_sum - target_sum) > TOL:
        if cur_sum > target_sum:  # need to decrease sum of 1/g, so grow some g
            best_i = best_ratio = None
            for i, (g_cur, c_i) in enumerate(zip(sizes, c)):
                g_next = nxt_large.get(g_cur)
                if g_next is None:
                    continue
                delta_sum = (1 / g_next) - (1 / g_cur)  # negative
                new_sum = cur_sum + delta_sum
                if new_sum + TOL < target_sum:  # would overshoot low
                    continue
                delta_err = c_i * (g_next - g_cur)
                ratio = delta_err / (-delta_sum)  # penalty per sum reduction
                if best_ratio is None or ratio < best_ratio:
                    best_ratio, best_i = ratio, i
            if best_i is None:
                break  # no legal move
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
                delta_sum = (1 / g_next) - (1 / g_cur)  # positive
                new_sum = cur_sum + delta_sum
                if new_sum - TOL > target_sum:  # would overshoot high
                    continue
                delta_err = c_i * (g_cur - g_next)
                gain = delta_err / delta_sum  # benefit per sum increase
                if best_gain is None or gain > best_gain:
                    best_gain, best_i = gain, i
            if best_i is None:
                break
            cur = sizes[best_i]
            nxt = nxt_small[cur]
            cur_sum += (1 / nxt) - (1 / cur)
            sizes[best_i] = nxt

    if abs(cur_sum - target_sum) > TOL:
        warnings.warn(
            f"Final sum 1/g = {cur_sum:.6f} vs target {target_sum:.6f} (>|{TOL}|)."
        )

    return {"g_k": sizes[0::2], "g_v": sizes[1::2]}


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

        print(f"Model: {model} | "
              f"Budget using fixed group size: {budget_using_fixed_group_size:.6f} | "
              f"Budget using dynamic group size: {budget_using_dynamic_group_size:.6f}")

        assert (
            abs(budget_using_fixed_group_size - budget_using_dynamic_group_size) < TOL
        ), f"Budget mismatch for model {model!r}."
        print("\n")

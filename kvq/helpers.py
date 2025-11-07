import json
import os
from pathlib import Path
from importlib.resources import files # we only support Python 3.9+

from kvq.const import model_dict, supported_models

ASSETS_PATH = "kvq.assets"


def load_kv_norms(model_name_or_path: str, score: int) -> dict:
    """Load KV norms from the assets."""
    model = extract_model_name(model_name_or_path)

    if model not in supported_models:
        raise ValueError(
            f"Model {model!r} is not supported. "
            f"Supported models: {', '.join(supported_models)}"
        )

    model_name = model_dict.get(model)
    if model_name is None:
        raise ValueError(f"No entry for {model!r} in kvq.const.model_dict.")

    norm_type = "frobenius_norm" if score == 0 else "spectral_norm"
    score_file = f"{norm_type}/{model_name}.json"

    with files(ASSETS_PATH).joinpath(score_file).open() as f:
        return json.load(f)


def extract_model_name(model_name_or_path):

    if os.sep not in model_name_or_path and "models--" not in model_name_or_path:
        return model_name_or_path

    p = Path(model_name_or_path)

    for part in p.parts:
        if part.startswith("models--"):
            model_part = part[len("models--") :]  # strip off the prefix
            owner, repo = model_part.split("--", 1)  # split into two pieces
            return f"{owner}/{repo}"

    if len(p.parts) >= 2:
        owner, repo = p.parts[-2], p.parts[-1]
        return f"{owner}/{repo}"

    return model_name_or_path

import os
from pathlib import Path


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

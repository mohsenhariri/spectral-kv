from os import getenv
from pathlib import Path

from .custom_type import PathType


def path_in(p: str, global_path=True) -> PathType:
    if p[0] == "/" or not global_path:  # Check whether p is absolute path.
        path_input = p
    else:
        root_in = getenv("ROOT_IN") or "."
        path_input = rf"{root_in}/{p}"

    path: PathType = Path(path_input)
    if not path.exists():
        raise Exception(f"{p} doesn't exist on the earth.")

    return path


def path_out(p: str, global_path=True) -> PathType:
    if p[0] == "/" or not global_path:  # Check whether p is absolute path
        path_output = p
    else:
        root_out = getenv("ROOT_OUT") or "."
        path_output = rf"{root_out}/{p}"

    path: PathType = Path(path_output)
    if not path.exists():
        print("The output path doesn't exist but no worries I'm making it.")
        path.mkdir(parents=True)

    return path

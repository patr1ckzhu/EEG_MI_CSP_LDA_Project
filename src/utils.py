
import os
from pathlib import Path

def ensure_dir(path: str) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

def edf_paths_in(dir_path: str):
    dirp = Path(dir_path)
    return sorted([str(p) for p in dirp.rglob("*.edf")])

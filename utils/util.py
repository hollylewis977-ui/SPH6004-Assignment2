"""util.py — 通用 IO 工具：JSON/pickle 读写、随机种子、路径辅助函数"""
from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def write_json(data: Any, path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_pickle(obj: Any, path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path | str) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def resolve_dynamic_sample_path(dynamic_dir: Path | str, fold_idx: int, split_name: str) -> Path:
    dynamic_dir = Path(dynamic_dir)
    flat_path = dynamic_dir / f"fold_{fold_idx}_{split_name}.csv"
    if flat_path.exists():
        return flat_path

    nested_path = dynamic_dir / f"fold_{fold_idx}" / f"fold_{fold_idx}_{split_name}.csv"
    if nested_path.exists():
        return nested_path

    return flat_path

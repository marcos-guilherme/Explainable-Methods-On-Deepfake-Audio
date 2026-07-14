"""I/O de artefatos de um run e marcadores de conclusão (resumibilidade)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RunPaths:
    """Resolve caminhos de artefatos dentro do diretório de um run."""
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "figures").mkdir(exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / name

    def figure(self, name: str) -> Path:
        return self.root / "figures" / name


def save_npy(arr: np.ndarray, path: str | Path) -> None:
    np.save(path, arr)


def load_npy(path: str | Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _done_path(rp: RunPaths, stage: str) -> Path:
    return rp.path(f"{stage}.done.json")


def write_done(rp: RunPaths, stage: str, config_hash: str, meta: dict[str, Any] | None = None) -> None:
    import datetime as _dt
    payload = {"stage": stage, "config_hash": config_hash,
               "finished_at": _dt.datetime.now().isoformat(timespec="seconds"),
               "meta": meta or {}}
    save_json(payload, _done_path(rp, stage))


def is_done(rp: RunPaths, stage: str, config_hash: str) -> bool:
    p = _done_path(rp, stage)
    if not p.exists():
        return False
    try:
        return load_json(p).get("config_hash") == config_hash
    except (ValueError, OSError):
        return False

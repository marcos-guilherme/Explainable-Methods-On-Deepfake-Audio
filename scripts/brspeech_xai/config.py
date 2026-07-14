"""Configuração de uma rodada: dataclasses + carregamento de YAML + overrides de CLI."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset_id: str = "AKCIT-Deepfake/BRSpeech-DF"
    loader: str = "auto"            # auto | stream | download
    train_split: str = "train"
    eval_split: str = "test"
    n_train_per_class: int = 1500
    n_test_per_class: int = 1500


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    num_samples: int = 64600


@dataclass
class ModelConfig:
    checkpoint: str = "nii-yamagishilab/mms-300m-anti-deepfake"
    spoof_index: int = 0


@dataclass
class OcclusionConfig:
    n_bands: int = 8
    f_min: float = 20.0
    f_max: float = 7900.0
    per_quadrant: int = 150


@dataclass
class ShapConfig:
    top_n: int = 10


@dataclass
class RunConfig:
    run_name: str = "default"
    seed: int = 42
    device: str = "auto"            # auto | cuda | cpu
    output_dir: str = "results"
    data: DataConfig = field(default_factory=DataConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    shap: ShapConfig = field(default_factory=ShapConfig)

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_NESTED = {"data": DataConfig, "audio": AudioConfig, "model": ModelConfig,
           "occlusion": OcclusionConfig, "shap": ShapConfig}


def _coerce(value: str) -> Any:
    """Converte string de CLI para int/float/bool/str (YAML scalar rules)."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _apply_override(cfg: RunConfig, dotted_key: str, raw_value: str) -> None:
    value = _coerce(raw_value)
    parts = dotted_key.split(".")
    target: Any = cfg
    for p in parts[:-1]:
        target = getattr(target, p)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise KeyError(f"override desconhecido: {dotted_key}")
    setattr(target, leaf, value)


def _from_dict(raw: dict[str, Any]) -> RunConfig:
    nested = {}
    for name, klass in _NESTED.items():
        sub = raw.get(name, {}) or {}
        valid = {f.name for f in fields(klass)}
        unknown = set(sub) - valid
        if unknown:
            raise KeyError(f"chaves desconhecidas em '{name}': {sorted(unknown)}")
        nested[name] = klass(**sub)
    top_valid = {f.name for f in fields(RunConfig)} - set(_NESTED)
    top = {k: v for k, v in raw.items() if k in top_valid}
    return RunConfig(**top, **nested)


def load_config(path: str | Path, overrides: list[str] | None = None) -> RunConfig:
    """Carrega o YAML, constrói o RunConfig e aplica overrides `chave.aninhada=valor`."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("config YAML deve ser um mapeamento no topo")
    cfg = _from_dict(raw)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override deve ser chave=valor: {ov!r}")
        key, val = ov.split("=", 1)
        _apply_override(cfg, key.strip(), val.strip())
    return cfg


def dump_resolved(cfg: RunConfig, path: str | Path) -> None:
    """Serializa a config efetiva (pós-overrides) em YAML para reprodutibilidade."""
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=True, allow_unicode=True))

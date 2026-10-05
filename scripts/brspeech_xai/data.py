"""Carregamento de dados: registry por ``dataset_kind`` (HF legado + manifest local)."""
from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from .logging_utils import get_logger

LOGGER = get_logger()
SPOOF_LABEL = 1
LABEL_NAMES = {0: "bonafide", 1: "spoof"}
CLASS_CONFIG_DIR = {0: "bonafide", 1: "spoof"}
_HF_API = HfApi()

# Espelha jmds_prepare.core.xai_sample.XAI_SAMPLE_COLUMNS (sem dependência runtime).
XAI_SAMPLE_COLUMNS = [
    "sample_id",
    "language",
    "role",
    "label",
    "corpus",
    "native_split",
    "original_ref",
    "processed_path",
    "speaker_id",
    "group_id",
    "attack_id",
    "sha256_source",
    "sha256_processed",
    "selection_seed",
    "selection_rank",
    "selection_reason",
    "selection_source",
]

_SUPPORTED_LANGUAGES = frozenset({"eng", "por", "zho"})
_SUPPORTED_ROLES = frozenset({"train", "calibration", "test"})
_SUPPORTED_LABELS = frozenset({0, 1})
_REQUIRED_TEXT_FIELDS = (
    "sample_id",
    "corpus",
    "native_split",
    "original_ref",
    "processed_path",
    "selection_reason",
    "selection_source",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DECIMAL_PATTERN = re.compile(r"^(0|[1-9]\d*)$")

LoaderFn = Callable[..., tuple[list[np.ndarray], list[int], list[int], list[dict[str, Any]]]]
_LOADERS: dict[str, LoaderFn] = {}


def register_loader(kind: str) -> Callable[[LoaderFn], LoaderFn]:
    def decorator(fn: LoaderFn) -> LoaderFn:
        _LOADERS[kind] = fn
        return fn

    return decorator


def list_dataset_kinds() -> list[str]:
    return sorted(_LOADERS)


def decode_audio(audio_field) -> tuple[np.ndarray, int]:
    data = audio_field.get("bytes")
    source = io.BytesIO(data) if data is not None else audio_field["path"]
    waveform, sample_rate = sf.read(source, dtype="float32", always_2d=False)
    return np.asarray(waveform, dtype=np.float32), int(sample_rate)


def list_split_shards(dataset_id: str, config_dir: str, split: str) -> list[str]:
    prefix = f"{config_dir}/{split}-"
    files = _HF_API.list_repo_files(dataset_id, repo_type="dataset")
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))


def _collect_download(dataset_id, label, split, n):
    """Baixa shards (xet nativo), lê e apaga; retorna (audios, srs, provenance)."""
    os.environ["HF_HUB_DISABLE_XET"] = "0"
    audios, srs, prov = [], [], []
    shards = list_split_shards(dataset_id, CLASS_CONFIG_DIR[label], split)
    with tempfile.TemporaryDirectory(dir=os.environ.get("HF_HOME") or None) as tmp:
        for shard in shards:
            if len(audios) >= n:
                break
            local = hf_hub_download(dataset_id, shard, repo_type="dataset", local_dir=tmp)
            audio_col = pq.read_table(local, columns=["audio"]).column("audio").to_pylist()
            for row_idx, rec in enumerate(audio_col):
                if len(audios) >= n:
                    break
                wav, sr = decode_audio(rec)
                audios.append(wav)
                srs.append(sr)
                prov.append(
                    {
                        "config": CLASS_CONFIG_DIR[label],
                        "shard": shard,
                        "row_index": row_idx,
                        "label": label,
                    }
                )
            os.remove(local)
    return audios, srs, prov


def _collect_stream(dataset_id, split, n_per_class):
    """Streaming do config 'default' (como no zeroshot); coleta balanceada."""
    from datasets import Audio, load_dataset

    ds = load_dataset(dataset_id, "default", split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False)).shuffle(seed=42, buffer_size=2000)
    audios, srs, labels, prov = [], [], [], []
    counts = {0: 0, 1: 0}
    for ex in ds:
        lbl = int(ex["label"])
        if counts.get(lbl, 0) >= n_per_class:
            if all(c >= n_per_class for c in counts.values()):
                break
            continue
        wav, sr = decode_audio(ex["audio"])
        audios.append(wav)
        srs.append(sr)
        labels.append(lbl)
        prov.append({"config": "default", "shard": None, "row_index": None, "label": lbl})
        counts[lbl] = counts.get(lbl, 0) + 1
    return audios, srs, labels, prov


def _is_xet_signature_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "403" in msg or "invalid key pair" in msg or "signature" in msg


def _shuffle(audios, srs, labels, prov, seed):
    order = np.random.default_rng(seed).permutation(len(audios))
    return (
        [audios[i] for i in order],
        [srs[i] for i in order],
        [int(labels[i]) for i in order],
        [prov[i] for i in order],
    )


def _require_positive_n_per_class(n_per_class: int | None, *, dataset_kind: str) -> None:
    if n_per_class is None or n_per_class <= 0:
        raise ValueError(
            f"n_per_class deve ser positivo para dataset_kind={dataset_kind!r}, "
            f"obteve {n_per_class!r}"
        )


@register_loader("hf_brspeech")
def _load_hf_brspeech(
    dataset_id: str,
    split: str,
    n_per_class: int | None,
    *,
    loader: str = "auto",
    seed: int = 42,
    **_: Any,
) -> tuple[list[np.ndarray], list[int], list[int], list[dict[str, Any]]]:
    _require_positive_n_per_class(n_per_class, dataset_kind="hf_brspeech")
    if loader in ("auto", "stream"):
        try:
            a, s, lbl, prov = _collect_stream(dataset_id, split, n_per_class)
            LOGGER.info(f"coleta via streaming: {len(a)} audios ({split})")
            return _shuffle(a, s, lbl, prov, seed)
        except (HfHubHTTPError, Exception) as exc:  # noqa: BLE001
            if loader == "stream" or not _is_xet_signature_error(exc):
                raise
            LOGGER.warning(f"streaming falhou ({type(exc).__name__}); fallback para download")
    audios, srs, labels, prov = [], [], [], []
    for label in (0, 1):
        a, s, p = _collect_download(dataset_id, label, split, n_per_class)
        audios += a
        srs += s
        labels += [label] * len(a)
        prov += p
        LOGGER.info(f"download {LABEL_NAMES[label]}/{split}: {len(a)} audios")
    return _shuffle(audios, srs, labels, prov, seed)


def _parse_manifest_label(value: str) -> int:
    if value not in {"0", "1"}:
        raise ValueError(f"Unsupported label: {value}")
    return int(value)


def _parse_manifest_int(value: str, *, field: str) -> int:
    if not _DECIMAL_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid {field}: {value}")
    return int(value)


def _parse_manifest_sha256(value: str, *, field: str) -> str:
    if value != value.lower() or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid {field}: {value}")
    return value


def _optional_text(value: str) -> str | None:
    text = value.strip()
    return text or None


def _find_bundle_root(manifest_dir: Path) -> Path:
    for candidate in (manifest_dir, *manifest_dir.parents):
        if (candidate / "bundle_receipt.json").is_file():
            return candidate
    raise ValueError(
        f"bundle_receipt.json não encontrado no diretório do manifesto ou ancestrais: "
        f"{manifest_dir}"
    )


def _resolve_processed_path(raw_path: str, manifest_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    bundle_root = _find_bundle_root(manifest_dir).resolve()
    resolved = (bundle_root / path).resolve()
    try:
        resolved.relative_to(bundle_root)
    except ValueError as exc:
        raise ValueError(
            f"processed_path relativo escapa do bundle root: {raw_path}"
        ) from exc
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256_preserving_mtime(path: Path, expected: str) -> None:
    stat = path.stat()
    mtime_ns = stat.st_mtime_ns
    actual = _sha256_file(path)
    current_mtime_ns = path.stat().st_mtime_ns
    if current_mtime_ns != mtime_ns:
        raise ValueError(f"mtime alterado durante verificação SHA-256: {path}")
    if actual != expected:
        raise ValueError(
            f"sha256_processed não confere para {path}: esperado {expected}, obtido {actual}"
        )


def _validate_canonical_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        info = sf.info(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"WAV indescodificável: {path}") from exc
    if info.format != "WAV":
        raise ValueError(
            f"WAV canônico deve ser format WAV: {path} (format={info.format})"
        )
    if info.subtype != "PCM_16":
        raise ValueError(
            f"WAV canônico deve ser subtype PCM_16: {path} (subtype={info.subtype})"
        )
    if info.samplerate != 16000:
        raise ValueError(f"WAV deve ser 16 kHz mono canônico: {path} (sr={info.samplerate})")
    if info.channels != 1:
        raise ValueError(f"WAV deve ser 16 kHz mono canônico: {path} (channels={info.channels})")
    if info.frames == 0:
        raise ValueError(f"WAV canônico não pode ser vazio: {path}")
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0 or not np.all(np.isfinite(waveform)):
        raise ValueError(
            f"WAV canônico deve conter amostras finitas não vazias: {path}"
        )
    return waveform, int(sample_rate)


def _read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("manifesto vazio") from exc
        if header != XAI_SAMPLE_COLUMNS:
            raise ValueError("Column order does not match the canonical schema")

        rows: list[dict[str, str]] = []
        for line_no, values in enumerate(reader, start=2):
            if not values:
                continue
            if len(values) != len(XAI_SAMPLE_COLUMNS):
                raise ValueError(
                    f"linha {line_no}: esperadas {len(XAI_SAMPLE_COLUMNS)} colunas, "
                    f"obteve {len(values)}"
                )
            row = dict(zip(XAI_SAMPLE_COLUMNS, values, strict=True))
            rows.append({**row, "manifest_row": str(line_no)})
    return rows


def _validate_manifest_row_semantics(row: dict[str, str]) -> None:
    line = row.get("manifest_row", "?")
    prefix = f"linha {line}"

    for field in _REQUIRED_TEXT_FIELDS:
        if not row[field].strip():
            raise ValueError(f"{prefix}: {field} must be non-empty")

    language = row["language"]
    if language not in _SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")

    role = row["role"]
    if role not in _SUPPORTED_ROLES:
        raise ValueError(f"Unsupported role: {role}")

    label = _parse_manifest_label(row["label"])
    _parse_manifest_sha256(row["sha256_source"], field="sha256_source")
    _parse_manifest_sha256(row["sha256_processed"], field="sha256_processed")
    _parse_manifest_int(row["selection_seed"], field="selection_seed")
    _parse_manifest_int(row["selection_rank"], field="selection_rank")

    if label == 0 and _optional_text(row["attack_id"]) is not None:
        raise ValueError(f"attack_id must be empty for label 0 ({row['sample_id']})")


def _validate_all_manifest_rows(rows: list[dict[str, str]]) -> None:
    for row in rows:
        _validate_manifest_row_semantics(row)


def _validate_manifest_uniqueness(rows: list[dict[str, str]], manifest_dir: Path) -> None:
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen_ids:
            raise ValueError(f"sample_id duplicado: {sample_id}")
        seen_ids.add(sample_id)

        resolved = str(_resolve_processed_path(row["processed_path"], manifest_dir))
        if resolved in seen_paths:
            raise ValueError(f"processed_path duplicado: {row['processed_path']}")
        seen_paths.add(resolved)


def _select_balanced_rows(
    rows: list[dict[str, str]],
    role: str,
    n_per_class: int | None,
) -> list[dict[str, str]]:
    filtered = [row for row in rows if row["role"] == role]
    by_label: dict[int, list[dict[str, str]]] = {0: [], 1: []}
    for row in filtered:
        label = _parse_manifest_label(row["label"])
        by_label[label].append(row)

    for label in (0, 1):
        by_label[label].sort(
            key=lambda row: (
                _parse_manifest_int(row["selection_rank"], field="selection_rank"),
                row["sample_id"],
            )
        )

    if not n_per_class:
        if not by_label[0] or not by_label[1]:
            raise ValueError("amostras insuficientes para balanceamento (classe ausente)")
        n_per_class = min(len(by_label[0]), len(by_label[1]))
        if n_per_class == 0:
            raise ValueError("amostras insuficientes para balanceamento")

    selected: list[dict[str, str]] = []
    for label in (0, 1):
        pool = by_label[label]
        if len(pool) < n_per_class:
            raise ValueError(
                f"amostras insuficientes para label {label}: "
                f"precisa {n_per_class}, disponível {len(pool)} (role={role})"
            )
        selected.extend(pool[:n_per_class])
    return selected


@register_loader("local_manifest")
def _load_local_manifest(
    _dataset_id: str,
    split: str,
    n_per_class: int | None,
    *,
    manifest_path: str = "",
    seed: int = 42,
    **_: Any,
) -> tuple[list[np.ndarray], list[int], list[int], list[dict[str, Any]]]:
    if not manifest_path:
        raise ValueError("manifest_path é obrigatório para dataset_kind=local_manifest")
    manifest = Path(manifest_path).resolve()
    if not manifest.is_file():
        raise ValueError(f"manifesto ausente: {manifest}")

    rows = _read_manifest_rows(manifest)
    manifest_dir = manifest.parent
    _validate_all_manifest_rows(rows)
    _validate_manifest_uniqueness(rows, manifest_dir)
    selected = _select_balanced_rows(rows, split, n_per_class)

    audios: list[np.ndarray] = []
    srs: list[int] = []
    labels: list[int] = []
    provenance: list[dict[str, Any]] = []

    for row in selected:
        label = _parse_manifest_label(row["label"])
        processed_raw = row["processed_path"]
        processed = _resolve_processed_path(processed_raw, manifest_dir)
        if not processed.is_file():
            raise ValueError(f"arquivo ausente: {processed}")
        expected_hash = _parse_manifest_sha256(row["sha256_processed"], field="sha256_processed")
        _verify_sha256_preserving_mtime(processed, expected_hash)
        wav, sr = _validate_canonical_wav(processed)
        audios.append(wav)
        srs.append(sr)
        labels.append(label)
        provenance.append(
            {
                "sample_id": row["sample_id"],
                "language": row["language"],
                "role": row["role"],
                "corpus": row["corpus"],
                "native_split": row["native_split"],
                "original_ref": row["original_ref"],
                "processed_path": str(processed),
                "speaker_id": _optional_text(row["speaker_id"]),
                "group_id": _optional_text(row["group_id"]),
                "attack_id": _optional_text(row["attack_id"]),
                "sha256_source": _parse_manifest_sha256(
                    row["sha256_source"], field="sha256_source"
                ),
                "sha256_processed": expected_hash,
                "selection_seed": _parse_manifest_int(
                    row["selection_seed"], field="selection_seed"
                ),
                "selection_rank": _parse_manifest_int(
                    row["selection_rank"], field="selection_rank"
                ),
                "selection_reason": row["selection_reason"],
                "selection_source": row["selection_source"],
                "manifest_path": str(manifest),
                "manifest_row": int(row["manifest_row"]),
                "label": label,
            }
        )

    LOGGER.info(
        f"coleta via manifest local: {len(audios)} audios (role={split}, manifest={manifest.name})"
    )
    return _shuffle(audios, srs, labels, provenance, seed)


def build_balanced_split(
    dataset_id: str,
    split: str,
    n_per_class: int | None,
    loader: str = "auto",
    seed: int = 42,
    *,
    dataset_kind: str = "hf_brspeech",
    manifest_path: str = "",
) -> tuple[list[np.ndarray], list[int], list[int], list[dict[str, Any]]]:
    """Monta subconjunto balanceado + proveniência via registry ``dataset_kind``.

    Contrato de saída (estável):
        ``(audios, srs, labels, provenance)``, listas do mesmo tamanho;
        ``labels`` usa ``SPOOF_LABEL=1`` para spoof e 0 para bonafide.

    ``dataset_kind=hf_brspeech`` (default): comportamento legado HF auto/stream/download;
    exige ``n_per_class`` positivo.
    ``dataset_kind=local_manifest``: lê ``manifest_path`` (CSV ``xai_samples`` canônico),
    valida semanticamente todas as linhas, filtra ``role == split``. Com ``n_per_class``
    None ou 0, usa todas as amostras balanceadas disponíveis (``min(count_por_classe)``),
    seleção determinística por ``selection_rank``/``sample_id`` antes do shuffle final
    por ``seed``. Paths ``processed_path`` absolutos permanecem suportados; relativos
    exigem ``bundle_receipt.json`` em um ancestral do manifest e são resolvidos contra
    essa raiz sem permitir escape. Integridade é garantida por SHA-256.
    """
    try:
        fn = _LOADERS[dataset_kind]
    except KeyError as exc:
        raise ValueError(f"dataset_kind desconhecido: {dataset_kind!r}") from exc

    if dataset_kind == "local_manifest":
        return fn(
            dataset_id,
            split,
            n_per_class,
            manifest_path=manifest_path,
            seed=seed,
        )
    return fn(dataset_id, split, n_per_class, loader=loader, seed=seed)

"""Carregamento do BRSpeech-DF: streaming, download nativo e modo auto com fallback."""
from __future__ import annotations

import io
import os
import tempfile

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
                audios.append(wav); srs.append(sr)
                prov.append({"config": CLASS_CONFIG_DIR[label], "shard": shard,
                             "row_index": row_idx, "label": label})
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
        audios.append(wav); srs.append(sr); labels.append(lbl)
        prov.append({"config": "default", "shard": None, "row_index": None, "label": lbl})
        counts[lbl] = counts.get(lbl, 0) + 1
    return audios, srs, labels, prov


def _is_xet_signature_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "403" in msg or "invalid key pair" in msg or "signature" in msg


def build_balanced_split(dataset_id, split, n_per_class, loader="auto", seed=42):
    """Ponto único de carga de dados. Monta subconjunto balanceado + proveniência.

    Contrato de saída (estável, para um segundo dataset plugar sem retrabalho):
        ``(audios, srs, labels, provenance)``, com listas do mesmo tamanho, onde
        ``labels`` usa ``SPOOF_LABEL=1`` para spoof e 0 para bonafide, e cada item de
        ``provenance`` é um dict com ao menos ``{"config", "shard", "row_index", "label"}``.

    Suposições específicas do BRSpeech-DF (rever ao adicionar outro dataset):
        configs por classe ``bonafide``/``spoof``; config de streaming ``default``;
        campo de rótulo ``label``. Quando entrar um segundo dataset, introduzir
        ``data.dataset_kind`` e despachar aqui (ou promover a um registry só então).
    """
    if loader in ("auto", "stream"):
        try:
            a, s, lbl, prov = _collect_stream(dataset_id, split, n_per_class)
            LOGGER.info("coleta via streaming: %d audios (%s)", len(a), split)
            return _shuffle(a, s, lbl, prov, seed)
        except (HfHubHTTPError, Exception) as exc:  # noqa: BLE001
            if loader == "stream" or not _is_xet_signature_error(exc):
                raise
            LOGGER.warning("streaming falhou (%s); fallback para download", type(exc).__name__)
    # download por classe
    audios, srs, labels, prov = [], [], [], []
    for label in (0, 1):
        a, s, p = _collect_download(dataset_id, label, split, n_per_class)
        audios += a; srs += s; labels += [label] * len(a); prov += p
        LOGGER.info("download %s/%s: %d audios", LABEL_NAMES[label], split, len(a))
    return _shuffle(audios, srs, labels, prov, seed)


def _shuffle(audios, srs, labels, prov, seed):
    order = np.random.default_rng(seed).permutation(len(audios))
    return ([audios[i] for i in order], [srs[i] for i in order],
            [int(labels[i]) for i in order], [prov[i] for i in order])

"""Audit simple class-correlated corpus and waveform shortcuts."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import roc_auc_score


_REQUIRED_COLUMNS = {
    "sample_id",
    "language",
    "role",
    "label",
    "corpus",
    "processed_path",
}


def _cramers_v(labels: np.ndarray, corpora: np.ndarray) -> float:
    table = pd.crosstab(labels, corpora).to_numpy(dtype=np.float64)
    total = float(table.sum())
    expected = table.sum(axis=1, keepdims=True) @ table.sum(
        axis=0, keepdims=True
    ) / total
    chi_square = float(np.sum((table - expected) ** 2 / expected))
    denominator = total * min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt(chi_square / denominator)) if denominator > 0 else 0.0


def _waveform_features(path: Path) -> tuple[dict[str, float], str]:
    info = sf.info(path)
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"invalid finite mono waveform: {path}")
    peak = float(np.max(np.abs(audio)))
    first = audio[: min(len(audio), int(0.25 * sample_rate))]
    active = np.flatnonzero(np.abs(audio) >= max(peak * 0.01, 1e-5))
    leading = float(active[0] / sample_rate) if len(active) else len(audio) / sample_rate
    features = {
        "duration_s": float(len(audio) / sample_rate),
        "peak": peak,
        "rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
        "first_250ms_rms": float(
            np.sqrt(np.mean(first.astype(np.float64) ** 2))
        ),
        "leading_silence_s": leading,
    }
    channels = "mono" if info.channels == 1 else f"{info.channels}ch"
    description = (
        f"{info.format}/{info.subtype}/{channels}/{info.samplerate}Hz"
    )
    return features, description


def audit_manifest(manifest_path: str | Path, *, role: str = "test") -> pd.DataFrame:
    """Summarize class/corpus confounding and trivial waveform feature AUCs."""
    manifest_path = Path(manifest_path)
    frame = pd.read_csv(manifest_path)
    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"manifest lacks columns {sorted(missing)}")
    frame = frame[frame["role"] == role].copy()
    if frame.empty:
        raise ValueError(f"manifest has no rows for role {role!r}")
    frame["label"] = frame["label"].astype(int)
    if set(frame["label"]) != {0, 1}:
        raise ValueError("audit role must contain labels 0 and 1")

    feature_rows: list[dict[str, float]] = []
    formats: Counter[str] = Counter()
    for row in frame.itertuples(index=False):
        features, description = _waveform_features(Path(row.processed_path))
        formats[description] += 1
        feature_rows.append(features)
    values = pd.DataFrame(feature_rows, index=frame.index)
    labels = frame["label"].to_numpy(dtype=int)
    bonafide = labels == 0
    spoof = labels == 1
    corpus_v = _cramers_v(labels, frame["corpus"].astype(str).to_numpy())
    format_text = "; ".join(
        name if count == len(frame) else f"{name} ({count})"
        for name, count in sorted(formats.items())
    )
    common = {
        "language": str(frame["language"].iloc[0]),
        "role": role,
        "n_bonafide": int(bonafide.sum()),
        "n_spoof": int(spoof.sum()),
        "bonafide_corpora": "|".join(
            sorted(frame.loc[bonafide, "corpus"].astype(str).unique())
        ),
        "spoof_corpora": "|".join(
            sorted(frame.loc[spoof, "corpus"].astype(str).unique())
        ),
        "corpus_label_cramers_v": corpus_v,
        "file_formats": format_text,
    }
    rows = []
    for feature in values.columns:
        auc = float(roc_auc_score(labels, values[feature]))
        rows.append(
            {
                **common,
                "feature": feature,
                "bonafide_median": float(values.loc[bonafide, feature].median()),
                "spoof_median": float(values.loc[spoof, feature].median()),
                "roc_auc": auc,
                "orientation_invariant_auc": max(auc, 1.0 - auc),
            }
        )
    return pd.DataFrame(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit corpus/label confounding and simple waveform shortcuts."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--role", default="test")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = audit_manifest(args.manifest, role=args.role)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, lineterminator="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""English rules held by the profile, processing free-space check and resume arity."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from jmds_prepare import audio as audio_module
from jmds_prepare import cli
from jmds_prepare.config import PreparationConfig
from jmds_prepare.manifest import MANIFEST_COLUMNS
from jmds_prepare.pipelines import acquisition
from jmds_prepare.profiles.english import ENGLISH_PROFILE
from jmds_prepare.sources.asvspoof import (
    ASV_COLUMNS,
    read_asvspoof_protocol,
    validate_correspondence,
)
from jmds_prepare.sources.jmds import JMDS_COLUMNS, read_jmds_protocol
from jmds_prepare.storage import estimates
from jmds_prepare.storage.layout import english_layout

# --- A: English rules held by the profile ---------------------------------------


def test_profile_holds_the_remaining_english_rules(tmp_path):
    profile = ENGLISH_PROFILE

    assert dict(profile.split_id_prefixes) == {"train": "T", "dev": "D", "eval": "E"}
    assert dict(profile.label_equivalence) == {
        "pristine": "bonafide",
        "generated": "spoof",
    }
    assert dict(profile.archive_prefix_splits) == {
        "flac_T_": "train",
        "flac_D_": "dev",
    }
    assert profile.generated_audio_subdir == "wav"
    assert [profile.jmds_protocol_filename(split) for split in profile.protocol_splits] == [
        "open_v2_train.cm.csv",
        "open_v2_dev.cm.csv",
        "open_v2_eval.cm.csv",
    ]
    assert profile.jmds_protocol_path(tmp_path, "dev") == (
        tmp_path / "cm_protocols" / "open_v2_dev.cm.csv"
    )
    assert profile.generated_split_dir(tmp_path, "train") == (
        tmp_path / "dataset" / "English_ASVspoof2024_Generated" / "train" / "wav"
    )


def test_profile_rule_mappings_are_read_only():
    for mapping in (
        ENGLISH_PROFILE.split_id_prefixes,
        ENGLISH_PROFILE.label_equivalence,
        ENGLISH_PROFILE.archive_prefix_splits,
    ):
        with pytest.raises(TypeError):
            mapping["train"] = "changed"  # type: ignore[index]


def test_profile_protocol_filename_rejects_splits_outside_the_protocols():
    with pytest.raises(ValueError, match="Unsupported split"):
        ENGLISH_PROFILE.jmds_protocol_filename("test")


def test_profile_archive_split_maps_every_required_archive():
    assert [ENGLISH_PROFILE.archive_split(name) for name in ENGLISH_PROFILE.required_archives] == [
        *(["train"] * 5),
        *(["dev"] * 3),
    ]
    with pytest.raises(ValueError, match="Cannot determine split for archive: x.tar"):
        ENGLISH_PROFILE.archive_split("x.tar")


def test_acquisition_archive_split_follows_the_profile():
    for name in ENGLISH_PROFILE.required_archives:
        assert acquisition.archive_split(name) == ENGLISH_PROFILE.archive_split(name)
    assert cli._archive_split is acquisition.archive_split
    with pytest.raises(ValueError, match="Cannot determine split for archive"):
        acquisition.archive_split("flac_E_aa.tar")


def _jmds_csv(path: Path, utt_id: str, label: str = "pristine") -> Path:
    row = {
        "spk_id": "S_1",
        "utt_id": utt_id,
        "gender": "F",
        "codec": "-",
        "attack_id": "pristine" if label == "pristine" else "A01",
        "label": label,
        "native": "yes",
        "language": "eng",
        "dataset": "ASVspoof2024",
    }
    pd.DataFrame([row], columns=JMDS_COLUMNS).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("split", ENGLISH_PROFILE.protocol_splits)
def test_protocol_readers_enforce_the_profile_id_prefix(tmp_path, split):
    prefix = ENGLISH_PROFILE.split_id_prefixes[split]
    good_id = f"{prefix}_0000000001"
    wrong_prefix = next(
        other for other in ENGLISH_PROFILE.split_id_prefixes.values() if other != prefix
    )
    bad_id = f"{wrong_prefix}_0000000001"

    assert read_jmds_protocol(_jmds_csv(tmp_path / "ok.csv", good_id), split)[
        "utt_id"
    ].tolist() == [good_id]
    with pytest.raises(ValueError, match="Malformed JMDS utt_id"):
        read_jmds_protocol(_jmds_csv(tmp_path / "bad.csv", bad_id), split)

    asv_ok = tmp_path / "ok.tsv"
    asv_ok.write_text(
        f"S_1 {good_id} F - - - - bonafide bonafide -\n", encoding="utf-8"
    )
    assert len(ASV_COLUMNS) == 10
    assert read_asvspoof_protocol(asv_ok, split)["utt_id"].tolist() == [good_id]


@pytest.mark.parametrize(
    ("jmds_label", "asv_key", "matches"),
    [
        ("pristine", "bonafide", True),
        ("generated", "spoof", True),
        ("pristine", "spoof", False),
        ("generated", "bonafide", False),
    ],
)
def test_correspondence_uses_the_profile_label_equivalence(
    jmds_label, asv_key, matches
):
    assert (ENGLISH_PROFILE.label_equivalence[jmds_label] == asv_key) is matches
    jmds = pd.DataFrame(
        [["S_1", "T_0000000001", "F", "-", "A01", jmds_label, "yes", "eng", "ASVspoof2024"]],
        columns=JMDS_COLUMNS,
    )
    asv = pd.DataFrame(
        [["S_1", "T_0000000001", "F", "-", "-", "-", "-", "A01", asv_key, "-"]],
        columns=ASV_COLUMNS,
    )

    report = validate_correspondence(jmds, asv, "train")

    assert ("T_0000000001" in report.metadata_mismatches) is not matches
    matched = report.pristine_matched + report.generated_matched
    assert matched == (1 if matches else 0)


def test_config_requires_the_profile_protocol_files(tmp_path):
    jmds_root = tmp_path / "jmds"
    (jmds_root / "cm_protocols").mkdir(parents=True)
    ENGLISH_PROFILE.generated_root(jmds_root).mkdir(parents=True)
    for split in ENGLISH_PROFILE.protocol_splits:
        ENGLISH_PROFILE.jmds_protocol_path(jmds_root, split).write_text("x")
    config = PreparationConfig(jmds_root=jmds_root, data_root=tmp_path / "data")

    config.validate()

    missing = ENGLISH_PROFILE.jmds_protocol_path(jmds_root, "eval")
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="open_v2_eval.cm.csv"):
        config.validate()


def test_audio_estimate_reads_generated_audio_from_the_profile_dir(tmp_path):
    config = PreparationConfig(jmds_root=tmp_path / "jmds", data_root=tmp_path / "data")
    generated = ENGLISH_PROFILE.generated_split_dir(config.jmds_root, "dev")
    generated.mkdir(parents=True)
    sf.write(generated / "D_0000000001.wav", np.zeros(160), 16_000, subtype="PCM_16")
    protocol = pd.DataFrame(
        {"utt_id": ["D_0000000001"], "label": ["generated"]}
    )

    result = estimates.audio_storage_estimate(config, {"dev": protocol})

    assert result["generated"]["files_measured"] == 1
    assert result["generated"]["processed_pcm16_bytes"] == 320


# --- B: free-space check before processing ---------------------------------------


class _Usage:
    def __init__(self, free: int) -> None:
        self.total = free
        self.used = 0
        self.free = free


def _write(path: Path, samples: np.ndarray, rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, rate, subtype="FLOAT")
    return path


def _row(source: Path, utt_id: str, split: str = "train", label: str = "pristine"):
    return {
        "utt_id": utt_id,
        "spk_id": "T_0001",
        "gender": "F",
        "language": "eng",
        "dataset": "ASVspoof2024",
        "split": split,
        "label": label,
        "attack_id": "pristine" if label == "pristine" else "A01",
        "source_path": str(source),
        "processed_path": "",
        "sha256_source": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sha256_processed": "",
        "metadata_source": "JMDS+ASVspoof5-verified",
    }


@pytest.fixture
def raw_manifest(tmp_path):
    first = _write(tmp_path / "src" / "a.wav", np.linspace(-0.5, 0.5, 81), 8_000)
    second = _write(
        tmp_path / "src" / "b.wav",
        np.column_stack([np.ones(100) * 0.1, np.zeros(100)]),
        22_050,
    )
    return pd.DataFrame(
        [
            _row(first, "T_1"),
            _row(second, "D_2", split="dev", label="generated"),
        ],
        columns=MANIFEST_COLUMNS,
    )


MIB = 1 << 20
GIB = 1 << 30
# Multiples of 100 ns, the NTFS timestamp resolution.
FIXED_TIMES_NS = (1_600_000_000_123_456_700, 1_500_000_000_987_654_300)
# a.wav: 81 frames at 8 kHz -> 162 frames -> 324 + 44; b.wav: 100 frames at
# 22.05 kHz -> 73 frames -> 146 + 44.
A_WAV_BYTES = 368
B_WAV_BYTES = 190


def _patch_free(monkeypatch, free: int) -> list[Path]:
    seen: list[Path] = []

    def disk_usage(path):
        seen.append(Path(path))
        return _Usage(free)

    monkeypatch.setattr(audio_module.shutil, "disk_usage", disk_usage)
    return seen


def _estimate(manifest: pd.DataFrame, data_root: Path):
    return estimates.processing_space_estimate(
        manifest, english_layout(data_root), 16_000
    )


def _create_outputs(data_root: Path, *keys: tuple[str, str, str]) -> None:
    layout = english_layout(data_root)
    for split, label, utt_id in keys:
        path = layout.processed_dir(split, label) / f"{utt_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"existing")


def test_remaining_processed_wav_bytes_counts_header_and_payload(tmp_path, raw_manifest):
    layout = english_layout(tmp_path / "data")

    required = estimates.remaining_processed_wav_bytes(raw_manifest, layout, 16_000)

    assert estimates.WAV_PCM16_HEADER_BYTES == 44
    assert required == A_WAV_BYTES + B_WAV_BYTES


def test_remaining_estimate_matches_the_written_wav_sizes(
    tmp_path, raw_manifest, monkeypatch
):
    data_root = tmp_path / "data"
    required = estimates.remaining_processed_wav_bytes(
        raw_manifest, english_layout(data_root), 16_000
    )
    _patch_free(monkeypatch, 1 << 62)

    result = audio_module.process_manifest(
        raw_manifest, data_root, tmp_path / "processed.csv"
    )

    written = sum(Path(path).stat().st_size for path in result["processed_path"])
    assert written == required


def test_remaining_estimate_skips_existing_outputs(tmp_path, raw_manifest):
    data_root = tmp_path / "data"
    _create_outputs(data_root, ("train", "pristine", "T_1"))

    required = estimates.remaining_processed_wav_bytes(
        raw_manifest, english_layout(data_root), 16_000
    )

    assert required == B_WAV_BYTES


def test_space_estimate_for_missing_outputs_has_every_reserve(tmp_path, raw_manifest):
    estimate = _estimate(raw_manifest, tmp_path / "data")

    assert estimate == estimates.ProcessingSpaceEstimate(
        rows=2,
        missing_outputs=2,
        missing_output_bytes=A_WAV_BYTES + B_WAV_BYTES,
        missing_output_allocated_bytes=2 * 4096,
        largest_temporary_bytes=4096,
        manifest_reserve_bytes=2 * MIB,
        safety_reserve_bytes=GIB,
    )
    assert estimate.subtotal_bytes == 2 * 4096 + 4096 + 2 * MIB
    assert estimate.required_bytes == estimate.subtotal_bytes + GIB


def test_space_estimate_rounds_each_output_up_to_whole_clusters(tmp_path):
    # 2027 frames at the target rate -> 4054 + 44 = 4098 bytes: two clusters.
    source = _write(tmp_path / "src" / "c.wav", np.zeros(2027), 16_000)
    frame = pd.DataFrame([_row(source, "T_9")], columns=MANIFEST_COLUMNS)

    estimate = _estimate(frame, tmp_path / "data")

    assert estimates.CLUSTER_BYTES == 4096
    assert estimate.missing_output_bytes == 4098
    assert estimate.missing_output_allocated_bytes == 8192
    assert estimate.largest_temporary_bytes == 8192


def test_space_estimate_with_all_outputs_existing_keeps_temp_and_reserves(
    tmp_path, raw_manifest
):
    data_root = tmp_path / "data"
    _create_outputs(
        data_root, ("train", "pristine", "T_1"), ("dev", "generated", "D_2")
    )

    estimate = _estimate(raw_manifest, data_root)

    assert estimate.missing_outputs == 0
    assert estimate.missing_output_bytes == 0
    assert estimate.missing_output_allocated_bytes == 0
    assert estimate.largest_temporary_bytes == 4096
    assert estimate.manifest_reserve_bytes == 2 * MIB
    assert estimate.required_bytes == 4096 + 2 * MIB + GIB


def test_space_estimate_counts_only_the_missing_outputs_as_new(tmp_path, raw_manifest):
    data_root = tmp_path / "data"
    _create_outputs(data_root, ("train", "pristine", "T_1"))

    estimate = _estimate(raw_manifest, data_root)

    assert estimate.missing_outputs == 1
    assert estimate.missing_output_bytes == B_WAV_BYTES
    assert estimate.missing_output_allocated_bytes == 4096
    assert estimate.largest_temporary_bytes == 4096


def test_manifest_reserve_covers_the_manifest_and_its_temporary():
    small = estimates.processing_space_estimate_from_sizes([], [], rows=10)
    large = estimates.processing_space_estimate_from_sizes([], [], rows=1_000)

    assert small.manifest_reserve_bytes == 2 * MIB
    assert large.manifest_reserve_bytes == 2 * 1_000 * 2048
    assert small.largest_temporary_bytes == 0


def test_safety_reserve_is_at_least_one_gib_or_one_percent():
    small = estimates.processing_space_estimate_from_sizes([10], [10], rows=1)
    size = 200 * GIB + 1
    large = estimates.processing_space_estimate_from_sizes([size], [size], rows=1)

    assert small.safety_reserve_bytes == GIB
    assert large.missing_output_allocated_bytes == 200 * GIB + 4096
    assert large.safety_reserve_bytes == -(-large.subtotal_bytes // 100)
    assert large.safety_reserve_bytes > GIB
    assert large.required_bytes == large.subtotal_bytes + large.safety_reserve_bytes


def test_source_payload_uses_integer_resampling_arithmetic(tmp_path, monkeypatch):
    frames = 2**53 + 1
    monkeypatch.setattr(estimates, "_read_frames_and_rate", lambda _p: (frames, 16_000))

    assert estimates._source_pcm16_payload(tmp_path / "x.wav", 16_000) == frames * 2
    monkeypatch.setattr(estimates, "_read_frames_and_rate", lambda _p: (3, 2))
    assert estimates._source_pcm16_payload(tmp_path / "x.wav", 3) == 5 * 2


@pytest.mark.parametrize(("frames", "rate"), [(0, 16_000), (10, 0), (-1, 8_000)])
def test_source_payload_rejects_invalid_headers(tmp_path, monkeypatch, frames, rate):
    monkeypatch.setattr(estimates, "_read_frames_and_rate", lambda _p: (frames, rate))

    with pytest.raises(ValueError, match="Invalid audio header"):
        estimates._source_pcm16_payload(tmp_path / "x.wav", 16_000)


def _touching(monkeypatch, module, name: str) -> None:
    """Make a header reader bump the file times, as a real filesystem may."""
    original = getattr(module, name)

    def reader(path, *args, **kwargs):
        os.utime(path, ns=(7, 7))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(module, name, reader)


def _fix_times(*paths: Path) -> None:
    for path in paths:
        os.utime(path, ns=FIXED_TIMES_NS)


def _times(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_atime_ns, stat.st_mtime_ns


@pytest.mark.parametrize("enough_space", [False, True])
def test_process_manifest_header_reads_preserve_source_times(
    tmp_path, raw_manifest, monkeypatch, enough_space
):
    sources = [Path(path) for path in raw_manifest["source_path"]]
    _fix_times(*sources)
    _touching(monkeypatch, estimates.wave, "open")
    _touching(monkeypatch, estimates.sf, "info")
    _patch_free(monkeypatch, (1 << 62) if enough_space else 0)

    if enough_space:
        audio_module.process_manifest(raw_manifest, tmp_path / "data", tmp_path / "m.csv")
    else:
        with pytest.raises(estimates.InsufficientProcessingSpaceError):
            audio_module.process_manifest(
                raw_manifest, tmp_path / "data", tmp_path / "m.csv"
            )

    assert [_times(path) for path in sources] == [FIXED_TIMES_NS] * 2


def test_failed_header_read_preserves_source_times(tmp_path, monkeypatch):
    invalid = tmp_path / "src" / "invalid.wav"
    invalid.parent.mkdir(parents=True)
    invalid.write_text("not audio", encoding="utf-8")
    frame = pd.DataFrame([_row(invalid, "T_3")], columns=MANIFEST_COLUMNS)
    _fix_times(invalid)
    _touching(monkeypatch, estimates.wave, "open")
    _touching(monkeypatch, estimates.sf, "info")

    with pytest.raises(ValueError, match="Could not read audio source"):
        audio_module.process_manifest(frame, tmp_path / "data", tmp_path / "m.csv")

    assert _times(invalid) == FIXED_TIMES_NS
    assert not (tmp_path / "data").exists()


def test_process_manifest_refuses_when_space_is_insufficient(
    tmp_path, raw_manifest, monkeypatch
):
    data_root = tmp_path / "data"
    manifest_path = tmp_path / "out" / "processed.csv"
    estimate = _estimate(raw_manifest, data_root)
    free = estimate.required_bytes - 1
    seen = _patch_free(monkeypatch, free)
    monkeypatch.setattr(
        audio_module,
        "process_audio",
        lambda *_args: pytest.fail("no audio may be processed"),
    )

    with pytest.raises(
        estimates.InsufficientProcessingSpaceError,
        match=(
            rf"required {estimate.required_bytes} bytes.*free {free} bytes.*"
            rf"missing outputs 2 \(allocated 8192\).*largest temporary 4096.*"
            rf"manifest reserve {2 * MIB}.*safety reserve {GIB}"
        ),
    ):
        audio_module.process_manifest(raw_manifest, data_root, manifest_path)

    assert seen == [tmp_path.resolve()]
    assert not data_root.exists()
    assert not manifest_path.exists()
    assert not manifest_path.parent.exists()


def test_process_manifest_failure_preserves_existing_manifest_and_outputs(
    tmp_path, raw_manifest, monkeypatch
):
    data_root = tmp_path / "data"
    layout = english_layout(data_root)
    _create_outputs(data_root, ("train", "pristine", "T_1"))
    existing = layout.processed_dir("train", "pristine") / "T_1.wav"
    manifest_path = tmp_path / "processed.csv"
    manifest_path.write_text("old,manifest\n", encoding="utf-8")
    _patch_free(monkeypatch, _estimate(raw_manifest, data_root).required_bytes - 1)

    with pytest.raises(estimates.InsufficientProcessingSpaceError):
        audio_module.process_manifest(raw_manifest, data_root, manifest_path)

    assert manifest_path.read_text(encoding="utf-8") == "old,manifest\n"
    assert existing.read_bytes() == b"existing"
    assert not layout.processed_dir("dev", "generated").exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "data",
        "processed.csv",
        "src",
    ]


def test_process_manifest_proceeds_when_free_equals_the_conservative_total(
    tmp_path, raw_manifest, monkeypatch
):
    data_root = tmp_path / "data"
    _patch_free(monkeypatch, _estimate(raw_manifest, data_root).required_bytes)
    manifest_path = tmp_path / "processed.csv"

    result = audio_module.process_manifest(raw_manifest, data_root, manifest_path)

    assert manifest_path.is_file()
    assert all(Path(path).is_file() for path in result["processed_path"])


def test_process_manifest_with_all_outputs_present_still_needs_temp_and_reserves(
    tmp_path, raw_manifest, monkeypatch
):
    data_root = tmp_path / "data"
    _patch_free(monkeypatch, 1 << 62)
    first = audio_module.process_manifest(raw_manifest, data_root, tmp_path / "a.csv")
    required = _estimate(first, data_root).required_bytes
    assert required == 4096 + 2 * MIB + GIB

    _patch_free(monkeypatch, required - 1)
    with pytest.raises(estimates.InsufficientProcessingSpaceError):
        audio_module.process_manifest(raw_manifest, data_root, tmp_path / "b.csv")
    assert not (tmp_path / "b.csv").exists()

    _patch_free(monkeypatch, required)
    again = audio_module.process_manifest(raw_manifest, data_root, tmp_path / "b.csv")
    reprocessed = audio_module.process_manifest(first, data_root, tmp_path / "c.csv")

    pd.testing.assert_frame_equal(again, first)
    pd.testing.assert_frame_equal(reprocessed, first)

def test_process_manifest_reports_unreadable_sources_as_audio_read_errors(
    tmp_path, raw_manifest
):
    invalid = tmp_path / "src" / "invalid.wav"
    invalid.write_text("not audio", encoding="utf-8")
    frame = pd.concat(
        [raw_manifest, pd.DataFrame([_row(invalid, "T_3")], columns=MANIFEST_COLUMNS)],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Could not read audio source"):
        audio_module.process_manifest(frame, tmp_path / "data", tmp_path / "m.csv")

    assert not (tmp_path / "data").exists()


def test_cli_replaced_process_manifest_skips_the_internal_space_check(
    tmp_path, monkeypatch
):
    config = PreparationConfig(jmds_root=tmp_path / "jmds", data_root=tmp_path / "data")
    raw_path = english_layout(config.data_root).raw_manifest_path
    raw_path.parent.mkdir(parents=True)
    pd.DataFrame(columns=MANIFEST_COLUMNS).to_csv(raw_path, index=False)
    monkeypatch.setattr(
        audio_module.shutil,
        "disk_usage",
        lambda _path: pytest.fail("replaced collaborator must own the check"),
    )
    calls: list[tuple] = []
    monkeypatch.setattr(
        cli,
        "process_manifest",
        lambda *args: calls.append(args) or pd.DataFrame(),
    )

    cli.process_english(config, output=lambda _message: None)

    assert len(calls) == 1


def test_dry_run_describes_the_existing_processing_space_check():
    from jmds_prepare.pipelines import acquisition_plan

    assert acquisition_plan.PROCESSING_SPACE_NOTE == (
        "process-english checks free space for the missing PCM-16 outputs "
        "before writing any audio or manifest"
    )


# --- resume_valid_ids arity ------------------------------------------------------


def test_resume_valid_ids_passes_all_arguments_to_full_validators(tmp_path):
    calls: list[tuple] = []

    def validator(output, targets, ledger_path, archive_md5s, split):
        calls.append((output, targets, ledger_path, archive_md5s, split))
        return {"T_1"}

    args = (tmp_path, {"T_1"}, tmp_path / "ledger.csv", {"a.tar": "md5"}, "train")

    assert acquisition.resume_valid_ids(validator, *args) == {"T_1"}
    assert calls == [args]


def test_resume_valid_ids_keeps_legacy_two_argument_validators(tmp_path):
    calls: list[tuple] = []

    def validator(output, targets):
        calls.append((output, targets))
        return set(targets)

    result = acquisition.resume_valid_ids(
        validator, tmp_path, {"T_1"}, tmp_path / "ledger.csv", {}, "train"
    )

    assert result == {"T_1"}
    assert calls == [(tmp_path, {"T_1"})]


def test_resume_valid_ids_propagates_internal_type_errors(tmp_path):
    calls: list[int] = []

    def validator(output, targets, ledger_path=None, archive_md5s=None, split=None):
        calls.append(1)
        raise TypeError("internal validator bug")

    with pytest.raises(TypeError, match="internal validator bug"):
        acquisition.resume_valid_ids(
            validator, tmp_path, {"T_1"}, tmp_path / "ledger.csv", {}, "train"
        )

    assert calls == [1]

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from jmds_prepare import cli
from jmds_prepare.config import PreparationConfig
from jmds_prepare.extraction import ArchiveInventory, ExtractionReport
from jmds_prepare.protocols import ASV_COLUMNS, JMDS_COLUMNS, CorrespondenceReport
from jmds_prepare.zenodo import REQUIRED_ARCHIVES, RecordFile


def _config(tmp_path: Path) -> PreparationConfig:
    return PreparationConfig(
        jmds_root=tmp_path / "jmds",
        data_root=tmp_path / "data",
        asvspoof_protocol_root=tmp_path / "asv",
    )


def _report(split: str, pristine: int = 1, generated: int = 1):
    return CorrespondenceReport(
        split=split,
        pristine_expected=pristine,
        generated_expected=generated,
        pristine_matched=pristine,
        generated_matched=generated,
        missing_ids=(),
        extra_ids=(),
        metadata_mismatches={},
        jmds_fingerprint=f"jmds-{split}",
        asv_fingerprint=f"asv-{split}",
    )


def _protocol(split: str, pristine_ids: tuple[str, ...]) -> pd.DataFrame:
    prefix = "T" if split == "train" else "D"
    rows = [
        {
            "spk_id": f"{prefix}_0001",
            "utt_id": utt_id,
            "gender": "F",
            "codec": "-",
            "attack_id": "pristine",
            "label": "pristine",
            "native": "yes",
            "language": "eng",
            "dataset": "ASVspoof2024",
        }
        for utt_id in pristine_ids
    ]
    return pd.DataFrame(rows)


def _record_files() -> dict[str, RecordFile]:
    return {
        name: RecordFile(
            name=name,
            size=index + 10,
            checksum=f"{index:032x}",
            download_url=f"https://example.test/{name}",
        )
        for index, name in enumerate(REQUIRED_ARCHIVES)
    }


def _eval_evidence(jmds_fingerprint="jmds-eval", asv_fingerprint="asv-eval"):
    return {
        "status": "excluded_known_id_mismatch",
        "reason": "known mismatch",
        "jmds_fingerprint": jmds_fingerprint,
        "asv_fingerprint": asv_fingerprint,
        "counts": {
            "jmds": {"pristine": 1, "generated": 1},
            "asv": {"bonafide": 1, "spoof": 2},
        },
        "overlap_by_label": {
            "pristine_bonafide": 1,
            "generated_spoof": 1,
        },
        "id_overlap_total": 2,
        "jmds_only_count": 0,
        "asv_only_count": 1,
    }


def test_configuration_adds_optional_asv_root_compatibly(tmp_path):
    legacy = PreparationConfig(tmp_path / "jmds", tmp_path / "data")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            (
                f"jmds_root: '{tmp_path / 'jmds'}'",
                f"data_root: '{tmp_path / 'data'}'",
                f"asvspoof_protocol_root: '{tmp_path / 'asv'}'",
            )
        ),
        encoding="utf-8",
    )

    loaded = PreparationConfig.load(config_path)

    assert legacy.asvspoof_protocol_root is None
    assert loaded.asvspoof_protocol_root == tmp_path / "asv"


def test_validate_protocols_requires_asv_root(tmp_path):
    config = replace(_config(tmp_path), asvspoof_protocol_root=None)

    with pytest.raises(ValueError, match="asvspoof_protocol_root"):
        cli.validate_protocols(config)


def test_correspondence_selects_jmds_ids_from_official_asv_superset(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    protocols = {
        split: _protocol(
            split,
            (("T" if split == "train" else "D") + "_0000000001",),
        )
        for split in ("train", "dev")
    }
    asv_protocols = {}
    for split, jmds in protocols.items():
        prefix = "T" if split == "train" else "D"
        matching = jmds.iloc[0]
        asv_protocols[split] = pd.DataFrame(
            [
                [
                    matching["spk_id"],
                    matching["utt_id"],
                    matching["gender"],
                    "-",
                    "-",
                    "-",
                    "-",
                    "bonafide",
                    "bonafide",
                    "-",
                ],
                [
                    f"{prefix}_9999",
                    f"{prefix}_0000000002",
                    "M",
                    "-",
                    "-",
                    "-",
                    "-",
                    "A01",
                    "spoof",
                    "-",
                ],
            ],
            columns=ASV_COLUMNS,
        )
    monkeypatch.setattr(
        cli,
        "read_jmds_protocol",
        lambda path, split, **kwargs: protocols[split],
    )
    monkeypatch.setattr(
        cli,
        "read_asvspoof_protocol",
        lambda path, split: asv_protocols[split],
    )
    monkeypatch.setattr(cli, "EXPECTED_PRISTINE", {"train": 1, "dev": 1})
    monkeypatch.setattr(cli, "EXPECTED_GENERATED", {"train": 0, "dev": 0})

    reports = cli._compute_correspondence(config)

    assert reports["train"].extra_ids == ()
    assert reports["dev"].extra_ids == ()
    assert reports["train"].pristine_matched == 1
    assert reports["dev"].pristine_matched == 1


def test_validate_protocols_writes_atomic_report_and_prints_counts(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    config.jmds_root.joinpath("cm_protocols").mkdir(parents=True)
    config.asvspoof_protocol_root.mkdir(parents=True)
    for split in ("train", "dev", "eval"):
        config.jmds_root.joinpath(
            "cm_protocols", f"open_v2_{split}.cm.csv"
        ).touch()
        config.asvspoof_protocol_root.joinpath(
            cli.ASV_PROTOCOL_FILES[split]
        ).touch()
    config.asvspoof_protocol_root.joinpath(
        cli.ASV_PROTOCOL_FILES["eval"]
    ).touch()

    reports = {"train": _report("train", 18_797, 65_424)}
    reports["dev"] = _report("dev", 15_667, 54_808)
    monkeypatch.setattr(cli, "_compute_correspondence", lambda *_: reports)
    monkeypatch.setattr(
        cli, "_compute_eval_evidence", lambda *_: _eval_evidence()
    )
    messages: list[str] = []

    result = cli.validate_protocols(config, output=messages.append)

    destination = config.data_root / "reports" / "correspondence.json"
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert result == reports
    assert payload["splits"]["train"]["pristine_matched"] == 18_797
    assert payload["splits"]["dev"]["generated_matched"] == 54_808
    assert payload["eval"]["status"] == "excluded_known_id_mismatch"
    assert payload["eval"]["jmds_fingerprint"] == "jmds-eval"
    assert payload["eval"]["asv_fingerprint"] == "asv-eval"
    assert payload["eval"]["asv_only_count"] == 1
    assert "missing_ids" not in payload["eval"]
    assert messages == [
        "train: pristine matched 18797; generated matched 65424",
        "dev: pristine matched 15667; generated matched 54808",
        "eval: excluded (known ID mismatch)",
    ]
    assert not list(destination.parent.glob("*.tmp"))


def test_eval_evidence_has_canonical_fingerprints_and_summary(tmp_path):
    config = _config(tmp_path)
    jmds_root = config.jmds_root / "cm_protocols"
    jmds_root.mkdir(parents=True)
    config.asvspoof_protocol_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "spk_id": "E_0001",
                "utt_id": "E_0000000001",
                "gender": "F",
                "codec": "-",
                "attack_id": "pristine",
                "label": "pristine",
                "native": "yes",
                "language": "eng",
                "dataset": "ASVspoof2024",
            },
            {
                "spk_id": "E_0002",
                "utt_id": "E_0000000002",
                "gender": "M",
                "codec": "-",
                "attack_id": "A01",
                "label": "generated",
                "native": "yes",
                "language": "eng",
                "dataset": "ASVspoof2024",
            },
        ],
        columns=JMDS_COLUMNS,
    ).to_csv(jmds_root / "open_v2_eval.cm.csv", index=False)
    (
        config.asvspoof_protocol_root / cli.ASV_PROTOCOL_FILES["eval"]
    ).write_text(
        "\n".join(
            (
                "E_0001 E_0000000001 F - - - - bonafide bonafide -",
                "E_9999 E_0000000009 M - - - - A01 spoof -",
            )
        ),
        encoding="utf-8",
    )

    evidence = cli._compute_eval_evidence(config)

    assert len(evidence["jmds_fingerprint"]) == 64
    assert len(evidence["asv_fingerprint"]) == 64
    assert evidence["counts"]["jmds"] == {"pristine": 1, "generated": 1}
    assert evidence["counts"]["asv"] == {"bonafide": 1, "spoof": 1}
    assert evidence["overlap_by_label"] == {
        "pristine_bonafide": 1,
        "generated_spoof": 0,
    }
    assert evidence["jmds_only_count"] == 1
    assert evidence["asv_only_count"] == 1


def test_eval_evidence_refuses_obsolete_mismatch_status(tmp_path):
    config = _config(tmp_path)
    jmds_root = config.jmds_root / "cm_protocols"
    jmds_root.mkdir(parents=True)
    config.asvspoof_protocol_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "spk_id": "E_0001",
                "utt_id": "E_0000000001",
                "gender": "F",
                "codec": "-",
                "attack_id": "pristine",
                "label": "pristine",
                "native": "yes",
                "language": "eng",
                "dataset": "ASVspoof2024",
            },
            {
                "spk_id": "E_0002",
                "utt_id": "E_0000000002",
                "gender": "M",
                "codec": "-",
                "attack_id": "A01",
                "label": "generated",
                "native": "yes",
                "language": "eng",
                "dataset": "ASVspoof2024",
            },
        ],
        columns=JMDS_COLUMNS,
    ).to_csv(jmds_root / "open_v2_eval.cm.csv", index=False)
    (
        config.asvspoof_protocol_root / cli.ASV_PROTOCOL_FILES["eval"]
    ).write_text(
        "\n".join(
            (
                "E_0001 E_0000000001 F - - - - bonafide bonafide -",
                "E_0002 E_0000000002 M - - - - A01 spoof -",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="correspond.*review|review.*correspond"):
        cli._compute_eval_evidence(config)


def test_acquire_rejects_correspondence_when_eval_fingerprint_changes(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    reports = {"train": _report("train"), "dev": _report("dev")}
    old_eval = _eval_evidence()
    destination = config.data_root / "reports" / "correspondence.json"
    destination.parent.mkdir(parents=True)
    destination.write_text(
        json.dumps(
            {
                "jmds_root": str(config.jmds_root.resolve()),
                "asvspoof_protocol_root": str(
                    config.asvspoof_protocol_root.resolve()
                ),
                "splits": {
                    split: {
                        **report.__dict__,
                        "missing_ids": [],
                        "extra_ids": [],
                    }
                    for split, report in reports.items()
                },
                "eval": old_eval,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "_require_protocol_paths", lambda _: None)
    monkeypatch.setattr(cli, "_compute_correspondence", lambda _: reports)
    monkeypatch.setattr(
        cli,
        "_compute_eval_evidence",
        lambda _: _eval_evidence(jmds_fingerprint="changed"),
    )

    with pytest.raises(ValueError, match="eval.*stale|stale.*eval"):
        cli._load_current_correspondence(config)


def test_dry_run_lists_plan_without_mutating_data_root(tmp_path, monkeypatch):
    config = _config(tmp_path)
    protocols = {
        "train": _protocol("train", ("T_0000000001",)),
        "dev": _protocol("dev", ("D_0000000001",)),
    }
    monkeypatch.setattr(
        cli,
        "_load_current_correspondence",
        lambda _: ({"train": _report("train"), "dev": _report("dev")}, protocols),
    )
    monkeypatch.setattr(cli, "get_record_files", lambda _: _record_files())
    usage = type("Usage", (), {"free": 900})()
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda _: usage)
    messages: list[str] = []

    cli.acquire_english(config, dry_run=True, output=messages.append)

    text = "\n".join(messages)
    assert not config.data_root.exists()
    assert all(name in text for name in REQUIRED_ARCHIVES)
    assert "exact TAR download bytes: 108" in text
    assert "maximum active TAR working set bytes" in text
    assert "conservative total acquisition peak bytes: 125" in text
    assert "generated processed PCM-16 payload bytes" in text
    assert "status=pending until extraction" in text
    assert (
        "process-english checks free space for the missing PCM-16 outputs "
        "before writing any audio or manifest"
    ) in text
    assert "free space: 900" in text
    assert "target=1 existing_valid=0 outstanding=1" in text
    assert str(config.data_root / "archives") in text
    assert str(config.data_root / "raw" / "asvspoof5") in text


def test_valid_extracted_ids_requires_decodable_flac(tmp_path):
    output = tmp_path / "raw"
    output.mkdir()
    valid_id = "T_0000000001"
    invalid_id = "T_0000000002"
    sf.write(
        output / f"{valid_id}.flac",
        np.zeros(160, dtype=np.float32),
        16_000,
        format="FLAC",
    )
    (output / f"{invalid_id}.flac").write_bytes(b"not audio")

    valid = cli._valid_extracted_ids(output, {valid_id, invalid_id})

    assert valid == {valid_id}


def test_valid_extracted_ids_rejects_truncated_flac_after_header(tmp_path):
    output = tmp_path / "raw"
    output.mkdir()
    utt_id = "T_0000000001"
    path = output / f"{utt_id}.flac"
    samples = np.sin(
        np.linspace(0.0, 100.0 * np.pi, 32_000, dtype=np.float64)
    )
    sf.write(path, samples, 16_000, format="FLAC")
    encoded = path.read_bytes()
    path.write_bytes(encoded[:-64])
    assert sf.info(path).frames == len(samples)

    valid = cli._valid_extracted_ids(output, {utt_id})

    assert valid == set()


def test_real_acquisition_stops_and_preserves_archive_on_extraction_failure(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    protocols = {
        "train": _protocol("train", ("T_0000000001",)),
        "dev": _protocol("dev", ("D_0000000001",)),
    }
    monkeypatch.setattr(
        cli,
        "_load_current_correspondence",
        lambda _: ({"train": _report("train"), "dev": _report("dev")}, protocols),
    )
    files = _record_files()
    monkeypatch.setattr(cli, "get_record_files", lambda _: files)
    downloaded: list[str] = []

    def download(file, destination):
        downloaded.append(file.name)
        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / file.name
        archive.write_bytes(b"tar")
        return archive

    monkeypatch.setattr(cli, "download_verified", download)
    monkeypatch.setattr(
        cli,
        "inventory_archive",
        lambda _: ArchiveInventory(("T_0000000001",), (), ()),
    )
    monkeypatch.setattr(
        cli,
        "extract_selected",
        lambda *_: (_ for _ in ()).throw(RuntimeError("broken TAR")),
    )

    with pytest.raises(RuntimeError, match="broken TAR"):
        cli.acquire_english(config, dry_run=False)

    first = REQUIRED_ARCHIVES[0]
    assert downloaded == [first]
    assert (config.data_root / "archives" / first).is_file()


def test_acquisition_distributes_targets_across_archives_and_removes_empty_tar(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    train_ids = ("T_0000000001", "T_0000000002")
    dev_ids = ("D_0000000001",)
    protocols = {
        "train": _protocol("train", train_ids),
        "dev": _protocol("dev", dev_ids),
    }
    reports = {"train": _report("train"), "dev": _report("dev")}
    monkeypatch.setattr(
        cli, "_load_current_correspondence", lambda _: (reports, protocols)
    )
    monkeypatch.setattr(cli, "get_record_files", lambda _: _record_files())
    monkeypatch.setattr(cli, "EXPECTED_PRISTINE", {"train": 2, "dev": 1})
    inventory_ids = {
        "flac_T_aa.tar": (train_ids[0],),
        "flac_T_ab.tar": ("T_9999999999",),
        "flac_T_ac.tar": (train_ids[1],),
        "flac_D_aa.tar": (dev_ids[0],),
    }
    events: list[tuple[str, str, tuple[str, ...]]] = []

    def download(file, destination):
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / file.name
        path.write_bytes(b"tar")
        return path

    def inventory(archive):
        ids = inventory_ids.get(archive.name, ())
        events.append(("inventory", archive.name, ids))
        return ArchiveInventory(ids, (), ())

    def extract(archive, wanted_ids, output):
        selected = tuple(sorted(wanted_ids))
        events.append(("extract", archive.name, selected))
        output.mkdir(parents=True, exist_ok=True)
        for utt_id in selected:
            (output / f"{utt_id}.flac").write_bytes(b"audio")
        return ExtractionReport(selected, (), (), ())

    monkeypatch.setattr(cli, "download_verified", download)
    monkeypatch.setattr(cli, "inventory_archive", inventory)
    monkeypatch.setattr(cli, "extract_selected", extract)
    monkeypatch.setattr(
        cli,
        "_valid_extracted_ids",
        lambda output, targets: {
            utt_id
            for utt_id in targets
            if (output / f"{utt_id}.flac").is_file()
        },
    )
    monkeypatch.setattr(cli, "build_raw_manifest", lambda *_: pd.DataFrame())
    monkeypatch.setattr(cli, "write_manifest_atomic", lambda *_: None)

    cli.acquire_english(config, dry_run=False)

    assert ("extract", "flac_T_aa.tar", (train_ids[0],)) in events
    assert ("extract", "flac_T_ab.tar", ()) in events
    assert ("extract", "flac_T_ac.tar", (train_ids[1],)) in events
    assert ("extract", "flac_D_aa.tar", (dev_ids[0],)) in events
    assert not (config.data_root / "archives" / "flac_T_ab.tar").exists()


def test_acquisition_preserves_last_tar_when_target_never_appears(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    missing_id = "T_0000000001"
    dev_id = "D_0000000001"
    protocols = {
        "train": _protocol("train", (missing_id,)),
        "dev": _protocol("dev", (dev_id,)),
    }
    reports = {"train": _report("train"), "dev": _report("dev")}
    monkeypatch.setattr(
        cli, "_load_current_correspondence", lambda _: (reports, protocols)
    )
    monkeypatch.setattr(cli, "get_record_files", lambda _: _record_files())
    monkeypatch.setattr(cli, "EXPECTED_PRISTINE", {"train": 1, "dev": 1})

    def download(file, destination):
        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / file.name
        archive.write_bytes(b"tar")
        return archive

    def inventory(archive):
        ids = (dev_id,) if archive.name == "flac_D_aa.tar" else ()
        return ArchiveInventory(ids, (), ())

    def extract(archive, wanted_ids, output):
        selected = tuple(sorted(wanted_ids))
        output.mkdir(parents=True, exist_ok=True)
        for utt_id in selected:
            (output / f"{utt_id}.flac").write_bytes(b"audio")
        return ExtractionReport(selected, (), (), ())

    monkeypatch.setattr(cli, "download_verified", download)
    monkeypatch.setattr(cli, "inventory_archive", inventory)
    monkeypatch.setattr(cli, "extract_selected", extract)
    monkeypatch.setattr(
        cli,
        "_valid_extracted_ids",
        lambda output, targets: {
            utt_id
            for utt_id in targets
            if (output / f"{utt_id}.flac").is_file()
        },
    )

    with pytest.raises(ValueError, match="never appeared") as error:
        cli.acquire_english(config, dry_run=False)

    assert missing_id in str(error.value)
    archive_root = config.data_root / "archives"
    assert (archive_root / REQUIRED_ARCHIVES[-1]).is_file()
    assert not (archive_root / REQUIRED_ARCHIVES[-2]).exists()


@pytest.mark.parametrize(
    "inventory",
    [
        ArchiveInventory(("T_0000000001",), ("T_0000000001",), ()),
        ArchiveInventory(("T_0000000001",), (), ("../unsafe.flac",)),
    ],
)
def test_acquisition_preserves_tar_when_inventory_is_unsafe(
    tmp_path, monkeypatch, inventory
):
    config = _config(tmp_path)
    protocols = {
        "train": _protocol("train", ("T_0000000001",)),
        "dev": _protocol("dev", ("D_0000000001",)),
    }
    monkeypatch.setattr(
        cli,
        "_load_current_correspondence",
        lambda _: ({"train": _report("train"), "dev": _report("dev")}, protocols),
    )
    monkeypatch.setattr(cli, "get_record_files", lambda _: _record_files())

    def download(file, destination):
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / file.name
        path.write_bytes(b"tar")
        return path

    monkeypatch.setattr(cli, "download_verified", download)
    monkeypatch.setattr(cli, "inventory_archive", lambda _: inventory)
    monkeypatch.setattr(
        cli,
        "extract_selected",
        lambda *_: pytest.fail("extraction must not start"),
    )

    with pytest.raises(ValueError, match="Duplicate|Unsafe"):
        cli.acquire_english(config, dry_run=False)

    first = config.data_root / "archives" / REQUIRED_ARCHIVES[0]
    assert first.is_file()


def test_extraction_validation_requires_exact_expected_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli,
        "_valid_extracted_ids",
        lambda output, targets: set(targets),
    )
    report = ExtractionReport(("T_0000000001",), (), (), ())

    with pytest.raises(ValueError, match="exact"):
        cli._validate_extraction(
            report,
            {"T_0000000001", "T_0000000002"},
            tmp_path,
        )


def test_real_acquisition_deletes_only_after_validation_and_builds_manifest(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    train_id = "T_0000000001"
    dev_id = "D_0000000001"
    protocols = {
        "train": _protocol("train", (train_id,)),
        "dev": _protocol("dev", (dev_id,)),
    }
    reports = {"train": _report("train"), "dev": _report("dev")}
    monkeypatch.setattr(
        cli, "_load_current_correspondence", lambda _: (reports, protocols)
    )
    monkeypatch.setattr(cli, "get_record_files", lambda _: _record_files())
    monkeypatch.setattr(cli, "EXPECTED_PRISTINE", {"train": 1, "dev": 1})
    events: list[tuple[str, str]] = []
    current_archive = {"name": ""}

    def download(file, destination):
        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / file.name
        archive.write_bytes(b"tar")
        events.append(("download", file.name))
        return archive

    def extract(archive, wanted_ids, output):
        current_archive["name"] = archive.name
        output.mkdir(parents=True, exist_ok=True)
        extracted = tuple(sorted(wanted_ids))
        for utt_id in extracted:
            (output / f"{utt_id}.flac").write_bytes(b"audio")
        events.append(("extract", archive.name))
        return ExtractionReport(extracted, (), (), ())

    monkeypatch.setattr(cli, "download_verified", download)
    monkeypatch.setattr(
        cli,
        "inventory_archive",
        lambda archive: ArchiveInventory(
            (train_id,) if archive.name.startswith("flac_T_") else (dev_id,),
            (),
            (),
        ),
    )
    monkeypatch.setattr(cli, "extract_selected", extract)
    monkeypatch.setattr(
        cli,
        "_valid_extracted_ids",
        lambda output, targets: {
            utt_id
            for utt_id in targets
            if (output / f"{utt_id}.flac").is_file()
        },
    )
    original_validate = cli._validate_extraction

    def validate(report, wanted_ids, output):
        events.append(("validate", current_archive["name"]))
        original_validate(report, wanted_ids, output)

    monkeypatch.setattr(cli, "_validate_extraction", validate)
    monkeypatch.setattr(
        cli,
        "update_extraction_ledger",
        lambda *_: events.append(("ledger", current_archive["name"])),
    )
    original_unlink = Path.unlink

    def tracked_unlink(path, *args, **kwargs):
        if path.suffix == ".tar":
            events.append(("delete", path.name))
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", tracked_unlink)
    raw_manifest = pd.DataFrame({"ok": [1]})
    monkeypatch.setattr(cli, "build_raw_manifest", lambda *_: raw_manifest)
    written: list[tuple[pd.DataFrame, Path]] = []
    monkeypatch.setattr(
        cli, "write_manifest_atomic", lambda frame, path: written.append((frame, path))
    )

    cli.acquire_english(config, dry_run=False)

    processed_archives = (
        next(name for name in REQUIRED_ARCHIVES if name.startswith("flac_T_")),
        next(name for name in REQUIRED_ARCHIVES if name.startswith("flac_D_")),
    )
    for archive in processed_archives:
        archive_events = [kind for kind, name in events if name == archive]
        assert archive_events == [
            "download",
            "extract",
            "validate",
            "ledger",
            "delete",
        ]
    assert not any(
        kind == "download" and name not in processed_archives
        for kind, name in events
    )
    assert written == [
        (raw_manifest, config.data_root / "manifests" / "english_raw.csv")
    ]


def test_process_and_audit_commands_delegate_to_existing_modules(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    raw = pd.DataFrame({"raw": [1]})
    processed = pd.DataFrame({"processed": [1]})
    reads = iter((raw, processed))
    monkeypatch.setattr(cli.pd, "read_csv", lambda *_, **__: next(reads))
    process_calls = []
    monkeypatch.setattr(
        cli,
        "process_manifest",
        lambda *args: process_calls.append(args) or processed,
    )
    audit_calls = []
    monkeypatch.setattr(
        cli,
        "audit_and_publish",
        lambda *args, **kwargs: audit_calls.append((args, kwargs)),
    )

    cli.process_english(config)
    cli.audit_english(config)

    assert process_calls[0][0] is raw
    assert process_calls[0][1:] == (
        config.data_root,
        config.data_root / "manifests" / "english_processed.csv",
        config.sample_rate,
    )
    assert audit_calls[0][0] == (processed, config.data_root / "reports")


def test_main_dispatches_and_parser_returns_nonzero_for_bad_command(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    config = _config(tmp_path)
    monkeypatch.setattr(cli.PreparationConfig, "load", lambda _: config)
    monkeypatch.setattr(cli.PreparationConfig, "validate", lambda _: None)
    called = []
    monkeypatch.setattr(
        cli, "acquire_english", lambda cfg, *, dry_run: called.append((cfg, dry_run))
    )

    assert (
        cli.main(["acquire-english", "--config", str(config_path), "--dry-run"])
        == 0
    )
    assert called == [(config, True)]
    with pytest.raises(SystemExit) as error:
        cli.main(["not-a-command", "--config", str(config_path)])
    assert error.value.code == 2


def test_main_validates_config_before_dispatch(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    config = _config(tmp_path)
    monkeypatch.setattr(cli.PreparationConfig, "load", lambda _: config)
    monkeypatch.setattr(
        config.__class__,
        "validate",
        lambda _self: (_ for _ in ()).throw(ValueError("invalid config")),
    )
    called = []
    monkeypatch.setattr(cli, "validate_protocols", lambda *_: called.append(True))

    with pytest.raises(ValueError, match="invalid config"):
        cli.main(["validate-protocols", "--config", str(config_path)])

    assert called == []

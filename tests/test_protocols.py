from pathlib import Path

import pandas as pd
import pytest

from jmds_prepare.protocols import (
    ASV_COLUMNS,
    CorrespondenceReport,
    JMDS_COLUMNS,
    protocol_fingerprint,
    read_asvspoof_protocol,
    read_jmds_protocol,
    validate_correspondence,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_reads_strict_jmds_schema():
    protocol = read_jmds_protocol(FIXTURES / "jmds_train.csv", "train")

    assert list(protocol.columns) == JMDS_COLUMNS
    assert protocol["utt_id"].tolist() == ["T_0000000001", "T_0000000002"]


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        (JMDS_COLUMNS[:-1], "missing.*dataset"),
        (JMDS_COLUMNS + ["duration"], "extra.*duration"),
    ],
)
def test_rejects_missing_or_extra_jmds_columns(tmp_path, columns, message):
    path = tmp_path / "jmds.csv"
    pd.DataFrame(columns=columns).to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        read_jmds_protocol(path, "train")


def test_rejects_reordered_jmds_columns(tmp_path):
    protocol = pd.read_csv(FIXTURES / "jmds_train.csv", dtype=str)
    reordered = [JMDS_COLUMNS[1], JMDS_COLUMNS[0], *JMDS_COLUMNS[2:]]
    path = tmp_path / "jmds.csv"
    protocol[reordered].to_csv(path, index=False)

    with pytest.raises(ValueError, match="required order"):
        read_jmds_protocol(path, "train")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("utt_id", "bad-id", "Malformed.*utt_id"),
        ("label", "fake", "Unsupported.*label"),
    ],
)
def test_rejects_invalid_jmds_values(tmp_path, field, value, message):
    protocol = pd.read_csv(FIXTURES / "jmds_train.csv", dtype=str)
    protocol.loc[0, field] = value
    path = tmp_path / "jmds.csv"
    protocol.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        read_jmds_protocol(path, "train")


def test_rejects_duplicate_jmds_utterance_ids(tmp_path):
    protocol = pd.read_csv(FIXTURES / "jmds_train.csv", dtype=str)
    protocol.loc[1, "utt_id"] = protocol.loc[0, "utt_id"]
    path = tmp_path / "jmds.csv"
    protocol.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Duplicate.*utt_id"):
        read_jmds_protocol(path, "train")


def test_language_filter_precedes_duplicate_validation(tmp_path):
    protocol = pd.read_csv(FIXTURES / "jmds_train.csv", dtype=str)
    reused = protocol.iloc[[0]].copy()
    reused.loc[:, "language"] = "jpn"
    reused.loc[:, "dataset"] = "MLAAD"
    path = tmp_path / "multilingual.csv"
    pd.concat([reused, protocol], ignore_index=True).to_csv(path, index=False)

    english = read_jmds_protocol(path, "train", language="eng")

    assert english["language"].tolist() == ["eng", "eng"]
    assert english["utt_id"].is_unique
    with pytest.raises(ValueError, match="Duplicate.*utt_id"):
        read_jmds_protocol(path, "train")


def test_reads_whitespace_delimited_asvspoof_protocol():
    protocol = read_asvspoof_protocol(
        FIXTURES / "asvspoof_train.tsv", "train"
    )

    assert list(protocol.columns) == ASV_COLUMNS
    assert protocol["key"].tolist() == ["bonafide", "spoof"]


def test_reads_and_validates_dev_protocols_with_d_ids(tmp_path):
    jmds_path = tmp_path / "jmds_dev.csv"
    jmds_path.write_text(
        (FIXTURES / "jmds_train.csv")
        .read_text(encoding="utf-8")
        .replace("T_", "D_"),
        encoding="utf-8",
    )
    asv_path = tmp_path / "asvspoof_dev.txt"
    asv_path.write_text(
        (FIXTURES / "asvspoof_train.tsv")
        .read_text(encoding="utf-8")
        .replace("T_", "D_")
        .replace("D_9999", "D_0002"),
        encoding="utf-8",
    )

    jmds = read_jmds_protocol(jmds_path, "dev")
    asv = read_asvspoof_protocol(asv_path, "dev")
    report = validate_correspondence(jmds, asv, "dev")

    assert jmds["utt_id"].tolist() == ["D_0000000001", "D_0000000002"]
    assert report.pristine_matched == 1
    assert report.generated_matched == 1
    report.raise_for_errors()


@pytest.mark.parametrize(
    ("reader", "path"),
    [
        (read_jmds_protocol, FIXTURES / "jmds_train.csv"),
        (read_asvspoof_protocol, FIXTURES / "asvspoof_train.tsv"),
    ],
)
def test_rejects_train_prefix_in_dev_protocol(reader, path):
    with pytest.raises(ValueError, match="Malformed.*dev"):
        reader(path, "dev")


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                "T_0001 T_0000000001 F - - - - pristine bonafide -",
                "T_0002 T_0000000001 M - - - A01 A01 spoof -",
            ],
            "Duplicate.*utt_id",
        ),
        (
            ["T_0001 invalid F - - - - pristine bonafide -"],
            "Malformed.*utt_id",
        ),
        (
            ["T_0001 T_0000000001 F - - - - pristine unknown -"],
            "Unsupported.*key",
        ),
        (
            ["T_0001 T_0000000001 F - - - - pristine bonafide"],
            "10 fields",
        ),
    ],
)
def test_rejects_invalid_asvspoof_rows(tmp_path, rows, message):
    path = tmp_path / "asv.txt"
    path.write_text("\n".join(rows), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_asvspoof_protocol(path, "train")


def test_reports_exact_correspondence_and_all_metadata_mismatches():
    jmds = read_jmds_protocol(FIXTURES / "jmds_train.csv", "train")
    asv = read_asvspoof_protocol(FIXTURES / "asvspoof_train.tsv", "train")

    report = validate_correspondence(jmds, asv, "train")

    assert report.pristine_expected == 1
    assert report.generated_expected == 1
    assert report.pristine_matched == 1
    assert report.generated_matched == 1
    assert report.missing_ids == ()
    assert report.extra_ids == ()
    assert report.metadata_mismatches == {
        "T_0000000002": {"spk_id": ("T_0002", "T_9999")}
    }
    assert report.jmds_fingerprint == protocol_fingerprint(jmds, "train")
    assert report.asv_fingerprint == protocol_fingerprint(asv, "train")
    with pytest.raises(ValueError, match="T_0000000002.*spk_id"):
        report.raise_for_errors()


def test_protocol_fingerprint_is_canonical_and_content_bound():
    jmds = read_jmds_protocol(FIXTURES / "jmds_train.csv", "train")
    reordered_rows = jmds.iloc[::-1].reset_index(drop=True)
    changed_value = jmds.copy()
    changed_value.loc[0, "gender"] = "M"
    reordered_schema = jmds[[JMDS_COLUMNS[1], JMDS_COLUMNS[0], *JMDS_COLUMNS[2:]]]

    fingerprint = protocol_fingerprint(jmds, "train")

    assert len(fingerprint) == 64
    assert protocol_fingerprint(reordered_rows, "train") == fingerprint
    assert protocol_fingerprint(changed_value, "train") != fingerprint
    assert protocol_fingerprint(jmds, "dev") != fingerprint
    assert protocol_fingerprint(reordered_schema, "train") != fingerprint


def test_reports_missing_extra_gender_and_label_differences():
    jmds = read_jmds_protocol(FIXTURES / "jmds_train.csv", "train")
    asv = read_asvspoof_protocol(FIXTURES / "asvspoof_train.tsv", "train")
    asv.loc[0, ["gender", "key"]] = ["M", "spoof"]
    asv.loc[1, "utt_id"] = "T_0000000003"

    report = validate_correspondence(jmds, asv, "train")

    assert report.pristine_matched == 0
    assert report.generated_matched == 0
    assert report.missing_ids == ("T_0000000002",)
    assert report.extra_ids == ("T_0000000003",)
    assert report.metadata_mismatches == {
        "T_0000000001": {
            "gender": ("F", "M"),
            "label": ("pristine", "spoof"),
        }
    }


def test_raise_for_errors_aggregates_multiple_differences():
    jmds = read_jmds_protocol(FIXTURES / "jmds_train.csv", "train")
    asv = read_asvspoof_protocol(FIXTURES / "asvspoof_train.tsv", "train")
    asv.loc[0, ["gender", "key"]] = ["M", "spoof"]
    asv.loc[1, "utt_id"] = "T_0000000003"

    with pytest.raises(ValueError) as error:
        validate_correspondence(jmds, asv, "train").raise_for_errors()

    message = str(error.value)
    assert "missing IDs: T_0000000002" in message
    assert "extra IDs: T_0000000003" in message
    assert "T_0000000001 metadata mismatch: gender, label" in message


def test_correspondence_report_is_mutable():
    jmds = read_jmds_protocol(FIXTURES / "jmds_train.csv", "train")
    asv = read_asvspoof_protocol(FIXTURES / "asvspoof_train.tsv", "train")
    report = validate_correspondence(jmds, asv, "train")

    report.split = "dev"

    assert report.split == "dev"


def test_correspondence_report_preserves_legacy_construction():
    positional = CorrespondenceReport(
        "train",
        1,
        2,
        1,
        2,
        (),
        (),
        {},
    )
    keyword = CorrespondenceReport(
        split="dev",
        pristine_expected=3,
        generated_expected=4,
        pristine_matched=3,
        generated_matched=4,
        missing_ids=(),
        extra_ids=(),
        metadata_mismatches={},
    )

    assert positional.pristine_expected == 1
    assert positional.generated_expected == 2
    assert positional.jmds_fingerprint == ""
    assert positional.asv_fingerprint == ""
    assert keyword.pristine_expected == 3
    assert keyword.generated_expected == 4
    assert keyword.jmds_fingerprint == ""
    assert keyword.asv_fingerprint == ""


def test_accepts_report_without_differences():
    jmds = read_jmds_protocol(FIXTURES / "jmds_train.csv", "train")
    asv = read_asvspoof_protocol(FIXTURES / "asvspoof_train.tsv", "train")
    asv.loc[1, "spk_id"] = "T_0002"

    validate_correspondence(jmds, asv, "train").raise_for_errors()


def test_rejects_non_english_jmds_rows():
    jmds = read_jmds_protocol(FIXTURES / "jmds_train.csv", "train")
    asv = read_asvspoof_protocol(FIXTURES / "asvspoof_train.tsv", "train")
    jmds.loc[0, "language"] = "jpn"

    with pytest.raises(ValueError, match="English.*jpn"):
        validate_correspondence(jmds, asv, "train")


@pytest.mark.parametrize("split", ["eval", "test"])
def test_rejects_unsupported_splits(split):
    with pytest.raises(ValueError, match=split):
        read_jmds_protocol(FIXTURES / "jmds_train.csv", split)

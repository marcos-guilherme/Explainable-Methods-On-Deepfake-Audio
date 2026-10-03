"""Tests for the standalone Mandarin metadata CLI."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from jmds_prepare import mandarin_metadata
from jmds_prepare.core.publication import DestinationConflictError

FIXTURES = Path(__file__).parent / "fixtures"
AISHELL_FIXTURE = FIXTURES / "aishell3_mini.tgz"
JMDS_HEADER = (
    FIXTURES / "jmds_add_metadata.csv"
).read_text(encoding="utf-8").splitlines()[0] + "\n"


def _valid_roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    jmds_root = tmp_path / "jmds"
    archive_dir = tmp_path / "sources" / "aishell3"
    output_root = tmp_path / "output"
    archive_dir.mkdir(parents=True)
    aishell_archive = archive_dir / "data_aishell3.tgz"
    shutil.copy(AISHELL_FIXTURE, aishell_archive)

    for split in ("train", "dev", "eval"):
        protocol = jmds_root / "cm_protocols" / f"open_v2_{split}.cm.csv"
        protocol.parent.mkdir(parents=True, exist_ok=True)
        protocol.write_text(JMDS_HEADER, encoding="utf-8")

    output_root.mkdir()
    return jmds_root, aishell_archive, output_root


def _write_config(
    path: Path,
    *,
    jmds_root: Path,
    aishell_archive: Path,
    output_root: Path,
) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "jmds_root": str(jmds_root),
                "aishell_archive": str(aishell_archive),
                "output_root": str(output_root),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_cli_accepts_required_config_and_returns_zero(tmp_path, monkeypatch):
    jmds_root, aishell_archive, output_root = _valid_roots(tmp_path)
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        jmds_root=jmds_root,
        aishell_archive=aishell_archive,
        output_root=output_root,
    )
    monkeypatch.setattr(
        mandarin_metadata,
        "extract_mandarin_metadata",
        lambda *_args, **_kwargs: None,
    )

    assert mandarin_metadata.main(["--config", str(config_path)]) == 0


def test_cli_prints_error_and_returns_one_for_missing_config(tmp_path, capsys):
    missing = tmp_path / "missing.yaml"

    code = mandarin_metadata.main(["--config", str(missing)])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out.startswith("error:")


def test_cli_prints_error_for_invalid_config(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("not: [a, mapping, of, scalars]\n", encoding="utf-8")

    code = mandarin_metadata.main(["--config", str(config_path)])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out.startswith("error:")
    assert "Traceback" not in captured.out + captured.err


def test_cli_prints_error_for_syntactically_invalid_yaml(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("jmds_root: [\n", encoding="utf-8")

    code = mandarin_metadata.main(["--config", str(config_path)])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out.startswith("error:")
    assert "Traceback" not in captured.out + captured.err


def test_cli_prints_error_when_required_input_missing(tmp_path, capsys):
    jmds_root, aishell_archive, output_root = _valid_roots(tmp_path)
    aishell_archive.unlink()
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        jmds_root=jmds_root,
        aishell_archive=aishell_archive,
        output_root=output_root,
    )

    code = mandarin_metadata.main(["--config", str(config_path)])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out.startswith("error:")
    assert "aishell_archive" in captured.out
    assert "Traceback" not in captured.out + captured.err


def test_cli_prints_error_on_destination_conflict(tmp_path, monkeypatch, capsys):
    jmds_root, aishell_archive, output_root = _valid_roots(tmp_path)
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        jmds_root=jmds_root,
        aishell_archive=aishell_archive,
        output_root=output_root,
    )

    def raise_conflict(*_args, **_kwargs):
        raise DestinationConflictError("destination conflict")

    monkeypatch.setattr(
        mandarin_metadata,
        "extract_mandarin_metadata",
        raise_conflict,
    )

    code = mandarin_metadata.main(["--config", str(config_path)])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out.startswith("error:")
    assert "destination conflict" in captured.out
    assert "Traceback" not in captured.out + captured.err


def test_cli_prints_one_line_for_expected_os_error(tmp_path, monkeypatch, capsys):
    jmds_root, aishell_archive, output_root = _valid_roots(tmp_path)
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        jmds_root=jmds_root,
        aishell_archive=aishell_archive,
        output_root=output_root,
    )

    def raise_os_error(*_args, **_kwargs):
        raise OSError("disk unavailable\nretry later")

    monkeypatch.setattr(
        mandarin_metadata,
        "extract_mandarin_metadata",
        raise_os_error,
    )

    assert mandarin_metadata.main(["--config", str(config_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["error: disk unavailable retry later"]
    assert captured.err == ""


def test_module_entrypoint_is_invokable():
    completed = subprocess.run(
        [sys.executable, "-m", "jmds_prepare.mandarin_metadata", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--config" in completed.stdout

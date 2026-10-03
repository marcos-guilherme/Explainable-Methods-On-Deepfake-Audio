# Mandarin Metadata Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce six local, reproducible Mandarin metadata artifacts from
AISHELL-3 (pristine, streamed from `data_aishell3.tgz`) and JMDS/ADD
(generated), without extracting WAVs, without claiming sample-level pairing,
and without duplicating the Portuguese config/layout/pipeline stack.

**Architecture:** First extract profile-parameterized two-source primitives
(artifact names, layout, config validation helpers, dependency-injected
pipeline orchestration, shared profile/provenance field builders) from the
existing Portuguese implementation while keeping all 583 Portuguese tests
green via thin compatibility wrappers. Then add `MandarinProfile`, a streaming
AISHELL-3 adapter with explicit metadata–WAV reconciliation, a JMDS/ADD
generated adapter scoped to `zho/generated/ADD`, Mandarin-specific
profile/summary/provenance builders, and a standalone CLI wired through the
generic pipeline.

**Tech Stack:** Python 3.11, pandas, PyYAML, pytest, standard-library
`csv`/`json`/`pathlib`/`tarfile`/`tempfile`/`os`.

## Global Constraints

- Do not edit the approved design at
  `docs/superpowers/specs/2026-10-02-mandarin-metadata-design.md`.
- Do not create git commits during implementation; the user will commit later.
- Preserve every source file byte-for-byte; never extract or decode AISHELL-3
  WAV members from `data_aishell3.tgz`.
- Never map JMDS pristine `zho/AISHELL3` IDs to AISHELL-3 archive utterance
  names; exclude all 4,410 pristine JMDS rows from the AISHELL manifest.
- Treat `Chinese_ADD_Generated` as the exact upstream directory spelling.
- Expected JMDS/ADD generated counts are train 7,146, dev 7,497 and eval 9,999
  (24,642 total); all selected IDs must resolve 1:1 to existing WAV files.
- Do not assume AISHELL-3 test/train utterance counts before reconciliation;
  the adapter must explain observed vs official test divergence in an explicit
  reconciliation report (observed `test/content.txt` has 24,773 lines; official
  publication reports 23,262 test samples).
- AISHELL-3 license is Apache-2.0; JMDS/ADD sparse fields (`attack_id`,
  `gender`, `spk_id` = `unk`, `codec` = `-`, `native` = `yes`) must not be
  inferred or filled.
- Develop each behavior test-first and observe the targeted test fail before
  implementing it.
- After every refactor task, run the full 583-test suite and keep it green.
- Do not duplicate Portuguese-specific config, layout, or pipeline modules;
  extend generic primitives and add Mandarin/Portuguese wrappers only.

---

## File Map

**Create — generic primitives (Phase 1)**

- `src/jmds_prepare/storage/two_source_metadata_layout.py`: frozen
  `TwoSourceArtifactNames` and profile-driven `TwoSourceMetadataLayout`.
- `src/jmds_prepare/metadata_config_common.py`: shared YAML scalar-path load,
  nested-root rejection, and JMDS protocol existence checks.
- `src/jmds_prepare/pipelines/metadata_profiling_common.py`: deterministic
  `_profile_field`, `_counts_by_split`, `_value_counts`, length and numeric
  statistics helpers.
- `src/jmds_prepare/pipelines/metadata_provenance_common.py`: sorted input
  evidence list and reusable artifact-schema section builders.
- `tests/test_two_source_metadata_layout.py`: generic layout contract tests.
- `tests/test_metadata_config_common.py`: shared validation helper tests.

**Modify — Portuguese wrappers delegate to generics (Phase 1)**

- `src/jmds_prepare/storage/metadata_layout.py`: thin
  `PortugueseMetadataLayout(TwoSourceMetadataLayout)` preserving every
  existing property name.
- `src/jmds_prepare/metadata_config.py`: delegate load/validate to
  `metadata_config_common`; keep `PortugueseMetadataConfig` public API.
- `src/jmds_prepare/pipelines/metadata_extraction.py`: add
  `extract_two_source_metadata`; keep `extract_portuguese_metadata` as a
  wrapper with identical behavior.
- `src/jmds_prepare/pipelines/metadata_profiling.py`: import shared helpers;
  keep `build_coraa_profile`, `build_jmds_mlaad_profile`,
  `build_portuguese_summary` signatures unchanged.
- `src/jmds_prepare/pipelines/metadata_provenance.py`: import shared helpers;
  keep `build_metadata_provenance` signature unchanged.
- `src/jmds_prepare/profiles/portuguese.py`: add frozen
  `artifact_names: TwoSourceArtifactNames` field on `PortugueseProfile`.

**Create — Mandarin feature (Phase 2)**

- `src/jmds_prepare/profiles/mandarin.py`: immutable Mandarin constants,
  AISHELL archive member paths, ADD layout derivation, artifact names.
- `src/jmds_prepare/sources/aishell3.py`: streaming `.tgz` adapter,
  reconciliation report, manifest columns.
- `src/jmds_prepare/sources/jmds_add.py`: strict `zho/generated/ADD` adapter.
- `src/jmds_prepare/pipelines/mandarin_metadata_profiling.py`:
  `build_aishell3_profile`, `build_jmds_add_profile`,
  `build_mandarin_summary`.
- `src/jmds_prepare/pipelines/mandarin_metadata_provenance.py`:
  `build_mandarin_metadata_provenance`.
- `src/jmds_prepare/mandarin_metadata_config.py`: YAML config with
  `aishell_archive`, `jmds_root`, `output_root`.
- `src/jmds_prepare/storage/mandarin_metadata_layout.py`: thin wrapper over
  `TwoSourceMetadataLayout`.
- `src/jmds_prepare/mandarin_metadata.py`: standalone CLI.
- `configs/mandarin-metadata.example.yaml`: portable template.
- `configs/mandarin-metadata.yaml`: local real-data configuration.
- `tests/fixtures/aishell3_mini.tgz`: small streaming fixture (built by test
  helper, committed once generated).
- `tests/fixtures/jmds_add_metadata.csv`: mixed pristine/generated fixture.
- `tests/test_mandarin_profile.py`
- `tests/test_aishell3.py`
- `tests/test_jmds_add.py`
- `tests/test_mandarin_metadata_profiling.py`
- `tests/test_mandarin_metadata_pipeline.py`
- `tests/test_mandarin_metadata_cli.py`

**Modify — integration (Phase 2)**

- `src/jmds_prepare/profiles/__init__.py`: export `MANDARIN_PROFILE`.
- `tests/test_profiles_sources.py`: pin Mandarin profile/source layer rules.
- `tests/test_pipelines.py`: include Mandarin pipeline import checks.
- `.gitignore`: ignore local Mandarin derived data/config where necessary.
- `README.md`, `docs/data-framework.md`, `docs/sources.md`, `BACKLOG.md`:
  document command, outputs, reconciliation behavior and limitations.

---

### Task 1: Generic artifact names and two-source layout

**Files:**
- Create: `src/jmds_prepare/storage/two_source_metadata_layout.py`
- Create: `tests/test_two_source_metadata_layout.py`
- Modify: `src/jmds_prepare/storage/metadata_layout.py`
- Modify: `src/jmds_prepare/profiles/portuguese.py`

**Interfaces:**
- Produces:
  `TwoSourceArtifactNames(pristine_manifest, generated_manifest,
  pristine_profile, generated_profile, summary, provenance)`.
- Produces `TwoSourceMetadataLayout(output_root, names)` with properties
  `pristine_metadata_csv`, `generated_metadata_csv`,
  `pristine_metadata_profile_json`, `generated_metadata_profile_json`,
  `summary_json`, `provenance_json`, plus `manifests_dir` and `reports_dir`.
- Consumes: `PortugueseProfile.artifact_names -> TwoSourceArtifactNames`.
- Preserves every existing `PortugueseMetadataLayout` property path unchanged.

- [ ] **Step 1: Write failing generic layout tests**

```python
from jmds_prepare.storage.two_source_metadata_layout import (
    TwoSourceArtifactNames,
    TwoSourceMetadataLayout,
)

NAMES = TwoSourceArtifactNames(
    pristine_manifest="coraa_metadata.csv",
    generated_manifest="jmds_mlaad_generated_metadata.csv",
    pristine_profile="coraa_metadata_profile.json",
    generated_profile="jmds_mlaad_generated_metadata_profile.json",
    summary="portuguese_metadata_summary.json",
    provenance="portuguese_metadata_provenance.json",
)


def test_two_source_layout_resolves_six_paths(tmp_path):
    layout = TwoSourceMetadataLayout(tmp_path / "out", NAMES)
    assert layout.pristine_metadata_csv == (
        tmp_path / "out" / "manifests" / "coraa_metadata.csv"
    )
    assert layout.generated_metadata_csv == (
        tmp_path / "out" / "manifests" / "jmds_mlaad_generated_metadata.csv"
    )
    assert layout.summary_json == (
        tmp_path / "out" / "reports" / "portuguese_metadata_summary.json"
    )
    assert layout.provenance_json == (
        tmp_path / "out" / "reports" / "portuguese_metadata_provenance.json"
    )
```

Add a test asserting `PortugueseMetadataLayout` paths still match
`tests/test_portuguese_metadata_pipeline.py::test_layout_exposes_six_artifact_paths`.

- [ ] **Step 2: Run and observe import failure**

Run:

```powershell
python -m pytest tests/test_two_source_metadata_layout.py tests/test_portuguese_metadata_pipeline.py::test_layout_exposes_six_artifact_paths -q
```

Expected: import failure for `two_source_metadata_layout`; Portuguese layout
test still collects.

- [ ] **Step 3: Implement generic layout and Portuguese delegation**

Add to `profiles/portuguese.py`:

```python
PORTUGUESE_ARTIFACT_NAMES = TwoSourceArtifactNames(
    pristine_manifest="coraa_metadata.csv",
    generated_manifest="jmds_mlaad_generated_metadata.csv",
    pristine_profile="coraa_metadata_profile.json",
    generated_profile="jmds_mlaad_generated_metadata_profile.json",
    summary="portuguese_metadata_summary.json",
    provenance="portuguese_metadata_provenance.json",
)
```

Extend `PortugueseProfile` with `artifact_names: TwoSourceArtifactNames =
PORTUGUESE_ARTIFACT_NAMES`. Refactor `PortugueseMetadataLayout` to subclass
`TwoSourceMetadataLayout`, passing `PORTUGUESE_ARTIFACT_NAMES`, and keep
alias properties `coraa_metadata_csv`, `jmds_mlaad_generated_metadata_csv`,
`coraa_metadata_profile_json`, `jmds_mlaad_generated_metadata_profile_json`,
`portuguese_metadata_summary_json`, `portuguese_metadata_provenance_json`.

- [ ] **Step 4: Verify green**

Run:

```powershell
python -m pytest tests/test_two_source_metadata_layout.py tests/test_portuguese_metadata_pipeline.py::test_layout_exposes_six_artifact_paths -q
```

Expected: 2+ tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/jmds_prepare/storage/two_source_metadata_layout.py src/jmds_prepare/storage/metadata_layout.py src/jmds_prepare/profiles/portuguese.py tests/test_two_source_metadata_layout.py
git commit -m "refactor: extract generic two-source metadata layout"
```

---

### Task 2: Shared config validation helpers

**Files:**
- Create: `src/jmds_prepare/metadata_config_common.py`
- Create: `tests/test_metadata_config_common.py`
- Modify: `src/jmds_prepare/metadata_config.py`

**Interfaces:**
- Produces `scalar_path(value, field_name) -> Path`.
- Produces `reject_nested_output(*, output_root, forbidden_roots) -> None`.
- Produces `validate_jmds_protocols(jmds_root, profile) -> None`.
- Preserves `PortugueseMetadataConfig.load(path)` and
  `.validate(PORTUGUESE_PROFILE)` behavior exactly.

- [ ] **Step 1: Write failing common validation tests**

```python
def test_reject_nested_output_detects_child(tmp_path):
    outer = tmp_path / "jmds"
    inner = outer / "nested_output"
    outer.mkdir()
    inner.mkdir()
    with pytest.raises(ValueError, match="nested"):
        reject_nested_output(
            output_root=inner.resolve(),
            forbidden_roots=[outer.resolve()],
        )


def test_validate_jmds_protocols_requires_all_splits(tmp_path, monkeypatch):
    profile = replace(
        PORTUGUESE_PROFILE,
        protocol_splits=("train",),
    )
    jmds_root = tmp_path / "jmds"
    (jmds_root / "cm_protocols").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="open_v2_train"):
        validate_jmds_protocols(jmds_root, profile)
```

- [ ] **Step 2: Run and observe failure**

Run:

```powershell
python -m pytest tests/test_metadata_config_common.py -q
```

Expected: import failure for `metadata_config_common`.

- [ ] **Step 3: Implement helpers and delegate Portuguese config**

Move `_scalar_path` and `_is_nested` from `metadata_config.py` into
`metadata_config_common.py` as public `scalar_path` and
`reject_nested_output`. Add `validate_jmds_protocols`. Rewrite
`PortugueseMetadataConfig.validate` to call the shared helpers without
changing error messages tested in `tests/test_portuguese_metadata_pipeline.py`.

- [ ] **Step 4: Verify Portuguese config tests still pass**

Run:

```powershell
python -m pytest tests/test_metadata_config_common.py tests/test_portuguese_metadata_pipeline.py -k config -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/jmds_prepare/metadata_config_common.py src/jmds_prepare/metadata_config.py tests/test_metadata_config_common.py
git commit -m "refactor: share two-source metadata config validation"
```

---

### Task 3: Generic two-source extraction pipeline

**Files:**
- Modify: `src/jmds_prepare/pipelines/metadata_extraction.py`
- Modify: `tests/test_portuguese_metadata_pipeline.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class TwoSourceMetadataExtractionDeps:
    layout: TwoSourceMetadataLayout
    collect_pristine: Callable[[Any], pd.DataFrame]
    collect_generated: Callable[[Any], pd.DataFrame]
    validate_pristine_counts: Callable[[pd.DataFrame], None] | None
    validate_generated_counts: Callable[[pd.DataFrame], None]
    build_pristine_profile: Callable[[pd.DataFrame], dict[str, Any]]
    build_generated_profile: Callable[[pd.DataFrame], dict[str, Any]]
    build_summary: Callable[[pd.DataFrame, pd.DataFrame], dict[str, Any]]
    build_provenance: Callable[..., dict[str, Any]]
    publish_artifacts: Callable[[Mapping[Path, bytes]], None]
    provenance_kwargs: Mapping[str, Any]


def extract_two_source_metadata(
    config: Any,
    deps: TwoSourceMetadataExtractionDeps,
    *,
    output: Output = print,
) -> None: ...
```

- Preserves `MetadataExtractionDeps` and `extract_portuguese_metadata` public
  API; Portuguese wrapper wires `collect_pristine` as the existing per-split
  CORAA loop and `collect_generated` as the existing JMDS loop.

- [ ] **Step 1: Write failing AST/behavior test for generic extraction**

Add to `tests/test_portuguese_metadata_pipeline.py`:

```python
def test_extract_two_source_metadata_has_no_language_literals():
    source = Path("src/jmds_prepare/pipelines/metadata_extraction.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value in {"por", "zho"}:
            raise AssertionError("metadata_extraction must not branch on language codes")
    assert "def extract_two_source_metadata" in source
```

- [ ] **Step 2: Run and observe failure**

Run:

```powershell
python -m pytest tests/test_portuguese_metadata_pipeline.py::test_extract_two_source_metadata_has_no_language_literals -q
```

Expected: FAIL because `extract_two_source_metadata` does not exist yet.

- [ ] **Step 3: Extract generic orchestration**

Move serialization helpers unchanged. Implement `extract_two_source_metadata`
to:

1. call `deps.collect_pristine(config)` and optional
   `deps.validate_pristine_counts`;
2. call `deps.collect_generated(config)` and
   `deps.validate_generated_counts`;
3. build four JSON payloads via injected builders;
4. assemble six `{layout property: bytes}` entries from
   `deps.layout`;
5. call `deps.publish_artifacts` once.

Rewrite `extract_portuguese_metadata` to construct
`TwoSourceMetadataExtractionDeps` with closures that preserve the current
CORAA/JMDS split loops, count checks, layout property names, and provenance
kwargs. Keep `MetadataExtractionDeps` as a compatibility alias or unchanged
dataclass consumed only by the wrapper.

- [ ] **Step 4: Verify all Portuguese pipeline tests**

Run:

```powershell
python -m pytest tests/test_portuguese_metadata_pipeline.py tests/test_portuguese_metadata_cli.py -q
```

Expected: all tests pass with identical artifact paths and publication behavior.

- [ ] **Step 5: Full regression gate**

Run:

```powershell
python -m pytest -q
```

Expected: `583 passed`.

- [ ] **Step 6: Commit**

```powershell
git add src/jmds_prepare/pipelines/metadata_extraction.py tests/test_portuguese_metadata_pipeline.py
git commit -m "refactor: generic two-source metadata extraction pipeline"
```

---

### Task 4: Shared profiling and provenance helpers

**Files:**
- Create: `src/jmds_prepare/pipelines/metadata_profiling_common.py`
- Create: `src/jmds_prepare/pipelines/metadata_provenance_common.py`
- Modify: `src/jmds_prepare/pipelines/metadata_profiling.py`
- Modify: `src/jmds_prepare/pipelines/metadata_provenance.py`
- Modify: `tests/test_metadata_profiling.py`

**Interfaces:**
- Produces unchanged `build_coraa_profile`, `build_jmds_mlaad_profile`,
  `build_portuguese_summary`, `build_metadata_provenance`.
- Produces shared `profile_field(...)`, `counts_by_split(...)`,
  `value_counts(...)`, `build_input_evidence(...)`.

- [ ] **Step 1: Write failing import test for shared profiling helpers**

```python
from jmds_prepare.pipelines.metadata_profiling_common import profile_field


def test_profile_field_reports_missing_and_distinct():
    frame = pd.DataFrame({"split": ["train", ""], "task": ["a", "b"]})
    payload = profile_field(
        frame,
        "split",
        description="split label",
        origin="derived",
        categorical=frozenset({"split"}),
        length_fields=frozenset(),
        vote_fields=frozenset(),
    )
    assert payload["missing_count"] == 1
    assert payload["distinct_count"] == 2
    assert payload["value_counts"] == [
        {"count": 1, "value": ""},
        {"count": 1, "value": "train"},
    ]
```

- [ ] **Step 2: Run and observe failure**

Run:

```powershell
python -m pytest tests/test_metadata_profiling.py -k profile_field -q
```

Expected: import failure for `metadata_profiling_common`.

- [ ] **Step 3: Move helpers and re-export from Portuguese modules**

Move `_profile_field`, `_counts_by_split`, `_length_statistics`,
`_numeric_statistics`, `_value_counts` into `metadata_profiling_common.py` as
public functions without leading underscores. Update Portuguese builders to
import them. Move sorted input list construction into
`metadata_provenance_common.build_input_evidence`.

- [ ] **Step 4: Verify profiling and provenance suites**

Run:

```powershell
python -m pytest tests/test_metadata_profiling.py tests/test_metadata_publication.py -q
```

Expected: all tests pass; Portuguese JSON payloads byte-identical for fixture
inputs (add a snapshot assertion if not already present).

- [ ] **Step 5: Full regression gate**

Run:

```powershell
python -m pytest -q
```

Expected: `583 passed`.

- [ ] **Step 6: Commit**

```powershell
git add src/jmds_prepare/pipelines/metadata_profiling_common.py src/jmds_prepare/pipelines/metadata_provenance_common.py src/jmds_prepare/pipelines/metadata_profiling.py src/jmds_prepare/pipelines/metadata_provenance.py tests/test_metadata_profiling.py
git commit -m "refactor: share metadata profile and provenance helpers"
```

---

### Task 5: Mandarin profile and artifact names

**Files:**
- Create: `tests/test_mandarin_profile.py`
- Create: `src/jmds_prepare/profiles/mandarin.py`
- Modify: `src/jmds_prepare/profiles/__init__.py`
- Modify: `tests/test_profiles_sources.py`

**Interfaces:**
- Produces `MandarinProfile` with:
  `language_code="zho"`, `language_slug="mandarin"`,
  `generated_dataset="ADD"`, `generated_dir_name="Chinese_ADD_Generated"`,
  `generated_audio_subdir="wav"`, `protocol_splits=("train", "dev", "eval")`,
  `aishell_splits=("train", "test")`,
  `expected_generated_counts={"train": 7146, "dev": 7497, "eval": 9999}`,
  `split_id_prefixes={"train": "T", "dev": "D", "eval": "E"}`,
  `excluded_jmds_pristine_count=4410`,
  `artifact_names` with the six Mandarin filenames from the spec.
- Produces path helpers:
  `jmds_protocol_path(jmds_root, split) -> Path`,
  `generated_wav_path(jmds_root, split, utt_id) -> Path`,
  `aishell_archive_members` mapping for
  `spk-info.txt`, `{split}/content.txt`, and
  `train/prosody_label_train-set.txt`.

- [ ] **Step 1: Write failing Mandarin profile tests**

```python
def test_mandarin_profile_pins_add_layout():
    root = Path("E:/JMDS")
    assert MANDARIN_PROFILE.generated_wav_path(
        root, "train", "T_0000123456"
    ) == (
        root / "dataset" / "Chinese_ADD_Generated"
        / "train" / "wav" / "T_0000123456.wav"
    )
    assert dict(MANDARIN_PROFILE.expected_generated_counts) == {
        "train": 7146, "dev": 7497, "eval": 9999
    }


def test_mandarin_artifact_names_match_spec():
    names = MANDARIN_PROFILE.artifact_names
    assert names.pristine_manifest == "aishell3_metadata.csv"
    assert names.generated_manifest == "jmds_add_generated_metadata.csv"
    assert names.summary == "mandarin_metadata_summary.json"
    assert names.provenance == "mandarin_metadata_provenance.json"
```

- [ ] **Step 2: Run and observe failure**

Run:

```powershell
python -m pytest tests/test_mandarin_profile.py tests/test_profiles_sources.py -q
```

Expected: import failure for `profiles.mandarin`.

- [ ] **Step 3: Implement immutable Mandarin profile**

Mirror `PortugueseProfile` immutability patterns (`MappingProxyType`, frozen
tuples). Do not add `expected_aishell_counts`; AISHELL cardinality is
reconciliation-driven, not pinned pre-scan.

- [ ] **Step 4: Verify green and full suite**

Run:

```powershell
python -m pytest tests/test_mandarin_profile.py tests/test_profiles_sources.py -q
python -m pytest -q
```

Expected: all tests pass (`583+` after new tests).

- [ ] **Step 5: Commit**

```powershell
git add src/jmds_prepare/profiles/mandarin.py src/jmds_prepare/profiles/__init__.py tests/test_mandarin_profile.py tests/test_profiles_sources.py
git commit -m "feat: add immutable Mandarin metadata profile"
```

---

### Task 6: AISHELL-3 streaming adapter and mini `.tgz` fixture

**Files:**
- Create: `tests/fixtures/aishell3_mini.tgz` (via helper script committed in repo)
- Create: `tests/test_aishell3.py`
- Create: `src/jmds_prepare/sources/aishell3.py`

**Interfaces:**
- Produces column tuple `AISHELL3_COLUMNS`:

```python
AISHELL3_COLUMNS = (
    "utt_id",
    "split",
    "archive_member_path",
    "speaker_id",
    "gender",
    "age",
    "accent",
    "transcription",
    "pinyin",
    "prosody_available",
    "metadata_source_file",
    "metadata_source_row",
)
```

- Produces:

```python
@dataclass(frozen=True)
class AishellReconciliation:
    by_split: Mapping[str, dict[str, Any]]
    official_test_sample_count: int
    observed_test_content_line_count: int
    divergence_explanation: str


def read_aishell3_metadata(
    archive_path: Path,
    *,
    profile: MandarinProfile,
) -> tuple[pd.DataFrame, AishellReconciliation]: ...
```

- Streaming contract: single sequential `tarfile.open(archive_path, "r:gz")`
  pass; read textual member bodies; for `.wav` members record
  `{member_name: size_bytes}` only; never call `extract` or read WAV payloads.

- [ ] **Step 1: Add fixture builder used once to create committed `.tgz`**

Create `tests/fixtures/build_aishell3_mini.py`:

```python
"""Build tests/fixtures/aishell3_mini.tgz — run once, commit the archive."""

import io
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "aishell3_mini.tgz"

MEMBERS = {
    "data_aishell3/spk-info.txt": (
        "SSB0005 female 52 北\n"
        "SSB0999 male 31 冀\n"
    ),
    "data_aishell3/train/content.txt": (
        "SSB00050001 你好\n"
        "SSB00050002 训练语句\n"
        "SSB09990001 无wav训练\n"
    ),
    "data_aishell3/test/content.txt": (
        "SSB00050003 测试一\n"
        "SSB00050004 测试二\n"
        "SSB00050005 缺wav测试\n"
    ),
    "data_aishell3/train/prosody_label_train-set.txt": (
        "SSB00050001 ni3 hao3\n"
        "SSB00050002 xun4 lian4\n"
    ),
}

WAV_STUBS = {
    "data_aishell3/train/wav/SSB00050001.wav": b"RIFFxxxxWAVEfmt ",
    "data_aishell3/train/wav/SSB00050002.wav": b"RIFFyyyyWAVEfmt ",
    "data_aishell3/test/wav/SSB00050003.wav": b"RIFFzzzzWAVEfmt ",
    "data_aishell3/test/wav/SSB00050004.wav": b"RIFFwwwwWAVEfmt ",
    "data_aishell3/test/wav/SSB00059999.wav": b"RIFFextraWAVEfmt ",
}


def build() -> None:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        for name, text in MEMBERS.items():
            payload = text.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        for name, payload in WAV_STUBS.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


if __name__ == "__main__":
    build()
    print(f"wrote {ARCHIVE}")
```

Run once:

```powershell
python tests/fixtures/build_aishell3_mini.py
```

Expected: `wrote tests/fixtures/aishell3_mini.tgz`.

- [ ] **Step 2: Write failing AISHELL adapter tests**

```python
FIXTURE = Path(__file__).parent / "fixtures" / "aishell3_mini.tgz"


def test_read_aishell3_streams_without_extracting_wavs(tmp_path, monkeypatch):
    extracted: list[str] = []
    original_open = tarfile.open

    def tracking_open(path, mode="r"):
        archive = original_open(path, mode)
        original_extractfile = archive.extractfile

        def wrapped(name):
            if str(name).endswith(".wav"):
                extracted.append(str(name))
            return original_extractfile(name)

        archive.extractfile = wrapped  # type: ignore[method-assign]
        return archive

    monkeypatch.setattr(tarfile, "open", tracking_open)
    frame, reconciliation = read_aishell3_metadata(
        FIXTURE, profile=MANDARIN_PROFILE
    )
    assert extracted == []
    assert list(frame.columns) == [*AISHELL3_COLUMNS]
    assert set(frame["split"]) == {"train", "test"}
    assert frame["archive_member_path"].str.endswith(".wav").all()
    assert (frame["prosody_available"] == "yes").sum() == 2
    test_report = reconciliation.by_split["test"]
    assert test_report["content_line_count"] == 3
    assert test_report["wav_member_count"] == 3
    assert test_report["matched_count"] == 2
    assert test_report["content_without_wav"] == ["SSB00050005"]
    assert test_report["wav_without_content"] == ["SSB00059999"]
    assert reconciliation.official_test_sample_count == 23262
    assert reconciliation.observed_test_content_line_count == 3
    assert "24,773" in reconciliation.divergence_explanation or "24773" in reconciliation.divergence_explanation
```

Add parametrized failures for duplicate tar members, path traversal (`../`),
non-regular members, unknown speaker IDs, and duplicate utterance IDs.

- [ ] **Step 3: Run and observe failure**

Run:

```powershell
python -m pytest tests/test_aishell3.py -q
```

Expected: import failure for `sources.aishell3`.

- [ ] **Step 4: Implement streaming adapter**

Parse `spk-info.txt` as whitespace-separated
`speaker_id gender age accent`. Parse `content.txt` as
`utt_id transcription` (split on first space). Parse prosody as
`utt_id pinyin_tokens...`. Derive `speaker_id` as first seven characters of
`utt_id` (AISHELL convention `SSB####` + utterance suffix). Join gender/age/
accent only from `spk-info.txt`. Set `prosody_available` to `"yes"` or `"no"`.
Build manifest rows only for utterances with content, speaker, and WAV present;
fail on ambiguous duplicates or unsafe paths. Populate `AishellReconciliation`
with per-split matched/content_without_wav/wav_without_content counts and a
`divergence_explanation` referencing observed 24,773 vs official 23,262 for
real archives (fixture uses smaller counts but same report shape).

- [ ] **Step 5: Verify green**

Run:

```powershell
python -m pytest tests/test_aishell3.py -q
```

Expected: all AISHELL tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/jmds_prepare/sources/aishell3.py tests/test_aishell3.py tests/fixtures/build_aishell3_mini.py tests/fixtures/aishell3_mini.tgz
git commit -m "feat: stream AISHELL-3 metadata with WAV reconciliation"
```

---

### Task 7: JMDS/ADD generated adapter

**Files:**
- Create: `tests/fixtures/jmds_add_metadata.csv`
- Create: `tests/test_jmds_add.py`
- Create: `src/jmds_prepare/sources/jmds_add.py`

**Interfaces:**
- Produces
  `read_mandarin_generated(path, *, split, jmds_root, profile) -> pd.DataFrame`.
- Validates full protocol schema first; selects only
  `language=zho`, `label=generated`, `dataset=ADD`; rejects other Mandarin
  rows; never reads pristine `zho/AISHELL3` rows into the manifest.

- [ ] **Step 1: Create exact small fixture**

Header: exact nine-column JMDS header from `sources/jmds.py`. Include:

- two generated `zho/ADD` train rows with valid `T_##########` IDs;
- one pristine `zho/AISHELL3` row;
- one generated `por/MLAAD` row.

- [ ] **Step 2: Write failing ADD contract tests**

```python
def test_reads_mandarin_generated_subset_with_audio_paths(jmds_root):
    profile = replace(
        MANDARIN_PROFILE,
        expected_generated_counts={"train": 2, "dev": 7497, "eval": 9999},
    )
    frame = read_mandarin_generated(
        FIXTURE, split="train", jmds_root=jmds_root, profile=profile
    )
    assert set(frame["language"]) == {"zho"}
    assert set(frame["label"]) == {"generated"}
    assert set(frame["dataset"]) == {"ADD"}
    assert len(frame) == 2
    assert frame["audio_path"].map(Path).map(Path.is_file).all()
```

Mirror Portuguese failure cases: wrong dataset/language/label, malformed IDs,
duplicates, missing WAV, path escape.

- [ ] **Step 3: Run, implement, verify**

Run before:

```powershell
python -m pytest tests/test_jmds_add.py -q
```

Expected: import failure.

Implement by adapting `sources/jmds_mlaad.py` patterns with Mandarin profile
fields and `_validate_mandarin_row_combinations` accepting only generated ADD
or pristine AISHELL3 references in the full-protocol scan (pristine excluded
from output). Use `jmds_common` validators unchanged.

Run after:

```powershell
python -m pytest tests/test_jmds_add.py tests/test_protocols.py -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Commit**

```powershell
git add src/jmds_prepare/sources/jmds_add.py tests/test_jmds_add.py tests/fixtures/jmds_add_metadata.csv
git commit -m "feat: add strict JMDS/ADD Mandarin generated adapter"
```

---

### Task 8: Mandarin profiles, summary and provenance builders

**Files:**
- Create: `tests/test_mandarin_metadata_profiling.py`
- Create: `src/jmds_prepare/pipelines/mandarin_metadata_profiling.py`
- Create: `src/jmds_prepare/pipelines/mandarin_metadata_provenance.py`

**Interfaces:**
- Produces `build_aishell3_profile(frame, *, reconciliation) -> dict`.
- Produces `build_jmds_add_profile(frame) -> dict`.
- Produces `build_mandarin_summary(aishell, jmds) -> dict`.
- Produces
  `build_mandarin_metadata_provenance(*, profile, input_paths, sha256_file,
  reconciliation) -> dict`.

- [ ] **Step 1: Write failing profile JSON tests**

Assert `build_aishell3_profile` includes:

```python
{
    "source": "AISHELL3",
    "reconciliation": {
        "by_split": {"test": {"matched_count": 2, "content_without_wav": ["SSB00050005"]}},
        "official_test_sample_count": 23262,
        "divergence_explanation": "...",
    },
    "fields": [...],
}
```

Sensitive columns (`utt_id`, `archive_member_path`, `transcription`, `pinyin`,
`metadata_source_file`, `metadata_source_row`) expose length statistics only.
`build_jmds_add_profile` mirrors MLAAD profile with source `JMDS_ADD`.

- [ ] **Step 2: Write failing summary and provenance tests**

```python
assert summary["comparison_limitations"]["paired_samples"] is False
assert "sample_pairs" not in summary
assert summary["sources"]["AISHELL3"]["label_role"] == "pristine"
assert summary["sources"]["JMDS_ADD"]["label_role"] == "generated"
assert provenance["excluded_jmds_pristine_count"] == 4410
assert provenance["aishell3"]["license"] == "Apache-2.0"
assert provenance["inputs"][0]["sha256"] == "deadbeef"
```

Provenance must record archive SHA-256/size for `aishell_archive` and protocol
SHA-256 for each JMDS split; document ADD unknown-field reasons matching the
spec.

- [ ] **Step 3: Run, implement, verify**

Run:

```powershell
python -m pytest tests/test_mandarin_metadata_profiling.py -q
```

Expected first: import failures. Expected after: all tests pass. Reuse
`metadata_profiling_common.profile_field` and
`metadata_provenance_common.build_input_evidence`.

- [ ] **Step 4: Commit**

```powershell
git add src/jmds_prepare/pipelines/mandarin_metadata_profiling.py src/jmds_prepare/pipelines/mandarin_metadata_provenance.py tests/test_mandarin_metadata_profiling.py
git commit -m "feat: add Mandarin metadata profile summary and provenance"
```

---

### Task 9: Mandarin config, layout, pipeline wiring and CLI

**Files:**
- Create: `tests/test_mandarin_metadata_pipeline.py`
- Create: `tests/test_mandarin_metadata_cli.py`
- Create: `src/jmds_prepare/mandarin_metadata_config.py`
- Create: `src/jmds_prepare/storage/mandarin_metadata_layout.py`
- Create: `src/jmds_prepare/mandarin_metadata.py`
- Create: `configs/mandarin-metadata.example.yaml`
- Create: `configs/mandarin-metadata.yaml`
- Modify: `tests/test_pipelines.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces `MandarinMetadataConfig.load(path)` with keys
  `jmds_root`, `aishell_archive`, `output_root`.
- Produces `MandarinMetadataLayout(TwoSourceMetadataLayout)` using
  `MANDARIN_PROFILE.artifact_names`.
- Produces `extract_mandarin_metadata(config, deps, *, output=print)` thin
  wrapper over `extract_two_source_metadata`.
- Produces standalone
  `python -m jmds_prepare.mandarin_metadata --config <yaml>`.

- [ ] **Step 1: Write failing config and layout tests**

YAML schema:

```yaml
jmds_root: E:\JMDS_open_v2_public\JMDS_open_v2_public
aishell_archive: E:\sources\aishell3\data_aishell3.tgz
output_root: E:\JMDS_open_v2_public\mandarin_preparation
```

Reject unknown/missing keys, non-scalar paths, missing archive/protocol files,
equal input/output roots, output nested inside `jmds_root` or parent of archive.
Assert six Mandarin artifact paths under `manifests/` and `reports/`.

- [ ] **Step 2: Write failing dependency-injected pipeline test**

Use `aishell3_mini.tgz`, fixture JMDS CSV, in-memory publication recorder, and
injected `sha256_file`. Assert:

```python
extract_mandarin_metadata(config, deps)
assert len(published) == 6
assert reconciliation["by_split"]["test"]["content_without_wav"] == ["SSB00050005"]
```

Add AST test: `mandarin_metadata.py` imports neither `jmds_prepare.cli` nor
`profiles.english`; `metadata_extraction.py` still has no `"zho"` literal branch.

- [ ] **Step 3: Run and observe failures**

Run:

```powershell
python -m pytest tests/test_mandarin_metadata_pipeline.py tests/test_mandarin_metadata_cli.py -q
```

Expected: import failures for Mandarin modules.

- [ ] **Step 4: Implement config, layout, wrapper, CLI**

`MandarinMetadataConfig.validate` checks archive is a file, runs
`validate_jmds_protocols`, and reuses `reject_nested_output`.
`extract_mandarin_metadata` wires:

```python
collect_pristine=lambda cfg: read_aishell3_metadata(cfg.aishell_archive, profile=MANDARIN_PROFILE)[0]
validate_pristine_counts=None  # reconciliation-driven, not pre-pinned
collect_generated=lambda cfg: pd.concat([
    read_mandarin_generated(
        MANDARIN_PROFILE.jmds_protocol_path(cfg.jmds_root, split),
        split=split,
        jmds_root=cfg.jmds_root,
        profile=MANDARIN_PROFILE,
    )
    for split in MANDARIN_PROFILE.protocol_splits
], ignore_index=True)
```

Pass reconciliation object into `build_aishell3_profile` and provenance via
closure or `provenance_kwargs`. CLI mirrors Portuguese error handling.

- [ ] **Step 5: Add configs and gitignore entries**

Example YAML uses quoted placeholders. Local config uses the three Windows paths
above. Ignore `configs/mandarin-metadata.yaml` and local Mandarin output roots.

- [ ] **Step 6: Verify Mandarin tests and full suite**

Run:

```powershell
python -m pytest tests/test_mandarin_metadata_pipeline.py tests/test_mandarin_metadata_cli.py tests/test_pipelines.py tests/test_layer_placement.py -q
python -m pytest -q
```

Expected: all tests pass (`583+`).

- [ ] **Step 7: Commit**

```powershell
git add src/jmds_prepare/mandarin_metadata_config.py src/jmds_prepare/storage/mandarin_metadata_layout.py src/jmds_prepare/mandarin_metadata.py configs/mandarin-metadata.example.yaml tests/test_mandarin_metadata_pipeline.py tests/test_mandarin_metadata_cli.py tests/test_pipelines.py .gitignore
git commit -m "feat: wire Mandarin metadata config layout pipeline and CLI"
```

---

### Task 10: Documentation, real extraction and idempotent verification

**Files:**
- Modify: `README.md`
- Modify: `docs/data-framework.md`
- Modify: `docs/sources.md`
- Modify: `BACKLOG.md`

**Interfaces:**
- Consumes standalone CLI and six Mandarin artifact paths from Task 9.
- Produces real local outputs under
  `E:\JMDS_open_v2_public\mandarin_preparation`.

- [ ] **Step 1: Document the metadata-only command and limitations**

Document:

```powershell
python -m jmds_prepare.mandarin_metadata --config configs/mandarin-metadata.yaml
```

List all six outputs, AISHELL streaming/no-extraction policy, reconciliation
report contents, ADD 24,642 row counts, exclusion of 4,410 pristine JMDS rows,
`paired_samples: false`, and ADD sparse-field unknowns.

- [ ] **Step 2: Run full automated verification**

```powershell
python -m pytest -q
python -m compileall -q src
```

Expected: every test passes; compileall exits 0.

- [ ] **Step 3: Execute real metadata extraction (first run)**

```powershell
python -m jmds_prepare.mandarin_metadata --config configs/mandarin-metadata.yaml
```

Expected stdout includes AISHELL row count after reconciliation, JMDS/ADD
`24642` rows, test-split reconciliation noting observed `24773` content lines
vs official `23262`, and `published 6 artifacts`.

- [ ] **Step 4: Verify real artifacts independently**

```python
import json
import pandas as pd
from pathlib import Path

root = Path(r"E:\JMDS_open_v2_public\mandarin_preparation")
aishell = pd.read_csv(root / "manifests" / "aishell3_metadata.csv", dtype=str, keep_default_na=False)
jmds = pd.read_csv(root / "manifests" / "jmds_add_generated_metadata.csv", dtype=str, keep_default_na=False)
profile = json.loads((root / "reports" / "aishell3_metadata_profile.json").read_text(encoding="utf-8"))
summary = json.loads((root / "reports" / "mandarin_metadata_summary.json").read_text(encoding="utf-8"))
provenance = json.loads((root / "reports" / "mandarin_metadata_provenance.json").read_text(encoding="utf-8"))

assert len(jmds) == 24_642
assert jmds["utt_id"].is_unique
assert jmds["audio_path"].map(Path).map(Path.is_file).all()
assert aishell["utt_id"].is_unique
assert profile["reconciliation"]["official_test_sample_count"] == 23_262
assert profile["reconciliation"]["observed_test_content_line_count"] == 24_773
assert summary["comparison_limitations"]["paired_samples"] is False
assert provenance["excluded_jmds_pristine_count"] == 4_410
```

Recompute SHA-256 for `data_aishell3.tgz` and each JMDS protocol; compare with
provenance inputs.

- [ ] **Step 5: Prove idempotence on real data (second run)**

Run the CLI again unchanged. Expected: success; SHA-256 identical for all six
artifacts; no temporary files left under `manifests/` or `reports/`.

- [ ] **Step 6: Final handoff**

Run `git status --short` and review only; do not stage or commit unless the user
asks.

---

## Self-Review

### 1. Spec coverage

| Spec requirement | Task |
| --- | --- |
| AISHELL-3 pristine via streaming `.tgz`, no WAV extraction | Task 6 |
| Metadata–WAV reconciliation with explicit test divergence report | Task 6, 8 |
| No silent row drops; fail on ambiguity | Task 6 |
| JMDS/ADD `zho/generated/ADD` only, 1:1 WAV resolution | Task 7 |
| Exclude 4,410 pristine `zho/AISHELL3` JMDS rows | Task 5, 7, 8 |
| Six artifact filenames from spec | Task 1, 5, 9 |
| Separate profiles, summary with `paired_samples: false`, provenance | Task 8, 9 |
| Content-idempotent publication of all six payloads | Task 3 (reuse), Task 9 |
| Generic pipeline without language branches | Task 3, 4 |
| Small `.tgz` fixture and real two-run execution | Task 6, 10 |
| Do not duplicate Portuguese config/layout/pipeline | Tasks 1–4 refactor first |
| Apache-2.0 AISHELL license; ADD unknowns not inferred | Task 8 |
| Memory: stream archive, index WAVs by name/size only | Task 6 |

No spec gaps identified.

### 2. Placeholder scan

No `TBD`, `TODO`, `implement later`, or `similar to Task N` shortcuts present.
Every task includes concrete file paths, test code, commands, and expected
outputs.

### 3. Type and interface consistency

- `TwoSourceArtifactNames` / `TwoSourceMetadataLayout` used by both Portuguese
  (Task 1) and Mandarin (Task 5, 9) wrappers.
- `extract_two_source_metadata` accepts injected collectors; Mandarin sets
  `validate_pristine_counts=None` while ADD keeps count validation — consistent
  with spec's reconciliation-first AISHELL policy.
- `read_aishell3_metadata` returns `(DataFrame, AishellReconciliation)`;
  Task 8 builders consume the same reconciliation object passed through Task 9
  provenance/profile closures.
- JMDS adapters share `jmds_common` validators and parallel signatures
  (`read_portuguese_generated` / `read_mandarin_generated`).

---

**Plan complete and saved to
`docs/superpowers/plans/2026-10-02-mandarin-metadata.md`. Two execution
options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task,
review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in one session using
superpowers:executing-plans with batch checkpoints.

**Which approach?**

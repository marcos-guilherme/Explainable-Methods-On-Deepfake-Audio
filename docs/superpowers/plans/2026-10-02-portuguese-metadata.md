# Portuguese Metadata Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce six local, reproducible artifacts that preserve and describe
the native CORAA and JMDS/MLAAD Portuguese metadata without claiming a
sample-level relationship between the corpora.

**Architecture:** Immutable corpus rules live in `profiles`; strict,
source-specific readers live in `sources`; generic content-idempotent
publication lives in `core`; a dependency-injected pipeline builds two CSV
manifests and four JSON reports; a standalone CLI wires concrete components.
The existing English CLI and outputs remain unchanged.

**Tech Stack:** Python 3.11, pandas, PyYAML, pytest, standard-library
`csv`/`json`/`pathlib`/`tempfile`/`os`.

## Global Constraints

- Do not edit the approved design at
  `docs/superpowers/specs/2026-10-02-portuguese-metadata-design.md`.
- Do not create git commits; the user will commit later.
- Preserve every source file byte-for-byte and never extract CORAA audio in
  this feature.
- Never map JMDS pristine IDs to CORAA paths.
- Treat `Portugese_MLAAD_Generated` as the exact upstream directory spelling.
- Expected CORAA counts are train 382,258, dev 7,522 and test 12,676.
- Expected JMDS/MLAAD generated counts are train 1,806, dev 602 and eval 603.
- All 3,011 generated IDs must resolve one-to-one to existing JMDS WAV files.
- Derived CORAA artifacts remain local and must not be committed or
  redistributed under the project's conservative CC BY-NC-ND 4.0 policy.
- Develop each behavior test-first and observe the targeted test fail before
  implementing it.

---

## File Map

**Create**

- `src/jmds_prepare/profiles/portuguese.py`: immutable Portuguese constants and
  source-path derivation.
- `src/jmds_prepare/sources/jmds_common.py`: profile-parameterized JMDS schema,
  split, ID and duplicate validation.
- `src/jmds_prepare/sources/coraa.py`: strict CORAA CSV adapter.
- `src/jmds_prepare/sources/jmds_mlaad.py`: strict Portuguese generated adapter.
- `src/jmds_prepare/core/publication.py`: content-idempotent artifact-set
  publication.
- `src/jmds_prepare/storage/metadata_layout.py`: six output paths.
- `src/jmds_prepare/metadata_config.py`: standalone YAML configuration.
- `src/jmds_prepare/pipelines/metadata_profiling.py`: two source profiles and
  comparison summary.
- `src/jmds_prepare/pipelines/metadata_provenance.py`: input evidence and field
  origins.
- `src/jmds_prepare/pipelines/metadata_extraction.py`: orchestration.
- `src/jmds_prepare/portuguese_metadata.py`: standalone CLI.
- `configs/portuguese-metadata.example.yaml`: portable configuration template.
- `configs/portuguese-metadata.yaml`: local real-data configuration.
- `tests/fixtures/coraa_metadata.csv`: small valid CORAA fixture.
- `tests/fixtures/jmds_mlaad_metadata.csv`: mixed pristine/generated fixture.
- `tests/test_portuguese_profile.py`
- `tests/test_coraa.py`
- `tests/test_jmds_mlaad.py`
- `tests/test_metadata_publication.py`
- `tests/test_metadata_profiling.py`
- `tests/test_portuguese_metadata_pipeline.py`
- `tests/test_portuguese_metadata_cli.py`

**Modify**

- `src/jmds_prepare/profiles/__init__.py`: export the Portuguese profile.
- `src/jmds_prepare/sources/jmds.py`: delegate shared validation while
  preserving the public English API.
- `src/jmds_prepare/pipelines/__init__.py`: expose no implementation; keep
  package importable.
- `tests/test_profiles_sources.py`: pin new source/profile layer rules.
- `tests/test_pipelines.py`: include metadata pipeline in import checks.
- `.gitignore`: ignore local Portuguese derived data/config where necessary.
- `README.md`, `docs/data-framework.md`, `docs/sources.md`, `BACKLOG.md`:
  document the command, outputs, evidence and limitations.

---

### Task 1: Portuguese profile and shared JMDS contracts

**Files:**
- Create: `tests/test_portuguese_profile.py`
- Create: `src/jmds_prepare/profiles/portuguese.py`
- Create: `src/jmds_prepare/sources/jmds_common.py`
- Modify: `src/jmds_prepare/profiles/__init__.py`
- Modify: `src/jmds_prepare/sources/jmds.py`
- Modify: `tests/test_profiles_sources.py`

**Interfaces:**
- Produces:
  `PortugueseProfile.jmds_protocol_path(jmds_root, split) -> Path`,
  `generated_wav_path(jmds_root, split, utt_id) -> Path`, and
  `coraa_metadata_path(coraa_root, split) -> Path`.
- Produces shared `validate_schema`, `validate_split`,
  `validate_unique_ids`, and `validate_utterance_ids`.
- Preserves `sources.jmds.read_jmds_protocol(path, split, *, language=None)`.

- [ ] **Step 1: Write failing profile and shared-contract tests**

```python
def test_portuguese_profile_pins_real_layout():
    root = Path("E:/JMDS")
    assert PORTUGUESE_PROFILE.generated_wav_path(
        root, "train", "T_0000025757"
    ) == (
        root / "dataset" / "Portugese_MLAAD_Generated"
        / "train" / "wav" / "T_0000025757.wav"
    )
    assert dict(PORTUGUESE_PROFILE.expected_generated_counts) == {
        "train": 1806, "dev": 602, "eval": 603
    }


def test_portuguese_profile_mappings_are_immutable():
    with pytest.raises(TypeError):
        PORTUGUESE_PROFILE.expected_coraa_counts["train"] = 1
```

Add characterization tests proving that the existing English reader still
rejects an unsupported split, wrong column order, malformed IDs and duplicates.

- [ ] **Step 2: Run the focused tests and observe the expected failure**

Run:

```powershell
python -m pytest tests/test_portuguese_profile.py tests/test_protocols.py -q
```

Expected: collection fails because `profiles.portuguese` and `jmds_common` do
not exist; existing protocol tests remain collectible.

- [ ] **Step 3: Implement the immutable profile**

Define:

```python
@dataclass(frozen=True)
class PortugueseProfile:
    language_code: str
    language_slug: str
    generated_dataset: str
    generated_dir_name: str
    generated_audio_subdir: str
    protocol_splits: tuple[str, ...]
    coraa_splits: tuple[str, ...]
    coraa_metadata_files: Mapping[str, str]
    expected_coraa_counts: Mapping[str, int]
    expected_generated_counts: Mapping[str, int]
    split_id_prefixes: Mapping[str, str]
    accepted_labels: frozenset[str]
    coraa_version: str
    coraa_revision: str
    coraa_license: str

    def jmds_protocol_path(self, jmds_root: Path, split: str) -> Path: ...
    def generated_wav_path(
        self, jmds_root: Path, split: str, utt_id: str
    ) -> Path: ...
    def coraa_metadata_path(self, coraa_root: Path, split: str) -> Path: ...
```

Instantiate `PORTUGUESE_PROFILE` with `language_code="por"`,
`generated_dataset="MLAAD"`, `generated_dir_name="Portugese_MLAAD_Generated"`,
CORAA revision `719c91226a79f5f9a8984145f15f29626eabc29a`, and the exact counts in
Global Constraints. Freeze mappings with private copies and
`MappingProxyType`.

- [ ] **Step 4: Extract parameterized JMDS validators without changing behavior**

Move schema/order, supported-split, prefix, duplicate and label checks to
`sources/jmds_common.py`:

```python
def validate_schema(
    actual: Sequence[str], expected: Sequence[str], *, source: str
) -> None: ...

def validate_split(split: str, supported: Iterable[str]) -> None: ...

def validate_unique_ids(
    frame: pd.DataFrame, *, source: str, id_column: str = "utt_id"
) -> None: ...

def validate_utterance_ids(
    frame: pd.DataFrame,
    *,
    split: str,
    prefixes: Mapping[str, str],
    source: str,
) -> None: ...
```

Make `sources/jmds.py` call these helpers with `ENGLISH_PROFILE`; preserve its
signature, exception types and message text pinned by existing tests.

- [ ] **Step 5: Verify green and the English regression boundary**

Run:

```powershell
python -m pytest tests/test_portuguese_profile.py tests/test_protocols.py tests/test_profiles_sources.py -q
```

Expected: all selected tests pass.

---

### Task 2: Strict CORAA and JMDS/MLAAD adapters

**Files:**
- Create: `tests/fixtures/coraa_metadata.csv`
- Create: `tests/fixtures/jmds_mlaad_metadata.csv`
- Create: `tests/test_coraa.py`
- Create: `tests/test_jmds_mlaad.py`
- Create: `src/jmds_prepare/sources/coraa.py`
- Create: `src/jmds_prepare/sources/jmds_mlaad.py`

**Interfaces:**
- Produces
  `read_coraa_metadata(path, *, split, profile) -> pd.DataFrame`.
- Produces
  `read_portuguese_generated(path, *, split, jmds_root, profile) ->
  pd.DataFrame`.
- Both return deterministic column order and preserve native strings.

- [ ] **Step 1: Create exact small fixtures**

The CORAA fixture header must be:

```csv
file_path,task,variety,dataset,accent,speech_genre,speech_style,up_votes,down_votes,votes_for_hesitation,votes_for_filled_pause,votes_for_noise_or_low_voice,votes_for_second_voice,votes_for_no_identified_problem,text
```

Include three `train/...wav` records: one annotation, one transcription with
empty vote fields, and one annotation_and_transcription. The JMDS fixture uses
the exact nine-column JMDS header and contains generated `por/MLAAD`, pristine
`por/CORAA`, and generated `eng/ASVspoof2024` rows with valid split prefixes.

- [ ] **Step 2: Write failing CORAA contract tests**

Tests must assert:

```python
frame = read_coraa_metadata(
    fixture, split="train", profile=PORTUGUESE_PROFILE
)
assert list(frame.columns) == [
    *CORAA_COLUMNS,
    "split", "metadata_source_file", "metadata_source_row",
]
assert frame["metadata_source_row"].tolist() == ["2", "3", "4"]
assert frame["file_path"].is_unique
```

Parametrize failures for missing/extra/reordered columns, wrong path split,
absolute paths, `..`, backslashes, duplicate/empty `file_path`, unsupported
split, and nonempty vote fields that are not valid non-negative integers.

- [ ] **Step 3: Observe CORAA tests fail, then implement minimally**

Run:

```powershell
python -m pytest tests/test_coraa.py -q
```

Expected before implementation: import failure for `sources.coraa`.

Implement `CORAA_COLUMNS` exactly as the fixture header and vectorized
validations. Read with `dtype=str, keep_default_na=False`; do not use
`iterrows`. Accept empty vote fields, preserve all native values and add
traceability columns.

- [ ] **Step 4: Write failing JMDS/MLAAD contract tests**

Tests must prove:

```python
frame = read_portuguese_generated(
    fixture,
    split="train",
    jmds_root=jmds_root,
    profile=profile_with_fixture_count,
)
assert set(frame["language"]) == {"por"}
assert set(frame["label"]) == {"generated"}
assert set(frame["dataset"]) == {"MLAAD"}
assert frame["audio_path"].map(Path).map(Path.is_file).all()
```

Also assert exclusion of pristine CORAA and English rows, exact expected count,
one-to-one WAV resolution, no duplicate IDs or resolved paths, schema/order
validation and rejection of wrong dataset/language/label in the selected
subset.

- [ ] **Step 5: Observe JMDS tests fail, then implement minimally**

Run:

```powershell
python -m pytest tests/test_jmds_mlaad.py -q
```

Expected before implementation: import failure for `sources.jmds_mlaad`.

Implement full-protocol validation before filtering. Preserve all nine native
columns; append `split`, `metadata_source_file`, `metadata_source_row` and
`audio_path`. Require each selected WAV to exist, remain under the exact
generated split root and resolve uniquely.

- [ ] **Step 6: Verify both adapters and English protocols**

Run:

```powershell
python -m pytest tests/test_coraa.py tests/test_jmds_mlaad.py tests/test_protocols.py -q
```

Expected: all selected tests pass.

---

### Task 3: Content-idempotent publication

**Files:**
- Create: `tests/test_metadata_publication.py`
- Create: `src/jmds_prepare/core/publication.py`

**Interfaces:**
- Produces `DestinationConflictError(FileExistsError)`.
- Produces
  `publish_artifact_set_idempotent(artifacts: Mapping[Path, bytes]) -> None`.
- Existing identical destinations are accepted; any divergent destination
  prevents new publication.

- [ ] **Step 1: Write failing publication tests**

Cover:

```python
publish_artifact_set_idempotent({csv_path: b"a,b\n1,2\n", json_path: b"{}\n"})
publish_artifact_set_idempotent({csv_path: b"a,b\n1,2\n", json_path: b"{}\n"})
assert csv_path.read_bytes() == b"a,b\n1,2\n"
```

Add tests where one existing destination differs, where publication fails
after one hard link, and where a stale temporary exists. Assert divergent
files are untouched and files newly linked by a failed call are rolled back.

- [ ] **Step 2: Run and observe import failure**

```powershell
python -m pytest tests/test_metadata_publication.py -q
```

Expected: import failure for `core.publication`.

- [ ] **Step 3: Implement preflight, fsync and atomic links**

For every destination, compare existing bytes first. If all existing files are
identical, create same-directory temporary files with flush/fsync, then use
`os.link(temp, destination)` so existing files are never overwritten. Track
links created by the current call and remove them on failure. Always remove
temporaries. Reject duplicate resolved destinations.

- [ ] **Step 4: Verify green**

```powershell
python -m pytest tests/test_metadata_publication.py -q
```

Expected: all tests pass.

---

### Task 4: Source profiles, comparison summary and provenance

**Files:**
- Create: `tests/test_metadata_profiling.py`
- Create: `src/jmds_prepare/pipelines/metadata_profiling.py`
- Create: `src/jmds_prepare/pipelines/metadata_provenance.py`

**Interfaces:**
- Produces `build_coraa_profile(frame) -> dict[str, Any]`.
- Produces `build_jmds_mlaad_profile(frame) -> dict[str, Any]`.
- Produces `build_portuguese_summary(coraa, jmds) -> dict[str, Any]`.
- Produces
  `build_metadata_provenance(*, profile, input_paths, sha256_file) ->
  dict[str, Any]`.

- [ ] **Step 1: Write failing profile JSON tests**

Assert each source profile contains:

```python
{
    "source": "CORAA",
    "sample_count": 3,
    "counts_by_split": {"train": 3},
    "fields": [
        {
            "name": "file_path",
            "description": ...,
            "origin": "CORAA metadata",
            "observed_type": "string",
            "nullable": False,
            "missing_count": 0,
            "distinct_count": 3,
        },
        ...
    ],
}
```

Low-cardinality categorical fields include sorted `value_counts`; `text`,
`file_path`, `utt_id`, `audio_path` and source-file fields must not expose
values and instead report length statistics. Vote fields report numeric
min/max/mean over nonempty values and missing counts.

- [ ] **Step 2: Write failing summary and provenance tests**

The summary must keep sources separate:

```python
assert summary["sources"]["CORAA"]["label_role"] == "pristine"
assert summary["sources"]["JMDS_MLAAD"]["label_role"] == "generated"
assert "sample_pairs" not in summary
assert summary["comparison_limitations"]["paired_samples"] is False
```

Provenance must contain exact input path, size, SHA-256, source version/license,
artifact schemas, field origins and a fixed statement that JMDS pristine IDs
were excluded. Inject a deterministic hash function in unit tests.

- [ ] **Step 3: Observe failures, implement deterministic builders, verify**

Run before and after implementation:

```powershell
python -m pytest tests/test_metadata_profiling.py -q
```

Expected first: imports fail. Expected after: all tests pass. Builders return
plain JSON-compatible values, sort all keys/value-count rows deterministically,
and never include timestamps.

---

### Task 5: Config, layout, pipeline and standalone CLI

**Files:**
- Create: `tests/test_portuguese_metadata_pipeline.py`
- Create: `tests/test_portuguese_metadata_cli.py`
- Create: `src/jmds_prepare/storage/metadata_layout.py`
- Create: `src/jmds_prepare/metadata_config.py`
- Create: `src/jmds_prepare/pipelines/metadata_extraction.py`
- Create: `src/jmds_prepare/portuguese_metadata.py`
- Create: `configs/portuguese-metadata.example.yaml`
- Create: `configs/portuguese-metadata.yaml`
- Modify: `tests/test_pipelines.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces `PortugueseMetadataConfig.load(path)` and `.validate(profile)`.
- Produces `PortugueseMetadataLayout` properties for all six artifacts.
- Produces `MetadataExtractionDeps`.
- Produces
  `extract_portuguese_metadata(config, deps, *, output=print) -> None`.
- Produces standalone
  `python -m jmds_prepare.portuguese_metadata --config <yaml>`.

- [ ] **Step 1: Write failing config and layout tests**

The YAML schema is exactly:

```yaml
jmds_root: E:\JMDS_open_v2_public\JMDS_open_v2_public
coraa_root: E:\JMDS_open_v2_public\future_sources\coraa-v1.1
output_root: E:\JMDS_open_v2_public\portuguese_preparation
```

Reject unknown/missing keys, non-scalar paths, missing protocol/metadata files,
equal input/output roots and output roots nested inside either source root.
Assert exact six layout paths under `manifests/` and `reports/`.

- [ ] **Step 2: Write failing dependency-injected pipeline test**

Use fixture readers and an in-memory publication recorder. Assert all three
CORAA splits and all three JMDS splits are concatenated in profile order,
expected counts are checked before payload creation, the six destination paths
are passed in one publication call and progress reports source and artifact
counts.

Add AST tests asserting `metadata_extraction.py` imports neither
`jmds_prepare.cli` nor `profiles.english`, and contains no comparison against
the literal language code `"por"`.

- [ ] **Step 3: Run and observe failures**

```powershell
python -m pytest tests/test_portuguese_metadata_pipeline.py tests/test_portuguese_metadata_cli.py -q
```

Expected: imports fail for config/layout/pipeline/CLI modules.

- [ ] **Step 4: Implement layout, strict config and pipeline**

Define:

```python
@dataclass(frozen=True)
class MetadataExtractionDeps:
    profile: PortugueseProfile
    read_coraa: Callable[..., pd.DataFrame]
    read_jmds_generated: Callable[..., pd.DataFrame]
    build_coraa_profile: Callable[[pd.DataFrame], dict[str, Any]]
    build_jmds_profile: Callable[[pd.DataFrame], dict[str, Any]]
    build_summary: Callable[[pd.DataFrame, pd.DataFrame], dict[str, Any]]
    build_provenance: Callable[..., dict[str, Any]]
    publish_artifacts: Callable[[Mapping[Path, bytes]], None]
    sha256_file: Callable[[Path], str]
```

Serialize CSV with UTF-8, `index=False`, `lineterminator="\n"` and JSON with
UTF-8, `ensure_ascii=False`, sorted keys, two-space indentation, no NaN and a
final newline. Build every payload before the single publication call.

- [ ] **Step 5: Implement thin standalone CLI**

`portuguese_metadata.main(argv)` accepts only required `--config`, loads and
validates it, wires `PORTUGUESE_PROFILE` and canonical collaborators, prints
one `error:` line and returns 1 for expected filesystem/schema/config/conflict
errors, otherwise returns 0. Do not modify the English `cli.py`.

- [ ] **Step 6: Add portable and local configs**

The example uses quoted placeholders. The local config uses the three exact
Windows paths shown in Step 1. Add
`configs/portuguese-metadata.yaml` and local output/data patterns to
`.gitignore` without ignoring the example or source code.

- [ ] **Step 7: Verify pipeline, CLI and architecture**

```powershell
python -m pytest tests/test_portuguese_metadata_pipeline.py tests/test_portuguese_metadata_cli.py tests/test_pipelines.py tests/test_layer_placement.py -q
```

Expected: all selected tests pass.

---

### Task 6: Documentation, real extraction and final verification

**Files:**
- Modify: `README.md`
- Modify: `docs/data-framework.md`
- Modify: `docs/sources.md`
- Modify: `BACKLOG.md`
- Modify: `.superpowers/sdd/progress.md`

**Interfaces:**
- Consumes the standalone CLI and six artifact paths from Task 5.
- Produces real local outputs under
  `E:\JMDS_open_v2_public\portuguese_preparation`.

- [ ] **Step 1: Document the metadata-only command and limitations**

Document:

```powershell
python -m jmds_prepare.portuguese_metadata --config configs/portuguese-metadata.yaml
```

List all six outputs, native row counts, the exact 1:1 mapping for 3,011 MLAAD
WAVs, absence of CORAA audio extraction, non-pairing of corpora, local-only
CORAA policy and the fact that Portuguese eval is inventoried but has no
automatically assigned experimental role.

- [ ] **Step 2: Run the full automated verification**

```powershell
python -m pytest -q
python -m compileall -q src
```

Expected: every test passes and compileall exits 0. Check IDE diagnostics for
all changed Python files and fix introduced errors.

- [ ] **Step 3: Execute the real metadata extraction**

```powershell
python -m jmds_prepare.portuguese_metadata --config configs/portuguese-metadata.yaml
```

Expected output reports CORAA 402,456 rows, JMDS/MLAAD 3,011 rows and six
published artifacts.

- [ ] **Step 4: Verify real artifacts independently**

Read the two CSVs and four JSONs and assert:

```python
assert len(coraa) == 402_456
assert coraa["file_path"].is_unique
assert len(jmds) == 3_011
assert jmds["utt_id"].is_unique
assert jmds["audio_path"].map(Path).map(Path.is_file).all()
assert summary["comparison_limitations"]["paired_samples"] is False
assert provenance["excluded_jmds_pristine_count"] == 1_000
```

Recompute every recorded input SHA-256 and compare it with provenance.

- [ ] **Step 5: Prove idempotence on real data**

Run the CLI a second time unchanged. Expected: success, identical hashes for
all six artifacts and no temporary files left behind.

- [ ] **Step 6: Final repository handoff**

Run `git status --short` and review only; do not stage, commit, delete the two
known English partial archives or publish any CORAA-derived artifact.


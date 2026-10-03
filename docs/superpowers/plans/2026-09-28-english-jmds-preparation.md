# English JMDS Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible English train/dev dataset whose JMDS metadata is
verified against ASVspoof 5 and whose pristine audio is downloaded, selectively
extracted, normalized, and audited.

**Architecture:** A small Python package separates protocol validation, Zenodo
acquisition, selective TAR extraction, manifest generation, audio processing,
and auditing. Raw files remain immutable under the data root; every derived WAV
and metadata field records its provenance and checksum. The official JMDS train
is training data, the official dev is validation data, and no final test result
is reported until the eval mismatch is resolved.

**Tech Stack:** Python 3.11+, pandas, requests, soundfile, scipy, scikit-learn,
PyYAML, pytest, standard-library `hashlib`, `tarfile`, and `pathlib`.

## Global Constraints

- Repository: `C:\Users\Marcos\Explainable-Methods-On-Deepfake-Audio`.
- Source JMDS:
  `E:\JMDS_open_v2_public\JMDS_open_v2_public`.
- Default data root: `E:\JMDS_open_v2_public\english_preparation`.
- Supported pristine splits in this phase: `train` and `dev`.
- Expected pristine counts: train `18_797`, dev `15_667`.
- Expected generated counts: train `65_424`, dev `54_808`.
- Never associate the public ASVspoof eval audio with JMDS eval IDs in this
  phase.
- Never overwrite or transcode source audio in place.
- Download one TAR at a time, validate its MD5, extract selected files, validate
  the extraction, then remove that TAR.
- Every sample must retain source path, source checksum, processed path,
  processed checksum, and field provenance.
- No git commit is performed unless the user explicitly requests it.

---

## Planned File Structure

```text
pyproject.toml
configs/
└── english.example.yaml
src/jmds_prepare/
├── __init__.py          # package version only
├── config.py            # typed configuration loading and path validation
├── protocols.py         # JMDS/ASVspoof parsing and cross-validation
├── zenodo.py            # record metadata and resumable archive download
├── extraction.py        # safe selective extraction from ASVspoof TAR files
├── manifest.py          # canonical sample manifest and provenance
├── audio.py             # immutable decoding/resampling/PCM export
├── audit.py             # duplicates, leakage, distributions, shortcut baseline
└── cli.py               # prepare-English command orchestration
tests/
├── fixtures/
│   ├── jmds_train.csv
│   ├── asvspoof_train.tsv
│   └── tiny_flac_T_aa.tar
├── test_config.py
├── test_protocols.py
├── test_zenodo.py
├── test_extraction.py
├── test_manifest.py
├── test_audio.py
├── test_audit.py
└── test_cli.py
```

## Task 1: Package, configuration, and test harness

**Files:**
- Create: `pyproject.toml`
- Create: `configs/english.example.yaml`
- Create: `src/jmds_prepare/__init__.py`
- Create: `src/jmds_prepare/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces:
  `PreparationConfig.load(path: Path) -> PreparationConfig`
- Produces:
  `PreparationConfig.validate() -> None`

- [ ] **Step 1: Write failing configuration tests**

Cover successful YAML loading, rejection of a missing JMDS root, rejection of
`eval` in `supported_splits`, and creation-free validation of output paths.

```python
def test_rejects_eval_split(tmp_path):
    cfg = PreparationConfig(
        jmds_root=tmp_path,
        data_root=tmp_path / "out",
        supported_splits=("train", "eval"),
    )
    with pytest.raises(ValueError, match="eval"):
        cfg.validate()
```

- [ ] **Step 2: Run the failing tests**

Run:

```powershell
python -m pytest tests/test_config.py -v
```

Expected: collection fails because `jmds_prepare.config` does not exist.

- [ ] **Step 3: Implement configuration and packaging**

Use an immutable dataclass with these fields:

```python
@dataclass(frozen=True)
class PreparationConfig:
    jmds_root: Path
    data_root: Path
    zenodo_record_id: int = 14498691
    supported_splits: tuple[str, ...] = ("train", "dev")
    sample_rate: int = 16_000
```

`validate()` must require the three JMDS protocol CSVs, the
`dataset/English_ASVspoof2024_Generated` directory, splits limited to
`{"train", "dev"}`, and positive sample rate. It must not create directories.

- [ ] **Step 4: Run the package tests**

Run:

```powershell
python -m pytest tests/test_config.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Review checkpoint**

Verify `git diff --check` succeeds. Do not commit without explicit user
authorization.

## Task 2: Protocol parsing and exact correspondence report

**Files:**
- Create: `src/jmds_prepare/protocols.py`
- Create: `tests/fixtures/jmds_train.csv`
- Create: `tests/fixtures/asvspoof_train.tsv`
- Create: `tests/test_protocols.py`

**Interfaces:**
- Produces:
  `read_jmds_protocol(path: Path, split: str) -> pd.DataFrame`
- Produces:
  `read_asvspoof_protocol(path: Path, split: str) -> pd.DataFrame`
- Produces:
  `validate_correspondence(jmds, asv, split) -> CorrespondenceReport`
- `CorrespondenceReport` contains `split`, expected and matched counts,
  missing IDs, extra IDs, and metadata mismatches.

- [ ] **Step 1: Write failing parser and validator tests**

Fixtures must include one pristine row, one generated row, and one deliberate
speaker mismatch. Tests must assert:

```python
assert report.pristine_matched == 1
assert report.generated_matched == 1
assert report.metadata_mismatches["T_0000000002"]["spk_id"] == (
    "T_0002",
    "T_9999",
)
```

Also test that JMDS `pristine` maps to ASVspoof `bonafide` and JMDS
`generated` maps to ASVspoof `spoof`.

- [ ] **Step 2: Run tests and observe failure**

Run:

```powershell
python -m pytest tests/test_protocols.py -v
```

Expected: import failure for `jmds_prepare.protocols`.

- [ ] **Step 3: Implement strict parsing**

JMDS must require exactly:

```python
JMDS_COLUMNS = [
    "spk_id", "utt_id", "gender", "codec", "attack_id",
    "label", "native", "language", "dataset",
]
```

ASVspoof must parse whitespace-delimited rows as:

```python
ASV_COLUMNS = [
    "spk_id", "utt_id", "gender", "codec", "codec_q",
    "codec_seed", "attack_tag", "attack_id", "key", "tmp",
]
```

Reject duplicate `utt_id`, unsupported labels, malformed IDs, and non-English
JMDS rows passed to the English validator.

- [ ] **Step 4: Implement correspondence rules**

For train/dev, require every English JMDS ID to exist in the matching ASVspoof
protocol. Require exact `spk_id` and gender equality. Treat
`pristine`/`bonafide` and `generated`/`spoof` as equivalent labels. Return all
differences in the report and provide `report.raise_for_errors()`.

- [ ] **Step 5: Run focused and full tests**

Run:

```powershell
python -m pytest tests/test_protocols.py -v
python -m pytest -q
```

Expected: all tests pass.

## Task 3: Zenodo metadata and sequential verified downloads

**Files:**
- Create: `src/jmds_prepare/zenodo.py`
- Create: `tests/test_zenodo.py`

**Interfaces:**
- Produces:
  `get_record_files(record_id: int) -> dict[str, RecordFile]`
- Produces:
  `download_verified(file: RecordFile, destination: Path) -> Path`
- `RecordFile` contains `name`, `size`, `checksum`, and `download_url`.

- [ ] **Step 1: Write failing tests with mocked HTTP**

Test parsing of a Zenodo API response, resumption after a partial file, restart
when the server ignores `Range`, MD5 success, and MD5 failure that preserves
the `.partial` file for diagnosis.

```python
def test_md5_mismatch_is_rejected(tmp_path, record_file, fake_session):
    with pytest.raises(ChecksumError):
        download_verified(record_file, tmp_path, session=fake_session)
    assert (tmp_path / f"{record_file.name}.partial").exists()
```

- [ ] **Step 2: Run tests and observe failure**

Run:

```powershell
python -m pytest tests/test_zenodo.py -v
```

Expected: import failure for `jmds_prepare.zenodo`.

- [ ] **Step 3: Implement record discovery**

Fetch `https://zenodo.org/api/records/{record_id}`. Select only:

```python
TRAIN_ARCHIVES = [f"flac_T_a{x}.tar" for x in "abcde"]
DEV_ARCHIVES = [f"flac_D_a{x}.tar" for x in "abc"]
```

Read file size, `md5:` checksum, and download URL from the API. Fail if any
required archive is absent.

- [ ] **Step 4: Implement resumable download and MD5 validation**

Write to `<name>.partial`, use `Range: bytes=<existing>-`, call
`response.raise_for_status()`, stream 8 MiB chunks, verify final size and MD5,
then atomically rename to `<name>`.

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_zenodo.py -v
python -m pytest -q
```

Expected: all tests pass without network access.

## Task 4: Safe selective pristine extraction

**Files:**
- Create: `src/jmds_prepare/extraction.py`
- Create: `tests/fixtures/tiny_flac_T_aa.tar`
- Create: `tests/test_extraction.py`

**Interfaces:**
- Produces:
  `extract_selected(archive: Path, wanted_ids: set[str], output: Path) -> ExtractionReport`
- `ExtractionReport` contains extracted IDs, missing IDs for that archive,
  duplicate members, and rejected unsafe members.

- [ ] **Step 1: Build a tiny fixture and failing tests**

The fixture must contain:

```text
flac_T/T_0000000001.flac
flac_T/T_0000000002.flac
../unsafe.flac
```

Assert only requested safe members are written, an existing destination with a
different checksum is rejected, and path traversal is reported.

- [ ] **Step 2: Run tests and observe failure**

Run:

```powershell
python -m pytest tests/test_extraction.py -v
```

Expected: import failure for `jmds_prepare.extraction`.

- [ ] **Step 3: Implement extraction**

Iterate TAR members without `extractall()`. Accept regular files only, derive
the utterance ID from `Path(member.name).stem`, and write directly to:

```text
<data_root>/raw/asvspoof5/<split>/<utt_id>.flac
```

Use a temporary output file, calculate SHA-256 while copying, then rename
atomically.

- [ ] **Step 4: Run extraction tests**

Run:

```powershell
python -m pytest tests/test_extraction.py -v
python -m pytest -q
```

Expected: all tests pass.

## Task 5: Canonical manifest

**Files:**
- Create: `src/jmds_prepare/manifest.py`
- Create: `tests/test_manifest.py`

**Interfaces:**
- Produces:
  `build_raw_manifest(config, reports) -> pd.DataFrame`
- Produces:
  `write_manifest_atomic(frame, destination) -> None`

- [ ] **Step 1: Write failing manifest tests**

Assert exact schema and provenance:

```python
MANIFEST_COLUMNS = [
    "utt_id", "spk_id", "gender", "language", "dataset", "split",
    "label", "attack_id", "source_path", "processed_path",
    "sha256_source", "sha256_processed", "metadata_source",
]
```

Test a pristine FLAC from ASVspoof, a generated WAV from JMDS, a missing file,
and duplicate IDs. Missing or duplicate files must fail rather than producing
partial output.

- [ ] **Step 2: Run tests and observe failure**

Run:

```powershell
python -m pytest tests/test_manifest.py -v
```

Expected: import failure for `jmds_prepare.manifest`.

- [ ] **Step 3: Implement manifest construction**

For pristine rows, use:

```text
raw/asvspoof5/<split>/<utt_id>.flac
```

For generated rows, use:

```text
<jmds_root>/dataset/English_ASVspoof2024_Generated/<split>/wav/<utt_id>.wav
```

Set `metadata_source` to `JMDS+ASVspoof5-verified`. Calculate source SHA-256
using streaming reads. Leave processed fields empty until Task 6.

- [ ] **Step 4: Run tests**

Run:

```powershell
python -m pytest tests/test_manifest.py -v
python -m pytest -q
```

Expected: all tests pass.

## Task 6: Immutable audio processing

**Files:**
- Create: `src/jmds_prepare/audio.py`
- Create: `tests/test_audio.py`

**Interfaces:**
- Produces:
  `process_audio(source: Path, destination: Path, sample_rate: int) -> AudioMetadata`
- `AudioMetadata` contains original and output sample rates, channels, frame
  counts, durations, peaks, RMS values, and SHA-256 checksums.

- [ ] **Step 1: Write failing signal-processing tests**

Generate temporary mono/stereo fixtures at 8 kHz and 22.05 kHz. Assert output
is mono, 16 kHz, PCM-16, duration differs by at most one output sample, source
bytes are unchanged, and repeated processing is byte-identical.

- [ ] **Step 2: Run tests and observe failure**

Run:

```powershell
python -m pytest tests/test_audio.py -v
```

Expected: import failure for `jmds_prepare.audio`.

- [ ] **Step 3: Implement deterministic processing**

Decode with `soundfile`, average channels in float64, resample with
`scipy.signal.resample_poly` using the reduced integer ratio, clip to
`[-1.0, 1.0]`, and write `WAV/PCM_16`. Write to:

```text
<data_root>/processed/english/<split>/<label>/<utt_id>.wav
```

Do not apply loudness normalization, silence trimming, denoising, or
augmentation in this phase.

- [ ] **Step 4: Update the manifest atomically**

Populate processed path/checksum and calculated technical metadata only after
the output WAV passes validation.

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_audio.py -v
python -m pytest -q
```

Expected: all tests pass.

## Task 7: Leakage and shortcut audit

**Files:**
- Create: `src/jmds_prepare/audit.py`
- Create: `tests/test_audit.py`

**Interfaces:**
- Produces:
  `audit_manifest(frame: pd.DataFrame, output_dir: Path) -> AuditSummary`
- Produces:
  `fit_technical_baseline(frame: pd.DataFrame, seed: int = 42) -> BaselineReport`

- [ ] **Step 1: Write failing audit tests**

Create a fixture with one cross-split speaker, one duplicate processed hash,
one class-specific sample rate, and separable technical features. Assert all
four conditions appear in the summary.

- [ ] **Step 2: Run tests and observe failure**

Run:

```powershell
python -m pytest tests/test_audit.py -v
```

Expected: import failure for `jmds_prepare.audit`.

- [ ] **Step 3: Implement deterministic audits**

Report:

- count by split, class, speaker, and attack;
- cross-split speaker intersections;
- duplicate source and processed hashes;
- missing values and paths;
- duration, RMS, peak, silence ratio, sample rate, and channel distributions.

Write both `audit.json` and sample-level `audit.csv`.

- [ ] **Step 4: Implement the technical shortcut baseline**

Use only duration, RMS, peak, silence ratio, original sample rate, and original
channel count. Fit a standardized logistic regression on train and evaluate on
dev. Report balanced accuracy, ROC-AUC, F1, and coefficients. Do not use audio
embeddings or content features.

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_audit.py -v
python -m pytest -q
```

Expected: all tests pass.

## Task 8: CLI orchestration and dry run

**Files:**
- Create: `src/jmds_prepare/cli.py`
- Create: `tests/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- Produces console command:
  `jmds-prepare validate-protocols`
- Produces console command:
  `jmds-prepare acquire-english`
- Produces console command:
  `jmds-prepare process-english`
- Produces console command:
  `jmds-prepare audit-english`

- [ ] **Step 1: Write failing CLI tests**

Mock network and filesystem-heavy boundaries. Assert `--dry-run` lists eight
archives, expected counts, required bytes, and output paths without downloading
or creating audio files.

- [ ] **Step 2: Run tests and observe failure**

Run:

```powershell
python -m pytest tests/test_cli.py -v
```

Expected: console entry point is absent.

- [ ] **Step 3: Implement orchestration**

`acquire-english` must execute, per archive:

```text
discover metadata
→ download/resume
→ verify size and MD5
→ selectively extract outstanding IDs
→ validate extracted outputs
→ delete verified TAR
→ continue
```

If any stage fails, preserve the TAR or `.partial` file and stop before the next
archive.

- [ ] **Step 4: Document exact commands**

Add to `README.md`:

```powershell
python -m pip install -e ".[dev]"
jmds-prepare validate-protocols --config configs/english.yaml
jmds-prepare acquire-english --config configs/english.yaml --dry-run
jmds-prepare acquire-english --config configs/english.yaml
jmds-prepare process-english --config configs/english.yaml
jmds-prepare audit-english --config configs/english.yaml
```

Document that dev is validation only and that final-test reporting remains
blocked.

- [ ] **Step 5: Run all automated checks**

Run:

```powershell
python -m pytest -q
python -m compileall src
git diff --check
```

Expected: tests pass, compilation succeeds, and no whitespace errors appear.

- [ ] **Step 6: Run the real protocol-only validation**

Run:

```powershell
jmds-prepare validate-protocols --config configs/english.yaml
```

Expected:

```text
train pristine: 18797/18797 matched
train generated: 65424/65424 matched
dev pristine: 15667/15667 matched
dev generated: 54808/54808 matched
eval: excluded (known ID mismatch)
```

- [ ] **Step 7: Execution checkpoint before large downloads**

Show the dry-run report, current free space, archive total size, selected
utterance counts, and output paths. Start the eight archive downloads only
after this checkpoint is accepted.

## Completion Evidence

Implementation is complete only when the following artifacts exist and agree:

```text
<data_root>/reports/correspondence.json
<data_root>/manifests/english_raw.csv
<data_root>/manifests/english_processed.csv
<data_root>/reports/audit.json
<data_root>/reports/audit.csv
<data_root>/reports/technical_baseline.json
```

The acquisition phase must report exactly 18,797 train and 15,667 dev pristine
files. The prepared data is suitable for training and validation, but not for a
final generalization claim until an independent English test set is defined.

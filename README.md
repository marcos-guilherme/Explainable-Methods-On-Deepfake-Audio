# Explainable Methods on Deepfake Audio

Reproducible preparation of the English JMDS v2 subset. JMDS supplies the
generated audio and metadata; the corresponding pristine train/dev audio comes
from the official ASVspoof 5 Zenodo record.

Only English audio is prepared today. Portuguese metadata (CORAA and
JMDS/MLAAD generated) and Mandarin metadata (AISHELL-3 and JMDS/ADD generated)
can be extracted without modifying source files (see
[Portuguese metadata extraction](#portuguese-metadata-extraction) and
[Mandarin metadata extraction](#mandarin-metadata-extraction)). For AISHELL-3
and CORAA, this repository also downloads the original source archives (see
[Future source acquisition](#future-source-acquisition-aishell-3-and-coraa)).

## Documentation

- [docs/data-framework.md](docs/data-framework.md) (Portuguese): layers,
  allowed dependencies, the English flow, temporary compatibility facades and
  how to add a corpus.
- [docs/sources.md](docs/sources.md): acquisition of external source datasets,
  official checksums versus observed SHA-256, and usage policies.
- [BACKLOG.md](BACKLOG.md) (Portuguese): status and open questions.

## Architecture overview

The package `src/jmds_prepare/` is split into layers; the full rules are in
[docs/data-framework.md](docs/data-framework.md).

- `core`: corpus-agnostic primitives (atomic writes, hashing, paths).
- `profiles`: immutable description of a subset; English rules live here.
- `storage`: layout, verified downloads, ledger, audio checks, space estimates
  and source-catalog acquisition.
- `sources`: strict readers for JMDS, ASVspoof 5 and Zenodo metadata.
- `pipelines`: orchestration of the commands.
- `cli`: argument parsing and wiring of the pieces above.

`jmds_prepare.protocols`, `jmds_prepare.zenodo` and the private aliases in
`jmds_prepare.cli` are temporary compatibility facades; new code should import
from the layers. The English commands keep their names and output layout.

## Setup

Python 3.11 or newer is required. From this repository:

```powershell
python -m pip install -e ".[dev]"
Copy-Item configs/english.example.yaml configs/english.yaml
```

Edit `configs/english.yaml` so that:

- `jmds_root` points to the extracted JMDS v2 public release;
- `asvspoof_protocol_root` contains `ASVspoof5.train.tsv`,
  `ASVspoof5.dev.track_1.tsv`, and `ASVspoof5.eval.track_1.tsv`;
- `data_root` is a separate writable destination with enough free space.

Raw JMDS and ASVspoof source files are treated as immutable.

## Commands

Run protocol correspondence first:

```powershell
python -m jmds_prepare.cli validate-protocols --config configs/english.yaml
```

This validates exact ID/metadata correspondence for train and dev and writes
`reports/correspondence.json`. Only `train` and `dev` are supported outputs.
Evaluation remains excluded. The English JMDS eval subset and the official ASV
eval protocol are parsed strictly (schema, IDs, duplicates and labels) before
recording the known mismatch; this does not claim strict semantic validation
of every non-English JMDS row.
Freshness covers SHA-256 of the exact bytes and canonical content fingerprints
for the complete train/dev/eval protocols, plus the English compared-subset
fingerprints. Acquisition rejects the report if any covered value changes.

Before downloading anything large, inspect the mandatory dry run:

```powershell
python -m jmds_prepare.cli acquire-english --config configs/english.yaml --dry-run
```

It may fetch the small Zenodo metadata record and read local audio headers, but
it creates and deletes nothing. It reports exact TAR download bytes; the
maximum active TAR working set (current TARs, partials and `.invalid*`
quarantines plus the largest remaining TAR); and the conservative total
acquisition peak: extracted FLAC bytes + current TAR/partial/quarantine bytes
+ an upper bound for retained FLACs equal to the official sizes of remaining
TARs + the largest active remaining TAR. It also reports calculated generated
PCM-16 output payload bytes
from actual WAV sizes after representative beginning/middle/end headers in
each split confirm the target PCM-16 mono 16 kHz format; pristine output bytes
from per-file headers when extracted, otherwise `pending until extraction`;
and the maximum of one TAR partial or one calculated output temporary. The
active TAR working set remains separate from the total acquisition peak. Only after
reviewing that output, explicitly omit `--dry-run` to acquire data:

```powershell
python -m jmds_prepare.cli acquire-english --config configs/english.yaml
python -m jmds_prepare.cli process-english --config configs/english.yaml
python -m jmds_prepare.cli audit-english --config configs/english.yaml
```

The short entry point `jmds-prepare` is also installed, but it may not be on
`PATH` in some Windows/Pyenv setups. The `python -m` form above uses the active
interpreter directly and avoids that dependency.

Before writing any WAV or manifest, `process-english` reads the source audio
headers (restoring their access/modification times) and requires: each missing
PCM-16 mono WAV rounded up to 4096-byte clusters, the largest planned WAV once
more as its temporary candidate, twice `max(1 MiB, 2048 bytes per row)` for
the manifest and its temporary, and a safety reserve of `max(1 GiB, 1%)` of
that subtotal. It refuses to start, listing required, free and each part,
when free space at `data_root` (or its nearest existing ancestor) is smaller.

Acquisition requires a successful, current correspondence report. Archives are
downloaded/resumed one at a time, checked by size and MD5, and deleted only
after a safe FLAC-member inventory and exact extraction validation. A valid
existing final TAR is reused without network access; an invalid final TAR is
quarantined for diagnosis before download/resume continues. A failure stops
the run and preserves the TAR or `.partial` file. Completion requires exactly
18,797 train and 15,667 dev decodable pristine FLACs before the raw manifest is
written. Every accepted FLAC is bound to TAR name/MD5, member path/size and
extracted SHA-256 in an atomic ledger. Reuse requires one unique ledger row,
the current Zenodo MD5 and matching FLAC bytes. Otherwise the member is
revalidated from the TAR. A crash after ledger publication resumes by
validating and deleting the retained TAR.

## Output layout

```text
data_root/
├── archives/                         # transient TAR/.partial files
├── raw/asvspoof5/{train,dev}/        # immutable extracted FLAC
├── processed/english/{train,dev}/
│   ├── generated/                    # deterministic mono PCM-16 WAV
│   └── pristine/
├── manifests/
│   ├── extraction_ledger.csv
│   ├── english_raw.csv
│   └── english_processed.csv
└── reports/
    ├── correspondence.json
    ├── provenance.json
    ├── audit.json
    ├── audit.csv
    └── technical_baseline.json
```

`provenance.json` records Zenodo ID, DOI, licence, version/date, official URLs,
archive size/MD5, protocol hashes, UTC validation time, and each manifest
field's origin. Missing official values are `null` with an explicit reason.
The manifest preserves `protocol_codec` and `source_group_id` only from ASV
evidence; unavailable values are empty. `attack_id` provenance is JMDS only;
the pipeline does not claim sample-level ASV validation for that field. Source
grouping ignores empty IDs.

The audit decodes each source and processed audio once into one reusable table,
then derives all three reports before content-idempotent set publication.
After a crash that published only part of the set, an identical rerun verifies
the existing content and completes missing files; a content conflict fails.
Progress is logged every 1,000 samples and at completion. Metrics include
source/output container format and subtype, RMS, RMS dBFS
(`20*log10(RMS)`, full scale 1.0), silence ratio, and `estimated_snr_db`.
The last is explicitly a proxy: overall mono RMS divided by the 10th
percentile of non-zero 20 ms frame RMS. It is not true SNR and is biased by
silence, speech dynamics, non-stationary noise and short clips. Unavailable
values are null with a reason column.
`audit.json` also contains structured split/label counts by protocol codec,
source format/subtype and output format/subtype, plus non-empty source groups
that cross splits or contain multiple samples.

Train is the model-fitting split. Dev is validation only. Because eval is
blocked, this pipeline has no final test set and must not support final
generalization claims.

## Future source acquisition (AISHELL-3 and CORAA)

Separate from the English commands, `jmds_prepare.source_acquisition` downloads
the original archives of future source datasets into a destination you choose.
It only acquires and verifies files. It does not extract, convert, build
manifests or integrate them with the English data; no adapter for these corpora
exists yet. Use one destination per dataset, outside `data_root`.

Set a writable location with enough free space, then always inspect the dry run
first. It makes no network requests and writes nothing:

```powershell
$SourcesRoot = "<writable path with enough free space>"
python -m jmds_prepare.source_acquisition --catalog configs/sources/aishell3.yaml --destination "$SourcesRoot\aishell3" --dry-run
python -m jmds_prepare.source_acquisition --catalog configs/sources/coraa-v1.1.yaml --destination "$SourcesRoot\coraa-v1.1" --dry-run
```

After reviewing the sizes, licence and usage policy printed by the dry run,
omit `--dry-run` to download:

```powershell
python -m jmds_prepare.source_acquisition --catalog configs/sources/aishell3.yaml --destination "$SourcesRoot\aishell3"
python -m jmds_prepare.source_acquisition --catalog configs/sources/coraa-v1.1.yaml --destination "$SourcesRoot\coraa-v1.1"
```

Each accepted file gets a receipt with its observed SHA-256. Only files with an
official checksum are reported as verified by it; the others are verified by
exact size and keep the observed SHA-256 as an observation. Details, including
the CORAA licence restrictions, are in [docs/sources.md](docs/sources.md).

## Portuguese metadata extraction

This step inventories native CORAA and JMDS/MLAAD generated metadata. It does
not extract CORAA audio, does not modify any source file, and does not claim a
sample-level relationship between the two corpora.

Copy the portable template and point the three paths at your local layout:

```powershell
Copy-Item configs/portuguese-metadata.example.yaml configs/portuguese-metadata.yaml
python -m jmds_prepare.portuguese_metadata --config configs/portuguese-metadata.yaml
```

Expected row counts: CORAA 402,456 (train 382,258, dev 7,522, test 12,676);
JMDS/MLAAD generated 3,011 (train 1,806, dev 602, eval 603). Every generated
row resolves one-to-one to an existing WAV under
`Portugese_MLAAD_Generated/{split}/wav/`. The 1,000 JMDS pristine Portuguese
rows are excluded from the generated manifest.

Six artifacts are published under `output_root`:

```text
output_root/
├── manifests/
│   ├── coraa_metadata.csv
│   └── jmds_mlaad_generated_metadata.csv
└── reports/
    ├── coraa_metadata_profile.json
    ├── jmds_mlaad_generated_metadata_profile.json
    ├── portuguese_metadata_summary.json
    └── portuguese_metadata_provenance.json
```

The summary keeps CORAA (pristine) and JMDS/MLAAD (generated) separate;
`paired_samples` is false. Portuguese `eval` is inventoried but has no
automatically assigned experimental role. CORAA-derived artifacts remain local
only and must not be committed or redistributed under the project's conservative
CC BY-NC-ND 4.0 policy. Re-running the command with unchanged inputs is
content-idempotent: identical artifacts are accepted, divergent ones fail.

## Mandarin metadata extraction

This step inventories native AISHELL-3 and JMDS/ADD generated metadata. It does
not extract AISHELL-3 WAV members from `data_aishell3.tgz`, does not modify any
source file, and does not claim a sample-level relationship between the two
corpora. The 4,410 pristine JMDS rows labelled `zho/AISHELL3` are excluded from
the generated manifest and are never mapped to AISHELL archive utterance names.

Copy the portable template and point the three paths at your local layout:

```powershell
Copy-Item configs/mandarin-metadata.example.yaml configs/mandarin-metadata.yaml
python -m jmds_prepare.mandarin_metadata --config configs/mandarin-metadata.yaml
```

Expected row counts: AISHELL-3 88,035 after metadata–WAV reconciliation (train
63,262, test 24,773 content lines matched to archive WAV members); JMDS/ADD
generated 24,642 (train 7,146, dev 7,497, eval 9,999). Every generated row
resolves one-to-one to an existing WAV under
`Chinese_ADD_Generated/{split}/wav/`. Sparse JMDS fields (`attack_id`, `gender`,
`spk_id` = `unk`, `codec` = `-`, `native` = `yes`) are recorded as published and
are not inferred.

AISHELL-3 is read with a single streaming pass over `data_aishell3.tgz`: textual
members (`spk-info.txt`, `{split}/content.txt`, train prosody labels) are parsed;
WAV members contribute member path and byte size only—no extraction or decode.
The pristine profile and provenance include an explicit reconciliation report.
For the test split, observed `test/content.txt` has 24,773 lines while the
official AISHELL-3 publication reports 23,262 test samples; the adapter explains
this divergence without silently dropping rows.

Six artifacts are published under `output_root`:

```text
output_root/
├── manifests/
│   ├── aishell3_metadata.csv
│   └── jmds_add_generated_metadata.csv
└── reports/
    ├── aishell3_metadata_profile.json
    ├── jmds_add_generated_metadata_profile.json
    ├── mandarin_metadata_summary.json
    └── mandarin_metadata_provenance.json
```

The summary keeps AISHELL-3 (pristine) and JMDS/ADD (generated) separate;
`paired_samples` is false. AISHELL-3 is Apache-2.0; derived metadata artifacts
should remain local and traceable to the licence. Re-running the command with
unchanged inputs is content-idempotent: identical artifacts are accepted,
divergent ones fail, and no temporary files remain under `manifests/` or
`reports/`.

Real-data execution is opt-in: copy `configs/mandarin-metadata.example.yaml`,
set `jmds_root`, `aishell_archive`, and `output_root` to paths outside the
English `data_root`, then run the command above. Local configs and derived
outputs are gitignored.

## XAI dataset preparation (English, Portuguese, Mandarin)

Standalone CLI (not wired into `jmds-prepare`):

```powershell
python -m jmds_prepare.xai_dataset eng `
  --output-root E:\xai_out\english `
  --seed 42 `
  --english-manifest E:\english_preparation\manifests\english_processed.csv

python -m jmds_prepare.xai_dataset por `
  --output-root E:\xai_out\portuguese `
  --seed 42 `
  --pristine-manifest E:\portuguese_preparation\manifests\coraa_metadata.csv `
  --generated-manifest E:\portuguese_preparation\manifests\jmds_mlaad_generated_metadata.csv `
  --coraa-train-rar-part1 E:\sources\coraa\train.part1.rar `
  --coraa-dev-zip E:\sources\coraa\dev.zip `
  --coraa-test-zip E:\sources\coraa\test.zip

python -m jmds_prepare.xai_dataset zho `
  --output-root E:\xai_out\mandarin `
  --seed 42 `
  --pristine-manifest E:\mandarin_preparation\manifests\aishell3_metadata.csv `
  --generated-manifest E:\mandarin_preparation\manifests\jmds_add_generated_metadata.csv `
  --aishell-archive E:\sources\aishell3\data_aishell3.tgz
```

Each run publishes four content-idempotent artifacts under `output_root`:
`manifests/<language>/xai_samples.csv` and three JSON reports under
`reports/<language>/`. English reuses existing processed WAVs; Portuguese and
Mandarin materialize only the selected subset into `data/<language>/`.

Documented limitations in `provenance.json`:

- **English**: paired selection by speaker (`paired_by_speaker`), but
  `paired_utterances=false`.
- **Portuguese/Mandarin**: `paired_samples=false`, class–corpus confound and
  external validation under corpus shift.
- **CORAA**: local-only, no redistribution (`CC-BY-NC-ND-4.0`).

After editing the three local YAML templates so that `manifest_path` points to
the published manifests, run the source→target matrix:

```powershell
$env:PYTHONPATH = "scripts"
python -m brspeech_xai.matrix `
  --eng-config scripts/configs/xai-eng-local.yaml `
  --por-config scripts/configs/xai-por-local.yaml `
  --zho-config scripts/configs/xai-zho-local.yaml `
  --output E:\xai_out\matrix-run `
  --with-xai
```

The command extracts each language/role embedding once, fits exactly one head
per source language, calibrates its threshold only on that source's
`calibration` role, and evaluates all nine source→target cells. Outputs are
separated into `embeddings/`, `models/`, `cells/`, `matrix/`, `xai/` and
`shift/`. Full H1/H2/H3 analysis runs only under `xai/<language>/`; the six
off-diagonal cells under `shift/` use H1 association over the complete target
test set and a reduced H2 occlusion budget, with no confirmatory H3. They are
explicitly labelled as external validation under corpus shift, not evidence of
universal deepfake explanations. Use `--reduced-per-quadrant N` to change the
off-diagonal H2 budget, `--xai-plots` to render figures, and `--force` to
rebuild embedding caches.

See [docs/data-framework.md](docs/data-framework.md) for architecture details.

## Layer-wise trilingual encoder XAI

The VM suite studies where the detector decision emerges by layer and how it
changes across English, Portuguese and Mandarin. It evaluates all 12
Transformer blocks of HuBERT Base, WavLM Base+ and Wav2Vec2 Base. The primary
explanation path is AttnLRP → DFT-LRP/STDFT-LRP; the existing H1/H2/H3 analysis
is an optional layer-12 diagonal audit and is disabled by default.

The Linux VM needs Python 3.11, project dependencies, a CUDA-compatible
PyTorch/Transformers environment, and mounts for the repository/configs,
manifest files, every WAV referenced by `processed_path`, and the persistent
output directory. All three configs must declare a separate `calibration`
split and a positive per-class calibration quota; in-sample calibration is
forbidden.

Run the mandatory preflight first:

```bash
PYTHONPATH=scripts python -m brspeech_xai.encoder_suite \
  --eng-config scripts/configs/xai-eng-local.yaml \
  --por-config scripts/configs/xai-por-local.yaml \
  --zho-config scripts/configs/xai-zho-local.yaml \
  --profiles hubert_base wavlm_base_plus wav2vec2_base \
  --output /mnt/results/layerwise-suite \
  --xai-per-class 25 \
  --dry-run
```

Inspect `execution_plan.json` for config/manifest hashes, disk-space formula,
108 probes, 324 cells and separate backprop estimates. Then run a one-profile
pilot by removing `--dry-run`, changing the output directory, and using:

```bash
--profiles hubert_base --xai-per-class 2
```

Verify AttnLRP/DFT/STDFT conservation, immutable-generation integrity, fixed
cohort IDs and all aggregate tables before the complete run. In the
one-profile pilot, `encoder_relevance_agreement.csv` is intentionally empty
with status `not_applicable_less_than_two_profiles`, and its plot is skipped;
the other five tables remain required. The complete command is:

```bash
PYTHONPATH=scripts python -m brspeech_xai.encoder_suite \
  --eng-config scripts/configs/xai-eng-local.yaml \
  --por-config scripts/configs/xai-por-local.yaml \
  --zho-config scripts/configs/xai-zho-local.yaml \
  --profiles hubert_base wavlm_base_plus wav2vec2_base \
  --output /mnt/results/layerwise-suite \
  --xai-per-class 25
```

Runs resume automatically from valid stage markers. Check `run_status.json`;
use `--force` only to recompute the complete DAG. Outputs include embeddings,
layer probes, 3×3 source→target cells, XAI/trace generations and suite
aggregates. Add `--classical-audit` only when the extra H1/H2/H3 cost is
intended.

Off-diagonal results are external validation under corpus shift, not causal
language effects. Emergence intervals use stratified percentile bootstrap;
encoder-agreement and reorganization summaries use normal 95% intervals of
the mean. Spectral language shift compares each off-diagonal target's mean
absolute-normalized band distribution with the same source/profile/layer/class
diagonal, reporting Jensen-Shannon and cosine distances without an estimated
confidence interval. Embeddings, XAI and final trace use the same configured
mono/resample/fixed-length waveform routine; STDFT artifacts carry their own
time↔time-frequency conservation certificate.

The deterministic local smoke does not demonstrate real checkpoint, CUDA or
audio compatibility. No local real-HuggingFace/GPU success is claimed; the VM
pilot remains the next validation step. Operational details are in
[scripts/README.md](scripts/README.md#suíte-layer-wise-trilíngue-na-vm).
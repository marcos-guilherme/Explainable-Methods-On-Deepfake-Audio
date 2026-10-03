# External source acquisition

This step only downloads and verifies the original archives of external
datasets. It is separate from the English JMDS commands. For the overall code
architecture see [data-framework.md](data-framework.md) (Portuguese).

Catalogs in `configs/sources/` describe each dataset: official page, licence
(and where it is declared), version/revision, usage policy and the exact
artifacts (URL, size, optional official checksum). The loader is strict:
unknown, missing or duplicated keys, non-HTTPS URLs and unsafe local names are
rejected. Local names are plain file names; `remote_path` keeps the upstream
location (for example `train_dividido/train.part1.rar`) for provenance only.

Current catalogs:

- `configs/sources/aishell3.yaml`: AISHELL-3 (OpenSLR SLR93), one archive.
- `configs/sources/coraa-v1.1.yaml`: CORAA v1.1, pinned to one Hugging Face
  revision (audio archives plus metadata CSVs).

## Usage

The command is `python -m jmds_prepare.source_acquisition`; it is not one of
the four English subcommands of `jmds_prepare.cli`. Always inspect the dry run
first. It makes no network requests and writes nothing. Choose a destination
outside the English `data_root`, with one directory per dataset:

```powershell
$SourcesRoot = "<writable path with enough free space>"
python -m jmds_prepare.source_acquisition --catalog configs/sources/aishell3.yaml --destination "$SourcesRoot\aishell3" --dry-run
python -m jmds_prepare.source_acquisition --catalog configs/sources/coraa-v1.1.yaml --destination "$SourcesRoot\coraa-v1.1" --dry-run
```

Omit `--dry-run` to download. Artifacts of one catalog are fetched one at a
time into `--destination`, under an exclusive `.source-acquisition.lock` file
that is removed when the run ends. If a crash leaves the lock behind, delete
it only after confirming no acquisition is running.

Space checks are conservative. Anything without a receipt-backed final file
(including complete `.partial` files and finals without receipts) counts its
full size, because it may be quarantined and downloaded again. The command
refuses to start if free space is below the remaining bytes plus
`--minimum-free-gb` (finite, non-negative; default 10, in units of 10^9
bytes). It checks again before each transfer and stops at the first error.
Expected errors print a single `error:` line and exit with status 1.

## Transfer and verification

Transfers request `Accept-Encoding: identity`, reject encoded responses and any
non-HTTPS URL in the redirect chain, resume `<name>.partial` with HTTP Range
requests, retry transient network errors, require the exact catalog size and
check the official checksum when one exists. Partials are fsynced before being
promoted. Invalid finals and partials are renamed to `*.invalid*` instead of
being overwritten.

Accepted files get an atomic `<name>.receipt.json` with the catalog identity
and SHA-256, official/licence URLs, usage policy, artifact URL, expected size
and checksum (with its source), observed size and SHA-256, `origin`
(`downloaded`, `resumed_download`, `promoted_partial` or
`adopted_existing_final`), `official_checksum_verified` and a UTC timestamp.

## Official checksum versus observed SHA-256

These are different things and the receipts keep them apart.

- Official checksum: a value published by the dataset's own distributor for
  that file (for CORAA, the Hugging Face LFS SHA-256 of each archive at the
  pinned revision). The download is compared with it, and the receipt has
  `official_checksum_verified: true`.
- Observed SHA-256: the hash computed locally from the bytes that were
  downloaded. It is recorded in every receipt, but when there is no official
  value to compare with it only proves what was received, not that it matches
  the distributor's file.

When no official checksum was found (the AISHELL-3 archive and the CORAA
metadata CSVs), the catalog stores the expected checksum as `null` with an
explicit note. Those files are verified by exact size only, and the receipt
records the observed SHA-256 with `official_checksum_verified: false`. The
final summary lists the two groups separately.

On reruns the receipt pins the observation, so divergent bytes are rejected. A
receipt that is unreadable or disagrees with the catalog (identity, licence or
usage policy) stops the run and is never overwritten.

## Scientific separation

Acquisition only preserves original archives. It never extracts them, modifies
them or mixes them with the English JMDS outputs under `data_root`. Use a
separate destination for each dataset. Dataset adapters and their protocols are
out of scope for this step.

## CORAA metadata inventory (local only)

After acquiring CORAA v1.1, the standalone command

```powershell
python -m jmds_prepare.portuguese_metadata --config configs/portuguese-metadata.yaml
```

reads the three native metadata CSVs and the JMDS/MLAAD generated protocol rows.
It does not extract CORAA audio archives, does not modify any source byte, and
does not map JMDS pristine IDs to CORAA paths. Six artifacts (two CSV manifests
and four JSON reports) are published under a separate `output_root`, outside the
English `data_root`.

Expected counts: CORAA 402,456 rows; JMDS/MLAAD generated 3,011 rows with
one-to-one WAV resolution under `Portugese_MLAAD_Generated`. The summary keeps
sources separate and records `paired_samples: false`. Provenance includes input
SHA-256 hashes, the excluded pristine count (1,000) and the conservative usage
policy.

Under the project's CC BY-NC-ND 4.0 operating policy, CORAA-derived artifacts
must remain local to this research: do not commit them to version control, do
not publish them, and do not redistribute them.

## Mandarin metadata inventory (local only, opt-in)

After acquiring AISHELL-3 and with JMDS v2 available locally, the standalone
command

```powershell
python -m jmds_prepare.mandarin_metadata --config configs/mandarin-metadata.yaml
```

streams `data_aishell3.tgz` without extracting or decoding WAV members, reads
JMDS/ADD generated protocol rows under `zho/generated/ADD`, and publishes six
artifacts under a separate `output_root`. It does not modify any source byte,
does not map JMDS pristine `zho/AISHELL3` IDs to AISHELL utterance names, and
excludes all 4,410 such pristine rows from the generated manifest.

Expected counts on real data: AISHELL-3 88,035 reconciled rows (train 63,262;
test split reconciliation records 24,773 observed `test/content.txt` lines
versus 23,262 official test samples); JMDS/ADD generated 24,642 rows (train
7,146, dev 7,497, eval 9,999) with one-to-one WAV resolution under
`Chinese_ADD_Generated/{split}/wav/`. Sparse ADD fields (`attack_id`, `gender`,
`spk_id` = `unk`, `codec` = `-`, `native` = `yes`) are preserved as published
and not inferred. The summary keeps sources separate and records
`paired_samples: false`. Provenance includes the AISHELL archive SHA-256/size,
protocol SHA-256 for each JMDS split, `excluded_jmds_pristine_count` (4,410),
and the AISHELL-3 Apache-2.0 licence.

Copy `configs/mandarin-metadata.example.yaml` to a local YAML with your paths;
local configs and derived outputs are gitignored. Re-runs with unchanged inputs
are content-idempotent.

## Licences and usage policy

The `usage_policy` in each catalog records the licence as declared by the
dataset's official source, plus this project's operating policy. It is a
working note for this research, not legal advice. Check the licence itself if
your use differs from what is described here.

- AISHELL-3: the catalog records Apache-2.0 as declared on the OpenSLR page.
  The policy keeps the archive byte-identical; redistribution should carry the
  licence and attribution notices.
- CORAA v1.1: the catalog records CC BY-NC-ND 4.0, declared in the official
  GitHub repository. The project policy is conservative: noncommercial use
  only, derived material may be produced for this research, and derived audio,
  features or subsets are not shared, published or redistributed.

Each receipt and the dry run print these fields, so the policy in force when a
file was accepted stays traceable.

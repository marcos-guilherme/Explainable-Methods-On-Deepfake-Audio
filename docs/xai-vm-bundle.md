# Portable XAI data bundle

The trilingual XAI experiments consume a self-contained data bundle so the
same selected cohort can be copied from a workstation to a VM without
rewriting host-specific paths.

## Layout

```text
xai-vm-bundle/
├── data/<language>/<role>/<class>/<sample_id>.wav
├── manifests/<language>/xai_samples.csv
└── bundle_receipt.json
```

The supported languages are English (`eng`), Portuguese (`por`), and Mandarin
(`zho`). Manifest `processed_path` values are POSIX-style paths relative to
the bundle root. Absolute paths remain supported for legacy manifests.

## Build and verify

```bash
python -m jmds_prepare.xai_bundle build \
  --eng-manifest /path/to/eng/xai_samples.csv \
  --por-manifest /path/to/por/xai_samples.csv \
  --zho-manifest /path/to/zho/xai_samples.csv \
  --output /path/to/xai-vm-bundle \
  --archive

python -m jmds_prepare.xai_bundle verify /path/to/xai-vm-bundle

# Retry archiving an already-published bundle:
python -m jmds_prepare.xai_bundle archive /path/to/xai-vm-bundle
```

`build` validates source manifests, unique sample IDs and destinations, WAV
format, expected profile counts, and available disk space before publishing
the bundle atomically. `--dry-run` reports the required file count and bytes
without copying data.

`--archive` streams the result to `xai-vm-bundle.tar.zst` and writes a matching
`.sha256` sidecar. Copy both files to the VM, change to the directory containing
the archive and sidecar, and verify the archive hash there with:

```bash
sha256sum -c xai-vm-bundle.tar.zst.sha256
```

Then extract it and run `verify` before starting experiments. If archiving
fails after the bundle is built, the standalone `archive` command verifies that
bundle before retrying the same streaming archive operation.

## Integrity and path safety

`bundle_receipt.json` records each bundled file, SHA-256 digest, byte size,
manifest, language, role, class, and aggregate counts. `verify` checks the
receipt against the extracted files and rejects missing, modified, duplicated,
or unsupported audio.

Relative manifest paths are accepted only when a `bundle_receipt.json` can be
found at or above the manifest directory. Resolved paths must remain inside
that bundle root; traversal paths such as `../outside.wav` fail closed.

Audio files must be mono, 16-bit PCM WAV at 16 kHz. The canonical cohort has
6,022 files per language (18,066 total); non-canonical fixture sizes require
an explicit build option intended for tests and development.

## Publication guarantees

The builder writes into a sibling staging directory. The final output
directory appears only after every copy and validation succeeds. A failed
build removes its staging directory and never replaces an existing bundle.
Archives and sidecars are published without replacement through same-directory
temporary files and atomic hard links. The archive destination filesystem must
support hard links; move the bundle to a suitable filesystem and run
`archive <bundle>` when the original filesystem does not.

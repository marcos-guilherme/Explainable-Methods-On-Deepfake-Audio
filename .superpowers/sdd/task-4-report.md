# Task 4 report

## Status

Implemented the redesigned relevance-shift and spectral-divergence main figures.
Both families are registered, rendered, referenced by the report, and removed
from the deferred set. Legacy group-conditioned figures remain as technical
detail, and the full comparison CSV/TeX exports remain available.

## RED

Command:

```bash
PYTHONPATH=scripts .venv-report/bin/python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "language_shift_main or spectral_divergence_keeps" -v
```

Result:

```text
2 failed, 322 deselected in 10.76s
```

The failures were the expected missing-builder behavior: no `errorbar` calls
for `language_shift_intervals` and no captured
`spectral_divergence_by_layer` figure.

## GREEN

Command:

```bash
PYTHONPATH=scripts .venv-report/bin/python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "language_shift_main or spectral_divergence_keeps" -v
```

Result:

```text
2 passed, 322 deselected in 10.73s
```

Final focused command:

```bash
PYTHONPATH=scripts .venv-report/bin/python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "language_shift or spectral_divergence or bootstrap_ci or report_reference or manifest" -q
```

Result:

```text
32 passed, 294 deselected in 153.38s
```

Additional regression command:

```bash
PYTHONPATH=scripts .venv-report/bin/python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "multiple_languages_render_every or write_report_tables_exports_full or comparison_summary_tex_column or comparison_tex_summaries_use_readable or comparison_tex_summaries_stay_bounded" -q
```

Result:

```text
5 passed, 321 deselected in 26.43s
```

Static checks:

```bash
git diff --check && PYTHONPATH=scripts .venv-report/bin/python -m py_compile \
  scripts/brspeech_xai/layerwise_report.py \
  scripts/tests/test_layerwise_report.py
```

Result: exit code 0, no output.

## Files

- `scripts/brspeech_xai/layerwise_report.py`
- `scripts/tests/test_layerwise_report.py`
- `.superpowers/sdd/task-4-report.md`

The generated report bundle under `escrita/relatorio-layerwise-xai/` was not
modified.

## Self-review

- Main relevance-shift panels use only `group == "all"` and draw persisted
  point estimates plus the persisted `ci_low`/`ci_high` endpoints.
- No density or sample distribution is inferred from confidence intervals.
- Main spectral panels use layer on x and Wasserstein distance in Hz on y,
  with deterministic pair ordering and one row per model.
- Captions state the question, visual encoding, and principal interpretive
  limit; spectral distance is explicitly not ranked as better or worse.
- Legacy group-conditioned profiles remain in the technical-detail subsection,
  while CSV/TeX comparison exports are independent of renamed display families.
- Narrative references point to the redesigned main labels; legacy labels are
  not used as main-result references.

## Concerns

No known Task 4 defect. A full-file regression run initially found five
expectation/export failures; all five were corrected and rerun successfully.
The final verification used the requested focused suite plus that exact
five-test regression set rather than repeating the full 12-minute file run.

## Review follow-up

Status: all Task 4 review findings were addressed. The group-conditioned legacy
profiles now follow one `\appendix` boundary in a technical annex; the main
comparisons section contains only the redesigned `group=all` shift and spectral
figures. Captions now describe the facet layout and mark/axis encodings.

### Review RED

Command:

```bash
PYTHONPATH=scripts .venv-report/bin/python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "language_shift_main_figure or spectral_main or shift_and_spectral_main_captions or legacy_group_profiles" -v
```

Result:

```text
4 passed, 2 failed, 324 deselected in 28.67s
```

The expected failures were the incomplete captions and the absence of an
appendix boundary for the legacy group-conditioned profiles. The four new
data-encoding/layout tests passed against the existing implementation and were
retained as stronger regression contracts.

### Review GREEN

Command:

```bash
PYTHONPATH=scripts .venv-report/bin/python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "language_shift_main_figure or spectral_main or shift_and_spectral_main_captions or legacy_group_profiles" -v
```

Result:

```text
6 passed, 324 deselected in 27.87s
```

Final focused command:

```bash
PYTHONPATH=scripts .venv-report/bin/python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "language_shift or spectral_divergence or bootstrap_ci or report_structure or manifest" -q
```

Result:

```text
33 passed, 297 deselected in 152.97s (0:02:32)
```

Main-section regression:

```bash
PYTHONPATH=scripts .venv-report/bin/python -m pytest \
  scripts/tests/test_layerwise_report.py::test_comparison_narrative_explains_method_and_links_each_result -q
```

Result:

```text
1 passed in 26.95s
```

Static checks:

```bash
git diff --check && PYTHONPATH=scripts .venv-report/bin/python -m py_compile \
  scripts/brspeech_xai/layerwise_report.py \
  scripts/tests/test_layerwise_report.py
```

Result: exit code 0, no output.

### Review self-review and concerns

- Main shift data are filtered to `group=all`; tests assert exact persisted
  estimate and CI endpoints and exclude class-conditioned sentinel values.
- Main spectral data are filtered to `group=all`; tests assert exact persisted
  estimate and CI endpoints.
- Both figures render safely when the estimate lies outside its persisted CI.
- Model rows, source columns, language-pair columns, and pair ordering follow
  the canonical report/model and `LANGUAGE_ORDER` contracts.
- The report contains exactly one appendix boundary when these technical
  profiles exist. This is the smallest placement compatible with a later
  multi-appendix refactor.
- No generated bundle files were modified.

No known remaining Task 4 concern. Verification was limited to the requested
focused suites and static checks; the full report test file was not rerun.

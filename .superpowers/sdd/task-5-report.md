# Task 5 report

## Status

Implemented the main faithfulness interval figure and the four-row results
synthesis table. The figure is registered in the normal renderer, removed from
the deferred path, included in the main report, and still fails closed when it
is planned without aligned faithfulness data. The synthesis is emitted as CSV
and TeX, referenced by the report, and included automatically in the bundle
manifest.

The generated bundle under `escrita/relatorio-layerwise-xai/` was not touched.

## RED evidence

Command:

```bash
PYTHONPATH=scripts .venv-report/bin/python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "faithfulness_figure or results_synthesis" -v
```

Result: exit 1; 2 failed, 330 deselected.

- `test_faithfulness_figure_exposes_scope_zero_line_and_both_controls` failed
  with `KeyError: 'layer_faithfulness_intervals'` because the planned family was
  still deferred.
- `test_results_synthesis_records_verified_transfer_counts` failed with
  `AttributeError` because `build_results_synthesis` did not exist.

## GREEN evidence

Initial focused command:

```bash
PYTHONPATH=scripts .venv-report/bin/python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "faithfulness or results_synthesis or write_report_tables" -q
```

Result: exit 0; 11 passed, 323 deselected in 52.77s.

Manifest and report-flow command:

```bash
PYTHONPATH=scripts .venv-report/bin/python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "full_comparison_bundle_manifest or report_references_redesigned_main_figures or faithfulness_and_synthesis_are_exported_and_referenced or render_emits_comparison_and_transfer_effect_artifacts" -q
```

Result: exit 0; 4 passed, 330 deselected in 86.94s.

Final combined focused command:

```bash
PYTHONPATH=scripts .venv-report/bin/python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "faithfulness or results_synthesis or write_report_tables or full_comparison_bundle_manifest or report_references_redesigned_main_figures or render_emits_comparison_and_transfer_effect_artifacts" -q
```

Result: exit 0; 14 passed, 320 deselected in 111.33s.

Diff hygiene:

```bash
git diff --check
```

Result: exit 0; no whitespace errors.

## Files

- `scripts/brspeech_xai/layerwise_report.py`
  - Added `_figure_layer_faithfulness_intervals`.
  - Registered the figure and removed the deferred-family exception.
  - Added `build_results_synthesis` and `_results_synthesis_tex`.
  - Exported `results_synthesis.csv` and `results_synthesis.tex`.
  - Referenced the synthesis table and faithfulness figure in the current
    report flow without adding another appendix boundary.
- `scripts/tests/test_layerwise_report.py`
  - Added RED/GREEN coverage for figure scope, both controls, zero reference,
    omission of Wilcoxon values, synthesis claim counts, exports, report
    references, manifest entries, and fail-closed rendering.
  - Expanded the faithfulness fixture to cover both true classes and both
    controls.
- `.superpowers/sdd/task-5-report.md`
  - This implementation report.

## Self-review

- The figure uses model columns and true-class rows, with `k` on the vertical
  axis and persisted mean differences and confidence limits on the horizontal
  axis.
- Only diagonal, final-layer rows with eight pairs per class are admitted; both
  top-minus-random and top-minus-bottom must be present.
- No Wilcoxon p-value is rendered in the main figure or its caption.
- Numeric synthesis claims come from `transfer_effect_claim`,
  `agreement_claim`, and `faithfulness_claim`; the relevance-shift row makes no
  numeric claim.
- The existing single `\appendix` boundary remains unchanged.
- Manifest inclusion is exercised through full bundle generation rather than
  by special-casing manifest code.

## Concerns

No known Task 5 blocker. Verification was intentionally limited to the focused
faithfulness, synthesis, table, manifest, and report-reference tests requested
for this task; the complete report test module was not run.

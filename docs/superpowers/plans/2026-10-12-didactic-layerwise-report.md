# Didactic Layer-wise Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the layer-wise report as a 16–20 page question-driven main narrative plus complete technical appendices, with seven effect- and uncertainty-oriented figures.

**Architecture:** Keep `ReportTables` and the validated CSV contracts as the scientific source of truth. Add pure reduction helpers for selected-layer effects and quantitative claims, then replace the current comparison builders and repetitive LaTeX sections while retaining detailed figures/tables in appendices. The report remains deterministic, read-only over experiment artifacts, atomically published, and compilable with two `pdflatex` passes.

**Tech Stack:** Python 3.11+, pandas, NumPy, matplotlib, pytest, LaTeX/pdfLaTeX.

## Global Constraints

- Main body target: 16–20 pages for the complete 3-model × 3-language bundle
  (user decision: keep the concise body; do not add content only to gain pages).
- Preserve every currently published cell, layer, metric, and source artifact in the PDF appendices or bundle CSVs.
- Do not rerun encoders, probes, AttnLRP, DFT-LRP, or STDFT-LRP.
- Do not persist or reconstruct bootstrap draw distributions from confidence limits.
- Keep `BOOTSTRAP_RESAMPLES=2000`, `BOOTSTRAP_SEED=42`, and `BOOTSTRAP_CI_LEVEL=0.95`.
- Describe off-diagonal results as transfer under joint corpus/language shift, never as a causal language effect.
- Keep figure generation deterministic and CPU-only.
- Keep output publication atomic and source result directories read-only.
- Keep all existing full CSV filenames and schemas stable.
- Every figure caption states the question, visual encoding, and principal interpretive limit.
- Use Portuguese visible text and the canonical model/language orders already defined by the report.

---

### Task 1: Freeze the editorial and reduction contracts

**Files:**
- Modify: `scripts/brspeech_xai/layerwise_report.py:255-284`
- Modify: `scripts/brspeech_xai/layerwise_report.py:450-520`
- Modify: `scripts/brspeech_xai/layerwise_report.py:3000-3105`
- Test: `scripts/tests/test_layerwise_report.py:425-445`
- Test: `scripts/tests/test_layerwise_report.py:4167-4320`

**Interfaces:**
- Consumes: existing `ReportTables.transfer_selection`, `ReportTables.diagonal_vs_offdiagonal`, `ReportTables.encoder_agreement`, and `ReportTables.layer_faithfulness`.
- Produces:
  - `MAIN_RESULT_FIGURES: tuple[str, ...]`
  - `APPENDIX_FIGURES: tuple[str, ...]`
  - `selected_transfer_effects(tables: ReportTables, metric: str) -> pd.DataFrame`
  - `transfer_effect_claim(frame: pd.DataFrame) -> Mapping[str, float | int]`
  - `agreement_claim(frame: pd.DataFrame) -> Mapping[str, float | int]`
  - `faithfulness_claim(frame: pd.DataFrame) -> pd.DataFrame`

- [ ] **Step 1: Write failing tests for figure families and selected-layer reduction**

```python
_MAIN_RESULT_FIGURES = (
    "transfer_performance_heatmaps",
    "transfer_delta_roc_auc",
    "transfer_delta_mcc",
    "encoder_agreement_heatmaps",
    "language_shift_intervals",
    "spectral_divergence_by_layer",
    "layer_faithfulness_intervals",
)


def test_selected_transfer_effects_reuses_the_roc_auc_selected_layer(
    full_comparison_tables,
):
    selected = report_module.selected_transfer_effects(
        full_comparison_tables, "delta_mcc"
    )
    expected = full_comparison_tables.transfer_selection.query("source != target")
    assert len(selected) == len(expected)
    assert (selected["layer"] == selected["selected_layer"]).all()
    assert set(selected["metric"]) == {"delta_mcc"}


def test_full_report_declares_seven_main_result_figures(full_comparison_tables):
    assert all(
        name in full_comparison_tables.planned_figures
        for name in _MAIN_RESULT_FIGURES
    )
```

- [ ] **Step 2: Run the tests and verify the contract is absent**

Run:

```bash
PYTHONPATH=scripts python -m pytest \
  scripts/tests/test_layerwise_report.py::test_selected_transfer_effects_reuses_the_roc_auc_selected_layer \
  scripts/tests/test_layerwise_report.py::test_full_report_declares_seven_main_result_figures -v
```

Expected: FAIL because the helper and new figure names do not exist.

- [ ] **Step 3: Add constants and the exact selected-layer join**

```python
MAIN_RESULT_FIGURES: tuple[str, ...] = (
    "transfer_performance_heatmaps",
    "transfer_delta_roc_auc",
    "transfer_delta_mcc",
    "encoder_agreement_heatmaps",
    "language_shift_intervals",
    "spectral_divergence_by_layer",
    "layer_faithfulness_intervals",
)

APPENDIX_FIGURES: tuple[str, ...] = (
    "performance_by_layer",
    "fixed_threshold_by_layer",
    "dft_relevance_heatmap",
    "class_relevance_by_layer",
    "decision_reorganization_by_layer",
    "encoder_agreement_by_layer",
    "stdft_examples",
    "conservation_diagnostics",
)


def selected_transfer_effects(
    tables: ReportTables, metric: str
) -> pd.DataFrame:
    selected = tables.transfer_selection.loc[
        tables.transfer_selection["source"] != tables.transfer_selection["target"],
        ["model", "source", "target", "selected_layer"],
    ]
    effects = tables.diagonal_vs_offdiagonal[
        tables.diagonal_vs_offdiagonal["metric"] == metric
    ]
    merged = effects.merge(
        selected,
        on=["model", "source", "target"],
        how="inner",
        validate="many_to_one",
    )
    return _ordered(
        merged[merged["layer"] == merged["selected_layer"]],
        ("model", "source", "target"),
    )
```

Update `_planned_figures` so complete comparisons add the corresponding main and appendix names, and add `layer_faithfulness_intervals` only when aligned faithfulness rows are available.

- [ ] **Step 4: Add quantitative claim helpers with finite/status filtering**

`transfer_effect_claim` must return `n`, `median`, `negative_n`,
`ci_below_zero_n`, and `ci_above_zero_n`. `agreement_claim` filters
`metric == "prediction_agreement"` and returns `n`, `median`, `minimum`, and
`maximum`. `faithfulness_claim` groups by model and counts total rows,
`ci_low > 0`, and `ci_high < 0`.

- [ ] **Step 5: Run contract and existing comparison tests**

Run:

```bash
PYTHONPATH=scripts python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "selected_transfer_effects or planned_figures or comparison" -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/brspeech_xai/layerwise_report.py scripts/tests/test_layerwise_report.py
git commit -m "refactor(report): define didactic results contracts"
```

---

### Task 2: Build the overview and transfer-effect figures

**Files:**
- Modify: `scripts/brspeech_xai/layerwise_report.py:3300-3560`
- Modify: `scripts/brspeech_xai/layerwise_report.py:4375-4430`
- Modify: `scripts/brspeech_xai/layerwise_report.py:4499-4512`
- Test: `scripts/tests/test_layerwise_report.py:4322-4400`

**Interfaces:**
- Consumes: `selected_transfer_effects`.
- Produces:
  - `_figure_transfer_heatmaps(...) -> FigureRecord`
  - `_figure_transfer_delta_roc_auc(...) -> FigureRecord`
  - `_figure_transfer_delta_mcc(...) -> FigureRecord`
  - `_figure_transfer_effect(..., metric: str, ...) -> FigureRecord`

- [ ] **Step 1: Write failing visual-contract tests**

```python
@pytest.mark.parametrize(
    ("family", "metric"),
    (
        ("transfer_delta_roc_auc", "ROC-AUC"),
        ("transfer_delta_mcc", "MCC"),
    ),
)
def test_transfer_effect_figures_show_selected_effects_and_zero_reference(
    full_comparison_tables, tmp_path, family, metric, monkeypatch
):
    zero_lines = []
    import matplotlib.pyplot as plt

    original = plt.Axes.axvline

    def spy(self, x, *args, **kwargs):
        zero_lines.append(float(x))
        return original(self, x, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "axvline", spy)
    records, captured = _capture_figures(
        full_comparison_tables, tmp_path / family
    )
    assert family in captured
    assert 0.0 in zero_lines
    assert metric in "\n".join(captured[family]["texts"])
    assert next(r for r in records if r.name == family).units == "adimensional"
```

Also assert that the overview keeps train on rows, evaluation on columns, and
that the MCC annotations come from the same ROC-AUC-selected layer.

- [ ] **Step 2: Run the tests and verify the new builders are missing**

Run:

```bash
PYTHONPATH=scripts python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "transfer_effect_figures or transfer_heatmap_caption" -v
```

Expected: FAIL with missing figure family/builder assertions.

- [ ] **Step 3: Refine the existing overview heatmap**

Keep `transfer_performance_heatmaps` as Figure 1. Use two metric rows
(ROC-AUC, MCC), one column per model, fixed `eng/por/zho` ordering, independent
color scales, value annotations, and selected-layer annotations. Do not create a
second cell selection for MCC.

- [ ] **Step 4: Implement the common horizontal dot-interval builder**

Use one row per off-diagonal cell at its ROC-AUC-selected layer. Plot
`estimate` on x, categorical `source→target` on y, `ci_low/ci_high` as horizontal
intervals, and facet by model. Draw `ax.axvline(0.0, ...)`. The ROC-AUC and MCC
wrappers differ only in metric, title, and caption.

- [ ] **Step 5: Register the builders and keep deterministic ordering**

```python
_FIGURE_BUILDERS.update(
    {
        "transfer_delta_roc_auc": _figure_transfer_delta_roc_auc,
        "transfer_delta_mcc": _figure_transfer_delta_mcc,
    }
)
```

- [ ] **Step 6: Run focused figure tests**

Run:

```bash
PYTHONPATH=scripts python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "transfer_performance_heatmaps or transfer_delta" -q
```

Expected: PASS with nonblank PDF and PNG artifacts.

- [ ] **Step 7: Commit**

```bash
git add scripts/brspeech_xai/layerwise_report.py scripts/tests/test_layerwise_report.py
git commit -m "feat(report): add transfer effect figures"
```

---

### Task 3: Replace the main encoder-agreement plot with heatmaps

**Files:**
- Modify: `scripts/brspeech_xai/layerwise_report.py:4080-4310`
- Modify: `scripts/brspeech_xai/layerwise_report.py:4499-4512`
- Test: `scripts/tests/test_layerwise_report.py:4355-4376`
- Test: `scripts/tests/test_layerwise_report.py:4444-4471`

**Interfaces:**
- Consumes: `ReportTables.encoder_agreement`.
- Produces:
  - `encoder_agreement_heatmap_data(frame: pd.DataFrame) -> pd.DataFrame`
  - `_figure_encoder_agreement_heatmaps(...) -> FigureRecord`
  - renamed appendix builder `_figure_encoder_agreement_by_layer(...)`.

- [ ] **Step 1: Write failing aggregation and scale tests**

```python
def test_encoder_agreement_heatmap_uses_layer_medians(full_comparison_tables):
    result = report_module.encoder_agreement_heatmap_data(
        full_comparison_tables.encoder_agreement
    )
    key = (
        (result["model_a"] == "hubert_base")
        & (result["model_b"] == "wavlm_base")
        & (result["source"] == "eng")
        & (result["target"] == "por")
        & (result["metric"] == "prediction_agreement")
    )
    source = full_comparison_tables.encoder_agreement.query(
        "model_a == 'hubert_base' and model_b == 'wavlm_base' "
        "and source == 'eng' and target == 'por' "
        "and metric == 'prediction_agreement'"
    )
    assert result.loc[key, "estimate"].item() == pytest.approx(
        source["estimate"].median()
    )


def test_encoder_agreement_heatmaps_do_not_share_incompatible_scales(
    full_comparison_tables, tmp_path
):
    _, captured = _capture_figures(
        full_comparison_tables, tmp_path / "agreement"
    )
    text = "\n".join(captured["encoder_agreement_heatmaps"]["texts"])
    for label in ("Acordo de decisão", "Kappa", "Spearman", "Distância JS"):
        assert label in text
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONPATH=scripts python -m pytest \
  scripts/tests/test_layerwise_report.py -k "encoder_agreement_heatmap" -v
```

Expected: FAIL because the heatmap reducer and builder do not exist.

- [ ] **Step 3: Implement the median reducer**

Group finite `estimate` values by model pair, source, target, and metric; emit
one median per 3×3 cell. Preserve `status` counts only in the CSV/appendix, not
in the heatmap value.

- [ ] **Step 4: Implement metric-separated heatmaps**

Use one panel per metric and model pair, fixed source/target order, metric-specific
limits, and annotations with two decimals. The caption must state that each cell
is the median across 12 layers. Keep the current per-layer line figure under the
new appendix name `encoder_agreement_by_layer`.

- [ ] **Step 5: Update manifest/order tests and run them**

Run:

```bash
PYTHONPATH=scripts python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "encoder_agreement or manifest_lists" -q
```

Expected: PASS and both main heatmap and appendix profile files appear in the
manifest when the comparison is complete.

- [ ] **Step 6: Commit**

```bash
git add scripts/brspeech_xai/layerwise_report.py scripts/tests/test_layerwise_report.py
git commit -m "feat(report): summarize encoder agreement as heatmaps"
```

---

### Task 4: Redesign relevance-shift and spectral-divergence figures

**Files:**
- Modify: `scripts/brspeech_xai/layerwise_report.py:4313-4497`
- Modify: `scripts/brspeech_xai/layerwise_report.py:4499-4512`
- Test: `scripts/tests/test_layerwise_report.py:4485-4748`

**Interfaces:**
- Consumes: `ReportTables.language_shift`, `ReportTables.spectral_divergence`.
- Produces:
  - `_figure_language_shift_intervals(...) -> FigureRecord`
  - `_figure_spectral_divergence_by_layer(...) -> FigureRecord`.

- [ ] **Step 1: Write failing tests for honest uncertainty encodings**

```python
def test_language_shift_main_figure_uses_points_and_persisted_intervals(
    full_comparison_tables, tmp_path, monkeypatch
):
    calls = []
    import matplotlib.pyplot as plt

    original = plt.Axes.errorbar

    def spy(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "errorbar", spy)
    render_report_figures(full_comparison_tables, tmp_path / "shift")
    assert calls
    record = next(
        r for r in render_report_figures(full_comparison_tables, tmp_path / "again")
        if r.name == "language_shift_intervals"
    )
    assert "violin" not in record.transformation.lower()
    assert "ci_low" in record.transformation


def test_spectral_divergence_keeps_hz_and_layer_on_separate_axes(
    full_comparison_tables, tmp_path
):
    _, captured = _capture_figures(full_comparison_tables, tmp_path / "spectral")
    text = "\n".join(captured["spectral_divergence_by_layer"]["texts"])
    assert "Hz" in text
    assert "Camada" in text
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
PYTHONPATH=scripts python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "language_shift_main or spectral_divergence_keeps" -v
```

Expected: FAIL on missing renamed builders.

- [ ] **Step 3: Implement the language-shift dot-interval layout**

The main figure uses group `all`, with model rows and source columns; each target
pair/layer is a point with its persisted bootstrap interval. Real/synthetic
groups remain available in `language_shift.csv` and an appendix profile. Do not
construct violin densities from layers or confidence limits.

- [ ] **Step 4: Implement Wasserstein small multiples**

Plot layer on x and Wasserstein estimate on y, with persisted CI, fixed language
pair order, and one row per model. Main figure uses group `all`; class-conditioned
profiles remain in the appendix/CSV. State explicitly that higher distance is
neither better nor worse.

- [ ] **Step 5: Run comparison figure regression tests**

Run:

```bash
PYTHONPATH=scripts python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "language_shift or spectral_divergence or bootstrap_ci" -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/brspeech_xai/layerwise_report.py scripts/tests/test_layerwise_report.py
git commit -m "feat(report): clarify relevance shift and spectral divergence"
```

---

### Task 5: Add the faithfulness figure and results-synthesis table

**Files:**
- Modify: `scripts/brspeech_xai/layerwise_report.py:4499-4512`
- Modify: `scripts/brspeech_xai/layerwise_report.py:5780-6068`
- Modify: `scripts/brspeech_xai/layerwise_report.py:6430-6493`
- Test: `scripts/tests/test_layerwise_report.py:4070-4165`
- Test: `scripts/tests/test_layerwise_report.py:4748-4805`

**Interfaces:**
- Consumes: `_aligned_layer_faithfulness(tables)`, claim helpers from Task 1.
- Produces:
  - `_figure_layer_faithfulness_intervals(...) -> FigureRecord`
  - `build_results_synthesis(tables: ReportTables) -> pd.DataFrame`
  - `_results_synthesis_tex(tables: ReportTables) -> str`
  - generated `tables/results_synthesis.csv`
  - generated `tables/results_synthesis.tex`.

- [ ] **Step 1: Write failing faithfulness and synthesis tests**

```python
def test_faithfulness_figure_exposes_scope_zero_line_and_both_controls(
    full_comparison_tables, tmp_path
):
    records, captured = _capture_figures(
        full_comparison_tables, tmp_path / "faithfulness"
    )
    text = "\n".join(captured["layer_faithfulness_intervals"]["texts"])
    assert "top-minus-random" in text
    assert "top-minus-bottom" in text
    record = next(r for r in records if r.name == "layer_faithfulness_intervals")
    assert "camada 12" in record.caption.lower()
    assert "diagonal" in record.caption.lower()
    assert "oito amostras por classe" in record.caption.lower()


def test_results_synthesis_records_verified_transfer_counts(
    full_comparison_tables,
):
    result = report_module.build_results_synthesis(full_comparison_tables)
    transfer = result[result["question"] == "Quanto o desempenho cai na transferência?"]
    joined = " ".join(transfer.astype(str).to_numpy().ravel())
    effects = full_comparison_tables.diagonal_vs_offdiagonal
    for metric in ("delta_roc_auc", "delta_mcc"):
        rows = effects[(effects["metric"] == metric) & (effects["status"] == "ok")]
        assert f"{int((rows['estimate'] < 0).sum())}/{len(rows)}" in joined
        assert f"{int((rows['ci_high'] < 0).sum())}/{len(rows)}" in joined
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
PYTHONPATH=scripts python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "faithfulness_figure or results_synthesis" -v
```

Expected: FAIL because the figure and synthesis builders do not exist.

- [ ] **Step 3: Implement the faithfulness dot-interval figure**

Use model columns and true-class rows. Put `k` on y, mean difference on x, and
different markers/colors for `top_minus_random` and `top_minus_bottom`. Draw a
zero line and use `ci_low/ci_high` directly. Do not display `wilcoxon_p` in the
main figure.

- [ ] **Step 4: Build the four-row synthesis table**

Rows cover transfer, encoder agreement, relevance shift/divergence, and
faithfulness. Columns are `question`, `evidence`, `result`, and `caution`.
Generate every number from the Task 1 helpers; no hard-coded observed value may
appear in the narrative source.

- [ ] **Step 5: Export CSV/TeX and run tests**

Run:

```bash
PYTHONPATH=scripts python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "faithfulness or results_synthesis or write_report_tables" -q
```

Expected: PASS and both synthesis files are listed in the bundle manifest.

- [ ] **Step 6: Commit**

```bash
git add scripts/brspeech_xai/layerwise_report.py scripts/tests/test_layerwise_report.py
git commit -m "feat(report): add faithfulness figure and result synthesis"
```

---

### Task 6: Replace repetitive body prose with a question-driven narrative

**Files:**
- Modify: `scripts/brspeech_xai/layerwise_report.py:4590-4610`
- Modify: `scripts/brspeech_xai/layerwise_report.py:4949-5775`
- Test: `scripts/tests/test_layerwise_report.py:2030-2310`
- Test: `scripts/tests/test_layerwise_report.py:4210-4305`

**Interfaces:**
- Consumes: seven main `FigureRecord`s, `build_results_synthesis`, claim helpers.
- Produces:
  - `_orientation_section(...) -> list[str]`
  - `_main_results_section(...) -> list[str]`
  - `_result_block(...) -> list[str]`
  - `_cross_cutting_limitations_section(...) -> list[str]`
  - `_short_conclusion_section(...) -> list[str]`.

- [ ] **Step 1: Write failing structure and anti-repetition tests**

```python
def test_full_report_body_is_question_driven_and_has_no_cell_by_cell_prose(
    full_comparison_tex,
):
    body = full_comparison_tex.split(r"\appendix", 1)[0]
    for heading in (
        "Visão geral do desempenho",
        "Quanto o desempenho cai na transferência?",
        "Até que ponto os encoders concordam?",
        "Como a relevância muda entre corpora e idiomas?",
        "As faixas apontadas sustentam a decisão?",
    ):
        assert heading in body
    assert body.count("Para \\texttt{hubert\\_base}") < 4
    assert body.count("eng$\\rightarrow$por") < 8


def test_each_main_result_block_has_result_interpretation_and_caution(
    full_comparison_tex,
):
    body = full_comparison_tex.split(r"\appendix", 1)[0]
    assert body.count(r"\textbf{Resultado:}") == 6
    assert body.count(r"\textbf{Interpretação:}") == 6
    assert body.count(r"\textbf{Cautela:}") == 6
```

- [ ] **Step 2: Run and verify current report fails the editorial contract**

Run:

```bash
PYTHONPATH=scripts python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "question_driven or main_result_block" -v
```

Expected: FAIL because the current body contains per-cell loops.

- [ ] **Step 3: Add orientation and glossary**

Insert a short “Como ler este relatório” section with diagonal, transfer,
ROC-AUC, MCC, bootstrap interval, and relevance definitions. Add a table of
contents. Keep protocol counts compact: one shared count statement plus an
exception list only when models/languages differ.

- [ ] **Step 4: Implement six result blocks in the approved order**

Order:

1. overview matrix;
2. ROC-AUC/MCC transfer effects;
3. encoder agreement;
4. relevance shift;
5. spectral divergence;
6. faithfulness.

Each block emits question, figure(s), at most three derived findings,
interpretation, caution, and links to appendix/CSV. Remove the 27-cell prose
loops from `_performance_section`, `_emergence_section`, `_relevance_section`,
`_reorganization_section`, and `_stdft_section` in the main body.

- [ ] **Step 5: Add synthesis, transverse limitations, and short conclusion**

Input `tables/results_synthesis.tex` after the result blocks. Move local
limitations into the corresponding blocks. Keep only cross-cutting limitations
at the end. The conclusion must not repeat every comparison paragraph.

- [ ] **Step 6: Run narrative tests**

Run:

```bash
PYTHONPATH=scripts python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "executive_summary or narrative or conclusion or limitations or question_driven" -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/brspeech_xai/layerwise_report.py scripts/tests/test_layerwise_report.py
git commit -m "refactor(report): make results question driven"
```

---

### Task 7: Move complete detail into navigable appendices

**Files:**
- Modify: `scripts/brspeech_xai/layerwise_report.py:5355-5572`
- Modify: `scripts/brspeech_xai/layerwise_report.py:5724-5775`
- Modify: `scripts/brspeech_xai/layerwise_report.py:6126-6428`
- Test: `scripts/tests/test_layerwise_report.py:2270-2838`
- Test: `scripts/tests/test_layerwise_report.py:4408-4483`

**Interfaces:**
- Consumes: existing detailed tables/figures and all generated CSVs.
- Produces:
  - `_appendices(...) -> list[str]`
  - appendix sections A–G with stable labels
  - `\label{appendix:start}` for page-boundary verification.

- [ ] **Step 1: Write failing appendix-completeness tests**

```python
def test_full_report_appendices_preserve_every_detail_family(
    full_comparison_tex,
):
    appendix = full_comparison_tex.split(r"\appendix", 1)[1]
    for label in (
        "app:protocolo",
        "app:desempenho",
        "app:emergencia",
        "app:relevancia",
        "app:comparacoes",
        "app:fidelidade",
        "app:stdft-conservacao",
    ):
        assert rf"\label{{{label}}}" in appendix
    for table in (
        "performance_by_layer",
        "emergence_layers",
        "encoder_agreement",
        "language_shift",
        "diagonal_vs_offdiagonal",
        "spectral_divergence",
        "conservation_by_layer",
    ):
        assert rf"\input{{tables/{table}.tex}}" in appendix


def test_full_csv_schemas_are_unchanged(full_comparison_tables, tmp_path):
    written = write_report_tables(full_comparison_tables, tmp_path / "tables")
    for family in _COMPARISON_FAMILIES:
        actual = pd.read_csv(tmp_path / "tables" / f"{family}.csv")
        expected = getattr(full_comparison_tables, family)
        assert tuple(actual.columns) == tuple(expected.columns)
        assert len(actual) == len(expected)
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
PYTHONPATH=scripts python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "appendices_preserve or csv_schemas_are_unchanged" -v
```

Expected: FAIL because there is no appendix boundary or appendix labels.

- [ ] **Step 3: Implement the seven appendix sections**

After conclusion, emit `\clearpage`, `\appendix`, and
`\label{appendix:start}`. Reuse complete table fragments and appendix figures.
Replace repeated narrative with concise provenance/context paragraphs; detailed
values remain in tables and CSVs.

- [ ] **Step 4: Shorten all body captions**

Remove repeated full model/cell inventories from captions. A caption contains:
question, encoding, and limit. Put the canonical scope once in orientation and
in each `FigureRecord`/manifest entry.

- [ ] **Step 5: Verify manifest completeness and deterministic ordering**

Run:

```bash
PYTHONPATH=scripts python -m pytest \
  scripts/tests/test_layerwise_report.py -k \
  "appendix or caption or manifest or deterministic or source_order" -q
```

Expected: PASS with every PDF/PNG/CSV/TeX file listed once and in sorted order.

- [ ] **Step 6: Commit**

```bash
git add scripts/brspeech_xai/layerwise_report.py scripts/tests/test_layerwise_report.py
git commit -m "feat(report): preserve technical detail in appendices"
```

---

### Task 8: Update report documentation and citation policy

**Files:**
- Modify: `scripts/README.md:276-340`
- Modify: `README.md:481-535`
- Modify: `scripts/tests/test_layerwise_report.py:2460-2490`
- Modify: `scripts/tests/test_layerwise_report.py:4264-4288`

**Interfaces:**
- Consumes: final bundle/figure/table names.
- Produces: documented main-body/appendix workflow and explicit primary-source citation policy.

- [ ] **Step 1: Write failing documentation tests**

```python
def test_docs_describe_question_driven_body_and_same_pdf_appendices():
    texts = (_read_text(_REPO / "README.md"), _read_text(_SCRIPTS / "README.md"))
    for text in texts:
        assert "16 a 20 páginas" in text
        assert "apêndices no mesmo PDF" in text
        assert "sete figuras principais" in text
        assert "não persistidos" in text and "draws bootstrap" in text
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
PYTHONPATH=scripts python -m pytest \
  scripts/tests/test_layerwise_report.py::test_docs_describe_question_driven_body_and_same_pdf_appendices -v
```

Expected: FAIL because the current docs describe the old report.

- [ ] **Step 3: Update both READMEs**

Document the seven figures, body/appendix split, selected-layer rule for transfer
effects, uncertainty limitations, build command, and unchanged CSV contracts.
State that suggested references must be checked against DOI/publisher metadata
before entering a bibliography; do not reproduce Consensus-generated prose or
its usage notice.

- [ ] **Step 4: Run documentation tests**

Run:

```bash
PYTHONPATH=scripts python -m pytest \
  scripts/tests/test_layerwise_report.py -k "official_docs or docs_describe" -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md scripts/README.md scripts/tests/test_layerwise_report.py
git commit -m "docs(report): explain didactic results workflow"
```

---

### Task 9: Verify the complete synthetic suite and real trilingual bundle

**Files:**
- Modify only if verification exposes a defect:
  - `scripts/brspeech_xai/layerwise_report.py`
  - `scripts/tests/test_layerwise_report.py`
- Regenerate locally: `escrita/relatorio-layerwise-xai/`

**Interfaces:**
- Consumes: three completed VM result directories.
- Produces: final deterministic report bundle and compiled `report.pdf`.

- [ ] **Step 1: Run the complete report/comparison tests**

Run in the VM container:

```bash
docker run --rm \
  -e PYTHONPATH=/workspace/scripts \
  -v "$HOME/audio_experiments/xai-trilingual:/workspace:ro" \
  -w /workspace \
  brspeech-deepfake:xai \
  python -m pytest \
    scripts/tests/test_layerwise_comparisons.py \
    scripts/tests/test_layerwise_report.py \
    -q -p no:cacheprovider \
  > "$HOME/audio_experiments/report-tests-didactic.log" 2>&1
status=$?
cat "$HOME/audio_experiments/report-tests-didactic.log"
exit "$status"
```

Expected: all tests PASS. Preserve full output in a VM log so an SSH disconnect
does not erase the failure trace.

- [ ] **Step 2: Regenerate the real bundle**

```bash
docker run --rm \
  -e PYTHONPATH=/workspace/scripts \
  -v "$HOME/audio_experiments/xai-trilingual/scripts:/workspace/scripts:ro" \
  -v "$HOME/audio_experiments/configs:/workspace/configs:ro" \
  -v "$HOME/audio_experiments/results:/workspace/results" \
  brspeech-deepfake:xai \
  python -m brspeech_xai.layerwise_report \
    --result /workspace/results/hubert_base__eng-por-zho__layerwise_xai__full \
    --result /workspace/results/wavlm_base__eng-por-zho__layerwise_xai__full \
    --result /workspace/results/wav2vec2_base__eng-por-zho__layerwise_xai__full \
    --output /workspace/results/layerwise_xai_report_consolidated
```

Expected: `Report bundle written to ...` and no staging sibling remains.

- [ ] **Step 3: Validate manifest hashes**

Run a Python check that iterates over every entry in `generated_files`, verifies
each SHA-256, and prints `missing=0` and `hash_mismatches=0`.

- [ ] **Step 4: Copy and compile locally**

```bash
scp -r \
  daniel.melo@35.184.231.121:~/audio_experiments/results/layerwise_xai_report_consolidated/. \
  escrita/relatorio-layerwise-xai/
cd escrita/relatorio-layerwise-xai
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

Expected: both passes exit 0 and `report.pdf` exists.

- [ ] **Step 5: Verify body length and PDF integrity**

Run:

```bash
pdfinfo report.pdf | rg '^(Pages|Page size|File size|PDF version):'
rg 'newlabel\\{appendix:start\\}' report.aux
```

Expected: A4 PDF; appendix starts after printed page 16 and no later than page 21
(body pages are numbered in arabic after the roman title/contents pages).
Search `report.log` for `Fatal error`, `Missing $ inserted`,
`Undefined control sequence`, and unresolved references; expect no matches.

- [ ] **Step 6: Verify scientific claims against generated CSVs**

Use `csv`/pandas to assert:

- ROC-AUC: 201/216 negative and 196/216 CIs below zero;
- MCC: 216/216 negative and 215/216 CIs below zero;
- agreement: 324 rows, median approximately 0.861, range 0.571–0.989;
- faithfulness IC-above-zero counts: HuBERT 10, Wav2Vec2 50, WavLM 11.

- [ ] **Step 7: Commit any verification-only fix, otherwise record no code change**

If a defect was fixed:

```bash
git add scripts/brspeech_xai/layerwise_report.py scripts/tests/test_layerwise_report.py
git commit -m "fix(report): satisfy final didactic bundle checks"
```

- [ ] **Step 8: Push and synchronize the VM branch**

```bash
git push origin feat/xai-trilingual
```

Expected: local, origin, and VM all point to the same final commit with clean
working trees; the generated local report remains untracked.

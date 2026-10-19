# Frequency Table Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the main spectral-relevance summary by language and add unambiguous layer-by-layer dominant-band tables to the appendix.

**Architecture:** Add one pure derivation for per-layer diagonal dominance and make the existing recurrence summary consume it. Keep rendering in `layerwise_report.py`: the main table pivots languages into rows and models into columns, while the appendix emits one compact table per language with layers as rows.

**Tech Stack:** Python 3, pandas, pytest, LaTeX generated as strings.

## Global Constraints

- Main-table comparisons must hold language fixed within each row.
- `n/12` means recurrence across encoder layers, never a selected layer index.
- Appendix order is English, Portuguese, Mandarin; model order is HuBERT Base, Wav2Vec2 Base, WavLM Base; layers are C1--C12.
- Preserve exact ties as co-dominant bands.
- Do not alter mel bands, recompute explanations, or imply acoustic causality.

---

### Task 1: Derive per-layer dominant bands

**Files:**
- Modify: `scripts/brspeech_xai/layerwise_report.py:7510-7579`
- Test: `scripts/tests/test_layerwise_report.py:5554-5645`

**Interfaces:**
- Consumes: `ReportTables.band_relevance`.
- Produces: `frequency_layer_dominance(tables: ReportTables) -> pd.DataFrame` with columns `model`, `language`, `layer`, `band`, `band_label`, `mean_mass`.
- Updates: `frequency_profile_claims(tables)` to aggregate recurrence from that result.

- [ ] **Step 1: Write failing coverage and tie tests**

Add tests that assert one result for every model-language-layer without ties and two results for an artificial exact tie:

```python
def test_frequency_layer_dominance_covers_each_diagonal_layer(
    full_comparison_tables,
):
    rows = report_module.frequency_layer_dominance(full_comparison_tables)
    observed = set(rows[["model", "language", "layer"]].itertuples(index=False, name=None))
    expected = {
        (model, language, layer)
        for model in full_comparison_tables.models
        for language in full_comparison_tables.languages
        for layer in report_module.LAYER_ORDER
    }
    assert observed == expected


def test_frequency_layer_dominance_preserves_exact_ties(full_comparison_tables):
    frame = full_comparison_tables.band_relevance.copy()
    model, language, layer = full_comparison_tables.models[0], full_comparison_tables.languages[0], 1
    mask = (
        (frame["model"] == model)
        & (frame["source"] == language)
        & (frame["target"] == language)
        & (frame["layer"] == layer)
        & (frame["measure"] == "absolute_normalized")
    )
    frame.loc[mask, "mean"] = 0.0
    chosen = frame.loc[mask].sort_values("band").index[:2]
    frame.loc[chosen, "mean"] = 1.0
    tables = dataclasses.replace(full_comparison_tables, band_relevance=frame)
    rows = report_module.frequency_layer_dominance(tables)
    cell = rows.query("model == @model and language == @language and layer == @layer")
    assert cell["band"].tolist() == sorted(frame.loc[chosen, "band"].tolist())
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest scripts/tests/test_layerwise_report.py -k "frequency_layer_dominance" -q
```

Expected: FAIL because `frequency_layer_dominance` does not exist.

- [ ] **Step 3: Implement the pure derivation**

Implement diagonal filtering, exact per-cell maximum selection, deterministic ordering, and the six-column contract:

```python
def frequency_layer_dominance(tables: ReportTables) -> pd.DataFrame:
    columns = ["model", "language", "layer", "band", "band_label", "mean_mass"]
    selected = tables.band_relevance[
        (tables.band_relevance["source"] == tables.band_relevance["target"])
        & (tables.band_relevance["measure"] == "absolute_normalized")
    ].copy()
    if selected.empty:
        return pd.DataFrame(columns=columns)
    selected["language"] = selected["target"]
    identity = ["model", "language", "layer"]
    selected["maximum"] = selected.groupby(identity)["mean"].transform("max")
    result = selected[selected["mean"] == selected["maximum"]].rename(
        columns={"mean": "mean_mass"}
    )
    model_order = {value: index for index, value in enumerate(tables.models)}
    language_order = {value: index for index, value in enumerate(tables.languages)}
    result["_model_order"] = result["model"].map(model_order)
    result["_language_order"] = result["language"].map(language_order)
    return (
        result.sort_values(
            ["_language_order", "_model_order", "layer", "band"],
            kind="mergesort",
        )
        .loc[:, columns]
        .reset_index(drop=True)
    )
```

Refactor `frequency_profile_claims` to group this frame by model, language, band and count unique dominant layers; merge mean band mass from the original diagonal data as today.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest scripts/tests/test_layerwise_report.py -k "frequency_layer_dominance or frequency_profile_claims" -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/brspeech_xai/layerwise_report.py scripts/tests/test_layerwise_report.py
git commit -m "refactor(report): derive layer band dominance"
```

---

### Task 2: Pivot the main summary by language

**Files:**
- Modify: `scripts/brspeech_xai/layerwise_report.py:7581-7612`
- Test: `scripts/tests/test_layerwise_report.py:5588-5616`

**Interfaces:**
- Consumes: `frequency_profile_claims(tables)`.
- Produces: `_frequency_profile_table(tables: ReportTables, claims: pd.DataFrame) -> str`.

- [ ] **Step 1: Replace the old shape assertions with a failing language-row contract**

```python
def test_frequency_profile_table_compares_encoders_within_language(
    full_comparison_tables,
):
    claims = report_module.frequency_profile_claims(full_comparison_tables)
    tex = report_module._frequency_profile_table(full_comparison_tables, claims)
    assert "Idioma & HuBERT Base & Wav2Vec2 Base & WavLM Base" in tex
    body = tex.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    assert body.count(r"\\") == len(full_comparison_tables.languages)
    for language in full_comparison_tables.languages:
        assert report_module._language_name(language) in body
    assert "recorrência" in tex.lower()
    assert "não o índice de uma camada" in tex.lower()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
python -m pytest scripts/tests/test_layerwise_report.py -k "frequency_profile_table" -q
```

Expected: FAIL because the table still has model-language rows.

- [ ] **Step 3: Implement the language-row pivot**

Change the helper signature and call site. For each language in `tables.languages`, create one row. For each model in `tables.models`, select the corresponding claim rows and format:

```python
cell = (
    f"{_join_pt(labels)} "
    f"({int(rows['layers_dominant'].iloc[0])}/{int(rows['n_layers'].iloc[0])})"
)
```

Use headers `["Idioma", *[_model_name(model) for model in tables.models]]`. State in the caption that recurrence is the number of layers where the band is dominant and not a layer index. Preserve co-dominant labels.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest scripts/tests/test_layerwise_report.py -k "frequency_profile_table or main_frequency_profile" -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/brspeech_xai/layerwise_report.py scripts/tests/test_layerwise_report.py
git commit -m "fix(report): compare frequency bands by language"
```

---

### Task 3: Add layer-by-layer appendix tables

**Files:**
- Modify: `scripts/brspeech_xai/layerwise_report.py:6961-7007`
- Test: `scripts/tests/test_layerwise_report.py`

**Interfaces:**
- Consumes: `frequency_layer_dominance(tables)`.
- Produces: `_frequency_layer_tables(tables: ReportTables) -> str`, containing one labeled table per language.

- [ ] **Step 1: Write failing appendix tests**

```python
def test_frequency_layer_tables_cover_every_language_model_and_layer(
    full_comparison_tables,
):
    tex = report_module._frequency_layer_tables(full_comparison_tables)
    for language in full_comparison_tables.languages:
        assert rf"\label{{tab:frequency-layer-{language}}}" in tex
        block = tex.split(rf"\label{{tab:frequency-layer-{language}}}", 1)[1]
        block = block.split(r"\end{table}", 1)[0]
        assert block.count(r"\\") == len(report_module.LAYER_ORDER) + 1
        for model in full_comparison_tables.models:
            assert report_module._model_name(model) in block
        for layer in report_module.LAYER_ORDER:
            assert f"C{layer}" in block


def test_relevance_appendix_includes_layer_dominance_tables(full_comparison_tex):
    appendix = full_comparison_tex.split(r"\label{app:relevancia}", 1)[1]
    appendix = appendix.split(r"\section{", 1)[0]
    for language in ("eng", "por"):
        assert rf"\label{{tab:frequency-layer-{language}}}" in appendix
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest scripts/tests/test_layerwise_report.py -k "frequency_layer_tables or relevance_appendix_includes_layer" -q
```

Expected: FAIL because `_frequency_layer_tables` does not exist.

- [ ] **Step 3: Implement three compact appendix tables**

For each language, build 12 rows. Each result cell joins all dominant labels for the model-language-layer. Use `_comparison_compact_table` with:

```python
caption = (
    f"Faixa(s) de maior massa média por camada em "
    f"{_language_name(language)}. Empates são mantidos; a Figura~"
    r"\ref{fig:dft_relevance_heatmap} preserva o perfil completo das 24 bandas."
)
label = f"tab:frequency-layer-{language}"
headers = ["Camada", *[_model_name(model) for model in tables.models]]
```

Append the generated string in `_relevance_appendix` after the main relevance figure and before the XAI-performance association subsection.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest scripts/tests/test_layerwise_report.py -k "frequency_layer or frequency_profile" -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/brspeech_xai/layerwise_report.py scripts/tests/test_layerwise_report.py
git commit -m "feat(report): add per-layer frequency tables"
```

---

### Task 4: Regression and publication verification

**Files:**
- Modify only if a verified regression requires it.
- Verify: `scripts/brspeech_xai/layerwise_report.py`
- Verify: `scripts/tests/test_layerwise_report.py`

**Interfaces:**
- Consumes all outputs from Tasks 1--3.
- Produces a tested implementation ready for report regeneration.

- [ ] **Step 1: Run the report test module**

Run:

```powershell
python -m pytest scripts/tests/test_layerwise_report.py -q
```

Expected: all tests pass, with only pre-existing skips.

- [ ] **Step 2: Check Python and diff integrity**

Run:

```powershell
python -m compileall -q scripts/brspeech_xai scripts/tests
git diff --check
```

Expected: exit code 0 from both commands.

- [ ] **Step 3: Inspect the final diff**

Confirm that:

- the main summary has one row per language;
- every appendix language table has C1--C12 and all available models;
- no test or prose describes `n/12` as a selected layer;
- no unrelated report behavior changed.

- [ ] **Step 4: Commit any final test-only corrections**

If Task 4 required changes:

```powershell
git add scripts/brspeech_xai/layerwise_report.py scripts/tests/test_layerwise_report.py
git commit -m "test(report): verify frequency table redesign"
```

If no changes were required, do not create an empty commit.

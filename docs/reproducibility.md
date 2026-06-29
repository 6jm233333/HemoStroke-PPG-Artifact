# Reproducibility Guide

This guide maps the study workflow to the repository commands. Paths are repository-relative unless otherwise noted.

## Reproduction Boundary

This repository supports code-path inspection, environment setup, configuration review, unit testing, and full numerical reproduction after authorized access to the required clinical datasets. Because MIMIC-III and MC-MED are credentialed clinical datasets, raw clinical records, waveforms, physician-reviewed onset anchors, derived arrays, checkpoints, and predictions are not redistributed.

Without restricted local data, users can run the unit tests and inspect the pipeline. Full reproduction of the reported tables and figures requires placing the credentialed datasets under the expected local directory structure described in `docs/data_access.md`.

## Historical LLM Replay Boundary

The original temporal-anchor extraction run used `gemini-3-pro-preview` between
2025-12-22 and 2025-12-29. Google discontinued that preview endpoint on
2026-03-09. The historical request wrapper and request-level decoding controls
were not preserved. Exact endpoint replay is therefore not possible. See
`configs/llm_historical_run.yaml` and `docs/llm_anchor_extraction.md` for the
preserved schema, prompt, lifecycle record, and explicit `not_preserved`
parameters.

## 1. Environment

```bash
conda env create -f environment.yml
conda activate hemostroke-ppg
pip install -e .
pytest
```

## 2. Cohort Mining and Temporal Anchoring

MIMIC note filtering and LLM chunk export:

```bash
python -m src.data.mimic.build_stroke_note_table --config configs/mimic_data.yaml
python -m src.data.mimic.export_llm_chunks --config configs/mimic_data.yaml
```

Use `prompts/stroke_timestamp_extraction.md` for structured onset extraction. The paper reports physician validation of extracted onset anchors; keep those reviewed outputs under `data/interim/` and do not commit them.

Normalize MC-MED radiology text, keep reviewed outputs local, and construct the
external Pleth segment index:

```bash
python -m src.data.mimic.anchor_waveforms_to_notes --config configs/mimic_data.yaml
python -m src.data.mcmed.build_llm_input --config configs/mcmed_data.yaml
python -m src.data.mcmed.build_stroke_index --config configs/mcmed_data.yaml
python -m src.data.mcmed.filter_prewarning_segments --config configs/mcmed_data.yaml
```

For MC-MED, `build_llm_input` standardizes `rads.csv` fields `Study` and
`Impression` into the canonical `Row_ID`, `CHARTTIME`, and `TEXT` extraction
schema. `CHARTTIME` is populated from `Result_time`, while `Order_time` is
retained locally for provenance checks. The reviewed local anchor table and local Pleth
waveform-segment manifest are then merged by `build_stroke_index`. See
`docs/mcmed_anchor_generation.md`.

## 3. PPG Feature Construction

```bash
python -m src.features.extract_ppg_features --dataset mimic --feature-config configs/feature_extraction.yaml --data-config configs/mimic_data.yaml
python -m src.features.extract_ppg_features --dataset mcmed --feature-config configs/feature_extraction.yaml --data-config configs/mcmed_data.yaml
```

Clean and engineer feature tables if needed:

```bash
python -m src.features.clean_feature_table --input-dir data/processed/mimic/features_raw --output-dir data/processed/mimic/features_cleaned
python -m src.features.clean_feature_table --input-dir data/processed/mcmed/features_raw --output-dir data/processed/mcmed/features_cleaned
python -m src.features.engineer_features --input-dir data/processed/mimic/features_cleaned --output-dir data/processed/mimic/features_engineered --baseline-method mean --baseline-frac 0.10 --baseline-min-rows 5
python -m src.features.engineer_features --input-dir data/processed/mcmed/features_cleaned --output-dir data/processed/mcmed/features_engineered --baseline-method mean --baseline-frac 0.10 --baseline-min-rows 5
python -m src.features.select_features --help
```

The final 17-feature set is fixed in `configs/feature_set_17.json`. Relative
features use the frozen paper-aligned rule `(x - mu_base) / abs(mu_base)`, where
`mu_base` is the mean over the initial stable period. The same MIMIC-defined
preprocessing rule is reused unchanged on MC-MED.

Waveform filtering parameters and the minimum beat count are read from
`configs/feature_extraction.yaml`. Relabeling preserves extracted beat-level
`Absolute_Time` values and reconstructs a time axis only for legacy tables that
do not contain usable beat timestamps.

## 4. Temporal Labeling and Horizon Packaging

The paper defines normal, warning, buffer, and lead-time regions around the documented onset anchor. Use:

```bash
python -m src.labels.relabel_time_windows --config configs/feature_extraction.yaml --dataset mimic --output-dir data/processed/mimic/features_labeled
python -m src.labels.relabel_time_windows --config configs/feature_extraction.yaml --dataset mcmed --output-dir data/processed/mcmed/features_labeled
```

The packaged `.npy` arrays expected by `src.models.train` are:

```text
train_data.npy, train_label.npy, train_pid.npy
val_data.npy, val_label.npy, val_pid.npy
test_data.npy, test_label.npy, test_pid.npy
```

For external MC-MED testing, `test_*` arrays are sufficient.

Build all main arrays with the standard entry point:

```bash
python -m src.datasets.build_main_horizon_sets --config configs/feature_extraction.yaml
```

This writes patient-disjoint MIMIC `train_*`, `val_*`, and `test_*` arrays and
MC-MED `test_*` arrays under the configured `240min`, `300min`, and `360min`
folders. Patient identifiers are resolved from `SUBJECT_ID` for MIMIC-III and
`MRN` for MC-MED when available; encounter and waveform identifiers are legacy
fallbacks only.

## 5. Main Model

ResNet-1D:

```bash
python -m src.models.train --config configs/training.yaml
python -m src.models.evaluate --config configs/training.yaml
```

## 6. Main Benchmark Table

Export the manuscript Table III benchmark reference values:

```bash
python scripts/reproduce/table3_main_benchmarks.py \
  --format markdown \
  --output outputs/table3_main_benchmarks.md
```

This command regenerates the reported table from the checked-in reference
values. It does not recompute clinical-score or structured-EHR baselines from
raw data. Recomputing those baselines requires the restricted local EHR tables,
physician-reviewed anchors, and patient-level partitions used in the manuscript.

## 7. Robustness and Figure Reproduction

```bash
python -m src.models.sensitivity --help
python scripts/qc/summarize_label_coverage.py --help
python scripts/reproduce/table3_main_benchmarks.py --help
python scripts/reproduce/figure_roc.py --help
python scripts/reproduce/figure_shap.py --help
python scripts/reproduce/figure_temporal_cases.py --help
python scripts/reproduce/figure_subgroup_f1.py --help
python scripts/reproduce/table4_false_alert_burden.py --help
```

Regenerated tables and figures should be written to `outputs/`. Manuscript source, compiled PDFs, and paper-ready figure files are intentionally outside the GitHub repository.

## 8. Frozen Operating Threshold and False-Alert Burden

The released model configs record the operating threshold selected on MIMIC
validation. `src.models.train` and `src.models.evaluate` apply the configured
threshold unchanged to internal reporting and frozen MC-MED testing. They do
not use `argmax` as an implicit 0.5 operating point.

For each high-risk non-stroke cohort and horizon, generate the Table IV row from
window-level prediction scores:

```bash
python scripts/reproduce/table4_false_alert_burden.py \
  --predictions outputs/controls/mimic_240min_predictions.csv \
  --output-csv outputs/controls/mimic_240min_false_alert.csv \
  --identifier-col file_id \
  --order-col window_index \
  --cohort MIMIC-III \
  --horizon-minutes 240 \
  --stroke-tpr 0.9833
```

The input CSV must retain a file-level packaging-group column and an explicit
within-file window-order column. The script reads the frozen threshold from
`configs/training.yaml`, reports packaged and NaN-free window counts, and
computes `ID+` as the fraction of file-level identifiers with at least five
consecutive positive windows.

## 9. Reproduction Checks

- Use patient-level splits only.
- Verify no patient ID appears in more than one fold partition.
- Keep MC-MED frozen during external testing.
- Report window counts before metrics.
- Report AUC and false-alert burden alongside F1/F2.
- Keep all restricted data outside version control.

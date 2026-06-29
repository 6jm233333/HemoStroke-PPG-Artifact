# HemoStroke-PPG-Artifact

Code repository for **In-Hospital Stroke Risk-State Classification from PPG-Derived Hemodynamic Features**.

This repository contains the reproducible code path for building anchor-aligned photoplethysmography (PPG) hemodynamic features and training pre-anchor stroke risk-state classification models across MIMIC-III and MC-MED cohorts. It is a source-code repository, not a manuscript archive: paper source, compiled PDFs, and paper-ready figure files are intentionally kept outside version control.

## Scope

Included:

- Cohort mining, onset anchoring, and waveform alignment utilities.
- MC-MED radiology-text normalization and reviewed-anchor-to-Pleth-index generation utilities.
- PPG feature extraction, feature cleaning, relative-feature engineering, and temporal relabeling.
- Standard main-experiment packaging for MIMIC train/validation/test arrays and frozen MC-MED test arrays.
- ResNet-1D main model and manuscript-aligned clinical-score and structured-EHR baseline reference outputs.
- Patient-level split checks, thresholded evaluation, false-alert burden, SHAP, ROC, subgroup, sensitivity, and quality-control scripts.
- Minimal tests for label logic, patient-level splitting, filter alignment, and model tensor shapes.

Not included:

- Restricted clinical records, waveforms, note-derived LLM outputs, physician-reviewed timestamps, derived patient tables, `.npy` datasets, checkpoints, predictions, manuscript source, manuscript PDFs, or paper figure PDFs.

## Main Results

The main manuscript comparison uses the proposed PPG model against available-component clinical-score and structured-EHR reference baselines: CHA2DS2-VASc_avail, Nwosu-EHR RF, Teoh-EHR XGB, and Yang-EHR ML.

The public repository provides the manuscript Table III values and an exporter
for regenerating the reported table. Recomputing the clinical-score and
structured-EHR baseline values requires restricted local EHR tables, reviewed
onset anchors, and the same patient-level partitions used in the study; those
non-redistributable inputs are not included.

| Horizon | Cohort    | Model        |          Accuracy |         Precision |            Recall |                F1 |                F2 |               AUC |
| ------: | --------- | ------------ | ----------------: | ----------------: | ----------------: | ----------------: | ----------------: | ----------------: |
| 240 min | MIMIC-III | Ours         | 0.6654 +/- 0.0047 | 0.6681 +/- 0.0041 | 0.9833 +/- 0.0106 | 0.7956 +/- 0.0027 | 0.8985 +/- 0.0062 | 0.6525 +/- 0.0530 |
| 240 min | MIMIC-III | CHA2DS2-VASc | 0.5550 +/- 0.0306 | 0.5890 +/- 0.0204 | 0.7440 +/- 0.0388 | 0.6570 +/- 0.0250 | 0.7068 +/- 0.0339 | 0.5061 +/- 0.0641 |
| 240 min | MIMIC-III | Nwosu-EHR RF | 0.5230 +/- 0.0352 | 0.5570 +/- 0.0378 | 0.8240 +/- 0.0344 | 0.6650 +/- 0.0314 | 0.7519 +/- 0.0367 | 0.6070 +/- 0.0940 |
| 240 min | MIMIC-III | Teoh-EHR XGB | 0.5730 +/- 0.0314 | 0.5730 +/- 0.0316 | 0.9920 +/- 0.0066 | 0.7260 +/- 0.0268 | 0.8654 +/- 0.0185 | 0.5655 +/- 0.0843 |
| 240 min | MIMIC-III | Yang-EHR ML  | 0.5690 +/- 0.0327 | 0.5710 +/- 0.0337 | 0.9920 +/- 0.0066 | 0.7250 +/- 0.0268 | 0.8645 +/- 0.0196 | 0.5767 +/- 0.0716 |
| 240 min | MC-MED    | Ours         | 0.8636 +/- 0.0350 | 0.9179 +/- 0.0020 | 0.9341 +/- 0.0413 | 0.9256 +/- 0.0211 | 0.9306 +/- 0.0332 | 0.6195 +/- 0.0807 |
| 240 min | MC-MED    | CHA2DS2-VASc | 0.6780 +/- 0.0469 | 0.8640 +/- 0.0191 | 0.7500 +/- 0.0503 | 0.8030 +/- 0.0321 | 0.7703 +/- 0.0456 | 0.4264 +/- 0.0771 |
| 240 min | MC-MED    | Nwosu-EHR RF | 0.6780 +/- 0.0469 | 0.9000 +/- 0.0416 | 0.7110 +/- 0.0495 | 0.7940 +/- 0.0385 | 0.7422 +/- 0.0487 | 0.5370 +/- 0.0497 |
| 240 min | MC-MED    | Teoh-EHR XGB | 0.6210 +/- 0.0500 | 0.9060 +/- 0.0390 | 0.6320 +/- 0.0559 | 0.7440 +/- 0.0441 | 0.6727 +/- 0.0550 | 0.2993 +/- 0.0530 |
| 240 min | MC-MED    | Yang-EHR ML  | 0.7240 +/- 0.0497 | 0.8610 +/- 0.0416 | 0.8160 +/- 0.0464 | 0.8380 +/- 0.0339 | 0.8246 +/- 0.0456 | 0.3098 +/- 0.0573 |
| 300 min | MIMIC-III | Ours         | 0.7860 +/- 0.0129 | 0.8028 +/- 0.0125 | 0.9647 +/- 0.0360 | 0.8759 +/- 0.0105 | 0.9269 +/- 0.0243 | 0.6124 +/- 0.0668 |
| 300 min | MIMIC-III | CHA2DS2-VASc | 0.5990 +/- 0.0298 | 0.6910 +/- 0.0196 | 0.7420 +/- 0.0372 | 0.7160 +/- 0.0235 | 0.7312 +/- 0.0334 | 0.4999 +/- 0.1313 |
| 300 min | MIMIC-III | Nwosu-EHR RF | 0.6790 +/- 0.0324 | 0.6810 +/- 0.0298 | 0.9930 +/- 0.0054 | 0.8080 +/- 0.0230 | 0.9096 +/- 0.0143 | 0.5063 +/- 0.1252 |
| 300 min | MIMIC-III | Teoh-EHR XGB | 0.6500 +/- 0.0273 | 0.6780 +/- 0.0273 | 0.9930 +/- 0.0054 | 0.8060 +/- 0.0212 | 0.9086 +/- 0.0135 | 0.5656 +/- 0.0630 |
| 300 min | MIMIC-III | Yang-EHR ML  | 0.6760 +/- 0.0298 | 0.6790 +/- 0.0304 | 0.9930 +/- 0.0054 | 0.8060 +/- 0.0214 | 0.9089 +/- 0.0144 | 0.5417 +/- 0.0444 |
| 300 min | MC-MED    | Ours         | 0.9229 +/- 0.0273 | 0.9802 +/- 0.0023 | 0.9401 +/- 0.0294 | 0.9595 +/- 0.0151 | 0.9478 +/- 0.0238 | 0.6847 +/- 0.1560 |
| 300 min | MC-MED    | CHA2DS2-VASc | 0.7330 +/- 0.0474 | 0.9380 +/- 0.0161 | 0.7620 +/- 0.0446 | 0.8410 +/- 0.0311 | 0.7917 +/- 0.0409 | 0.5427 +/- 0.0807 |
| 300 min | MC-MED    | Nwosu-EHR RF | 0.7790 +/- 0.0444 | 0.9690 +/- 0.0209 | 0.7870 +/- 0.0467 | 0.8690 +/- 0.0316 | 0.8177 +/- 0.0433 | 0.7031 +/- 0.0685 |
| 300 min | MC-MED    | Teoh-EHR XGB | 0.9110 +/- 0.0217 | 0.9070 +/- 0.0242 | 0.9630 +/- 0.0337 | 0.9340 +/- 0.0288 | 0.9513 +/- 0.0317 | 0.4450 +/- 0.0771 |
| 300 min | MC-MED    | Yang-EHR ML  | 0.9300 +/- 0.0268 | 0.9210 +/- 0.0247 | 0.9740 +/- 0.0276 | 0.9470 +/- 0.0260 | 0.9629 +/- 0.0270 | 0.4542 +/- 0.0475 |
| 360 min | MIMIC-III | Ours         | 0.8880 +/- 0.0028 | 0.8894 +/- 0.0013 | 0.9981 +/- 0.0025 | 0.9406 +/- 0.0015 | 0.9743 +/- 0.0020 | 0.6492 +/- 0.1147 |
| 360 min | MIMIC-III | CHA2DS2-VASc | 0.6400 +/- 0.0291 | 0.7950 +/- 0.0163 | 0.7380 +/- 0.0334 | 0.7650 +/- 0.0222 | 0.7487 +/- 0.0304 | 0.5149 +/- 0.0807 |
| 360 min | MIMIC-III | Nwosu-EHR RF | 0.7940 +/- 0.0265 | 0.7980 +/- 0.0265 | 0.9920 +/- 0.0059 | 0.8840 +/- 0.0186 | 0.9460 +/- 0.0117 | 0.4337 +/- 0.0630 |
| 360 min | MIMIC-III | Teoh-EHR XGB | 0.7970 +/- 0.0278 | 0.7930 +/- 0.0278 | 0.9890 +/- 0.0064 | 0.8800 +/- 0.0196 | 0.9424 +/- 0.0125 | 0.4792 +/- 0.0640 |
| 360 min | MIMIC-III | Yang-EHR ML  | 0.7990 +/- 0.0278 | 0.8010 +/- 0.0278 | 0.9950 +/- 0.0048 | 0.8880 +/- 0.0189 | 0.9490 +/- 0.0113 | 0.4184 +/- 0.0532 |
| 360 min | MC-MED    | Ours         | 0.9797 +/- 0.0192 | 0.9975 +/- 0.0008 | 0.9804 +/- 0.0054 | 0.9888 +/- 0.0025 | 0.9837 +/- 0.0042 | 0.7079 +/- 0.0748 |
| 360 min | MC-MED    | CHA2DS2-VASc | 0.7470 +/- 0.0441 | 0.9850 +/- 0.0008 | 0.7560 +/- 0.0444 | 0.8550 +/- 0.0291 | 0.7929 +/- 0.0393 | 0.7198 +/- 0.0573 |
| 360 min | MC-MED    | Nwosu-EHR RF | 0.9560 +/- 0.0204 | 0.9880 +/- 0.0041 | 0.9770 +/- 0.0145 | 0.9820 +/- 0.0107 | 0.9792 +/- 0.0125 | 0.8221 +/- 0.3502 |
| 360 min | MC-MED    | Teoh-EHR XGB | 0.0800 +/- 0.0293 | 1.0000 +/- 0.0000 | 0.0700 +/- 0.0268 | 0.1300 +/- 0.0464 | 0.0860 +/- 0.0323 | 0.1535 +/- 0.1955 |
| 360 min | MC-MED    | Yang-EHR ML  | 0.6900 +/- 0.0500 | 0.9840 +/- 0.0133 | 0.6980 +/- 0.0492 | 0.8160 +/- 0.0352 | 0.7411 +/- 0.0459 | 0.1023 +/- 0.2159 |

Full benchmark tables are in `docs/benchmark_results.md`. The 17-feature definition is documented in `docs/feature_dictionary.md`.

## Repository Layout

```text
configs/                 Experiment and path configuration
data/                    Local-only data mount point; tracked README only
docs/                    Data access, reproducibility, results, and supporting notes
prompts/                 LLM onset timestamp extraction prompt
scripts/qc/              Quality-control utilities
scripts/reproduce/       Table and figure reproduction entry points
src/                     Maintained implementation
tests/                   Lightweight unit tests
outputs/                 Local-only generated outputs; tracked README only
```

New work should use `src/`, `scripts/`, and `configs/`.

Legacy auxiliary sequence-model code is retained in `src/models/lstm.py` and
`configs/lstm_baseline.yaml` for compatibility with earlier experiments. It is
not part of the current manuscript benchmark.

## Historical LLM Extraction Run

The original onset-anchor extraction run used the model identifier
`gemini-3-pro-preview` between **2025-12-22** and **2025-12-29**. Google later
discontinued that preview endpoint on **2026-03-09**. The retired endpoint
cannot be replayed exactly.

The repository includes the prompt, canonical input/output schema, and
local post-processing code. The original request wrapper and request-level
decoding controls (`temperature`, `top_p`, `top_k`, `max_output_tokens`, seed,
and thinking configuration) were not preserved. They are therefore documented
as `not_preserved` rather than reconstructed from current defaults. See
`configs/llm_historical_run.yaml` and `docs/llm_anchor_extraction.md`.

## Installation

```bash
conda env create -f environment.yml
conda activate hemostroke-ppg
pip install -e .
pytest
```

Without Conda:

```bash
python -m venv .venv
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
pytest
```

## Data

Because MIMIC-III and MC-MED are credentialed clinical datasets, this repository cannot provide raw clinical records, waveforms, note-derived timestamps, derived feature arrays, checkpoints, or predictions. Full numerical reproduction requires authorized local access to the required datasets and placement of the files under the expected local directory structure. Without restricted data, users can still inspect the code path, configuration files, feature definitions, model implementation, split checks, and unit tests.

Full reproduction requires credentialed local access to:

- MIMIC-III Clinical Database.
- MIMIC-III Waveform Database Matched Subset.
- MC-MED waveform/EHR resources.

Expected local layout:

```text
data/raw/mimic/tables/
data/raw/mimic/waveforms/
data/raw/mcmed/
data/interim/mimic/
data/interim/mcmed/
data/processed/mimic/
data/processed/mcmed/
```

See `docs/data_access.md` before running experiments. The `.gitignore` excludes raw data, intermediate clinical artifacts, processed datasets, checkpoints, predictions, manuscript files, and generated outputs.

## Reproduction

Run from the repository root.

```bash
# 1. Build MIMIC note candidates and LLM chunks.
python -m src.data.mimic.build_stroke_note_table --config configs/mimic_data.yaml
python -m src.data.mimic.export_llm_chunks --config configs/mimic_data.yaml

# 2. After LLM extraction and physician review, anchor waveforms to onset times.
python -m src.data.mimic.anchor_waveforms_to_notes --config configs/mimic_data.yaml
python -m src.data.mcmed.build_llm_input --config configs/mcmed_data.yaml
python -m src.data.mcmed.build_stroke_index --config configs/mcmed_data.yaml
python -m src.data.mcmed.filter_prewarning_segments --config configs/mcmed_data.yaml

# 3. Extract, clean, engineer, and label PPG windows.
python -m src.features.extract_ppg_features --dataset mimic --feature-config configs/feature_extraction.yaml --data-config configs/mimic_data.yaml
python -m src.features.extract_ppg_features --dataset mcmed --feature-config configs/feature_extraction.yaml --data-config configs/mcmed_data.yaml
python -m src.features.clean_feature_table --input-dir data/processed/mimic/features_raw --output-dir data/processed/mimic/features_cleaned
python -m src.features.clean_feature_table --input-dir data/processed/mcmed/features_raw --output-dir data/processed/mcmed/features_cleaned
python -m src.features.engineer_features --input-dir data/processed/mimic/features_cleaned --output-dir data/processed/mimic/features_engineered --baseline-method mean --baseline-frac 0.10 --baseline-min-rows 5
python -m src.features.engineer_features --input-dir data/processed/mcmed/features_cleaned --output-dir data/processed/mcmed/features_engineered --baseline-method mean --baseline-frac 0.10 --baseline-min-rows 5
python -m src.labels.relabel_time_windows --config configs/feature_extraction.yaml --dataset mimic --output-dir data/processed/mimic/features_labeled
python -m src.labels.relabel_time_windows --config configs/feature_extraction.yaml --dataset mcmed --output-dir data/processed/mcmed/features_labeled

# 4. Build the paper's MIMIC train/validation/test arrays and frozen MC-MED test arrays.
python -m src.datasets.build_main_horizon_sets --config configs/feature_extraction.yaml

# 5. Train and evaluate the main model.
python -m src.models.train --config configs/training.yaml
python -m src.models.evaluate --config configs/training.yaml
```

Relative features use the paper-aligned frozen rule `(x - mu_base) / abs(mu_base)`,
where `mu_base` is the mean over the initial stable period. The same
MIMIC-defined preprocessing rule is applied unchanged to MC-MED. Main evaluation
uses the `evaluation.threshold` operating point selected on MIMIC validation and
applies it unchanged to internal reporting, frozen MC-MED evaluation, and
false-alert analysis.

The main array builder prefers true patient identifiers (`SUBJECT_ID` for
MIMIC-III and `MRN` for MC-MED) over waveform- or encounter-level fallbacks.

Figure, table, and robustness scripts:

```bash
python scripts/reproduce/table3_main_benchmarks.py --help
python scripts/reproduce/figure_roc.py --help
python scripts/reproduce/figure_shap.py --help
python scripts/reproduce/figure_temporal_cases.py --help
python scripts/reproduce/figure_subgroup_f1.py --help
python scripts/reproduce/table1_mcmed_cohort_stats.py --help
python scripts/reproduce/table4_false_alert_burden.py --help
python -m src.models.sensitivity --help
```

See `docs/reproducibility.md` for the staged checklist.

## Citation

Use `CITATION.cff` for repository-level citation metadata.

## Clinical Disclaimer

This code is for retrospective research reproduction. It is not a medical device, not a clinical alerting system, and must not be used for patient care without prospective validation, calibration, governance review, and local clinical oversight.

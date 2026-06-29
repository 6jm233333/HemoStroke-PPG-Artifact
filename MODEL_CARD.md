# Model Card: HemoStroke-PPG

## Model Details

- Model family: one-dimensional neural time-series classifiers.
- Main model: ResNet-1D over PPG-derived hemodynamic feature windows.
- Reference comparisons: available-component clinical-score and structured-EHR baselines reported in `docs/benchmark_results.md`.
- Task: retrospective within-case classification of pre-anchor warning windows versus earlier reference windows around clinically documented stroke anchors.
- Horizons: 240, 300, and 360 minutes before documented stroke onset anchors.

## Intended Use

This repository supports retrospective research reproduction and method comparison. It is intended for researchers with credentialed access to the required clinical waveform and EHR datasets.

## Out-of-Scope Use

The model must not be used as a clinical alarm, triage tool, diagnosis system, or patient-care decision aid. Prospective validation, site-specific calibration, clinical governance review, privacy review, and safety monitoring would be required before any clinical deployment study.

## Data

The study uses MIMIC-III as the internal development cohort and MC-MED as the frozen external evaluation cohort. Raw clinical data, note-derived timestamps, derived feature tables, packaged arrays, and predictions are not redistributed.

## Metrics

Primary reporting includes accuracy, recall, precision, F1, F2, AUC, false-positive rate, true-positive rate, and file-level false-alert burden. Threshold-dependent metrics use the frozen operating point selected on MIMIC validation. Summary benchmark tables are in `docs/benchmark_results.md`.

## Known Limitations

- Retrospective labels depend on documented onset anchors and physician-reviewed timestamp validation.
- Waveform availability, signal quality, and ICU monitoring practices may introduce selection bias.
- External validation is limited to the included MC-MED evaluation setting.
- The public repository cannot numerically reproduce the full tables without restricted local data.
- High warning-window recall may still produce clinically unacceptable alert burden without prospective calibration.

- Reported numerical results require the same restricted local data, reviewed onset-anchor files, preprocessing configuration, and patient-level partitions used in the paper.

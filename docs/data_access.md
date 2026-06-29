# Data Access and Local Layout

This repository does not redistribute clinical notes, waveform records, labels, derived patient tables, model checkpoints, or prediction files. MIMIC-III and MC-MED require credentialed access and local data-use compliance.

## Required Data Sources

| Source | Role in paper | Local target |
|---|---|---|
| MIMIC-III Clinical Database | ICU structured tables and unstructured notes for internal cohort mining | `data/raw/mimic/tables/` |
| MIMIC-III Waveform Database Matched Subset | PPG waveform source for internal development | `data/raw/mimic/waveforms/` |
| MC-MED | External EHR/waveform cohort for frozen testing | `data/raw/mcmed/` |

For MC-MED reconstruction, authorized users must place the credentialed
`rads.csv` table under `data/raw/mcmed/` and generate a local Pleth
waveform-segment manifest with the columns documented in
`docs/mcmed_anchor_generation.md`.

## Expected Local Directories

```text
data/raw/mimic/tables/
data/raw/mimic/waveforms/
data/raw/mcmed/
data/interim/mimic/
data/interim/mcmed/
data/processed/mimic/
data/processed/mcmed/
outputs/
```

The config files use repository-relative paths by default. If your credentialed data live elsewhere, edit:

- `configs/mimic_data.yaml`
- `configs/mcmed_data.yaml`
- `configs/feature_extraction.yaml`
- `configs/training.yaml`

## Files That Must Not Be Committed

The following are intentionally ignored:

- Raw clinical tables and waveforms.
- Intermediate note chunks and LLM outputs.
- Extracted patient-level anchors.
- MC-MED radiology-text adapters and Pleth waveform-segment manifests.
- Per-window feature tables.
- `.npy` packaged datasets.
- Model checkpoints and prediction CSVs.

Only code, prompts, configuration, public documentation, tests, citation metadata, and license files should be versioned.

## Reproduction Boundary

The public repository can be checked syntactically and unit-tested without restricted data. Full numerical reproduction requires local credentialed data and the physician-validated timestamp review outputs described in the study.

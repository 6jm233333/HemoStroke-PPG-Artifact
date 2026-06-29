# MC-MED External Anchor Generation

## Scope

MC-MED is used only as a frozen external evaluation cohort. No MC-MED records
are used for model training, feature selection, or threshold tuning. The
released code reconstructs the local processing path without redistributing
credentialed MC-MED records, reviewed anchors, or derived patient-level indices.

## Source-Field Adapter

The MIMIC-III and MC-MED pipelines apply the same semantic extraction and
timestamp-fallback rules, but their source fields differ.

| Cohort | Source text | Canonical record timestamp | Waveform modality |
|---|---|---|---|
| MIMIC-III | `NOTEEVENTS.TEXT` | `NOTEEVENTS.CHARTTIME` | PPG / Pleth |
| MC-MED | `rads.csv`: `Study + Impression` | `rads.csv`: `Result_time` | PPG / Pleth |

For MC-MED, `Order_time` is retained locally for provenance checks. The adapter writes the
canonical fields `Row_ID`, `CHARTTIME`, and `TEXT`, which match the released
prompt. The generated table remains local because it contains restricted text.

```bash
python -m src.data.mcmed.build_llm_input --config configs/mcmed_data.yaml
```

## Reviewed Anchors

Run structured extraction with the released prompt and an available model under
the applicable data-use agreement. Review the extracted candidates locally.
Store the reviewed table at:

```text
data/interim/mcmed/reviewed_anchor_candidates.csv
```

Required columns:

```text
CSN, Extracted_Timestamp
```

If multiple distinct reviewed timestamps remain for one `CSN`, the public
builder fails instead of silently choosing an anchor. Resolve the ambiguity
locally before continuing.

## Local Pleth Segment Manifest

Create a local segment-level manifest from the credentialed MC-MED waveform
files:

```text
data/interim/mcmed/pleth_waveform_segments.csv
```

Required columns:

```text
CSN, Wave_Type, WAVE_PATH, WAVE_START, WAVE_END
```

`WAVE_PATH` points to the local waveform record. Only rows whose `Wave_Type`
matches `Pleth` are retained.

## Build and Filter the External Index

Merge reviewed anchors with the Pleth segment manifest:

```bash
python -m src.data.mcmed.build_stroke_index --config configs/mcmed_data.yaml
```

Then retain waveform segments that cross the reviewed anchor or occur strictly
before it within the configured six-hour pre-warning window:

```bash
python -m src.data.mcmed.filter_prewarning_segments --config configs/mcmed_data.yaml
```

The generated `stroke_index_raw.csv`, `stroke_index_filtered.csv`, logs, and
all patient-level tables are restricted local artifacts and must not be
committed.

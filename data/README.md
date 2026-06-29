# Local Data Directory

This directory is a local-only mount point for credentialed clinical data. Do not commit raw records, waveform files, intermediate tables, processed arrays, checkpoints, predictions, or patient-level files.

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

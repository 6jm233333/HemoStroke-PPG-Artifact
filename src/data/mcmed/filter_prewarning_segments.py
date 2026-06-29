from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


INPUT_COLUMN_RENAME_MAP = {
    "Stroke_Time": "Extracted_Timestamp",
    "Start_Time": "WAVE_START",
    "End_Time": "WAVE_END",
    "File_Path": "WAVE_PATH",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        for enc in ("utf-8-sig", "utf-8", "gbk", "latin1"):
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except Exception:
                continue
        raise RuntimeError(f"Failed to read CSV: {path}")

    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type for input: {path}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_parent_dir(path)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return

    if suffix in (".xlsx", ".xls"):
        df.to_excel(path, index=False)
        return

    raise ValueError(f"Unsupported file type for output: {path}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_candidates = {
        old: new
        for old, new in INPUT_COLUMN_RENAME_MAP.items()
        if old in df.columns and new not in df.columns
    }
    if rename_candidates:
        df = df.rename(columns=rename_candidates)

    return df


def validate_required_columns(df: pd.DataFrame, required: list[str], file_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{file_name} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter MC-MED waveform segments into crossing and prewarning segments."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configs/mcmed_data.yaml",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    input_path = cfg["paths"]["stroke_index_csv"]
    output_path = cfg["paths"]["filtered_index_csv"]

    warning_hours = float(cfg["segment_filter"].get("warning_hours", 6))
    keep_crossing = bool(cfg["segment_filter"].get("keep_crossing", True))
    keep_prewarning = bool(cfg["segment_filter"].get("keep_prewarning", True))
    deduplicate_by = list(cfg["segment_filter"].get("deduplicate_by", ["WAVE_PATH", "Extracted_Timestamp"]))
    save_processing_log = bool(cfg["quality_control"].get("save_processing_log", True))

    stroke_col = cfg["time_columns"]["extracted_stroke_time"]
    wave_start_col = cfg["time_columns"]["waveform_start"]
    wave_end_col = cfg["time_columns"]["waveform_end"]
    wave_path_col = cfg["waveform_columns"]["waveform_path"]

    print(f"[filter_prewarning_segments] reading input: {input_path}")
    df = read_table(input_path)
    df = normalize_columns(df)

    validate_required_columns(
        df,
        [stroke_col, wave_start_col, wave_end_col, wave_path_col],
        str(input_path),
    )

    original_count = len(df)

    df = df.copy()
    df[stroke_col] = pd.to_datetime(df[stroke_col], errors="coerce")
    df[wave_start_col] = pd.to_datetime(df[wave_start_col], errors="coerce")
    df[wave_end_col] = pd.to_datetime(df[wave_end_col], errors="coerce")

    df = df.dropna(subset=[stroke_col, wave_start_col, wave_end_col]).copy()
    valid_time_count = len(df)

    if df.empty:
        raise RuntimeError("No valid rows remain after datetime parsing and NA filtering.")

    df["Gap_Hours"] = (df[stroke_col] - df[wave_end_col]).dt.total_seconds() / 3600.0

    crossing_mask = (df[wave_start_col] <= df[stroke_col]) & (df[stroke_col] <= df[wave_end_col])
    post_event_mask = df[wave_start_col] > df[stroke_col]
    pre_event_mask = df[wave_end_col] <= df[stroke_col]

    df["Segment_Type"] = "Other"
    df.loc[crossing_mask, "Segment_Type"] = "Crossing(StrokeInsideSegment)"
    df.loc[post_event_mask, "Segment_Type"] = "PostEvent(SegmentAfterStroke)"
    df.loc[pre_event_mask & ~crossing_mask, "Segment_Type"] = "PreEvent(StrictlyBeforeStroke)"

    prewarning_mask = (
        df["Segment_Type"].str.startswith("PreEvent")
        & (df["Gap_Hours"] > 0)
        & (df["Gap_Hours"] <= warning_hours)
    )

    keep_mask = pd.Series(False, index=df.index)
    if keep_crossing:
        keep_mask = keep_mask | crossing_mask
    if keep_prewarning:
        keep_mask = keep_mask | prewarning_mask

    df_keep = df.loc[keep_mask].copy()
    before_dedup = len(df_keep)

    df_keep["Keep_Reason"] = ""
    if keep_crossing:
        df_keep.loc[crossing_mask.reindex(df_keep.index, fill_value=False), "Keep_Reason"] += "Crossing;"
    if keep_prewarning:
        df_keep.loc[
            prewarning_mask.reindex(df_keep.index, fill_value=False),
            "Keep_Reason",
        ] += f"PreWarning<={warning_hours}h;"

    if stroke_col in df_keep.columns:
        df_keep[stroke_col] = pd.to_datetime(df_keep[stroke_col], errors="coerce").dt.floor("s")

    existing_dedup_cols = [c for c in deduplicate_by if c in df_keep.columns]
    if existing_dedup_cols:
        df_keep = df_keep.drop_duplicates(subset=existing_dedup_cols, keep="first").copy()

    after_dedup = len(df_keep)

    if df_keep.empty:
        raise RuntimeError(
            "Filtered result is empty after applying crossing/prewarning rules."
        )

    time_cols = [stroke_col, wave_start_col, wave_end_col]
    for col in time_cols:
        if col in df_keep.columns:
            df_keep[col] = pd.to_datetime(df_keep[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    if "Gap_Hours" in df_keep.columns:
        df_keep["Gap_Hours"] = pd.to_numeric(df_keep["Gap_Hours"], errors="coerce").round(2)

    write_table(df_keep, output_path)

    print("[filter_prewarning_segments] done.")
    print(f"  output={output_path}")
    print(f"  original_rows={original_count:,}")
    print(f"  valid_time_rows={valid_time_count:,}")
    print(f"  kept_before_dedup={before_dedup:,}")
    print(f"  kept_after_dedup={after_dedup:,}")
    print(f"  removed_as_duplicates={before_dedup - after_dedup:,}")
    print("  segment_type_counts:")
    print(df["Segment_Type"].value_counts(dropna=False).to_string())

    if save_processing_log:
        log_path = Path(output_path).with_name("filter_prewarning_segments_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "warning_hours": warning_hours,
                    "keep_crossing": keep_crossing,
                    "keep_prewarning": keep_prewarning,
                    "deduplicate_by": existing_dedup_cols,
                    "original_rows": int(original_count),
                    "valid_time_rows": int(valid_time_count),
                    "kept_before_dedup": int(before_dedup),
                    "kept_after_dedup": int(after_dedup),
                    "removed_as_duplicates": int(before_dedup - after_dedup),
                    "segment_type_counts": {
                        str(k): int(v)
                        for k, v in df["Segment_Type"].value_counts(dropna=False).to_dict().items()
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"  log={log_path}")


if __name__ == "__main__":
    main()

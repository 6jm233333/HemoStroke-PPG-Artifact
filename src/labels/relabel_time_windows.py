from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


EXCLUDE_PREFIXES = ("Summary_",)
EXCLUDE_NAMES = {
    "processing_summary.csv",
    "processing_summary_groups.csv",
    "Summary_All.csv",
    "relabel_summary.csv",
}


@dataclass
class DatasetSchema:
    dataset_name: str
    group_candidates: list[str]
    beat_col_candidates: list[str]
    wave_start_col_candidates: list[str]
    wave_end_col_candidates: list[str]
    stroke_time_col_candidates: list[str]
    label_col_candidates: list[str]
    abs_time_col_candidates: list[str]
    rel_time_col_candidates: list[str]
    stroke_subject_col_candidates: list[str]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def is_feature_csv(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    if path.name in EXCLUDE_NAMES:
        return False
    for prefix in EXCLUDE_PREFIXES:
        if path.name.startswith(prefix):
            return False
    return True


def parse_dt(value: Any) -> pd.Timestamp | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        return None
    return dt


def first_existing(columns: Iterable[str], candidates: list[str]) -> str | None:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def build_schema(dataset: str) -> DatasetSchema:
    dataset = dataset.lower()
    if dataset == "mimic":
        return DatasetSchema(
            dataset_name="mimic",
            group_candidates=["Source_File", "WAVE_PATH", "Wave_Path", "wave_path"],
            beat_col_candidates=["Beat_Idx", "Beat_Index"],
            wave_start_col_candidates=["Wave_Start", "WAVE_START", "Start_Time"],
            wave_end_col_candidates=["Wave_End", "WAVE_END", "End_Time"],
            stroke_time_col_candidates=["Actual_Stroke_Time", "Extracted_Timestamp", "Stroke_Time"],
            label_col_candidates=["Label"],
            abs_time_col_candidates=["Absolute_Time"],
            rel_time_col_candidates=["Time_Rel_Min"],
            stroke_subject_col_candidates=["Is_Stroke_Subject"],
        )
    if dataset == "mcmed":
        return DatasetSchema(
            dataset_name="mcmed",
            group_candidates=["WAVE_PATH", "Wave_Path", "wave_path", "Source_File"],
            beat_col_candidates=["Beat_Idx", "Beat_Index"],
            wave_start_col_candidates=["Wave_Start", "WAVE_START", "Start_Time"],
            wave_end_col_candidates=["Wave_End", "WAVE_END", "End_Time"],
            stroke_time_col_candidates=["Actual_Stroke_Time", "Extracted_Timestamp", "Stroke_Time"],
            label_col_candidates=["Label"],
            abs_time_col_candidates=["Absolute_Time"],
            rel_time_col_candidates=["Time_Rel_Min"],
            stroke_subject_col_candidates=["Is_Stroke_Subject"],
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_labeling_config(cfg: dict[str, Any]) -> dict[str, Any]:
    labeling = cfg.get("labeling", {})
    required = ["normal_range", "positive_range", "ignore_ranges"]
    missing = [k for k in required if k not in labeling]
    if missing:
        raise ValueError(
            f"feature_extraction.yaml -> labeling is missing keys: {missing}"
        )
    return labeling


def assign_label(rel_min: float, labeling_cfg: dict[str, Any]) -> int:
    if pd.isna(rel_min):
        return -1

    pos_start = float(labeling_cfg["positive_range"]["start"])
    pos_end = float(labeling_cfg["positive_range"]["end"])
    neg_start = float(labeling_cfg["normal_range"]["start"])
    neg_end = float(labeling_cfg["normal_range"]["end"])

    if pos_start <= rel_min < pos_end:
        return 1
    if neg_start <= rel_min < neg_end:
        return 0

    ignore_ranges = labeling_cfg.get("ignore_ranges", [])
    for r in ignore_ranges:
        if len(r) != 2:
            continue
        lo, hi = float(r[0]), float(r[1])
        if lo <= rel_min < hi:
            return -1

    return -1


def candidate_output_order(df: pd.DataFrame) -> list[str]:
    preferred = [
        "Group_ID",
        "CSN",
        "Wave_Type",
        "WAVE_PATH",
        "Source_File",
        "Stroke_Occurred_Here",
        "Wave_Start",
        "WAVE_START",
        "Wave_End",
        "WAVE_END",
        "Actual_Stroke_Time",
        "Extracted_Timestamp",
        "Is_Stroke_Subject",
        "MRN",
        "Visit_no",
        "Age",
        "Gender",
        "Race",
        "Ethnicity",
        "Dx_ICD10",
        "Dx_name",
        "Segment_Type",
        "Keep_Reason",
        "Gap_Hours",
        "Beat_Idx",
        "Beat_Index",
        "T_pi",
        "T_sys",
        "T_dia",
        "T_sp",
        "IPR",
        "Tsys_Tdia",
        "Tsp_Tpi",
        "A_on",
        "A_sp",
        "A_off",
        "Pulse_Amplitude",
        "SI",
        "T_u",
        "Tu_Tpi",
        "T_v",
        "T_a",
        "Ta_Tpi",
        "T_b",
        "Tb_Tpi",
        "T_c",
        "Tc_Tpi",
        "T_d",
        "Td_Tpi",
        "T_e",
        "Te_Tpi",
        "T_f",
        "Tf_Tpi",
        "T_p1",
        "Tp1_Tpi",
        "T_p2",
        "Tp2_Tpi",
        "Tu_Ta_Tpi",
        "CV_T_pi",
        "CV_T_sys",
        "CV_Pulse_Amplitude",
        "Label",
        "Absolute_Time",
        "Time_Rel_Min",
    ]
    existing = df.columns.tolist()
    ordered = [c for c in preferred if c in existing]
    ordered += [c for c in existing if c not in ordered]
    return ordered


def rebuild_group_time_axis(
    g: pd.DataFrame,
    beat_col: str | None,
    abs_time_col: str | None,
    wave_start_col: str,
    wave_end_col: str,
    stroke_col: str,
    stroke_subject_col: str,
    labeling_cfg: dict[str, Any],
) -> pd.DataFrame | None:
    g = g.copy()

    t_start = parse_dt(g[wave_start_col].iloc[0])
    t_end = parse_dt(g[wave_end_col].iloc[0])
    t_stroke = parse_dt(g[stroke_col].iloc[0])

    if t_start is None or t_end is None:
        return None

    if beat_col and beat_col in g.columns:
        g[beat_col] = pd.to_numeric(g[beat_col], errors="coerce")
        g = g.sort_values(by=beat_col, kind="mergesort")
    else:
        g = g.reset_index(drop=True)

    n = len(g)
    if n == 0:
        return None

    existing_abs_times = None
    if abs_time_col is not None and abs_time_col in g.columns:
        parsed_abs_times = pd.to_datetime(g[abs_time_col], errors="coerce")
        if parsed_abs_times.notna().all():
            existing_abs_times = parsed_abs_times

    if existing_abs_times is not None:
        abs_times = existing_abs_times
    elif n == 1:
        abs_times = pd.Index([t_start])
    else:
        if t_end <= t_start:
            abs_times = t_start + pd.to_timedelta(np.arange(n), unit="s")
        else:
            abs_times = pd.date_range(start=t_start, end=t_end, periods=n)

    g["Absolute_Time"] = pd.to_datetime(abs_times)

    if t_stroke is not None:
        g["Time_Rel_Min"] = (g["Absolute_Time"] - t_stroke).dt.total_seconds() / 60.0
        g["Label"] = g["Time_Rel_Min"].apply(lambda x: assign_label(x, labeling_cfg))
        g[stroke_subject_col] = 1
    else:
        g["Time_Rel_Min"] = np.nan
        g["Label"] = 0
        g[stroke_subject_col] = 0

    g["Absolute_Time"] = pd.to_datetime(g["Absolute_Time"], errors="coerce").dt.strftime(
        "%Y/%m/%d %H:%M:%S"
    )

    return g


def process_one_file(
    csv_path: Path,
    output_dir: Path,
    schema: DatasetSchema,
    labeling_cfg: dict[str, Any],
) -> dict[str, Any]:
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    group_col = first_existing(df.columns, schema.group_candidates)
    beat_col = first_existing(df.columns, schema.beat_col_candidates)
    wave_start_col = first_existing(df.columns, schema.wave_start_col_candidates)
    wave_end_col = first_existing(df.columns, schema.wave_end_col_candidates)
    stroke_col = first_existing(df.columns, schema.stroke_time_col_candidates)
    abs_time_col = first_existing(df.columns, schema.abs_time_col_candidates)
    stroke_subject_col = first_existing(df.columns, schema.stroke_subject_col_candidates)

    required_map = {
        "group_col": group_col,
        "wave_start_col": wave_start_col,
        "wave_end_col": wave_end_col,
    }
    missing = [k for k, v in required_map.items() if v is None]
    if missing:
        raise ValueError(
            f"{csv_path.name} missing required schema fields: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    if stroke_col is None:
        raise ValueError(
            f"{csv_path.name} does not contain a stroke time column. "
            f"Expected one of {schema.stroke_time_col_candidates}"
        )

    if stroke_subject_col is None:
        stroke_subject_col = "Is_Stroke_Subject"

    rebuilt_parts: list[pd.DataFrame] = []
    skipped_groups = 0

    for _, g in df.groupby(group_col, sort=False):
        rebuilt = rebuild_group_time_axis(
            g=g,
            beat_col=beat_col,
            abs_time_col=abs_time_col,
            wave_start_col=wave_start_col,
            wave_end_col=wave_end_col,
            stroke_col=stroke_col,
            stroke_subject_col=stroke_subject_col,
            labeling_cfg=labeling_cfg,
        )
        if rebuilt is None or rebuilt.empty:
            skipped_groups += 1
            continue
        rebuilt_parts.append(rebuilt)

    if not rebuilt_parts:
        raise RuntimeError(f"{csv_path.name}: no valid groups remained after relabeling.")

    out_df = pd.concat(rebuilt_parts, ignore_index=True)
    out_df = out_df[candidate_output_order(out_df)]

    output_path = output_dir / csv_path.name
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return {
        "file_name": csv_path.name,
        "input_path": str(csv_path),
        "output_path": str(output_path),
        "rows_in": int(len(df)),
        "rows_out": int(len(out_df)),
        "groups_total": int(df[group_col].nunique(dropna=False)),
        "groups_skipped": int(skipped_groups),
        "count_label_0": int((out_df["Label"] == 0).sum()) if "Label" in out_df.columns else 0,
        "count_label_1": int((out_df["Label"] == 1).sum()) if "Label" in out_df.columns else 0,
        "count_label_neg1": int((out_df["Label"] == -1).sum()) if "Label" in out_df.columns else 0,
    }


def resolve_default_input_dir(dataset: str, feature_cfg: dict[str, Any]) -> str | None:
    paths = feature_cfg.get("paths", {})
    engineered_roots = paths.get("engineered_feature_root", {})
    if dataset in engineered_roots:
        return engineered_roots[dataset]
    feature_roots = paths.get("raw_feature_root", {})
    return feature_roots.get(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign Time_Rel_Min and Label while preserving available beat-level timestamps."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to configs/feature_extraction.yaml")
    parser.add_argument("--dataset", type=str, required=True, choices=["mimic", "mcmed"])
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    feature_cfg = load_yaml(args.config)
    labeling_cfg = get_labeling_config(feature_cfg)
    schema = build_schema(args.dataset)

    input_dir = args.input_dir or resolve_default_input_dir(args.dataset, feature_cfg)
    if input_dir is None:
        raise ValueError("No input directory provided and no default found in feature_extraction.yaml")

    input_dir = Path(input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    files = sorted([p for p in input_dir.iterdir() if p.is_file() and is_feature_csv(p)])
    if not files:
        raise RuntimeError(f"No feature CSV files found under: {input_dir}")

    logs: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    print(f"[relabel_time_windows] dataset={args.dataset}")
    print(f"[relabel_time_windows] input_dir={input_dir}")
    print(f"[relabel_time_windows] output_dir={output_dir}")
    print(f"[relabel_time_windows] files={len(files)}")

    for file_path in tqdm(files, desc="Relabeling feature files"):
        try:
            info = process_one_file(
                csv_path=file_path,
                output_dir=output_dir,
                schema=schema,
                labeling_cfg=labeling_cfg,
            )
            logs.append(info)
        except Exception as e:
            errors.append({"file_name": file_path.name, "error": f"{type(e).__name__}: {e}"})

    summary_df = pd.DataFrame(logs)
    summary_path = output_dir / "relabel_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    error_path = output_dir / "relabel_errors.json"
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    print("[relabel_time_windows] done.")
    print(f"  success_files={len(logs)}")
    print(f"  failed_files={len(errors)}")
    print(f"  summary={summary_path}")
    print(f"  errors={error_path}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.features.engineer_features import (
    DEFAULT_BASELINE_FRAC,
    DEFAULT_BASELINE_METHOD,
    DEFAULT_BASELINE_MIN_ROWS,
    COMPOSITE_FEATURES,
    RAW_BASE_FEATURES,
    engineer_composite_features,
    engineer_kinematic_features,
    infer_group_col,
    infer_sort_cols,
)


EXCLUDE_PREFIXES = ("Summary_",)
EXCLUDE_NAMES = {
    "processing_summary.csv",
    "processing_summary_groups.csv",
    "cleaning_summary.csv",
    "engineer_summary.csv",
    "selection_summary.csv",
    "external_eval_summary.csv",
}

DEFAULT_METADATA_COLS = [
    "CSN",
    "Wave_Type",
    "WAVE_PATH",
    "Source_File",
    "Stroke_Occurred_Here",
    "Wave_Start",
    "Wave_End",
    "Actual_Stroke_Time",
    "Is_Stroke_Subject",
    "Beat_Idx",
    "Label",
    "Absolute_Time",
    "Time_Rel_Min",
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
]

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


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].interpolate(
            method="linear",
            axis=0,
            limit_direction="both",
        )
        df[numeric_cols] = df[numeric_cols].bfill().ffill().fillna(0)

    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    if non_numeric_cols:
        df[non_numeric_cols] = df[non_numeric_cols].ffill().bfill().fillna("Unknown")

    return df


def resolve_selected_features(
    selected_features_json: str | None,
    config: dict[str, Any],
) -> list[str]:
    # 1) explicit JSON file takes priority
    if selected_features_json:
        with open(selected_features_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Selected_Features.json must contain a JSON list.")
        return [str(x) for x in data]

    # 2) fallback to training config fixed list
    fs_cfg = config.get("feature_selection", {})
    fixed_list = fs_cfg.get("fixed_feature_list", [])
    if fixed_list:
        return [str(x) for x in fixed_list]

    raise ValueError(
        "No selected feature list found. Provide --selected-features-json or "
        "set feature_selection.fixed_feature_list in training.yaml."
    )


def resolve_metadata_columns(config: dict[str, Any]) -> list[str]:
    fs_cfg = config.get("feature_selection", {})
    by_dataset = fs_cfg.get("metadata_columns", {})
    if "mcmed" in by_dataset:
        return list(by_dataset["mcmed"])
    return DEFAULT_METADATA_COLS.copy()


def process_one_file(
    file_path: Path,
    output_dir: Path,
    selected_features: list[str],
    metadata_cols: list[str],
    engineer_features_flag: bool,
    baseline_fraction: float,
    min_baseline_points: int,
    baseline_method: str,
) -> dict[str, Any]:
    df = pd.read_csv(file_path, low_memory=False)
    rows_in = len(df)

    if "Source_File" not in df.columns:
        df["Source_File"] = file_path.name

    df = fill_missing_values(df)

    if engineer_features_flag:
        df = engineer_composite_features(df)
        df = engineer_kinematic_features(
            df,
            group_col=infer_group_col(df),
            sort_cols=infer_sort_cols(df),
            base_features=[c for c in RAW_BASE_FEATURES + COMPOSITE_FEATURES if c in df.columns],
            baseline_frac=baseline_fraction,
            baseline_min_rows=min_baseline_points,
            baseline_method=baseline_method,
        )

    existing_metadata = [c for c in metadata_cols if c in df.columns]
    existing_selected = [c for c in selected_features if c in df.columns]
    missing_selected = [c for c in selected_features if c not in df.columns]

    final_columns = list(dict.fromkeys(existing_metadata + selected_features))
    out_df = df.reindex(columns=final_columns, fill_value=0)

    output_path = output_dir / file_path.name
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return {
        "file": file_path.name,
        "rows_in": int(rows_in),
        "rows_out": int(len(out_df)),
        "metadata_cols": int(len(existing_metadata)),
        "selected_feature_cols_expected": int(len(selected_features)),
        "selected_feature_cols_present": int(len(existing_selected)),
        "selected_feature_cols_missing": int(len(missing_selected)),
        "missing_features": missing_selected,
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build external evaluation feature tables by applying a fixed selected feature set to MC-MED."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to configs/training.yaml")
    parser.add_argument("--input-dir", type=str, required=True, help="MC-MED cleaned or engineered feature directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for packaged external-eval files")
    parser.add_argument(
        "--selected-features-json",
        type=str,
        default=None,
        help="Path to Selected_Features.json learned from internal data (recommended)",
    )
    parser.add_argument(
        "--skip-engineering",
        action="store_true",
        help="Skip composite/kinematic feature engineering if input-dir is already engineered",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    fx_cfg = cfg.get("feature_engineering", {})
    baseline_fraction = float(fx_cfg.get("baseline_fraction", DEFAULT_BASELINE_FRAC))
    min_baseline_points = int(fx_cfg.get("min_baseline_points", DEFAULT_BASELINE_MIN_ROWS))
    baseline_method = str(fx_cfg.get("baseline_method", DEFAULT_BASELINE_METHOD))

    selected_features = resolve_selected_features(
        selected_features_json=args.selected_features_json,
        config=cfg,
    )
    metadata_cols = resolve_metadata_columns(cfg)

    files = sorted([p for p in input_dir.iterdir() if p.is_file() and is_feature_csv(p)])
    if not files:
        raise RuntimeError(f"No feature CSV files found under: {input_dir}")

    logs: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    print(f"[build_external_eval_set] input_dir={input_dir}")
    print(f"[build_external_eval_set] output_dir={output_dir}")
    print(f"[build_external_eval_set] files={len(files)}")
    print(f"[build_external_eval_set] selected_features={len(selected_features)}")
    print(f"[build_external_eval_set] skip_engineering={args.skip_engineering}")

    for p in tqdm(files, desc="Packaging external eval set"):
        try:
            info = process_one_file(
                file_path=p,
                output_dir=output_dir,
                selected_features=selected_features,
                metadata_cols=metadata_cols,
                engineer_features_flag=not args.skip_engineering,
                baseline_fraction=baseline_fraction,
                min_baseline_points=min_baseline_points,
                baseline_method=baseline_method,
            )
            logs.append(info)
        except Exception as e:
            errors.append({"file": p.name, "error": f"{type(e).__name__}: {e}"})

    summary_df = pd.DataFrame(logs)
    summary_path = output_dir / "external_eval_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    error_path = output_dir / "external_eval_errors.json"
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    feature_manifest = {
        "selected_features": selected_features,
        "metadata_columns": metadata_cols,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "skip_engineering": bool(args.skip_engineering),
        "frozen_preprocessing": {
            "relative_feature_formula": "(x - mu_base) / abs(mu_base)",
            "baseline_method": baseline_method,
            "baseline_fraction": baseline_fraction,
            "min_baseline_points": min_baseline_points,
        },
    }
    manifest_path = output_dir / "external_eval_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(feature_manifest, f, ensure_ascii=False, indent=2)

    print("[build_external_eval_set] done.")
    print(f"  success_files={len(logs)}")
    print(f"  failed_files={len(errors)}")
    print(f"  summary={summary_path}")
    print(f"  errors={error_path}")
    print(f"  manifest={manifest_path}")


if __name__ == "__main__":
    main()

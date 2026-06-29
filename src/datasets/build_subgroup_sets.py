
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_METADATA_EXCLUDE = {
    "File_Name",
    "CSN",
    "MRN",
    "Visit_no",
    "Visits",
    "Age",
    "Age_Group",
    "Gender",
    "Race",
    "Ethnicity",
    "Means_of_arrival",
    "Risk_Group",
    "All_PMH_Codes",
    "All_PMH_Descs",
    "Label",
    "Time_Rel_Min",
    "Absolute_Time",
    "Beat_Idx",
    "Source_File",
    "Wave_Start",
    "Wave_End",
    "Actual_Stroke_Time",
    "Is_Stroke_Subject",
    "Stroke_Occurred_Here",
    "WAVE_PATH",
    "Wave_Type",
    "Segment_Type",
    "Keep_Reason",
    "Gap_Hours",
}


@dataclass
class BuildConfig:
    input_dir: Path
    metadata_index_csv: Path
    output_dir: Path
    selected_features_json: Path | None
    window_size: int
    stride: int
    max_gap_minutes: float
    warning_window_minutes: float
    age_threshold: float


def read_csv_with_fallback(path: str | Path) -> pd.DataFrame:
    last_error = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Failed to read CSV: {path}\nLast error: {last_error}")


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_group_name(name: Any) -> str:
    return str(name).replace("(", "").replace(")", "").replace(" ", "_").replace("/", "_")


def load_selected_features(selected_features_json: Path | None, input_dir: Path) -> list[str]:
    if selected_features_json is not None:
        with open(selected_features_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Selected_Features.json must contain a JSON list.")
        return [str(x) for x in data]

    csv_files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])
    if not csv_files:
        raise RuntimeError(f"No CSV files found under: {input_dir}")

    sample_df = read_csv_with_fallback(csv_files[0])
    feature_cols = []
    for c in sample_df.columns:
        if c in DEFAULT_METADATA_EXCLUDE:
            continue
        if pd.api.types.is_numeric_dtype(sample_df[c]):
            feature_cols.append(c)

    if not feature_cols:
        raise RuntimeError("Could not infer any numeric feature columns from input CSVs.")

    return feature_cols


def load_metadata_index(path: Path, age_threshold: float) -> dict[str, dict[str, Any]]:
    df = read_csv_with_fallback(path)
    df.columns = [str(c).strip() for c in df.columns]

    if "File_Name" not in df.columns:
        raise ValueError(
            "Metadata index CSV must contain a 'File_Name' column so subgroup metadata "
            "can be mapped back to each external-eval CSV file."
        )

    if "Age_Group" not in df.columns and "Age" in df.columns:
        age_num = pd.to_numeric(df["Age"], errors="coerce")
        df["Age_Group"] = np.where(age_num >= age_threshold, "Elderly", "Non_Elderly")

    if "Risk_Group" not in df.columns:
        print("[Warning] 'Risk_Group' column not found in metadata index. Risk subgroup packaging will be skipped.")

    meta_dict = df.set_index("File_Name").to_dict("index")
    print(f"[build_subgroup_sets] metadata loaded: {len(meta_dict)} records")
    return meta_dict


def save_subset_npy(
    output_sub_folder: Path,
    subset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    pid: np.ndarray,
    file_names: np.ndarray,
) -> dict[str, Any]:
    safe_name = safe_group_name(subset_name)
    save_dir = output_sub_folder / safe_name
    ensure_dir(save_dir)

    if len(X) == 0:
        print(f"   -> [{subset_name}] no data, skipped.")
        return {
            "subset_group": output_sub_folder.name,
            "subset_name": subset_name,
            "saved": False,
            "n_samples": 0,
            "n_neg": 0,
            "n_pos": 0,
            "save_dir": str(save_dir),
        }

    np.save(save_dir / "test_data.npy", X)
    np.save(save_dir / "test_label.npy", y)
    np.save(save_dir / "test_pid.npy", pid)
    np.save(save_dir / "test_file_name.npy", file_names)

    unique_labels, counts = np.unique(y, return_counts=True)
    count_dict = {int(k): int(v) for k, v in zip(unique_labels, counts)}

    manifest_df = pd.DataFrame({
        "file_name": file_names,
        "pid": pid,
        "label": y,
    })
    manifest_df.to_csv(save_dir / "sample_manifest.csv", index=False, encoding="utf-8-sig")

    print(
        f"   -> [{subset_name}] saved: n={len(y)} | "
        f"0(normal)={count_dict.get(0, 0)} | 1(warning)={count_dict.get(1, 0)}"
    )

    return {
        "subset_group": output_sub_folder.name,
        "subset_name": subset_name,
        "saved": True,
        "n_samples": int(len(y)),
        "n_neg": int(count_dict.get(0, 0)),
        "n_pos": int(count_dict.get(1, 0)),
        "save_dir": str(save_dir),
    }


def build_samples_for_one_file(
    file_path: Path,
    feature_cols: list[str],
    patient_id: int,
    window_size: int,
    stride: int,
    max_gap_minutes: float,
    warning_window_minutes: float,
) -> tuple[list[np.ndarray], list[int], list[int], list[str]]:
    cols_to_use = feature_cols + ["Label", "Time_Rel_Min"]
    df = read_csv_with_fallback(file_path)

    missing = [c for c in cols_to_use if c not in df.columns]
    if missing:
        raise ValueError(f"{file_path.name} is missing required columns: {missing}")

    df = df[cols_to_use].copy()
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df["Time_Rel_Min"] = pd.to_numeric(df["Time_Rel_Min"], errors="coerce")

    # Match the main array-building path by dropping ignore labels.
    df = df[df["Label"] != -1].copy()
    if df.empty:
        return [], [], [], []

    df = df.sort_values("Time_Rel_Min", kind="mergesort")

    # Feature values are already produced by the frozen upstream pipeline.
    # Keep window-level NaN exclusion consistent with the main packaging path.
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Split continuous time segments.
    time_diffs = df["Time_Rel_Min"].diff().fillna(0)
    df["segment_id"] = (time_diffs > max_gap_minutes).cumsum()

    X_chunks: list[np.ndarray] = []
    y_chunks: list[int] = []
    pid_chunks: list[int] = []
    fname_chunks: list[str] = []

    for _, group in df.groupby("segment_id", sort=False):
        if len(group) < window_size:
            continue

        X_group = group[feature_cols].values.astype(np.float32)
        y_group = group["Label"].values.astype(np.int32)
        t_group = group["Time_Rel_Min"].values.astype(np.float32)

        num_samples = (len(X_group) - window_size) // stride + 1

        for i in range(num_samples):
            start = i * stride
            end = start + window_size

            x_win = X_group[start:end]
            y_win_raw = y_group[start:end]
            t_win = t_group[start:end]

            if not np.isfinite(x_win).all():
                continue

            # Keep pre-onset windows only.
            if t_win[0] > 0:
                continue

            has_original_label = np.any(y_win_raw == 1)
            is_in_warning_time = np.any((t_win >= -warning_window_minutes) & (t_win <= 0))

            label = 1 if (has_original_label or is_in_warning_time) else 0

            X_chunks.append(x_win)
            y_chunks.append(label)
            pid_chunks.append(patient_id)
            fname_chunks.append(file_path.name)

    return X_chunks, y_chunks, pid_chunks, fname_chunks


def collect_all_samples(
    input_dir: Path,
    meta_dict: dict[str, dict[str, Any]],
    feature_cols: list[str],
    cfg: BuildConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    all_files = sorted([f.name for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() == ".csv"])
    valid_files = [f for f in all_files if f in meta_dict]
    skipped_files = len(all_files) - len(valid_files)

    if skipped_files > 0:
        print(f"[build_subgroup_sets] skipped {skipped_files} files not found in metadata index.")

    if not valid_files:
        raise RuntimeError("No valid files remained after matching input CSVs with metadata index.")

    file_to_id = {f: i for i, f in enumerate(valid_files)}

    total_X: list[np.ndarray] = []
    total_y: list[int] = []
    total_pid: list[int] = []
    total_filenames: list[str] = []
    errors: list[dict[str, Any]] = []

    print(f"[build_subgroup_sets] processing {len(valid_files)} valid files...")

    for file_name in tqdm(valid_files, desc="Building subgroup-ready samples"):
        file_path = input_dir / file_name
        patient_id = file_to_id[file_name]

        try:
            X_chunks, y_chunks, pid_chunks, fname_chunks = build_samples_for_one_file(
                file_path=file_path,
                feature_cols=feature_cols,
                patient_id=patient_id,
                window_size=cfg.window_size,
                stride=cfg.stride,
                max_gap_minutes=cfg.max_gap_minutes,
                warning_window_minutes=cfg.warning_window_minutes,
            )
            total_X.extend(X_chunks)
            total_y.extend(y_chunks)
            total_pid.extend(pid_chunks)
            total_filenames.extend(fname_chunks)
        except Exception as e:
            errors.append({"file": file_name, "error": f"{type(e).__name__}: {e}"})

    if len(total_X) == 0:
        raise RuntimeError("No valid samples were generated from the external-eval CSV files.")

    X_all = np.asarray(total_X, dtype=np.float32)
    y_all = np.asarray(total_y, dtype=np.int64)
    p_all = np.asarray(total_pid)
    f_all = np.asarray(total_filenames)

    return X_all, y_all, p_all, f_all, errors


def subgroup_indices(
    f_all: np.ndarray,
    meta_dict: dict[str, dict[str, Any]],
    field: str,
    value: Any,
) -> list[int]:
    return [i for i, fname in enumerate(f_all) if meta_dict[str(fname)].get(field) == value]


def package_subgroups(
    X_all: np.ndarray,
    y_all: np.ndarray,
    p_all: np.ndarray,
    f_all: np.ndarray,
    meta_dict: dict[str, dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []

    print("\n>>> Packaging: Risk (Risk_Group)")
    risk_save_path = output_dir / "Risk"
    unique_risks = set()
    for f in sorted(set(f_all.tolist())):
        val = meta_dict[str(f)].get("Risk_Group")
        if pd.notna(val) and str(val) != "Unknown":
            unique_risks.add(val)

    for risk_grp in sorted(list(unique_risks), key=lambda x: str(x)):
        idx = subgroup_indices(f_all, meta_dict, "Risk_Group", risk_grp)
        logs.append(save_subset_npy(risk_save_path, str(risk_grp), X_all[idx], y_all[idx], p_all[idx], f_all[idx]))

    print("\n>>> Packaging: Age (Elderly vs Non_Elderly)")
    age_save_path = output_dir / "Age"
    for age_grp in ["Elderly", "Non_Elderly"]:
        idx = subgroup_indices(f_all, meta_dict, "Age_Group", age_grp)
        if idx:
            logs.append(save_subset_npy(age_save_path, age_grp, X_all[idx], y_all[idx], p_all[idx], f_all[idx]))

    print("\n>>> Packaging: Sex (Gender)")
    sex_save_path = output_dir / "Sex"
    for sex in ["M", "F"]:
        idx = subgroup_indices(f_all, meta_dict, "Gender", sex)
        if idx:
            logs.append(save_subset_npy(sex_save_path, sex, X_all[idx], y_all[idx], p_all[idx], f_all[idx]))

    print("\n>>> Packaging: Race (White / Asian / Black / Other)")
    race_save_path = output_dir / "Race"
    race_groups = {
        "White": "White",
        "Asian": "Asian",
        "Black": "Black or African American",
        "Other": "Other",
    }
    for race_name, race_value in race_groups.items():
        idx = subgroup_indices(f_all, meta_dict, "Race", race_value)
        logs.append(save_subset_npy(race_save_path, race_name, X_all[idx], y_all[idx], p_all[idx], f_all[idx]))

    return logs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build subgroup-specific external evaluation sets from per-file external-eval CSVs."
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Directory of external-eval CSV files")
    parser.add_argument("--metadata-index-csv", type=str, required=True, help="CSV keyed by File_Name with subgroup metadata")
    parser.add_argument("--output-dir", type=str, required=True, help="Output root for subgroup npy folders")
    parser.add_argument("--selected-features-json", type=str, default=None, help="Optional Selected_Features.json")
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--stride", type=int, default=500)
    parser.add_argument("--max-gap-minutes", type=float, default=1.0)
    parser.add_argument("--warning-window-minutes", type=float, default=180.0)
    parser.add_argument("--age-threshold", type=float, default=65.0)
    args = parser.parse_args()

    cfg = BuildConfig(
        input_dir=Path(args.input_dir),
        metadata_index_csv=Path(args.metadata_index_csv),
        output_dir=Path(args.output_dir),
        selected_features_json=Path(args.selected_features_json) if args.selected_features_json else None,
        window_size=args.window_size,
        stride=args.stride,
        max_gap_minutes=args.max_gap_minutes,
        warning_window_minutes=args.warning_window_minutes,
        age_threshold=args.age_threshold,
    )

    ensure_dir(cfg.output_dir)

    feature_cols = load_selected_features(cfg.selected_features_json, cfg.input_dir)
    meta_dict = load_metadata_index(cfg.metadata_index_csv, cfg.age_threshold)

    X_all, y_all, p_all, f_all, build_errors = collect_all_samples(
        input_dir=cfg.input_dir,
        meta_dict=meta_dict,
        feature_cols=feature_cols,
        cfg=cfg,
    )

    print(f"\n[build_subgroup_sets] total samples built: {len(X_all)}")
    print("=" * 60)

    logs = package_subgroups(
        X_all=X_all,
        y_all=y_all,
        p_all=p_all,
        f_all=f_all,
        meta_dict=meta_dict,
        output_dir=cfg.output_dir,
    )

    summary_path = cfg.output_dir / "subgroup_summary.csv"
    pd.DataFrame(logs).to_csv(summary_path, index=False, encoding="utf-8-sig")

    error_path = cfg.output_dir / "build_subgroup_errors.json"
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(build_errors, f, ensure_ascii=False, indent=2)

    config_path = cfg.output_dir / "build_subgroup_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": str(cfg.input_dir),
                "metadata_index_csv": str(cfg.metadata_index_csv),
                "output_dir": str(cfg.output_dir),
                "selected_features_json": str(cfg.selected_features_json) if cfg.selected_features_json else None,
                "feature_cols": feature_cols,
                "window_size": cfg.window_size,
                "stride": cfg.stride,
                "max_gap_minutes": cfg.max_gap_minutes,
                "warning_window_minutes": cfg.warning_window_minutes,
                "age_threshold": cfg.age_threshold,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 60)
    print(f"All subgroup packages finished. Output: {cfg.output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Errors: {error_path}")


if __name__ == "__main__":
    main()

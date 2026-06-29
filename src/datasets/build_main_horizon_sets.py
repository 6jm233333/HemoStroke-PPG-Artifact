from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml


PATIENT_COL_CANDIDATES = ["SUBJECT_ID", "MRN", "CSN", "Group_ID", "Visit_no"]
FILE_COL_CANDIDATES = ["Source_File", "WAVE_PATH", "Wave_Path", "wave_path"]
TIME_COL_CANDIDATES = ["Absolute_Time", "Beat_Idx", "Time_Rel_Min"]
EXCLUDE_NAMES = {
    "relabel_summary.csv",
    "processing_summary.csv",
    "processing_summary_groups.csv",
    "cleaning_summary.csv",
    "feature_engineering_log.csv",
}


@dataclass
class PackagingSummary:
    dataset: str
    horizon_minutes: int
    n_packaged_before_nan_filter: int
    n_saved_windows: int
    n_excluded_nan_windows: int
    n_patients: int
    n_negative: int
    n_positive: int
    output_dir: str


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_feature_list(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("Feature list JSON must contain a non-empty list.")
    return [str(value) for value in data]


def first_existing(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    colset = set(columns)
    for candidate in candidates:
        if candidate in colset:
            return candidate
    return None


def is_feature_csv(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".csv" and path.name not in EXCLUDE_NAMES


def classify_relative_minutes(
    rel_minutes: pd.Series | np.ndarray,
    *,
    horizon_minutes: int,
    normal_start_minutes: float,
    boundary_buffer_minutes: float,
    onset_buffer_minutes: float,
) -> np.ndarray:
    values = pd.to_numeric(pd.Series(rel_minutes), errors="coerce").to_numpy(dtype=float)
    labels = np.full(len(values), -1, dtype=np.int64)
    normal_end = -float(horizon_minutes) - float(boundary_buffer_minutes)
    warning_start = -float(horizon_minutes) + float(boundary_buffer_minutes)
    labels[(values >= float(normal_start_minutes)) & (values < normal_end)] = 0
    labels[(values >= warning_start) & (values < -float(onset_buffer_minutes))] = 1
    return labels


def split_patient_ids(
    patient_ids: Sequence[str],
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> dict[str, str]:
    unique_ids = np.asarray(sorted({str(value) for value in patient_ids}), dtype=object)
    if len(unique_ids) == 0:
        raise ValueError("No patient IDs were found.")
    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Fractions must satisfy train > 0, val >= 0, and train + val < 1.")

    rng = np.random.default_rng(seed)
    shuffled = unique_ids.copy()
    rng.shuffle(shuffled)

    n_train = max(1, int(round(len(shuffled) * train_fraction)))
    n_val = int(round(len(shuffled) * val_fraction))
    if n_train + n_val >= len(shuffled) and len(shuffled) > 1:
        n_train = max(1, len(shuffled) - n_val - 1)

    split_map: dict[str, str] = {}
    for value in shuffled[:n_train]:
        split_map[str(value)] = "train"
    for value in shuffled[n_train : n_train + n_val]:
        split_map[str(value)] = "val"
    for value in shuffled[n_train + n_val :]:
        split_map[str(value)] = "test"
    return split_map


def discover_patient_ids(input_dir: Path) -> list[str]:
    patient_ids: set[str] = set()
    for csv_path in sorted(path for path in input_dir.iterdir() if is_feature_csv(path)):
        df = pd.read_csv(csv_path, low_memory=False)
        patient_col = first_existing(df.columns, PATIENT_COL_CANDIDATES)
        if patient_col is None:
            patient_ids.add(csv_path.stem)
            continue
        patient_ids.update(df[patient_col].dropna().astype(str).tolist())
    return sorted(patient_ids)


def _sort_group(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [col for col in TIME_COL_CANDIDATES if col in df.columns]
    if not sort_cols:
        return df.reset_index(drop=True)
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def build_windows_for_dataframe(
    df: pd.DataFrame,
    *,
    fallback_file_id: str,
    feature_cols: Sequence[str],
    horizon_minutes: int,
    normal_start_minutes: float,
    boundary_buffer_minutes: float,
    onset_buffer_minutes: float,
    window_size: int,
    stride: int,
    max_gap_minutes: float,
) -> tuple[list[np.ndarray], list[int], list[str], list[dict[str, Any]], int, int]:
    missing = [col for col in [*feature_cols, "Time_Rel_Min"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    patient_col = first_existing(df.columns, PATIENT_COL_CANDIDATES)
    file_col = first_existing(df.columns, FILE_COL_CANDIDATES)
    if patient_col is None:
        df = df.copy()
        patient_col = "_patient_id"
        df[patient_col] = fallback_file_id
    if file_col is None:
        df = df.copy()
        file_col = "_file_id"
        df[file_col] = fallback_file_id

    windows: list[np.ndarray] = []
    labels: list[int] = []
    patient_ids: list[str] = []
    manifest_rows: list[dict[str, Any]] = []
    n_packaged = 0
    n_nan_excluded = 0

    for file_id, file_df in df.groupby(file_col, dropna=False, sort=False):
        ordered = _sort_group(file_df.copy())
        ordered["_horizon_label"] = classify_relative_minutes(
            ordered["Time_Rel_Min"],
            horizon_minutes=horizon_minutes,
            normal_start_minutes=normal_start_minutes,
            boundary_buffer_minutes=boundary_buffer_minutes,
            onset_buffer_minutes=onset_buffer_minutes,
        )
        ordered["_time_numeric"] = pd.to_numeric(ordered["Time_Rel_Min"], errors="coerce")
        time_gap = ordered["_time_numeric"].diff().abs().fillna(0.0) > float(max_gap_minutes)
        label_change = ordered["_horizon_label"].ne(ordered["_horizon_label"].shift()).fillna(True)
        ordered["_segment_id"] = (time_gap | label_change).cumsum()

        for _, segment in ordered.groupby("_segment_id", sort=False):
            label = int(segment["_horizon_label"].iloc[0])
            if label not in (0, 1) or len(segment) < window_size:
                continue
            features = segment[list(feature_cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            for start in range(0, len(segment) - window_size + 1, stride):
                end = start + window_size
                n_packaged += 1
                window = features[start:end]
                if not np.isfinite(window).all():
                    n_nan_excluded += 1
                    continue
                patient_id = str(segment[patient_col].iloc[start])
                windows.append(window.astype(np.float32))
                labels.append(label)
                patient_ids.append(patient_id)
                manifest_rows.append(
                    {
                        "patient_id": patient_id,
                        "file_id": str(file_id),
                        "horizon_minutes": int(horizon_minutes),
                        "label": label,
                        "start_relative_min": float(segment["_time_numeric"].iloc[start]),
                        "end_relative_min": float(segment["_time_numeric"].iloc[end - 1]),
                    }
                )

    return windows, labels, patient_ids, manifest_rows, n_packaged, n_nan_excluded


def collect_dataset_windows(
    input_dir: Path,
    *,
    feature_cols: Sequence[str],
    horizon_minutes: int,
    normal_start_minutes: float,
    boundary_buffer_minutes: float,
    onset_buffer_minutes: float,
    window_size: int,
    stride: int,
    max_gap_minutes: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, int, int]:
    all_windows: list[np.ndarray] = []
    all_labels: list[int] = []
    all_patient_ids: list[str] = []
    all_manifest_rows: list[dict[str, Any]] = []
    n_packaged = 0
    n_nan_excluded = 0

    csv_paths = sorted(path for path in input_dir.iterdir() if is_feature_csv(path))
    if not csv_paths:
        raise RuntimeError(f"No labeled feature CSV files found under: {input_dir}")

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, low_memory=False)
        built = build_windows_for_dataframe(
            df,
            fallback_file_id=csv_path.stem,
            feature_cols=feature_cols,
            horizon_minutes=horizon_minutes,
            normal_start_minutes=normal_start_minutes,
            boundary_buffer_minutes=boundary_buffer_minutes,
            onset_buffer_minutes=onset_buffer_minutes,
            window_size=window_size,
            stride=stride,
            max_gap_minutes=max_gap_minutes,
        )
        windows, labels, patient_ids, manifest_rows, packaged, nan_excluded = built
        all_windows.extend(windows)
        all_labels.extend(labels)
        all_patient_ids.extend(patient_ids)
        all_manifest_rows.extend(manifest_rows)
        n_packaged += packaged
        n_nan_excluded += nan_excluded

    if not all_windows:
        raise RuntimeError(f"No valid {horizon_minutes}-minute windows were generated from: {input_dir}")
    return (
        np.asarray(all_windows, dtype=np.float32),
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_patient_ids, dtype=object),
        pd.DataFrame(all_manifest_rows),
        n_packaged,
        n_nan_excluded,
    )


def save_partition(
    output_dir: Path,
    prefix: str,
    x: np.ndarray,
    y: np.ndarray,
    pid: np.ndarray,
    manifest_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{prefix}_data.npy", x)
    np.save(output_dir / f"{prefix}_label.npy", y)
    np.save(output_dir / f"{prefix}_pid.npy", pid)
    manifest_df.to_csv(output_dir / f"{prefix}_sample_manifest.csv", index=False, encoding="utf-8-sig")


def package_dataset(
    *,
    dataset: str,
    input_dir: Path,
    output_root: Path,
    feature_cols: Sequence[str],
    horizons: Sequence[int],
    split_map: dict[str, str] | None,
    normal_start_minutes: float,
    boundary_buffer_minutes: float,
    onset_buffer_minutes: float,
    window_size: int,
    stride: int,
    max_gap_minutes: float,
) -> list[PackagingSummary]:
    summaries: list[PackagingSummary] = []
    for horizon_minutes in horizons:
        output_dir = output_root / f"{int(horizon_minutes)}min"
        x, y, pid, manifest_df, n_packaged, n_nan_excluded = collect_dataset_windows(
            input_dir,
            feature_cols=feature_cols,
            horizon_minutes=int(horizon_minutes),
            normal_start_minutes=normal_start_minutes,
            boundary_buffer_minutes=boundary_buffer_minutes,
            onset_buffer_minutes=onset_buffer_minutes,
            window_size=window_size,
            stride=stride,
            max_gap_minutes=max_gap_minutes,
        )

        if dataset == "mcmed":
            save_partition(output_dir, "test", x, y, pid, manifest_df)
        else:
            if split_map is None:
                raise ValueError("MIMIC packaging requires a patient-level split map.")
            for prefix in ("train", "val", "test"):
                mask = np.asarray([split_map[str(value)] == prefix for value in pid])
                save_partition(output_dir, prefix, x[mask], y[mask], pid[mask], manifest_df.loc[mask].copy())

        summary = PackagingSummary(
            dataset=dataset,
            horizon_minutes=int(horizon_minutes),
            n_packaged_before_nan_filter=int(n_packaged),
            n_saved_windows=int(len(y)),
            n_excluded_nan_windows=int(n_nan_excluded),
            n_patients=int(len(np.unique(pid))),
            n_negative=int(np.sum(y == 0)),
            n_positive=int(np.sum(y == 1)),
            output_dir=str(output_dir),
        )
        with open(output_dir / "packaging_summary.json", "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, ensure_ascii=False, indent=2)
        summaries.append(summary)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MIMIC train/val/test and frozen MC-MED test arrays for all main horizons."
    )
    parser.add_argument("--config", default="configs/feature_extraction.yaml")
    parser.add_argument("--feature-list", default="configs/feature_set_17.json")
    parser.add_argument("--dataset", choices=["both", "mimic", "mcmed"], default="both")
    parser.add_argument("--mimic-input-dir", default=None)
    parser.add_argument("--mcmed-input-dir", default=None)
    parser.add_argument("--mimic-output-root", default=None)
    parser.add_argument("--mcmed-output-root", default=None)
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--stride", type=int, default=500)
    parser.add_argument("--max-gap-minutes", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    feature_cols = load_feature_list(args.feature_list)
    packaging_cfg = cfg.get("multi_horizon_packaging", {})
    labeling_cfg = cfg.get("labeling", {})
    paths_cfg = cfg.get("paths", {})

    horizons = [int(item["minutes"]) for item in packaging_cfg.get("horizons", [])]
    if not horizons:
        raise ValueError("No multi_horizon_packaging.horizons found in the config.")

    labeled_roots = paths_cfg.get("labeled_feature_root", {})
    output_roots = paths_cfg.get("packaged_output_root", {})
    normal_start_minutes = float(labeling_cfg["normal_range"]["start"])
    boundary_buffer_minutes = float(labeling_cfg.get("pre_buffer_minutes", 15))
    onset_buffer_minutes = float(labeling_cfg.get("onset_buffer_minutes", 15))

    all_summaries: list[PackagingSummary] = []
    if args.dataset in {"both", "mimic"}:
        mimic_input = Path(args.mimic_input_dir or labeled_roots["mimic"])
        mimic_output = Path(args.mimic_output_root or output_roots["mimic"])
        split_map = split_patient_ids(
            discover_patient_ids(mimic_input),
            seed=args.seed,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        )
        all_summaries.extend(
            package_dataset(
                dataset="mimic",
                input_dir=mimic_input,
                output_root=mimic_output,
                feature_cols=feature_cols,
                horizons=horizons,
                split_map=split_map,
                normal_start_minutes=normal_start_minutes,
                boundary_buffer_minutes=boundary_buffer_minutes,
                onset_buffer_minutes=onset_buffer_minutes,
                window_size=args.window_size,
                stride=args.stride,
                max_gap_minutes=args.max_gap_minutes,
            )
        )

    if args.dataset in {"both", "mcmed"}:
        mcmed_input = Path(args.mcmed_input_dir or labeled_roots["mcmed"])
        mcmed_output = Path(args.mcmed_output_root or output_roots["mcmed"])
        all_summaries.extend(
            package_dataset(
                dataset="mcmed",
                input_dir=mcmed_input,
                output_root=mcmed_output,
                feature_cols=feature_cols,
                horizons=horizons,
                split_map=None,
                normal_start_minutes=normal_start_minutes,
                boundary_buffer_minutes=boundary_buffer_minutes,
                onset_buffer_minutes=onset_buffer_minutes,
                window_size=args.window_size,
                stride=args.stride,
                max_gap_minutes=args.max_gap_minutes,
            )
        )

    print(pd.DataFrame([asdict(summary) for summary in all_summaries]).to_string(index=False))


if __name__ == "__main__":
    main()

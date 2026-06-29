from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from multiprocessing import Manager, Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb
import yaml
from dotmap import DotMap
from pyPPG.fiducials import FpCollection
from pyPPG.preproc import Preprocess
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


@dataclass
class DatasetSpec:
    dataset_name: str
    index_table_path: str
    wave_path_col: str
    wave_start_col: str
    extracted_stroke_time_col: str
    grouping_mode: str
    visit_id_col: str | None = None
    waveform_type_col: str | None = None
    passthrough_cols: list[str] | None = None


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_time(value: Any) -> pd.Timestamp | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.isna(dt):
            return None
        return dt.tz_convert(None)
    except Exception:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None
        return dt


def get_group_id_from_path(wave_path: str) -> str:
    base = os.path.basename(str(wave_path).strip())
    m = re.match(r"^(.*)_\d{4}$", base)
    return m.group(1) if m else base


def resolve_output_dir(
    dataset: str,
    feature_cfg: dict[str, Any],
    output_dir_cli: str | None,
) -> Path:
    if output_dir_cli:
        return Path(output_dir_cli)

    paths = feature_cfg.get("paths", {})
    roots = paths.get("raw_feature_root", {})
    if dataset in roots:
        return Path(roots[dataset])

    raise ValueError(
        "No output directory was provided and no default raw_feature_root was found."
    )


def resolve_dataset_spec(
    dataset: str,
    data_cfg: dict[str, Any],
) -> DatasetSpec:
    dataset = dataset.lower()
    dataset_name = data_cfg.get("dataset_name", dataset).lower()

    if dataset != dataset_name:
        raise ValueError(
            f"Dataset mismatch: CLI dataset={dataset}, config dataset_name={dataset_name}"
        )

    time_cols = data_cfg["time_columns"]
    wave_cols = data_cfg["waveform_columns"]
    paths = data_cfg["paths"]

    if dataset == "mimic":
        index_table_path = paths["feature_index_csv"]
        subject_id_col = data_cfg["id_columns"].get("subject_id")
        visit_id_col = data_cfg["id_columns"].get("hadm_id")
        return DatasetSpec(
            dataset_name="mimic",
            index_table_path=index_table_path,
            wave_path_col=wave_cols["waveform_path"],
            wave_start_col=time_cols["waveform_start"],
            extracted_stroke_time_col=time_cols["extracted_stroke_time"],
            grouping_mode="mimic_group_from_path",
            visit_id_col=visit_id_col,
            waveform_type_col=None,
            passthrough_cols=[
                col for col in [subject_id_col, visit_id_col] if col is not None
            ],
        )

    if dataset == "mcmed":
        index_table_path = paths["filtered_index_csv"]
        return DatasetSpec(
            dataset_name="mcmed",
            index_table_path=index_table_path,
            wave_path_col=wave_cols["waveform_path"],
            wave_start_col=time_cols["waveform_start"],
            extracted_stroke_time_col=time_cols["extracted_stroke_time"],
            grouping_mode="mcmed_visit_wave_type",
            visit_id_col=data_cfg["id_columns"].get("visit_id"),
            waveform_type_col=wave_cols.get("waveform_type"),
            passthrough_cols=[
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
            ],
        )

    raise ValueError(f"Unsupported dataset: {dataset}")


def choose_group_stroke_time(group_df: pd.DataFrame, stroke_col: str, dataset: str) -> pd.Timestamp | None:
    times: list[pd.Timestamp] = []
    for value in group_df[stroke_col].tolist():
        t = parse_time(value)
        if t is not None:
            times.append(t)

    if not times:
        return None

    if dataset == "mimic":
        return times[0]

    s = pd.Series(times)
    mode_vals = s.mode()
    if len(mode_vals) > 0:
        return mode_vals.min()
    return min(times)


def extract_features_optimized(
    ppg: np.ndarray,
    fp_df: pd.DataFrame,
    fs: float,
    min_beats: int = 3,
) -> pd.DataFrame:
    if len(fp_df) < min_beats:
        return pd.DataFrame()

    valid_mask = fp_df["on"].notna() & fp_df["off"].notna() & fp_df["sp"].notna()
    df = fp_df[valid_mask].copy()
    if len(df) < min_beats:
        return pd.DataFrame()

    on = df["on"].values.astype(int)
    off = df["off"].values.astype(int)
    sp = df["sp"].values.astype(int)

    res = pd.DataFrame(index=df.index)
    res["Beat_Idx"] = df.index
    res["On_Sample_Index"] = on
    res["Sp_Sample_Index"] = sp

    res["T_pi"] = (off - on) / fs
    res["T_sys"] = (sp - on) / fs
    res["T_dia"] = (off - sp) / fs
    res["T_sp"] = sp / fs

    res["IPR"] = np.where(res["T_pi"] > 0, 60 / res["T_pi"], np.nan)
    res["Tsys_Tdia"] = np.where(res["T_dia"] > 0, res["T_sys"] / res["T_dia"], np.nan)
    res["Tsp_Tpi"] = np.where(res["T_pi"] > 0, res["T_sp"] / res["T_pi"], np.nan)

    res["A_on"] = ppg[on]
    res["A_sp"] = ppg[sp]
    res["A_off"] = ppg[off]
    res["Pulse_Amplitude"] = res["A_sp"] - res["A_on"]
    res["SI"] = np.where(res["T_sys"] > 0, res["Pulse_Amplitude"] / res["T_sys"], np.nan)

    cols_to_calc_time = ["u", "v", "a", "b", "c", "d", "e", "f", "p1", "p2"]
    for pt in cols_to_calc_time:
        if pt not in df.columns:
            continue
        mask = df[pt].notna()
        if not mask.any():
            continue
        idx_vals = df.loc[mask, pt].values.astype(int)
        on_vals = on[mask]
        t_col = f"T_{pt}"
        res.loc[mask, t_col] = (idx_vals - on_vals) / fs
        if pt != "v":
            ratio_col = f"T{pt}_Tpi"
            t_pi_vals = res.loc[mask, "T_pi"].values
            res.loc[mask, ratio_col] = np.divide(
                res.loc[mask, t_col].values,
                t_pi_vals,
                out=np.full_like(t_pi_vals, np.nan),
                where=t_pi_vals != 0,
            )

    if "T_u" in res.columns and "T_a" in res.columns:
        res["Tu_Ta_Tpi"] = (res["T_u"] - res["T_a"]) / res["T_pi"]

    if len(res) >= 5:
        for col in ["T_pi", "T_sys", "Pulse_Amplitude"]:
            if col in res.columns:
                roll = res[col].rolling(window=5, min_periods=3)
                res[f"CV_{col}"] = roll.std() / roll.mean()

    return res


def process_chunk(
    raw_chunk: np.ndarray,
    fs: float,
    start_offset_idx: int,
    correction_flags: dict[str, bool],
    preprocessing_cfg: dict[str, Any],
    min_beats: int,
) -> pd.DataFrame | None:
    try:
        if len(raw_chunk) < 1000:
            return None

        s = DotMap()
        s.v = raw_chunk
        s.fs = fs
        s.filtering = bool(preprocessing_cfg.get("filtering", True))

        smoothing_cfg = preprocessing_cfg.get("smoothing_windows", {})
        preprocessor = Preprocess(
            fL=float(preprocessing_cfg.get("lowcut_hz", 0.5)),
            fH=float(preprocessing_cfg.get("highcut_hz", 5.0)),
            order=int(preprocessing_cfg.get("filter_order", 4)),
            sm_wins={
                "ppg": int(smoothing_cfg.get("ppg", 50)),
                "vpg": int(smoothing_cfg.get("vpg", 10)),
                "apg": int(smoothing_cfg.get("apg", 10)),
                "jpg": int(smoothing_cfg.get("jpg", 10)),
            },
        )
        ppg, vpg, apg, jpg = preprocessor.get_signals(s)

        ppg_data = DotMap()
        ppg_data.ppg = ppg
        ppg_data.vpg = vpg
        ppg_data.apg = apg
        ppg_data.jpg = jpg
        ppg_data.fs = fs

        fp_coll = FpCollection(ppg_data)
        corr = DotMap()
        corr.correction = pd.DataFrame({k: [v] for k, v in correction_flags.items()})
        fp_df = fp_coll.get_fiducials(corr)

        if len(fp_df) < min_beats:
            return None

        features = extract_features_optimized(ppg, fp_df, fs, min_beats=min_beats)
        if features.empty:
            return None

        features["On_Sample_Index"] += start_offset_idx
        features["Sp_Sample_Index"] += start_offset_idx
        return features

    except Exception:
        return None


def provisional_labels_from_rel_minutes(
    rel_min: np.ndarray,
    warning_start: float,
    warning_end: float,
    detection_start: float,
    detection_end: float,
) -> np.ndarray:
    labels = np.full(len(rel_min), -1, dtype=int)

    mask_warn = (rel_min >= warning_start) & (rel_min < warning_end)
    labels[mask_warn] = 1

    mask_detect = (rel_min >= detection_start) & (rel_min <= detection_end)
    labels[mask_detect] = 2

    mask_normal = rel_min < warning_start
    labels[mask_normal] = 0

    return labels


def build_group_args(
    valid_df: pd.DataFrame,
    spec: DatasetSpec,
    error_log_queue: Any,
) -> list[tuple[Any, pd.DataFrame, Any]]:
    args_list = []

    if spec.grouping_mode == "mimic_group_from_path":
        valid_df = valid_df.copy()
        valid_df["Group_ID"] = valid_df[spec.wave_path_col].apply(get_group_id_from_path)
        grouped = list(valid_df.groupby("Group_ID", sort=False))
        for group_key, group_df in grouped:
            args_list.append((group_key, group_df, error_log_queue))
        return args_list

    if spec.grouping_mode == "mcmed_visit_wave_type":
        if spec.visit_id_col is None or spec.waveform_type_col is None:
            raise ValueError("MC-MED grouping requires visit_id_col and waveform_type_col.")

        grouped = list(valid_df.groupby([spec.visit_id_col, spec.waveform_type_col], sort=False))
        for group_key, group_df in grouped:
            args_list.append((group_key, group_df, error_log_queue))
        return args_list

    raise ValueError(f"Unsupported grouping mode: {spec.grouping_mode}")


def process_group(
    group_key: Any,
    group_df: pd.DataFrame,
    spec: DatasetSpec,
    feature_cfg: dict[str, Any],
    output_dir: Path,
    error_log_queue: Any,
) -> dict[str, Any]:
    result = {
        "group_key": group_key,
        "status": "Failed",
        "error": "",
        "output": "",
        "n_files": len(group_df),
        "n_rows": 0,
    }

    try:
        fx = feature_cfg["feature_extraction"]
        chunk_minutes = int(fx.get("chunk_minutes", 30))
        min_signal_length = int(fx.get("min_signal_length", 5000))
        min_beats = max(1, int(fx.get("min_beats", 3)))
        target_signal = str(fx.get("target_signal", "PLETH")).upper()
        preprocessing_cfg = fx.get("preprocessing", {})

        provisional = feature_cfg.get("provisional_labeling", {})
        warning_start = float(provisional.get("warning_start_min", -120))
        warning_end = float(provisional.get("warning_end_min", 0))
        detection_start = float(provisional.get("detection_start_min", 0))
        detection_end = float(provisional.get("detection_end_min", 120))

        correction_flags = feature_cfg["feature_extraction"]["fiducial_correction"]

        stroke_time = choose_group_stroke_time(
            group_df=group_df,
            stroke_col=spec.extracted_stroke_time_col,
            dataset=spec.dataset_name,
        )
        is_stroke = 1 if stroke_time is not None else 0

        if is_stroke == 0 and "Is_Stroke" in group_df.columns:
            try:
                is_stroke = int(pd.to_numeric(group_df["Is_Stroke"], errors="coerce").fillna(0).max())
            except Exception:
                is_stroke = 0

        all_group_features: list[pd.DataFrame] = []

        for _, row in group_df.iterrows():
            file_path = str(row[spec.wave_path_col]).strip()
            if file_path == "" or file_path.lower() == "nan":
                continue

            file_name = os.path.basename(file_path)
            wave_start_time = parse_time(row.get(spec.wave_start_col, ""))

            if wave_start_time is None:
                error_log_queue.put({"file": file_name, "error": "ValueError:Missing or invalid waveform start time"})
                continue

            base_path = file_path.replace(".dat", "").replace(".hea", "")
            if not os.path.exists(base_path + ".hea"):
                error_log_queue.put({"file": file_name, "error": f"FileNotFoundError:Header not found {base_path}.hea"})
                continue

            signals, fields = wfdb.rdsamp(base_path)
            sig_names = [str(n).upper() for n in fields["sig_name"]]
            fs = float(fields["fs"])

            if target_signal not in sig_names:
                error_log_queue.put({"file": file_name, "error": f"ValueError:no_{target_signal.lower()}"})
                continue

            signal_idx = sig_names.index(target_signal)
            full_raw = np.nan_to_num(signals[:, signal_idx])

            if len(full_raw) < min_signal_length:
                error_log_queue.put({"file": file_name, "error": f"ValueError:too_short_{len(full_raw)}"})
                continue

            chunk_size = int(chunk_minutes * 60 * fs)
            feats_list: list[pd.DataFrame] = []

            for start_idx in range(0, len(full_raw), chunk_size):
                end_idx = min(start_idx + chunk_size, len(full_raw))
                raw_chunk = full_raw[start_idx:end_idx]
                chunk_feats = process_chunk(
                    raw_chunk=raw_chunk,
                    fs=fs,
                    start_offset_idx=start_idx,
                    correction_flags=correction_flags,
                    preprocessing_cfg=preprocessing_cfg,
                    min_beats=min_beats,
                )
                if chunk_feats is not None:
                    feats_list.append(chunk_feats)

            if not feats_list:
                error_log_queue.put({"file": file_name, "error": "ValueError:no_features_extracted_from_any_chunk"})
                continue

            features = pd.concat(feats_list, ignore_index=True)
            features = features.sort_values("On_Sample_Index").reset_index(drop=True)

            total_samples = len(full_raw)
            wave_end_time = wave_start_time + pd.Timedelta(seconds=total_samples / fs)

            seconds_from_start = features["On_Sample_Index"] / fs
            abs_time_series = wave_start_time + pd.to_timedelta(seconds_from_start, unit="s")

            rel_min = np.full(len(features), np.nan)
            labels = np.full(len(features), 0 if is_stroke == 0 else -1, dtype=int)

            stroke_in_file = False
            if stroke_time is not None and (wave_start_time <= stroke_time <= wave_end_time):
                stroke_in_file = True

            if is_stroke == 1 and stroke_time is not None:
                rel_min_series = (abs_time_series - stroke_time).dt.total_seconds() / 60.0
                rel_min = rel_min_series.values
                labels = provisional_labels_from_rel_minutes(
                    rel_min=rel_min,
                    warning_start=warning_start,
                    warning_end=warning_end,
                    detection_start=detection_start,
                    detection_end=detection_end,
                )

            features["Label"] = labels
            features["Absolute_Time"] = abs_time_series.dt.strftime("%Y/%m/%d %H:%M:%S.%f")
            features["Time_Rel_Min"] = rel_min

            if spec.dataset_name == "mimic":
                features.insert(0, "Source_File", file_name)
                features.insert(0, "Group_ID", str(group_key))
            else:
                visit_id, wave_type = group_key
                features.insert(0, "Source_File", file_name)
                features.insert(0, spec.wave_path_col, file_path)
                features.insert(0, spec.waveform_type_col or "Wave_Type", wave_type)
                features.insert(0, spec.visit_id_col or "CSN", visit_id)

            features.insert(0, "Stroke_Occurred_Here", stroke_in_file)
            features.insert(0, "Wave_End", wave_end_time)
            features.insert(0, "Wave_Start", wave_start_time)
            features.insert(0, "Actual_Stroke_Time", stroke_time)
            features.insert(0, "Is_Stroke_Subject", is_stroke)

            if spec.passthrough_cols:
                for col in spec.passthrough_cols:
                    if col in group_df.columns and col not in features.columns:
                        values = group_df[col].dropna().unique()
                        if len(values) > 1:
                            raise ValueError(
                                f"Group {group_key} contains multiple values for {col}: "
                                f"{values.tolist()}"
                            )
                        features[col] = values[0] if len(values) == 1 else np.nan

            if "On_Sample_Index" in features.columns:
                del features["On_Sample_Index"]
            if "Sp_Sample_Index" in features.columns:
                del features["Sp_Sample_Index"]

            all_group_features.append(features)

        if not all_group_features:
            raise RuntimeError(f"no_valid_files_in_group:{group_key}")

        merged = pd.concat(all_group_features, ignore_index=True)
        merged["_abs_dt"] = pd.to_datetime(merged["Absolute_Time"], errors="coerce")
        sort_cols = ["_abs_dt"]
        if "Source_File" in merged.columns:
            sort_cols.append("Source_File")
        merged = merged.sort_values(sort_cols).drop(columns=["_abs_dt"])

        if spec.dataset_name == "mimic":
            save_name = f"{group_key}.csv"
        else:
            visit_id, wave_type = group_key
            safe_wave = str(wave_type).replace(os.sep, "_").replace(" ", "_")
            save_name = f"{visit_id}_{safe_wave}.csv"

        save_path = output_dir / save_name
        merged.to_csv(save_path, index=False, encoding="utf-8-sig")

        result["status"] = "Success"
        result["output"] = str(save_path)
        result["n_rows"] = int(len(merged))
        return result

    except Exception as e:
        error_msg = f"{type(e).__name__}:{e}"
        result["error"] = error_msg
        error_log_queue.put({"file": f"GROUP::{group_key}", "error": error_msg})
        return result


def process_group_wrapper(args: tuple[Any, pd.DataFrame, Any, DatasetSpec, dict[str, Any], Path]) -> dict[str, Any]:
    group_key, group_df, error_log_queue, spec, feature_cfg, output_dir = args
    return process_group(
        group_key=group_key,
        group_df=group_df,
        spec=spec,
        feature_cfg=feature_cfg,
        output_dir=output_dir,
        error_log_queue=error_log_queue,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract beat-level PPG features from waveform segments for MIMIC or MC-MED."
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["mimic", "mcmed"])
    parser.add_argument("--feature-config", type=str, required=True, help="Path to configs/feature_extraction.yaml")
    parser.add_argument("--data-config", type=str, required=True, help="Path to dataset config (mimic_data.yaml or mcmed_data.yaml)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-processes", type=int, default=6)
    args = parser.parse_args()

    feature_cfg = load_yaml(args.feature_config)
    data_cfg = load_yaml(args.data_config)
    spec = resolve_dataset_spec(args.dataset, data_cfg)
    output_dir = resolve_output_dir(args.dataset, feature_cfg, args.output_dir)
    ensure_dir(output_dir)

    print(f"[extract_ppg_features] dataset={args.dataset}")
    print(f"[extract_ppg_features] index_table={spec.index_table_path}")
    print(f"[extract_ppg_features] output_dir={output_dir}")

    df_index = pd.read_csv(spec.index_table_path, low_memory=False)

    if spec.wave_path_col not in df_index.columns:
        raise ValueError(
            f"Index table missing path column: {spec.wave_path_col}. "
            f"Available columns: {list(df_index.columns)}"
        )
    if spec.wave_start_col not in df_index.columns:
        raise ValueError(
            f"Index table missing waveform start column: {spec.wave_start_col}. "
            f"Available columns: {list(df_index.columns)}"
        )
    if spec.extracted_stroke_time_col not in df_index.columns:
        raise ValueError(
            f"Index table missing stroke time column: {spec.extracted_stroke_time_col}. "
            f"Available columns: {list(df_index.columns)}"
        )

    pleth_flag_column = "\u5305\u542bPLETH"
    if pleth_flag_column in df_index.columns:
        df_index = df_index[df_index[pleth_flag_column].astype(str).str.upper() != "FALSE"].copy()

    if args.dataset == "mcmed" and spec.waveform_type_col and spec.waveform_type_col in df_index.columns:
        df_index = df_index[
            df_index[spec.waveform_type_col].astype(str).str.lower().str.contains("pleth", na=False)
        ].copy()

    df_index = df_index[df_index[spec.wave_path_col].astype(str).str.strip().ne("")].copy()

    manager = Manager()
    error_log_queue = manager.Queue()

    raw_args = build_group_args(df_index, spec, error_log_queue)
    args_list = [
        (group_key, group_df, error_log_queue, spec, feature_cfg, output_dir)
        for group_key, group_df, error_log_queue in raw_args
    ]

    print(f"[extract_ppg_features] rows={len(df_index):,}")
    print(f"[extract_ppg_features] groups={len(args_list):,}")
    print(f"[extract_ppg_features] num_processes={args.num_processes}")

    start_time = time.time()
    results: list[dict[str, Any]] = []

    with Pool(processes=args.num_processes) as pool:
        iterator = pool.imap_unordered(process_group_wrapper, args_list, chunksize=1)
        for res in tqdm(iterator, total=len(args_list), desc="Extracting PPG features"):
            results.append(res)
            if res["status"] == "Success":
                tqdm.write(f"[Success] {res['group_key']} -> {Path(res['output']).name}")
            else:
                tqdm.write(f"[Failed] {res['group_key']} -> {res['error']}")

    elapsed = time.time() - start_time

    summary_df = pd.DataFrame(results)
    summary_path = output_dir / "processing_summary_groups.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    errors: list[dict[str, Any]] = []
    while not error_log_queue.empty():
        errors.append(error_log_queue.get())

    error_path = output_dir / "processing_errors.json"
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    success_count = int((summary_df["status"] == "Success").sum()) if not summary_df.empty else 0

    print("[extract_ppg_features] done.")
    print(f"  success_groups={success_count}/{len(args_list)}")
    print(f"  elapsed_minutes={elapsed / 60:.2f}")
    print(f"  summary={summary_path}")
    print(f"  errors={error_path}")


if __name__ == "__main__":
    main()

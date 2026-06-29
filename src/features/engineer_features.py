
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Config
# =============================================================================

EXCLUDE_PREFIX = ("Summary_",)
EXCLUDE_NAMES = {
    "processing_summary.csv",
    "preprocess_log.csv",
    "feature_engineering_log.csv",
    "feature_manifest.json",
    "Summary_All.csv",
}

#  MIMIC / MC-MED
GROUP_COL_CANDIDATES = [
    "WAVE_PATH",
    "Source_File",
    "Wave_Path",
    "wave_path",
    "source_file",
]

SORT_COL_CANDIDATES = [
    "Absolute_Time",
    "Beat_Idx",
    "Time_Rel_Min",
]

RAW_BASE_FEATURES = [
    "T_pi", "T_sys", "T_dia", "T_sp",
    "IPR", "Tsys_Tdia", "Tsp_Tpi",
    "A_on", "A_sp", "A_off",
    "Pulse_Amplitude", "SI",
    "T_u", "Tu_Tpi", "T_v", "T_a", "Ta_Tpi",
    "T_b", "Tb_Tpi", "T_c", "Tc_Tpi", "T_d", "Td_Tpi",
    "T_e", "Te_Tpi", "T_f", "Tf_Tpi",
    "T_p1", "Tp1_Tpi", "T_p2", "Tp2_Tpi", "Tu_Ta_Tpi",
    "CV_T_pi", "CV_T_sys", "CV_Pulse_Amplitude",
]

COMPOSITE_FEATURES = ["NVI", "DSI", "NCI", "VSSI"]
DEFAULT_BASELINE_FRAC = 0.10
DEFAULT_BASELINE_MIN_ROWS = 5
DEFAULT_BASELINE_METHOD = "mean"

#  manifest
COMMON_METADATA_COLS = [
    "Group_ID", "CSN", "Wave_Type", "WAVE_PATH", "Source_File",
    "Stroke_Occurred_Here", "Wave_Start", "Wave_End", "Actual_Stroke_Time",
    "Is_Stroke_Subject", "Beat_Idx", "Label", "Absolute_Time", "Time_Rel_Min",
    "MRN", "Visit_no", "Age", "Gender", "Sex", "Race", "Ethnicity",
    "Dx_ICD10", "Dx_name", "ICD9_CODE", "Segment_Type", "Keep_Reason", "Gap_Hours",
]


# =============================================================================
# IO helpers
# =============================================================================

def is_feature_csv(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    if path.name in EXCLUDE_NAMES:
        return False
    return not any(path.name.startswith(p) for p in EXCLUDE_PREFIX)


def read_csv_fallback(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV: {path}") from last_err


# =============================================================================
# Schema helpers
# =============================================================================

def infer_group_col(df: pd.DataFrame, explicit: Optional[str] = None) -> Optional[str]:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(
                f"Specified group_col='{explicit}' not found. Available columns: {list(df.columns)}"
            )
        return explicit

    for col in GROUP_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def infer_sort_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in SORT_COL_CANDIDATES:
        if c in df.columns:
            cols.append(c)
    return cols


def pick_existing_cols(df: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


# =============================================================================
# Numeric sanitation
# =============================================================================

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.replace([np.inf, -np.inf], np.nan)

    #  datetime
    for col in ["Absolute_Time", "Wave_Start", "Wave_End", "Actual_Stroke_Time"]:
        if col in out.columns:
            try:
                out[col] = pd.to_datetime(out[col], errors="coerce")
            except Exception:
                pass

    return out


# =============================================================================
# Feature engineering: composite
# =============================================================================

def engineer_composite_features(df: pd.DataFrame, epsilon: float = 1e-6) -> pd.DataFrame:
    """
      NVI  = (T_f + T_p2 + IPR) / CV_T_pi
      DSI  = (T_f + T_p2) / T_pi
      NCI  = T_f / (CV_T_pi^2)
      VSSI = IPR^2 / T_dia
    """
    out = df.copy()
    cols = set(out.columns)

    if {"T_f", "T_p2", "IPR", "CV_T_pi"}.issubset(cols):
        denom = pd.to_numeric(out["CV_T_pi"], errors="coerce").replace(0, np.nan)
        out["NVI"] = (
            pd.to_numeric(out["T_f"], errors="coerce")
            + pd.to_numeric(out["T_p2"], errors="coerce")
            + pd.to_numeric(out["IPR"], errors="coerce")
        ) / (denom + epsilon)

    if {"T_f", "T_p2", "T_pi"}.issubset(cols):
        denom = pd.to_numeric(out["T_pi"], errors="coerce").replace(0, np.nan)
        out["DSI"] = (
            pd.to_numeric(out["T_f"], errors="coerce")
            + pd.to_numeric(out["T_p2"], errors="coerce")
        ) / (denom + epsilon)

    if {"T_f", "CV_T_pi"}.issubset(cols):
        denom = pd.to_numeric(out["CV_T_pi"], errors="coerce")
        out["NCI"] = pd.to_numeric(out["T_f"], errors="coerce") / ((denom ** 2) + epsilon)

    if {"IPR", "T_dia"}.issubset(cols):
        denom = pd.to_numeric(out["T_dia"], errors="coerce").replace(0, np.nan)
        ipr = pd.to_numeric(out["IPR"], errors="coerce")
        out["VSSI"] = (ipr ** 2) / (denom + epsilon)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# =============================================================================
# Feature engineering: relative / velocity / acceleration
# =============================================================================

def _baseline_size(n_rows: int, baseline_frac: float, baseline_min_rows: int) -> int:
    return min(n_rows, max(baseline_min_rows, int(math.ceil(n_rows * baseline_frac))))


def _compute_baseline(series: pd.Series, method: str, eps: float) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None

    method = method.lower()
    if method == "median":
        base = float(s.median())
    elif method == "mean":
        base = float(s.mean())
    else:
        raise ValueError(f"Unsupported baseline method: {method}")

    if not np.isfinite(base):
        return None

    if abs(base) < eps:
        base = eps if base >= 0 else -eps
    return base


def _engineer_group_kinematics(
    g: pd.DataFrame,
    *,
    base_features: Sequence[str],
    baseline_frac: float,
    baseline_min_rows: int,
    baseline_method: str,
    eps: float,
) -> pd.DataFrame:
    out = g.copy()

    for feat in base_features:
        if feat not in out.columns:
            continue

        x = pd.to_numeric(out[feat], errors="coerce")
        if x.notna().sum() == 0:
            continue

        n0 = _baseline_size(len(out), baseline_frac=baseline_frac, baseline_min_rows=baseline_min_rows)
        base = _compute_baseline(x.iloc[:n0], method=baseline_method, eps=eps)
        if base is None:
            continue

        denom = max(abs(base), eps)
      
        # (x - mu_base) / |mu_base|
        rel = (x - base) / denom
        vel = rel.diff().fillna(0.0)
        acc = vel.diff().fillna(0.0)

        out[f"{feat}_Rel"] = rel
        out[f"{feat}_Vel"] = vel
        out[f"{feat}_Accel"] = acc

    return out


def engineer_kinematic_features(
    df: pd.DataFrame,
    *,
    group_col: Optional[str],
    sort_cols: Sequence[str],
    base_features: Sequence[str],
    baseline_frac: float = DEFAULT_BASELINE_FRAC,
    baseline_min_rows: int = DEFAULT_BASELINE_MIN_ROWS,
    baseline_method: str = DEFAULT_BASELINE_METHOD,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
      *_Rel
      *_Vel
      *_Accel
    """
    out = df.copy()

    actual_sort_cols = [c for c in sort_cols if c in out.columns]
    if group_col is not None and group_col in out.columns:
        sort_keys = [group_col] + actual_sort_cols if actual_sort_cols else [group_col]
    else:
        sort_keys = actual_sort_cols

    if sort_keys:
        out = out.sort_values(sort_keys, kind="mergesort").copy()

    actual_base_features = [c for c in base_features if c in out.columns]
    if not actual_base_features:
        return out

    if group_col is not None and group_col in out.columns:
        parts = []
        for _, g in out.groupby(group_col, dropna=False, sort=False):
            parts.append(
                _engineer_group_kinematics(
                    g,
                    base_features=actual_base_features,
                    baseline_frac=baseline_frac,
                    baseline_min_rows=baseline_min_rows,
                    baseline_method=baseline_method,
                    eps=eps,
                )
            )
        out = pd.concat(parts, axis=0)
    else:
        out = _engineer_group_kinematics(
            out,
            base_features=actual_base_features,
            baseline_frac=baseline_frac,
            baseline_min_rows=baseline_min_rows,
            baseline_method=baseline_method,
            eps=eps,
        )

    return out


# =============================================================================
# Per-file processing
# =============================================================================

def process_one_file(
    in_path: Path,
    out_path: Path,
    *,
    group_col: Optional[str],
    baseline_frac: float,
    baseline_min_rows: int,
    baseline_method: str,
    verbose: bool = False,
) -> Dict[str, object]:
    df = read_csv_fallback(in_path)
    df = sanitize_dataframe(df)

    rows_in = len(df)
    if rows_in == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return {
            "file": in_path.name,
            "rows_in": 0,
            "rows_out": 0,
            "group_col": None,
            "n_raw_base_features": 0,
            "n_composite_added": 0,
            "n_kinematic_added": 0,
            "status": "empty",
        }

    actual_group_col = infer_group_col(df, explicit=group_col)
    sort_cols = infer_sort_cols(df)

    existing_raw = pick_existing_cols(df, RAW_BASE_FEATURES)

    df_comp = engineer_composite_features(df)

    composite_added = [c for c in COMPOSITE_FEATURES if c in df_comp.columns and c not in df.columns]

    base_for_kinematics = existing_raw + [c for c in COMPOSITE_FEATURES if c in df_comp.columns]
    df_feat = engineer_kinematic_features(
        df_comp,
        group_col=actual_group_col,
        sort_cols=sort_cols,
        base_features=base_for_kinematics,
        baseline_frac=baseline_frac,
        baseline_min_rows=baseline_min_rows,
        baseline_method=baseline_method,
        eps=1e-6,
    )

    kinematic_added = [
        c for c in df_feat.columns
        if (c.endswith("_Rel") or c.endswith("_Vel") or c.endswith("_Accel"))
        and c not in df.columns
    ]

    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    for col in ["Absolute_Time", "Wave_Start", "Wave_End", "Actual_Stroke_Time"]:
        if col in df_feat.columns and pd.api.types.is_datetime64_any_dtype(df_feat[col]):
            df_feat[col] = df_feat[col].dt.strftime("%Y/%m/%d %H:%M:%S")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(out_path, index=False, encoding="utf-8-sig")

    if verbose:
        print(
            f"[OK] {in_path.name} | rows={rows_in} | group={actual_group_col} | "
            f"raw={len(existing_raw)} | composite+={len(composite_added)} | "
            f"kinematic+={len(kinematic_added)}"
        )

    return {
        "file": in_path.name,
        "rows_in": int(rows_in),
        "rows_out": int(len(df_feat)),
        "group_col": actual_group_col,
        "n_raw_base_features": int(len(existing_raw)),
        "n_composite_added": int(len(composite_added)),
        "n_kinematic_added": int(len(kinematic_added)),
        "status": "ok",
    }


# =============================================================================
# Manifest
# =============================================================================

def build_feature_manifest(
    output_dir: Path,
    logs: List[Dict[str, object]],
    *,
    baseline_frac: float,
    baseline_min_rows: int,
    baseline_method: str,
) -> Dict[str, object]:
    manifest = {
        "raw_base_features": RAW_BASE_FEATURES,
        "composite_features": COMPOSITE_FEATURES,
        "derived_suffixes": ["_Rel", "_Vel", "_Accel"],
        "common_metadata_cols": COMMON_METADATA_COLS,
        "frozen_preprocessing": {
            "relative_feature_formula": "(x - mu_base) / abs(mu_base)",
            "baseline_method": baseline_method,
            "baseline_frac": baseline_frac,
            "baseline_min_rows": baseline_min_rows,
        },
        "n_processed_files": int(sum(1 for x in logs if str(x.get("status", "")).startswith("ok"))),
    }

    with open(output_dir / "feature_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Engineer composite and kinematic PPG features from cleaned per-file feature tables."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing cleaned feature CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. If omitted, writes to <input-dir>/engineered_features",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default=None,
        help="Optional explicit grouping column, e.g. Source_File or WAVE_PATH.",
    )
    parser.add_argument(
        "--baseline-frac",
        type=float,
        default=DEFAULT_BASELINE_FRAC,
        help="Fraction of initial rows used as subject/file baseline for *_Rel.",
    )
    parser.add_argument(
        "--baseline-min-rows",
        type=int,
        default=DEFAULT_BASELINE_MIN_ROWS,
        help="Minimum number of initial rows used for baseline calculation.",
    )
    parser.add_argument(
        "--baseline-method",
        type=str,
        default=DEFAULT_BASELINE_METHOD,
        choices=["median", "mean"],
        help="How to compute the baseline value within the initial stable window.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else (input_dir / "engineered_features")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in input_dir.iterdir() if is_feature_csv(p)])
    if not files:
        print(f"No feature CSV files found under: {input_dir}")
        return

    print(f"Input dir : {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Files     : {len(files)}")
    print(f"Group col : {args.group_col or 'auto'}")
    print(f"Baseline  : {args.baseline_method}, frac={args.baseline_frac}, min_rows={args.baseline_min_rows}")
    print("-" * 88)

    logs: List[Dict[str, object]] = []
    ok_count = 0
    err_count = 0

    for fp in files:
        out_path = output_dir / fp.name
        try:
            info = process_one_file(
                fp,
                out_path,
                group_col=args.group_col,
                baseline_frac=args.baseline_frac,
                baseline_min_rows=args.baseline_min_rows,
                baseline_method=args.baseline_method,
                verbose=args.verbose,
            )
            logs.append(info)
            ok_count += 1
        except Exception as e:
            err_count += 1
            logs.append(
                {
                    "file": fp.name,
                    "rows_in": np.nan,
                    "rows_out": np.nan,
                    "group_col": args.group_col,
                    "n_raw_base_features": np.nan,
                    "n_composite_added": np.nan,
                    "n_kinematic_added": np.nan,
                    "status": f"error: {type(e).__name__}: {e}",
                }
            )
            print(f"[ERROR] {fp.name}: {type(e).__name__}: {e}")

    log_df = pd.DataFrame(logs)
    log_path = output_dir / "feature_engineering_log.csv"
    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")

    build_feature_manifest(
        output_dir,
        logs,
        baseline_frac=args.baseline_frac,
        baseline_min_rows=args.baseline_min_rows,
        baseline_method=args.baseline_method,
    )

    print("-" * 88)
    print(f"Done. Success: {ok_count} | Failed: {err_count}")
    print(f"Log saved to     : {log_path}")
    print(f"Manifest saved to: {output_dir / 'feature_manifest.json'}")


if __name__ == "__main__":
    main()

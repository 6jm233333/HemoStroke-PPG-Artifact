from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# =============================================================================
# Default config
# =============================================================================

EXCLUDE_PREFIX = ("Summary_",)
EXCLUDE_NAMES = {
    "processing_summary.csv",
    "preprocess_log.csv",
    "Summary_All.csv",
}

# MC-MED  WAVE_PATH | MIMIC  Source_File
GROUP_COL_CANDIDATES = [
    "WAVE_PATH",
    "Wave_Path",
    "wave_path",
    "Source_File",
    "source_file",
]

ORDER_COL_CANDIDATES = [
    "Beat_Idx",
    "Absolute_Time",
    "Time_Rel_Min",
]

#  ffill+bfill
STATIC_COLS_CANDIDATES = [
    "Age", "Gender", "Sex",
    "CSN", "MRN", "Visit_no", "SUBJECT_ID", "HADM_ID",
    "Race", "Ethnicity", "Dx_ICD10", "Dx_name", "ICD9_CODE",
    "Group_ID", "Wave_Type",
    "Is_Stroke_Subject", "Stroke_Occurred_Here",
    "Segment_Type", "Keep_Reason",
    "Wave_Start", "Wave_End", "WAVE_START", "WAVE_END",
    "Actual_Stroke_Time", "Extracted_Timestamp",
]

FEATURE_COLS_CANDIDATES = [
    "T_pi", "T_sys", "T_dia", "T_sp",
    "IPR", "Tsys_Tdia", "Tsp_Tpi",
    "A_on", "A_sp", "A_off", "Pulse_Amplitude", "SI",
    "T_u", "Tu_Tpi", "T_v", "T_a", "Ta_Tpi", "T_b", "Tb_Tpi",
    "T_c", "Tc_Tpi", "T_d", "Td_Tpi", "T_e", "Te_Tpi", "T_f", "Tf_Tpi",
    "T_p1", "Tp1_Tpi", "T_p2", "Tp2_Tpi", "Tu_Ta_Tpi",
    "CV_T_pi", "CV_T_sys", "CV_Pulse_Amplitude",
]

RELATIVE_FEATURE_HINTS = (
    "_Rel", "_rel", "Rel_", "rel_"
)


# =============================================================================
# Utilities
# =============================================================================

def is_feature_csv(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    if path.name in EXCLUDE_NAMES:
        return False
    return not any(path.name.startswith(p) for p in EXCLUDE_PREFIX)


def pick_existing_cols(df: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def pick_group_col(df: pd.DataFrame, user_group_col: Optional[str] = None) -> str:
    if user_group_col:
        if user_group_col not in df.columns:
            raise ValueError(
                f"Specified group_col='{user_group_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        return user_group_col

    for c in GROUP_COL_CANDIDATES:
        if c in df.columns:
            return c

    raise ValueError(
        "Could not determine group column automatically. "
        f"Tried: {GROUP_COL_CANDIDATES}. "
        f"Available columns: {list(df.columns)}"
    )


def pick_order_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in ORDER_COL_CANDIDATES if c in df.columns]
    return cols


def detect_feature_cols(df: pd.DataFrame) -> List[str]:
    feature_cols = set(c for c in FEATURE_COLS_CANDIDATES if c in df.columns)

    for c in df.columns:
        if str(c).startswith("CV_"):
            feature_cols.add(c)

    for c in df.columns:
        if any(h in str(c) for h in RELATIVE_FEATURE_HINTS):
            if pd.api.types.is_numeric_dtype(df[c]):
                feature_cols.add(c)

    return sorted(feature_cols)


def split_feature_types(df: pd.DataFrame, feature_cols: Sequence[str]) -> Dict[str, List[str]]:
    cv_cols = [c for c in feature_cols if str(c).startswith("CV_")]

    phys_cols = []
    for c in feature_cols:
        if c in cv_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            phys_cols.append(c)

    static_cols = [
        c for c in STATIC_COLS_CANDIDATES
        if c in df.columns and c not in feature_cols
    ]

    return {
        "static_cols": static_cols,
        "cv_cols": cv_cols,
        "phys_cols": phys_cols,
    }


def coerce_time_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Absolute_Time", "Wave_Start", "Wave_End", "WAVE_START", "WAVE_END", "Actual_Stroke_Time", "Extracted_Timestamp"]:
        if col in out.columns:
            try:
                out[col] = pd.to_datetime(out[col], errors="ignore")
            except Exception:
                pass
    return out


def preprocess_group(
    g: pd.DataFrame,
    *,
    static_cols: Sequence[str],
    cv_cols: Sequence[str],
    phys_cols: Sequence[str],
    interp_limit: int,
) -> pd.DataFrame:
    g = g.copy()

    if static_cols:
        existing = [c for c in static_cols if c in g.columns]
        if existing:
            g[existing] = g[existing].ffill().bfill()
          
    if cv_cols:
        existing = [c for c in cv_cols if c in g.columns]
        if existing:
            g[existing] = g[existing].bfill()

    if phys_cols:
        existing = [c for c in phys_cols if c in g.columns]
        if existing:
            g[existing] = g[existing].interpolate(
                method="linear",
                limit=interp_limit,
                limit_area="inside",
            )

    return g

def preprocess_one_file(
    in_path: Path,
    out_path: Path,
    *,
    group_col: Optional[str] = None,
    interp_limit: int = 5,
    verbose: bool = False,
) -> Dict[str, object]:
    df = pd.read_csv(in_path)
    df.columns = [str(c).strip() for c in df.columns]

    rows_in = len(df)
    if rows_in == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return {
            "file": in_path.name,
            "rows_in": 0,
            "rows_out": 0,
            "drop_cnt": 0,
            "group_col": None,
            "n_groups": 0,
            "n_static_cols": 0,
            "n_cv_cols": 0,
            "n_phys_cols": 0,
            "n_feature_cols": 0,
            "status": "empty",
        }

    # Inf -> NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = coerce_time_like_columns(df)

    actual_group_col = pick_group_col(df, user_group_col=group_col)
    order_cols = pick_order_cols(df)
    feature_cols = detect_feature_cols(df)

    if not feature_cols:
        raise ValueError(
            f"No usable feature columns detected in {in_path.name}. "
            f"Available columns: {list(df.columns)}"
        )

    split = split_feature_types(df, feature_cols)
    static_cols = split["static_cols"]
    cv_cols = split["cv_cols"]
    phys_cols = split["phys_cols"]
    df["_orig_row_id"] = np.arange(len(df))

    sort_cols = [actual_group_col] + order_cols + ["_orig_row_id"]
    df = df.sort_values(sort_cols, kind="mergesort").copy()

    processed = (
        df.groupby(actual_group_col, dropna=False, group_keys=False)
        .apply(
            preprocess_group,
            static_cols=static_cols,
            cv_cols=cv_cols,
            phys_cols=phys_cols,
            interp_limit=interp_limit,
        )
        .copy()
    )

    rows_before_drop = len(processed)
    processed = processed.dropna(subset=feature_cols, how="any").copy()
    rows_out = len(processed)
    drop_cnt = rows_before_drop - rows_out
    processed = processed.drop(columns=["_orig_row_id"], errors="ignore")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(out_path, index=False, encoding="utf-8-sig")

    if verbose:
        print(
            f"[OK] {in_path.name} | rows {rows_in}->{rows_out} | "
            f"drop={drop_cnt} | group={actual_group_col} | "
            f"features={len(feature_cols)}"
        )

    return {
        "file": in_path.name,
        "rows_in": int(rows_in),
        "rows_out": int(rows_out),
        "drop_cnt": int(drop_cnt),
        "group_col": actual_group_col,
        "n_groups": int(processed[actual_group_col].nunique(dropna=False)) if actual_group_col in processed.columns else 0,
        "n_static_cols": int(len(static_cols)),
        "n_cv_cols": int(len(cv_cols)),
        "n_phys_cols": int(len(phys_cols)),
        "n_feature_cols": int(len(feature_cols)),
        "status": "ok",
    }


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean per-file PPG feature tables by group-wise fill/interpolation."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing feature CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. If omitted and --inplace is not set, a subdir named "
             "'cleaned_features' will be created under input-dir.",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default=None,
        help="Optional explicit group column. Example: Source_File or WAVE_PATH.",
    )
    parser.add_argument(
        "--interp-limit",
        type=int,
        default=5,
        help="Maximum consecutive missing rows to interpolate inside each group.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite files in-place.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file processing details.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    if args.inplace:
        output_dir = input_dir
    else:
        output_dir = Path(args.output_dir) if args.output_dir else (input_dir / "cleaned_features")
        output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in input_dir.iterdir() if is_feature_csv(p)])
    if not files:
        print(f"No feature CSV files found under: {input_dir}")
        return

    logs: List[Dict[str, object]] = []
    ok_count = 0
    err_count = 0

    print(f"Input dir : {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Files     : {len(files)}")
    print(f"In-place  : {args.inplace}")
    print("-" * 80)

    for fp in files:
        out_path = fp if args.inplace else (output_dir / fp.name)
        try:
            info = preprocess_one_file(
                fp,
                out_path,
                group_col=args.group_col,
                interp_limit=args.interp_limit,
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
                    "drop_cnt": np.nan,
                    "group_col": args.group_col,
                    "n_groups": np.nan,
                    "n_static_cols": np.nan,
                    "n_cv_cols": np.nan,
                    "n_phys_cols": np.nan,
                    "n_feature_cols": np.nan,
                    "status": f"error: {type(e).__name__}: {e}",
                }
            )
            print(f"[ERROR] {fp.name}: {type(e).__name__}: {e}")

    log_df = pd.DataFrame(logs)
    log_path = output_dir / "preprocess_log.csv"
    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")

    print("-" * 80)
    print(f"Done. Success: {ok_count} | Failed: {err_count}")
    print(f"Log saved to: {log_path}")

    if not log_df.empty and "rows_in" in log_df.columns and "rows_out" in log_df.columns:
        valid_rows = log_df.dropna(subset=["rows_in", "rows_out"])
        if not valid_rows.empty:
            total_in = int(valid_rows["rows_in"].sum())
            total_out = int(valid_rows["rows_out"].sum())
            print(f"Total rows: {total_in} -> {total_out} (dropped {total_in - total_out})")


if __name__ == "__main__":
    main()

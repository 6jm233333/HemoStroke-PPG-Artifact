
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Constants
# =============================================================================

EXCLUDE_PREFIX = ("Summary_",)
EXCLUDE_NAMES = {
    "processing_summary.csv",
    "preprocess_log.csv",
    "feature_engineering_log.csv",
    "feature_selection_log.csv",
    "feature_manifest.json",
    "selection_manifest.json",
    "Summary_All.csv",
}

COMMON_METADATA_COLS = [
    "Group_ID",
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
    "Sex",
    "Race",
    "Ethnicity",
    "Dx_ICD10",
    "Dx_name",
    "ICD9_CODE",
    "Segment_Type",
    "Keep_Reason",
    "Gap_Hours",
]

GROUP_COL_CANDIDATES = [
    "WAVE_PATH",
    "Source_File",
    "Group_ID",
    "CSN",
]

LABEL_COL_CANDIDATES = [
    "Label",
    "label",
    "target",
    "y",
]

# Optional prettified labels for plots
SYMBOL_MAPPING = {
    "NVI": r"\mathrm{NVI}",
    "DSI": r"\mathrm{DSI}",
    "NCI": r"\mathrm{NCI}",
    "VSSI": r"\mathrm{VSSI}",
    "SI": r"\mathrm{SI}",
    "IPR": r"\mathrm{IPR}",
}


# =============================================================================
# File helpers
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


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# =============================================================================
# Schema inference
# =============================================================================

def infer_group_col(df: pd.DataFrame, explicit: Optional[str] = None) -> Optional[str]:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(
                f"Specified group column '{explicit}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        return explicit

    for col in GROUP_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def infer_label_col(df: pd.DataFrame, explicit: Optional[str] = None) -> str:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(
                f"Specified label column '{explicit}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        return explicit

    for col in LABEL_COL_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError(f"Could not infer label column. Available columns: {list(df.columns)}")


def infer_metadata_cols(df: pd.DataFrame, explicit_metadata: Optional[Sequence[str]] = None) -> List[str]:
    if explicit_metadata is not None:
        return [c for c in explicit_metadata if c in df.columns]
    return [c for c in COMMON_METADATA_COLS if c in df.columns]


def detect_candidate_feature_cols(
    df: pd.DataFrame,
    *,
    metadata_cols: Sequence[str],
    label_col: str,
) -> List[str]:
    excluded = set(metadata_cols) | {label_col}
    out: List[str] = []

    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)

    return sorted(out)


# =============================================================================
# Effect-size utilities
# =============================================================================

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) < 2 or len(y) < 2:
        return 0.0

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = np.sqrt(((len(x) - 1) * vx + (len(y) - 1) * vy) / max(len(x) + len(y) - 2, 1))

    if not np.isfinite(pooled) or pooled == 0:
        return 0.0

    return float(abs((np.mean(x) - np.mean(y)) / pooled))


def build_effect_analysis_table(
    df: pd.DataFrame,
    *,
    label_col: str,
    feature_cols: Sequence[str],
    group_col: Optional[str],
    effect_unit: str = "row",
) -> pd.DataFrame:
    """
    Build the table used for Cohen's d computation.

    effect_unit:
      - "row": use all row-level observations with Label in {0, 1}
      - "group": aggregate within (group, label) before Cohen's d
    """
    work = df.copy()

    work[label_col] = pd.to_numeric(work[label_col], errors="coerce")
    work = work[work[label_col].isin([0, 1])].copy()

    keep_cols = [label_col] + [c for c in feature_cols if c in work.columns]
    if group_col is not None and group_col in work.columns:
        keep_cols = [group_col] + keep_cols

    work = work[keep_cols].copy()

    if work.empty:
        return work

    if effect_unit == "row":
        return work

    if effect_unit == "group":
        if group_col is None or group_col not in work.columns:
            raise ValueError("effect_unit='group' requires a valid group column.")
        agg = (
            work.groupby([group_col, label_col], dropna=False, sort=False)[list(feature_cols)]
            .mean(numeric_only=True)
            .reset_index()
        )
        return agg

    raise ValueError(f"Unsupported effect_unit: {effect_unit}")


def compute_effect_size_table(
    analysis_df: pd.DataFrame,
    *,
    label_col: str,
    feature_cols: Sequence[str],
    min_effect_size: float,
) -> pd.DataFrame:
    rows = []

    g0 = analysis_df[analysis_df[label_col] == 0]
    g1 = analysis_df[analysis_df[label_col] == 1]

    for feat in feature_cols:
        if feat not in analysis_df.columns:
            continue

        x = pd.to_numeric(g0[feat], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g1[feat], errors="coerce").to_numpy(dtype=float)

        d = cohens_d(x, y)

        rows.append(
            {
                "feature": feat,
                "cohens_d": float(d),
                "n_label_0": int(np.isfinite(x).sum()),
                "n_label_1": int(np.isfinite(y).sum()),
                "passed_effect_threshold": bool(d > min_effect_size),
            }
        )

    score_df = pd.DataFrame(rows)
    if score_df.empty:
        return score_df

    score_df = score_df.sort_values(["cohens_d", "feature"], ascending=[False, True]).reset_index(drop=True)
    return score_df


# =============================================================================
# Correlation pruning
# =============================================================================

def build_correlation_analysis_table(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    group_col: Optional[str],
    corr_unit: str = "row",
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build the table used for correlation computation.

    corr_unit:
      - "row": use all rows
      - "group": aggregate per group over all retained rows
      - "group_label": aggregate per (group, label)
    """
    work = df.copy()
    keep = [c for c in feature_cols if c in work.columns]
    if not keep:
        return pd.DataFrame()

    if corr_unit == "row":
        return work[keep].copy()

    if group_col is None or group_col not in work.columns:
        raise ValueError(f"corr_unit='{corr_unit}' requires a valid group column.")

    if corr_unit == "group":
        return (
            work[[group_col] + keep]
            .groupby(group_col, dropna=False, sort=False)[keep]
            .mean(numeric_only=True)
            .reset_index(drop=True)
        )

    if corr_unit == "group_label":
        if label_col is None or label_col not in work.columns:
            raise ValueError("corr_unit='group_label' requires a valid label column.")
        return (
            work[[group_col, label_col] + keep]
            .groupby([group_col, label_col], dropna=False, sort=False)[keep]
            .mean(numeric_only=True)
            .reset_index(drop=True)
        )

    raise ValueError(f"Unsupported corr_unit: {corr_unit}")


def compute_pairwise_correlation_table(corr_df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    usable = [c for c in feature_cols if c in corr_df.columns]
    if not usable:
        return pd.DataFrame(columns=["feature_1", "feature_2", "pearson_r_abs"])

    mat = corr_df[usable].corr(method="pearson").abs()
    rows = []

    for i, f1 in enumerate(usable):
        for j in range(i + 1, len(usable)):
            f2 = usable[j]
            rows.append(
                {
                    "feature_1": f1,
                    "feature_2": f2,
                    "pearson_r_abs": float(mat.loc[f1, f2]),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["pearson_r_abs", "feature_1", "feature_2"], ascending=[False, True, True]).reset_index(drop=True)
    return out


def greedy_correlation_pruning(
    score_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    *,
    corr_threshold: float,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Keep the highest-ranked features by Cohen's d, then prune any later feature
    whose absolute Pearson correlation with an already selected feature exceeds
    corr_threshold.
    """
    ranked_features = score_df["feature"].tolist()
    corr_features = [c for c in ranked_features if c in corr_df.columns]
    corr_mat = corr_df[corr_features].corr(method="pearson").abs()

    selected: List[str] = []
    prune_log: List[Dict[str, object]] = []

    for feat in ranked_features:
        if feat not in corr_mat.columns:
            selected.append(feat)
            prune_log.append(
                {
                    "feature": feat,
                    "status": "kept_no_corr_column",
                    "blocked_by": None,
                    "pearson_r_abs": np.nan,
                }
            )
            continue

        blocked = False
        for kept in selected:
            if kept not in corr_mat.index:
                continue
            r = float(corr_mat.loc[feat, kept])
            if r > corr_threshold:
                prune_log.append(
                    {
                        "feature": feat,
                        "status": "dropped_correlated",
                        "blocked_by": kept,
                        "pearson_r_abs": r,
                    }
                )
                blocked = True
                break

        if not blocked:
            selected.append(feat)
            prune_log.append(
                {
                    "feature": feat,
                    "status": "kept",
                    "blocked_by": None,
                    "pearson_r_abs": np.nan,
                }
            )

    log_df = pd.DataFrame(prune_log)
    return selected, log_df


# =============================================================================
# Plotting
# =============================================================================

def get_latex_label(feature_name: str) -> str:
    suffix_latex = ""
    clean_name = feature_name

    if clean_name.endswith("_Accel"):
        suffix_latex = r"^{\mathrm{Acc}}"
        clean_name = clean_name[:-6]
    elif clean_name.endswith("_Vel"):
        suffix_latex = r"^{\mathrm{Vel}}"
        clean_name = clean_name[:-4]
    elif clean_name.endswith("_Rel"):
        suffix_latex = r"^{\mathrm{Rel}}"
        clean_name = clean_name[:-4]

    if clean_name in SYMBOL_MAPPING:
        core = SYMBOL_MAPPING[clean_name]
        return f"${core}{suffix_latex}$"

    if clean_name.startswith("CV_"):
        base = clean_name.replace("CV_", "")
        if "_" in base:
            head, tail = base.split("_", 1)
            core = f"CV_{{{head}_{{{tail}}}}}"
        else:
            core = f"CV_{{{base}}}"
        return f"${core}{suffix_latex}$"

    if "_" in clean_name:
        parts = clean_name.split("_")
        core = f"{parts[0]}_{{{''.join(parts[1:])}}}"
        return f"${core}{suffix_latex}$"

    return f"${clean_name}{suffix_latex}$"


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    features: Sequence[str],
    output_png: Path,
    output_pdf: Path,
) -> None:
    valid_features = [f for f in features if f in corr_df.columns]
    if not valid_features:
        return

    mat = corr_df[valid_features].corr(method="pearson")
    labels = [get_latex_label(f) for f in valid_features]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "font.size": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(valid_features)))
    ax.set_yticks(np.arange(len(valid_features)))
    ax.set_xticklabels(labels, rotation=90, ha="center")
    ax.set_yticklabels(labels)

    ax.set_title("Feature Correlation Matrix", fontsize=15, fontweight="bold", pad=16)

    # Draw cell borders
    ax.set_xticks(np.arange(-0.5, len(valid_features), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(valid_features), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Pearson Correlation Coefficient (r)", rotation=90)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Export reduced tables
# =============================================================================

def apply_feature_subset_to_directory(
    *,
    input_dir: Path,
    output_dir: Path,
    selected_features: Sequence[str],
    metadata_cols: Sequence[str],
    verbose: bool = False,
) -> pd.DataFrame:
    logs: List[Dict[str, object]] = []

    files = sorted([p for p in input_dir.iterdir() if is_feature_csv(p)])
    for fp in files:
        try:
            df = sanitize_dataframe(read_csv_fallback(fp))
            keep_cols = [c for c in metadata_cols if c in df.columns] + [c for c in selected_features if c in df.columns]
            keep_cols = list(dict.fromkeys(keep_cols))

            reduced = df[keep_cols].copy()

            out_path = output_dir / fp.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            reduced.to_csv(out_path, index=False, encoding="utf-8-sig")

            logs.append(
                {
                    "file": fp.name,
                    "rows": int(len(reduced)),
                    "n_metadata_cols": int(sum(c in reduced.columns for c in metadata_cols)),
                    "n_selected_features": int(sum(c in reduced.columns for c in selected_features)),
                    "status": "ok",
                }
            )

            if verbose:
                print(f"[OK] {fp.name} -> {out_path.name} | kept_cols={len(keep_cols)}")

        except Exception as e:
            logs.append(
                {
                    "file": fp.name,
                    "rows": np.nan,
                    "n_metadata_cols": np.nan,
                    "n_selected_features": np.nan,
                    "status": f"error: {type(e).__name__}: {e}",
                }
            )
            print(f"[ERROR] {fp.name}: {type(e).__name__}: {e}")

    export_log = pd.DataFrame(logs)
    export_log.to_csv(output_dir / "reduced_table_export_log.csv", index=False, encoding="utf-8-sig")
    return export_log


# =============================================================================
# End-to-end pipeline
# =============================================================================

def load_directory_tables(input_dir: Path) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []

    files = sorted([p for p in input_dir.iterdir() if is_feature_csv(p)])
    if not files:
        raise FileNotFoundError(f"No eligible CSV files found under: {input_dir}")

    for fp in files:
        df = sanitize_dataframe(read_csv_fallback(fp))
        df["__file_name__"] = fp.name
        parts.append(df)

    merged = pd.concat(parts, ignore_index=True)
    return merged


def run_selection_pipeline(
    *,
    input_dir: Path,
    output_dir: Path,
    group_col: Optional[str],
    label_col: Optional[str],
    effect_threshold: float,
    corr_threshold: float,
    effect_unit: str,
    corr_unit: str,
    heatmap_top_k: int,
    export_reduced_tables: bool,
    verbose: bool,
) -> Dict[str, object]:
    merged = load_directory_tables(input_dir)

    actual_label_col = infer_label_col(merged, explicit=label_col)
    actual_group_col = infer_group_col(merged, explicit=group_col)
    metadata_cols = infer_metadata_cols(merged)

    candidate_features = detect_candidate_feature_cols(
        merged,
        metadata_cols=metadata_cols,
        label_col=actual_label_col,
    )

    if not candidate_features:
        raise ValueError("No numeric candidate feature columns found after excluding metadata.")

    # 1) Cohen's d table
    effect_df = build_effect_analysis_table(
        merged,
        label_col=actual_label_col,
        feature_cols=candidate_features,
        group_col=actual_group_col,
        effect_unit=effect_unit,
    )

    score_df = compute_effect_size_table(
        effect_df,
        label_col=actual_label_col,
        feature_cols=candidate_features,
        min_effect_size=effect_threshold,
    )

    if score_df.empty:
        raise ValueError("No effect-size scores could be computed.")

    passed_df = score_df[score_df["passed_effect_threshold"]].copy()
    ranked_after_effect = passed_df["feature"].tolist()

    # 2) Correlation table and greedy pruning
    corr_analysis_df = build_correlation_analysis_table(
        merged,
        feature_cols=ranked_after_effect,
        group_col=actual_group_col,
        corr_unit=corr_unit,
        label_col=actual_label_col,
    )

    pairwise_corr_df = compute_pairwise_correlation_table(corr_analysis_df, ranked_after_effect)
    selected_features, prune_log_df = greedy_correlation_pruning(
        passed_df,
        corr_analysis_df,
        corr_threshold=corr_threshold,
    )

    selected_df = passed_df[passed_df["feature"].isin(selected_features)].copy()
    selected_df = selected_df.sort_values(["cohens_d", "feature"], ascending=[False, True]).reset_index(drop=True)

    # 3) Save analysis outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    score_df.to_csv(output_dir / "feature_effect_sizes.csv", index=False)
    passed_df.to_csv(output_dir / "features_passing_effect_threshold.csv", index=False)
    pairwise_corr_df.to_csv(output_dir / "feature_pairwise_correlations.csv", index=False)
    prune_log_df.to_csv(output_dir / "feature_pruning_log.csv", index=False)
    selected_df.to_csv(output_dir / "selected_features.csv", index=False)

    with open(output_dir / "selected_feature_names.txt", "w", encoding="utf-8") as f:
        for feat in selected_features:
            f.write(f"{feat}\n")

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "group_col": actual_group_col,
        "label_col": actual_label_col,
        "metadata_cols": metadata_cols,
        "effect_threshold": effect_threshold,
        "corr_threshold": corr_threshold,
        "effect_unit": effect_unit,
        "corr_unit": corr_unit,
        "n_candidate_features": len(candidate_features),
        "n_after_effect_threshold": len(ranked_after_effect),
        "n_selected_features": len(selected_features),
        "selected_features": selected_features,
    }
    with open(output_dir / "selection_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 4) Heatmap
    if heatmap_top_k > 0 and selected_features:
        heatmap_features = selected_features[: min(heatmap_top_k, len(selected_features))]
        plot_correlation_heatmap(
            corr_analysis_df,
            heatmap_features,
            output_png=output_dir / "selected_feature_correlation_heatmap.png",
            output_pdf=output_dir / "selected_feature_correlation_heatmap.pdf",
        )

    # 5) Export reduced tables
    export_log_df = pd.DataFrame()
    if export_reduced_tables:
        reduced_dir = output_dir / "reduced_tables"
        export_log_df = apply_feature_subset_to_directory(
            input_dir=input_dir,
            output_dir=reduced_dir,
            selected_features=selected_features,
            metadata_cols=metadata_cols,
            verbose=verbose,
        )

    if verbose:
        print(f"Label column      : {actual_label_col}")
        print(f"Group column      : {actual_group_col}")
        print(f"Candidate features: {len(candidate_features)}")
        print(f"After d-threshold : {len(ranked_after_effect)}")
        print(f"Final selected    : {len(selected_features)}")

    return {
        "manifest": manifest,
        "scores": score_df,
        "selected": selected_df,
        "prune_log": prune_log_df,
        "pairwise_corr": pairwise_corr_df,
        "export_log": export_log_df,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select engineered PPG features using Cohen's d and correlation pruning."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing engineered feature CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <input-dir>/selected_features",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default=None,
        help="Optional grouping column for grouped analysis, e.g. Source_File or WAVE_PATH.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Optional label column. Defaults to auto-detection.",
    )
    parser.add_argument(
        "--effect-threshold",
        type=float,
        default=0.05,
        help="Minimum Cohen's d threshold. Paper default: 0.05.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.80,
        help="Maximum allowed absolute Pearson correlation. Paper default: 0.80.",
    )
    parser.add_argument(
        "--effect-unit",
        type=str,
        default="row",
        choices=["row", "group"],
        help="Unit for Cohen's d computation: row or group.",
    )
    parser.add_argument(
        "--corr-unit",
        type=str,
        default="row",
        choices=["row", "group", "group_label"],
        help="Unit for correlation pruning.",
    )
    parser.add_argument(
        "--heatmap-top-k",
        type=int,
        default=30,
        help="Plot the correlation heatmap for the top-k selected features. Set 0 to disable.",
    )
    parser.add_argument(
        "--no-export-reduced-tables",
        action="store_true",
        help="Do not write reduced per-file CSV tables.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else (input_dir / "selected_features")

    result = run_selection_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        group_col=args.group_col,
        label_col=args.label_col,
        effect_threshold=args.effect_threshold,
        corr_threshold=args.corr_threshold,
        effect_unit=args.effect_unit,
        corr_unit=args.corr_unit,
        heatmap_top_k=args.heatmap_top_k,
        export_reduced_tables=not args.no_export_reduced_tables,
        verbose=args.verbose,
    )

    manifest = result["manifest"]
    print("-" * 88)
    print("Feature selection finished.")
    print(f"Input dir           : {manifest['input_dir']}")
    print(f"Output dir          : {manifest['output_dir']}")
    print(f"Label column        : {manifest['label_col']}")
    print(f"Group column        : {manifest['group_col']}")
    print(f"Candidates          : {manifest['n_candidate_features']}")
    print(f"After Cohen's d     : {manifest['n_after_effect_threshold']}")
    print(f"Final selected      : {manifest['n_selected_features']}")
    print(f"Manifest            : {output_dir / 'selection_manifest.json'}")
    print(f"Selected features   : {output_dir / 'selected_features.csv'}")
    print(f"Reduced tables dir  : {output_dir / 'reduced_tables'}")


if __name__ == "__main__":
    main()

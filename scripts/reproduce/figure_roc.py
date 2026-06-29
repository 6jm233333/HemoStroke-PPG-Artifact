
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# allow "python scripts/reproduce/figure_roc.py" from repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.roc_analysis import summarize_multi_horizon_roc, export_roc_points  # noqa: E402


TRUE_ALIASES = ["y_true", "label", "target", "gt", "true_label"]
SCORE_ALIASES = ["y_prob", "prob", "score", "pred_prob", "positive_prob", "prob_1", "p1"]
HORIZON_ALIASES = ["horizon", "window", "window_min", "lead_time_min", "minutes", "horizon_min"]
FOLD_ALIASES = ["fold", "cv_fold", "split", "fold_id"]


def _find_first_existing(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    colset = {str(c).strip(): c for c in columns}
    for a in aliases:
        if a in colset:
            return colset[a]
    return None


def _infer_horizon_from_name(name: str) -> Optional[int]:
    text = name.lower()

    for val in (240, 300, 360):
        if str(val) in text:
            return val

    m = re.search(r"([456])h\b", text)
    if m:
        return int(m.group(1)) * 60

    return None


def _infer_fold_from_name(name: str) -> int:
    text = name.lower()
    m = re.search(r"(?:fold|cv|split)[_\-]?(\d+)", text)
    if m:
        return int(m.group(1))
    return 0


def _collect_csv_files(path_like: str) -> List[Path]:
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.csv"))


def _load_prediction_csvs(path_like: str, dataset_name: str) -> pd.DataFrame:
    frames = []
    files = _collect_csv_files(path_like)

    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue

        if df.empty:
            continue

        df.columns = [str(c).strip() for c in df.columns]

        true_col = _find_first_existing(df.columns, TRUE_ALIASES)
        score_col = _find_first_existing(df.columns, SCORE_ALIASES)
        horizon_col = _find_first_existing(df.columns, HORIZON_ALIASES)
        fold_col = _find_first_existing(df.columns, FOLD_ALIASES)

        if true_col is None or score_col is None:
            continue

        keep = [true_col, score_col]
        rename = {true_col: "y_true", score_col: "y_score"}

        if horizon_col is not None:
            keep.append(horizon_col)
            rename[horizon_col] = "horizon"
        if fold_col is not None:
            keep.append(fold_col)
            rename[fold_col] = "fold"

        part = df[keep].rename(columns=rename).copy()
        part["dataset"] = dataset_name

        if "horizon" not in part.columns:
            inferred_horizon = _infer_horizon_from_name(fp.stem)
            if inferred_horizon is None:
                raise ValueError(
                    f"Could not infer horizon from filename '{fp.name}'. "
                    "Please include a horizon column or encode 240/300/360 or 4h/5h/6h in the filename."
                )
            part["horizon"] = inferred_horizon

        if "fold" not in part.columns:
            part["fold"] = _infer_fold_from_name(fp.stem)

        frames.append(part)

    if not frames:
        raise ValueError(f"No usable prediction CSVs found under: {path_like}")

    out = pd.concat(frames, ignore_index=True)
    out["horizon"] = out["horizon"].astype(str).map(lambda x: int(float(x)))
    out["fold"] = out["fold"].astype(str).map(lambda x: int(float(x)))
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out["y_score"] = pd.to_numeric(out["y_score"], errors="coerce")
    out = out.dropna(subset=["y_true", "y_score"])
    out["y_true"] = out["y_true"].astype(int)
    return out


def _horizon_label(minutes: int) -> str:
    try:
        return f"{int(round(minutes / 60.0))}h"
    except Exception:
        return str(minutes)


def plot_roc_figure(summary_dict: dict, output_png: Path, output_pdf: Path, title: str) -> None:
    roc_points = summary_dict["roc_points"].copy()
    summary = summary_dict["summary"].copy()

    if roc_points.empty or summary.empty:
        raise ValueError("ROC summary is empty.")

    plt.figure(figsize=(8.2, 6.4))
    ax = plt.gca()

    horizon_values = sorted(pd.unique(summary["horizon"]), key=lambda x: float(x))
    color_map = {
        240: "#4C72B0",
        300: "#DD8452",
        360: "#55A868",
    }
    linestyle_map = {
        "MIMIC-III (Internal)": "-",
        "MC-MED (External)": "--",
    }

    datasets = list(pd.unique(summary["dataset"]))
    if len(datasets) == 2:
        ds0, ds1 = datasets[0], datasets[1]
        linestyle_map = {
            ds0: "-",
            ds1: "--",
        }

    for dataset_name in datasets:
        for horizon in horizon_values:
            part = roc_points[(roc_points["dataset"] == dataset_name) & (roc_points["horizon"] == horizon)].copy()
            row = summary[(summary["dataset"] == dataset_name) & (summary["horizon"] == horizon)].copy()
            if part.empty or row.empty:
                continue

            auc_mean = float(row["auc_mean"].iloc[0])
            color = color_map.get(int(float(horizon)), None)
            ls = linestyle_map.get(dataset_name, "-")

            ax.plot(
                part["fpr"],
                part["mean_tpr"],
                linewidth=2.2,
                linestyle=ls,
                color=color,
                alpha=0.98,
            )

    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1.2, color="gray", alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)

    window_handles = []
    for horizon in horizon_values:
        minutes = int(float(horizon))
        window_handles.append(
            Line2D(
                [0],
                [0],
                color=color_map.get(minutes, "black"),
                lw=2.2,
                linestyle="-",
                label=f"{_horizon_label(minutes)} window",
            )
        )

    style_handles = []
    for dataset_name in datasets:
        style_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                lw=2.2,
                linestyle=linestyle_map.get(dataset_name, "-"),
                label=dataset_name,
            )
        )

    leg1 = ax.legend(handles=window_handles, loc="lower right", frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="lower center", frameon=False)

    ax.grid(True, linestyle="--", alpha=0.25)
    plt.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce ROC analysis figure.")
    parser.add_argument("--internal", required=True, help="Internal prediction CSV or directory of CSVs.")
    parser.add_argument("--external", required=True, help="External prediction CSV or directory of CSVs.")
    parser.add_argument("--internal-name", default="MIMIC-III (Internal)")
    parser.add_argument("--external-name", default="MC-MED (External)")
    parser.add_argument("--out-dir", default="outputs/figures/roc")
    parser.add_argument(
        "--title",
        default="ResNet ROC across warning horizons",
    )
    args = parser.parse_args()

    internal_df = _load_prediction_csvs(args.internal, args.internal_name)
    external_df = _load_prediction_csvs(args.external, args.external_name)
    pred_df = pd.concat([internal_df, external_df], ignore_index=True)

    roc_result = summarize_multi_horizon_roc(
        pred_df,
        y_true_col="y_true",
        y_score_col="y_score",
        dataset_col="dataset",
        horizon_col="horizon",
        fold_col="fold",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "roc_summary.csv"
    points_csv = out_dir / "roc_points.csv"
    per_fold_auc_csv = out_dir / "roc_per_fold_auc.csv"

    roc_result["summary"].to_csv(summary_csv, index=False)
    export_roc_points(roc_result, points_csv)
    roc_result["per_fold_auc"].to_csv(per_fold_auc_csv, index=False)

    plot_roc_figure(
        summary_dict=roc_result,
        output_png=out_dir / "figure_roc.png",
        output_pdf=out_dir / "figure_roc.pdf",
        title=args.title,
    )

    print(f"[OK] Figure saved to: {out_dir / 'figure_roc.png'}")
    print(f"[OK] Figure saved to: {out_dir / 'figure_roc.pdf'}")
    print(f"[OK] Summary saved to: {summary_csv}")
    print(f"[OK] ROC points saved to: {points_csv}")
    print(f"[OK] Per-fold AUC saved to: {per_fold_auc_csv}")


if __name__ == "__main__":
    main()

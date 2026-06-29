from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def label_relative_minutes(
    rel_min: float,
    *,
    horizon_minutes: int,
    normal_start_minutes: int = -480,
    boundary_buffer_minutes: int = 15,
    lead_time_minutes: int = 15,
) -> int:
    """Return paper labels after applying an onset-anchor-relative timeline."""
    if pd.isna(rel_min):
        return -1

    warning_start = -int(horizon_minutes) + int(boundary_buffer_minutes)
    normal_end = -int(horizon_minutes) - int(boundary_buffer_minutes)
    lead_cutoff = -int(lead_time_minutes)

    if normal_start_minutes <= rel_min < normal_end:
        return 0
    if warning_start <= rel_min < lead_cutoff:
        return 1
    return -1


def shifted_relative_minutes(rel_min: pd.Series, shift_minutes: float) -> pd.Series:
    """Shift the documented onset anchor and recompute relative time.

    Positive shift means the anchor is moved later. Since relative time is
    sample_time - anchor_time, the shifted relative time is old_rel - shift.
    """
    return pd.to_numeric(rel_min, errors="coerce") - float(shift_minutes)


def relabel_dataframe(
    df: pd.DataFrame,
    *,
    shift_minutes: float,
    horizon_minutes: int,
    rel_time_col: str = "Time_Rel_Min",
    label_col: str = "Label",
) -> pd.DataFrame:
    if rel_time_col not in df.columns:
        raise KeyError(f"Missing relative-time column: {rel_time_col}")

    out = df.copy()
    shifted_col = f"{rel_time_col}_Shifted"
    out[shifted_col] = shifted_relative_minutes(out[rel_time_col], shift_minutes)
    out[label_col] = [
        label_relative_minutes(x, horizon_minutes=horizon_minutes)
        for x in out[shifted_col].to_numpy()
    ]
    return out


def iter_csv_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.rglob("*.csv")):
        if path.name.startswith("Summary") or path.name.startswith("summary"):
            continue
        yield path


def relabel_directory(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for csv_path in iter_csv_files(input_dir):
        rel = csv_path.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(csv_path)
        out = relabel_dataframe(
            df,
            shift_minutes=args.shift_minutes,
            horizon_minutes=args.horizon_minutes,
            rel_time_col=args.rel_time_col,
            label_col=args.label_col,
        )
        out.to_csv(out_path, index=False)
        counts = out[args.label_col].value_counts(dropna=False).to_dict()
        rows.append(
            {
                "file": str(rel),
                "rows": int(len(out)),
                "label_0": int(counts.get(0, 0)),
                "label_1": int(counts.get(1, 0)),
                "label_ignore": int(counts.get(-1, 0)),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "relabel_shift_summary.csv", index=False)
    manifest = {
        "shift_minutes": float(args.shift_minutes),
        "horizon_minutes": int(args.horizon_minutes),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_files": int(len(rows)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def summarize_metrics(args: argparse.Namespace) -> None:
    metrics_dir = Path(args.metrics_dir)
    frames = []
    for path in sorted(metrics_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if args.metric not in df.columns:
            continue
        df = df.copy()
        if "shift_minutes" not in df.columns:
            shift = None
            for token in path.stem.replace("-", "_").split("_"):
                try:
                    shift = float(token)
                    break
                except ValueError:
                    continue
            df["shift_minutes"] = shift
        frames.append(df[["shift_minutes", args.metric]].dropna())

    if not frames:
        raise ValueError(f"No metric CSVs with column '{args.metric}' found under {metrics_dir}")

    all_df = pd.concat(frames, ignore_index=True)
    summary = (
        all_df.groupby("shift_minutes", as_index=False)[args.metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)

    if args.output_plot:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.errorbar(summary["shift_minutes"], summary["mean"], yerr=summary["std"], marker="o", capsize=3)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Anchor shift (min)")
        ax.set_ylabel(args.metric)
        ax.set_title("Onset-anchor perturbation sensitivity")
        fig.tight_layout()
        plot_path = Path(args.output_plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Onset-anchor perturbation utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    relabel = sub.add_parser("relabel", help="Relabel feature CSVs after shifting the documented onset anchor.")
    relabel.add_argument("--input-dir", required=True)
    relabel.add_argument("--output-dir", required=True)
    relabel.add_argument("--shift-minutes", type=float, required=True)
    relabel.add_argument("--horizon-minutes", type=int, required=True)
    relabel.add_argument("--rel-time-col", default="Time_Rel_Min")
    relabel.add_argument("--label-col", default="Label")
    relabel.set_defaults(func=relabel_directory)

    summarize = sub.add_parser("summarize", help="Summarize per-shift metric CSV files.")
    summarize.add_argument("--metrics-dir", required=True)
    summarize.add_argument("--metric", default="f1")
    summarize.add_argument("--output-csv", required=True)
    summarize.add_argument("--output-plot", default=None)
    summarize.set_defaults(func=summarize_metrics)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

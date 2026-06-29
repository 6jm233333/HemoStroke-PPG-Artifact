
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FEATURE_GROUPS = [
    ["T_pi", "T_sys", "T_dia"],
    ["T_sp", "IPR", "Pulse_Amplitude"],
    ["A_on", "A_sp", "A_off"],
    ["Te_Tpi", "Td_Tpi", "Tf_Tpi"],
    ["Tp1_Tpi", "Tp2_Tpi", "Tu_Ta_Tpi"],
    ["CV_T_pi", "CV_T_sys", "CV_Pulse_Amplitude"],
]

SYMBOL_MAPPING = {
    "T_pi": r"T_{pi}",
    "T_sys": r"T_{sys}",
    "T_dia": r"T_{dia}",
    "T_sp": r"T_{sp}",
    "IPR": r"PR_{inst}",
    "Pulse_Amplitude": r"A_{pulse}",
    "A_on": r"A_{on}",
    "A_sp": r"A_{sp}",
    "A_off": r"A_{off}",
    "Te_Tpi": r"Te_{Tpi}",
    "Td_Tpi": r"Td_{Tpi}",
    "Tf_Tpi": r"Tf_{Tpi}",
    "Tp1_Tpi": r"Tp1_{Tpi}",
    "Tp2_Tpi": r"Tp2_{Tpi}",
    "Tu_Ta_Tpi": r"TuTa_{Tpi}",
    "CV_T_pi": r"CV_{Tpi}",
    "CV_T_sys": r"CV_{Tsys}",
    "CV_Pulse_Amplitude": r"CV_{Apulse}",
}

LINE_COLORS = ["#009E73", "#D55E00", "#56B4E9"]

LABEL_COLORS = {
    -1: ("#E5E5E5", 0.35),   # ignore / buffer
    0: ("#C7E9C0", 0.45),    # normal
    1: ("#FDD0A2", 0.35),    # warning
    2: ("#F4A6A6", 0.35),    # detection / onset window
}

ONSET_LINE_COLOR = "#D62728"


@dataclass
class TrajectoryConfig:
    feature_groups: list[list[str]]
    gap_seconds: int = 10
    min_span_seconds: int = 60
    max_points_per_seg: int = 800
    max_segments: int = 1
    ewma_half_life_seconds: float = 6.0
    quantile_low: float = 0.01
    quantile_high: float = 0.99


def set_publication_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.30
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


def save_dual_formats(fig: plt.Figure, out_prefix: str | Path) -> tuple[Path, Path]:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    return png_path, pdf_path


def get_feature_label(name: str) -> str:
    sym = SYMBOL_MAPPING.get(name, name.replace("_", r"\_"))
    return fr"${sym}$"


def robust_minmax_01(s: pd.Series, q_low: float = 0.01, q_high: float = 0.99) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo = x.quantile(q_low)
    hi = x.quantile(q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        return pd.Series(0.5, index=s.index)
    y = (x - lo) / (hi - lo)
    return y.clip(0.0, 1.0)


def normalize_features_inplace(
    df: pd.DataFrame,
    target_features: Sequence[str],
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> None:
    for c in target_features:
        if c in df.columns:
            df[c] = robust_minmax_01(df[c], q_low=q_low, q_high=q_high)


def smooth_ewma(y: pd.Series, half_life_seconds: float = 6.0, sample_dt_seconds: float = 1.0) -> pd.Series:
    y = pd.to_numeric(y, errors="coerce")
    if y.isna().all():
        return y
    alpha = 1.0 - np.exp(-np.log(2) * sample_dt_seconds / max(1e-6, half_life_seconds))
    return y.ewm(alpha=alpha, adjust=False).mean()


def read_case_csv(path: str | Path, target_features: Sequence[str]) -> pd.DataFrame:
    path = Path(path)
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)

    required_candidates = [
        "CSN", "Group_ID", "Wave_Type", "Source_File",
        "Label", "Absolute_Time", "Actual_Stroke_Time",
    ]
    use_cols = [c for c in required_candidates + list(target_features) if c in cols]

    df = pd.read_csv(path, usecols=use_cols, low_memory=False)
    df["File_ID"] = path.name

    if "Absolute_Time" not in df.columns:
        raise ValueError(f"{path.name}: missing Absolute_Time")

    df["Absolute_Time"] = pd.to_datetime(df["Absolute_Time"], errors="coerce")
    if "Actual_Stroke_Time" in df.columns:
        df["Actual_Stroke_Time"] = pd.to_datetime(df["Actual_Stroke_Time"], errors="coerce")

    if "Source_File" not in df.columns:
        df["Source_File"] = "UnknownSource"
    if "Label" not in df.columns:
        raise ValueError(f"{path.name}: missing Label")

    df = df.dropna(subset=["Absolute_Time"]).sort_values("Absolute_Time").reset_index(drop=True)
    return df


def label_spans(df_sorted: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    spans: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    if len(df_sorted) < 2:
        return spans

    y = df_sorted["Label"].values
    start_idx = 0
    for i in range(1, len(df_sorted)):
        if y[i] != y[i - 1]:
            spans.append((
                df_sorted["Absolute_Time"].iloc[start_idx],
                df_sorted["Absolute_Time"].iloc[i - 1],
                int(y[i - 1]) if pd.notna(y[i - 1]) else -1,
            ))
            start_idx = i

    spans.append((
        df_sorted["Absolute_Time"].iloc[start_idx],
        df_sorted["Absolute_Time"].iloc[len(df_sorted) - 1],
        int(y[-1]) if pd.notna(y[-1]) else -1,
    ))
    return spans


def merge_short_spans(
    spans: list[tuple[pd.Timestamp, pd.Timestamp, int]],
    min_seconds: int = 60,
) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    if not spans:
        return spans

    merged = [spans[0]]
    for ts, te, lab in spans[1:]:
        pts, pte, plab = merged[-1]
        if lab == plab:
            merged[-1] = (pts, te, plab)
        else:
            merged.append((ts, te, lab))

    out = []
    for ts, te, lab in merged:
        dur = (te - ts).total_seconds()
        if dur < min_seconds:
            out.append((ts, te, -1))
        else:
            out.append((ts, te, lab))
    return out


def build_segments(df: pd.DataFrame, gap_seconds: int = 10) -> list[tuple[str, pd.DataFrame]]:
    segs: list[tuple[str, pd.DataFrame]] = []
    for src, part in df.groupby("Source_File", dropna=False):
        part = part.sort_values("Absolute_Time").copy()
        if len(part) < 5:
            continue

        dt = part["Absolute_Time"].diff().dt.total_seconds()
        contiguous = (dt.isna()) | (dt <= gap_seconds)
        seg_id = (~contiguous).cumsum()

        for _, seg in part.groupby(seg_id, sort=False):
            if len(seg) >= 5:
                segs.append((str(src), seg.copy()))

    segs.sort(key=lambda x: x[1]["Absolute_Time"].iloc[0])
    return segs


def pick_best_segments(
    segs: list[tuple[str, pd.DataFrame]],
    onset_time: pd.Timestamp | None,
    topk: int = 1,
) -> list[tuple[str, pd.DataFrame]]:
    scored = []
    for src, seg in segs:
        t0 = seg["Absolute_Time"].iloc[0]
        t1 = seg["Absolute_Time"].iloc[-1]
        mid = t0 + (t1 - t0) / 2

        if onset_time is None:
            dist_sec = 0.0
        else:
            dist_sec = abs((mid - onset_time).total_seconds())

        valid_ratio = np.mean(seg["Label"].values != -1) if "Label" in seg.columns else 0.0
        score = dist_sec - 600.0 * valid_ratio
        scored.append((score, src, seg))

    scored.sort(key=lambda x: x[0])
    return [(src, seg) for _, src, seg in scored[:max(1, topk)]]


def infer_case_id(df: pd.DataFrame) -> str:
    for c in ["CSN", "Group_ID", "File_ID"]:
        if c in df.columns and df[c].notna().any():
            return str(df[c].dropna().iloc[0])
    return "UnknownCase"


def infer_wave_type(df: pd.DataFrame) -> str:
    if "Wave_Type" in df.columns and df["Wave_Type"].notna().any():
        return str(df["Wave_Type"].dropna().iloc[0])
    return "Pleth"


def plot_case_feature_trajectories(
    df: pd.DataFrame,
    config: TrajectoryConfig,
) -> plt.Figure | None:
    set_publication_style()

    target_features = [c for group in config.feature_groups for c in group]
    present = [c for c in target_features if c in df.columns]
    if not present:
        return None

    df = df.copy()
    normalize_features_inplace(
        df,
        target_features=present,
        q_low=config.quantile_low,
        q_high=config.quantile_high,
    )

    case_id = infer_case_id(df)
    wave_type = infer_wave_type(df)

    segs = build_segments(df, gap_seconds=config.gap_seconds)
    if not segs:
        return None

    onset_time = None
    if "Actual_Stroke_Time" in df.columns:
        onset = df["Actual_Stroke_Time"].dropna()
        if not onset.empty:
            onset_time = pd.to_datetime(onset.iloc[0])

    segs = pick_best_segments(segs, onset_time=onset_time, topk=config.max_segments)

    all_times = [t for _, seg in segs for t in seg["Absolute_Time"]]
    if not all_times:
        return None
    t_min, t_max = min(all_times), max(all_times)

    raw_spans = label_spans(df)
    spans = merge_short_spans(raw_spans, min_seconds=config.min_span_seconds)

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 9), sharex=True, constrained_layout=True)
    axes = axes.flatten()

    date_str = t_min.strftime("%Y-%m-%d")
    fig.suptitle(
        f"Case: {case_id} | Waveform: {wave_type} | Date: {date_str}",
        fontsize=12,
        fontweight="bold",
    )

    for i, group in enumerate(config.feature_groups):
        if i >= len(axes):
            break
        ax = axes[i]

        for ts, te, lab in spans:
            if te < t_min or ts > t_max:
                continue
            plot_ts = max(ts, t_min)
            plot_te = min(te, t_max)
            color, alpha = LABEL_COLORS.get(lab, LABEL_COLORS[-1])
            ax.axvspan(plot_ts, plot_te, color=color, alpha=alpha, lw=0, zorder=0)

        if onset_time is not None:
            ax.axvline(onset_time, color=ONSET_LINE_COLOR, lw=1.5, linestyle="--", alpha=0.9, zorder=3)

        plotted_any = False
        for f_idx, feat in enumerate(group):
            if feat not in df.columns:
                continue

            color = LINE_COLORS[f_idx % len(LINE_COLORS)]

            for _, seg in segs:
                if feat not in seg.columns:
                    continue

                step = max(1, len(seg) // config.max_points_per_seg)
                segp = seg.iloc[::step]

                x_data = segp["Absolute_Time"]
                y_data = pd.to_numeric(segp[feat], errors="coerce")

                ax.plot(x_data, y_data, color=color, lw=0.5, alpha=0.3)

                ys = smooth_ewma(
                    y_data,
                    half_life_seconds=config.ewma_half_life_seconds,
                    sample_dt_seconds=1.0,
                )
                ax.plot(x_data, ys, color=color, lw=1.2, alpha=0.95)
                plotted_any = True

            ax.plot([], [], color=color, lw=1.2, label=get_feature_label(feat))

        ax.set_xlim(t_min, t_max)
        ax.set_ylim(-0.05, 1.05)

        if i % 3 == 0:
            ax.set_ylabel("Normalized value", fontsize=10)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))

        if i >= 3:
            ax.set_xlabel(f"Time on {date_str}", fontsize=10)
            ax.tick_params(axis="x", labelsize=9, rotation=0)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        if plotted_any:
            bg_handles = [
                plt.Rectangle((0, 0), 1, 1, fc=LABEL_COLORS[0][0], alpha=LABEL_COLORS[0][1]),
                plt.Rectangle((0, 0), 1, 1, fc=LABEL_COLORS[1][0], alpha=LABEL_COLORS[1][1]),
                plt.Rectangle((0, 0), 1, 1, fc=LABEL_COLORS[-1][0], alpha=LABEL_COLORS[-1][1]),
                plt.Line2D([0], [0], color=ONSET_LINE_COLOR, lw=1.5, linestyle="--"),
            ]
            bg_labels = ["Normal", "Warning", "Buffer/Ignore", "Onset"]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles + bg_handles,
                labels + bg_labels,
                loc="upper right",
                ncol=2,
                fontsize=7,
                frameon=False,
                columnspacing=1.0,
            )

    return fig


def load_feature_groups(path: str | Path | None) -> list[list[str]]:
    if path is None:
        return DEFAULT_FEATURE_GROUPS
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("feature-groups JSON must be a list of feature-group lists")
    return [[str(x) for x in group] for group in data]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot publication-quality temporal feature trajectories for one case CSV.")
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--feature-groups-json", type=str, default=None)
    parser.add_argument("--gap-seconds", type=int, default=10)
    parser.add_argument("--min-span-seconds", type=int, default=60)
    parser.add_argument("--max-points-per-seg", type=int, default=800)
    parser.add_argument("--max-segments", type=int, default=1)
    parser.add_argument("--ewma-half-life-seconds", type=float, default=6.0)
    args = parser.parse_args()

    feature_groups = load_feature_groups(args.feature_groups_json)
    cfg = TrajectoryConfig(
        feature_groups=feature_groups,
        gap_seconds=args.gap_seconds,
        min_span_seconds=args.min_span_seconds,
        max_points_per_seg=args.max_points_per_seg,
        max_segments=args.max_segments,
        ewma_half_life_seconds=args.ewma_half_life_seconds,
    )

    target_features = [c for g in feature_groups for c in g]
    df = read_case_csv(args.input_csv, target_features=target_features)
    fig = plot_case_feature_trajectories(df, cfg)

    if fig is None:
        raise RuntimeError("No valid trajectory figure could be built from the input CSV.")

    case_name = Path(args.input_csv).stem
    out_prefix = Path(args.output_dir) / f"feature_trajectories_{case_name}"
    png_path, pdf_path = save_dual_formats(fig, out_prefix)
    plt.close(fig)

    print(f"[plot_feature_trajectories] saved png -> {png_path}")
    print(f"[plot_feature_trajectories] saved pdf -> {pdf_path}")


if __name__ == "__main__":
    main()

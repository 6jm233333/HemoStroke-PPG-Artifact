from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.subgroup_statistics import run_multi_group_test, summarize_subgroup_f1

PANEL_ALIASES = ["panel", "domain", "category", "subgroup_type", "panel_name"]
SUBGROUP_ALIASES = ["subgroup", "group", "group_name", "level", "label"]
FOLD_ALIASES = ["fold", "cv_fold", "split", "fold_id"]
F1_ALIASES = ["f1", "f1_score", "macro_f1", "binary_f1"]


DEFAULT_PANEL_ORDER = ["Clinical Risk", "Race", "Age", "Gender"]

DEFAULT_SUBGROUP_ORDER = {
    "Clinical Risk": ["Low Risk", "Medium Risk", "High Risk"],
    "Race": ["White", "Asian", "Black"],
    "Age": ["Elderly (≥65)", "Non-Elderly"],
    "Gender": ["Male", "Female"],
}

# close to the screenshot
DEFAULT_PANEL_COLORS = {
    "Clinical Risk": "#E878A0",   # pink
    "Race": "#ECEA7A",            # pale yellow
    "Age": "#5CA4F2",             # blue
    "Gender": "#53D8C0",          # teal
}

FIG_BG = "#F2F2F2"
AX_BG = "#F2F2F2"
GRID_COLOR = "#CFCFCF"
AVG_LINE_COLOR = "#7A7A7A"
EDGE_COLOR = "#2D2D2D"

def _find_first_existing(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    colset = {str(c).strip(): c for c in columns}
    for a in aliases:
        if a in colset:
            return colset[a]
    return None


def _collect_csv_files(path_like: str) -> List[Path]:
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.csv"))


def _normalize_panel_name(x: str) -> str:
    s = str(x).strip().lower()
    mapping = {
        "clinical risk": "Clinical Risk",
        "risk": "Clinical Risk",
        "comorbidity": "Clinical Risk",
        "race": "Race",
        "ethnicity": "Race",
        "age": "Age",
        "gender": "Gender",
        "sex": "Gender",
    }
    return mapping.get(s, str(x).strip())


def _normalize_subgroup_name(panel: str, subgroup: str) -> str:
    s = str(subgroup).strip()
    s_low = s.lower()

    if panel == "Clinical Risk":
        lut = {
            "low": "Low Risk",
            "low risk": "Low Risk",
            "medium": "Medium Risk",
            "medium risk": "Medium Risk",
            "mid": "Medium Risk",
            "mid risk": "Medium Risk",
            "high": "High Risk",
            "high risk": "High Risk",
        }
        return lut.get(s_low, s)

    if panel == "Race":
        lut = {
            "white": "White",
            "asian": "Asian",
            "black": "Black",
            "black or african american": "Black",
        }
        return lut.get(s_low, s)

    if panel == "Age":
        if "elder" in s_low or "≥65" in s or ">=65" in s:
            return "Elderly (≥65)"
        if "non" in s_low or "<65" in s_low or "under" in s_low:
            return "Non-Elderly"
        return s

    if panel == "Gender":
        lut = {
            "m": "Male",
            "male": "Male",
            "f": "Female",
            "female": "Female",
            "man": "Male",
            "woman": "Female",
        }
        return lut.get(s_low, s)

    return s


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": FIG_BG,
            "axes.facecolor": AX_BG,
            "savefig.facecolor": FIG_BG,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.linewidth": 1.2,
            "hatch.linewidth": 1.2,
        }
    )


def _load_fold_level_f1(path_like: str) -> pd.DataFrame:
    """
    Reads one CSV or a directory of CSVs.

    Required semantic columns:
    - subgroup
    - f1

    Optional:
    - panel
    - fold

    If panel is missing, infer from filename.
    If fold is missing, infer from filename; otherwise default 0.
    """
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

        panel_col = _find_first_existing(df.columns, PANEL_ALIASES)
        subgroup_col = _find_first_existing(df.columns, SUBGROUP_ALIASES)
        fold_col = _find_first_existing(df.columns, FOLD_ALIASES)
        f1_col = _find_first_existing(df.columns, F1_ALIASES)

        if subgroup_col is None or f1_col is None:
            continue

        keep = [subgroup_col, f1_col]
        rename = {subgroup_col: "subgroup", f1_col: "f1"}

        if panel_col is not None:
            keep.append(panel_col)
            rename[panel_col] = "panel"

        if fold_col is not None:
            keep.append(fold_col)
            rename[fold_col] = "fold"

        part = df[keep].rename(columns=rename).copy()

        # infer panel from filename if absent
        if "panel" not in part.columns:
            stem = fp.stem.lower()
            inferred_panel = None
            for key in ["risk", "race", "age", "gender", "sex"]:
                if key in stem:
                    inferred_panel = _normalize_panel_name(key)
                    break
            if inferred_panel is None:
                raise ValueError(
                    f"Could not infer panel for '{fp.name}'. "
                    "Please include a panel column or encode risk/race/age/gender in filename."
                )
            part["panel"] = inferred_panel

        # infer fold from filename if absent
        if "fold" not in part.columns:
            m = re.search(r"(?:fold|cv|split)[_\-]?(\d+)", fp.stem.lower())
            part["fold"] = int(m.group(1)) if m else 0

        part["panel"] = part["panel"].map(_normalize_panel_name)
        part["subgroup"] = [
            _normalize_subgroup_name(panel, subgroup)
            for panel, subgroup in zip(part["panel"], part["subgroup"])
        ]
        part["f1"] = pd.to_numeric(part["f1"], errors="coerce")
        part["fold"] = pd.to_numeric(part["fold"], errors="coerce")

        part = part.dropna(subset=["f1", "fold"])
        part["fold"] = part["fold"].astype(int)

        frames.append(part)

    if not frames:
        raise ValueError(f"No usable fold-level subgroup F1 CSVs found under: {path_like}")

    out = pd.concat(frames, ignore_index=True)

    # keep only panels in desired order if possible
    out = out[out["panel"].isin(DEFAULT_PANEL_ORDER)].copy()
    if out.empty:
        raise ValueError("After normalization, no valid panels remained.")

    return out


def analyze_all_panels(
    df: pd.DataFrame,
    *,
    panel_order: Sequence[str],
    panel_col: str,
    subgroup_col: str,
    fold_col: str,
    f1_col: str,
    correction: str,
    equal_var: bool,
) -> Dict[str, object]:
    summary_df = summarize_subgroup_f1(
        df,
        panel_col=panel_col,
        subgroup_col=subgroup_col,
        fold_col=fold_col,
        f1_col=f1_col,
        panel_order=panel_order,
        subgroup_order_map=DEFAULT_SUBGROUP_ORDER,
    )

    pairwise_frames: List[pd.DataFrame] = []
    global_frames: List[pd.DataFrame] = []
    report_lines = ["Exploratory subgroup F1 summary", ""]

    for panel_name in panel_order:
        panel_df = df[df[panel_col] == panel_name].copy()
        if panel_df.empty:
            continue

        report_lines.append(f"[{panel_name}]")
        panel_summary = summary_df[summary_df["panel"] == panel_name]
        for _, row in panel_summary.iterrows():
            report_lines.append(
                f"- {row['subgroup']}: F1={row['f1_mean']:.4f} +/- {row['f1_std']:.4f} "
                f"(n_folds={int(row['n_folds'])})"
            )

        try:
            stats_bundle = run_multi_group_test(
                panel_df,
                group_col=subgroup_col,
                value_col=f1_col,
                correction=correction,
                equal_var=equal_var,
            )
        except ValueError as exc:
            report_lines.append(f"- Statistical test skipped: {exc}")
            report_lines.append("")
            continue

        panel_pairwise = stats_bundle["pairwise"].copy()
        panel_pairwise.insert(0, "panel", panel_name)
        pairwise_frames.append(panel_pairwise)

        panel_global = stats_bundle["global"].copy()
        panel_global.insert(0, "panel", panel_name)
        global_frames.append(panel_global)
        report_lines.append("")

    pairwise_df = (
        pd.concat(pairwise_frames, ignore_index=True)
        if pairwise_frames
        else pd.DataFrame(columns=["panel", "group_1", "group_2", "test", "statistic", "p_raw", "p_adj"])
    )
    global_df = (
        pd.concat(global_frames, ignore_index=True)
        if global_frames
        else pd.DataFrame(columns=["panel", "test", "statistic", "p_raw", "p_adj", "n_groups"])
    )

    return {
        "all_summary": summary_df,
        "all_pairwise": pairwise_df,
        "all_global": global_df,
        "merged_report_text": "\n".join(report_lines).rstrip() + "\n",
    }


def _get_panel_ylim(
    panel_summary: pd.DataFrame,
    annotations: List[Dict[str, float]],
    base_min: float = 0.82,
) -> tuple[float, float]:
    y = panel_summary["f1_mean"].to_numpy(dtype=float)
    err = panel_summary["f1_std"].to_numpy(dtype=float)
    top = float(np.max(y + err)) if len(y) > 0 else 1.0

    if annotations:
        top = max(top, max(a["text_y"] for a in annotations) + 0.010)

    ymin = base_min
    ymax = max(1.005, top + 0.006)
    ymax = min(ymax, 1.03)
    return ymin, ymax


def _format_value_label(v: float) -> str:
    return f"{v:.3f}"


def _plot_single_panel(
    ax,
    panel_name: str,
    panel_summary: pd.DataFrame,
    panel_pairs: pd.DataFrame,
    panel_letter: str,
) -> None:
    subgroup_order = DEFAULT_SUBGROUP_ORDER.get(panel_name, panel_summary["subgroup"].tolist())
    panel_summary = panel_summary.copy()
    panel_summary["subgroup"] = pd.Categorical(
        panel_summary["subgroup"],
        categories=subgroup_order,
        ordered=True,
    )
    panel_summary = panel_summary.sort_values("subgroup").reset_index(drop=True)

    x = np.arange(len(panel_summary), dtype=float)
    y = panel_summary["f1_mean"].to_numpy(dtype=float)
    err = panel_summary["f1_std"].to_numpy(dtype=float)

    bar_color = DEFAULT_PANEL_COLORS.get(panel_name, "#5B8FF9")
    bars = ax.bar(
        x,
        y,
        width=0.50,
        color=bar_color,
        edgecolor=EDGE_COLOR,
        linewidth=1.2,
        hatch="//",
        yerr=err,
        error_kw={
            "elinewidth": 1.1,
            "ecolor": "black",
            "capsize": 4,
            "capthick": 1.1,
        },
        zorder=3,
    )

    # group average dashed line
    group_avg = float(np.mean(y)) if len(y) else 0.95
    ax.axhline(
        group_avg,
        linestyle="--",
        linewidth=1.2,
        color=AVG_LINE_COLOR,
        alpha=0.95,
        zorder=2,
    )

    # Descriptive subgroup analysis only.
    # The paper does not use significance markers in the subgroup figure.
    annotations = []

    # value labels
    for rect, yi in zip(bars, y):
        cx = rect.get_x() + rect.get_width() / 2.0
        ax.text(
            cx,
            yi + 0.004,
            _format_value_label(yi),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black",
            zorder=8,
        )

    ymin, ymax = _get_panel_ylim(panel_summary, annotations, base_min=0.82)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.55, len(panel_summary) - 0.45)

    ax.set_title(panel_name, pad=8)

    ax.set_xticks(x)
    ax.set_xticklabels(subgroup_order, fontweight="bold")

    # only left column has ylabel in the screenshot
    ax.set_ylabel("F1-Score" if panel_letter in ["a", "c"] else "")

    ax.grid(axis="y", linestyle="--", linewidth=0.7, color=GRID_COLOR, alpha=0.7, zorder=1)
    ax.grid(axis="x", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", width=1.0)

    # panel letters: a / b / c / d
    ax.text(
        -0.12,
        1.02,
        panel_letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="black",
    )


def plot_subgroup_figure(
    summary_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
) -> None:
    _set_plot_style()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.0, 5.8),
        facecolor=FIG_BG,
    )
    axes = axes.flatten()
    panel_letters = ["a", "b", "c", "d"]

    for i, panel_name in enumerate(DEFAULT_PANEL_ORDER):
        ax = axes[i]
        panel_summary = summary_df[summary_df["panel"] == panel_name].copy()
        panel_pairs = pairwise_df[pairwise_df["panel"] == panel_name].copy()

        if panel_summary.empty:
            ax.axis("off")
            continue

        _plot_single_panel(
            ax=ax,
            panel_name=panel_name,
            panel_summary=panel_summary,
            panel_pairs=panel_pairs,
            panel_letter=panel_letters[i],
        )

    legend_handles = [
        Patch(
            facecolor="#C7C7C7",
            edgecolor=EDGE_COLOR,
            hatch="//",
            linewidth=1.0,
            label="Subgroup F1",
        ),
        Line2D(
            [0],
            [0],
            color=AVG_LINE_COLOR,
            linewidth=1.2,
            linestyle="--",
            label="Group Average",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=2,
        frameon=False,
        fontsize=10,
        handlelength=1.8,
        columnspacing=1.6,
    )

    plt.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.10,
        top=0.90,
        wspace=0.10,
        hspace=0.20,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=FIG_BG)
    plt.savefig(output_pdf, dpi=300, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce exploratory subgroup F1 figure.")
    parser.add_argument(
        "--input",
        required=True,
        help="Fold-level subgroup F1 CSV or directory of CSVs.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/figures/subgroup_f1",
        help="Output directory.",
    )
    args = parser.parse_args()

    raw_df = _load_fold_level_f1(args.input)
    stats_bundle = analyze_all_panels(
        raw_df,
        panel_order=DEFAULT_PANEL_ORDER,
        panel_col="panel",
        subgroup_col="subgroup",
        fold_col="fold",
        f1_col="f1",
        correction="bonferroni",
        equal_var=False, 
    )

    summary_df = stats_bundle["all_summary"]
    pairwise_df = stats_bundle["all_pairwise"]
    global_df = stats_bundle["all_global"]
    report_text = stats_bundle["merged_report_text"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "subgroup_summary.csv"
    pvalue_csv = out_dir / "subgroup_pairwise.csv"
    global_csv = out_dir / "subgroup_global_tests.csv"
    report_txt = out_dir / "subgroup_statistical_report.txt"
    figure_png = out_dir / "figure_subgroup_f1.png"
    figure_pdf = out_dir / "figure_subgroup_f1.pdf"

    summary_df.to_csv(summary_csv, index=False)
    pairwise_df.to_csv(pvalue_csv, index=False)
    global_df.to_csv(global_csv, index=False)

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)

    plot_subgroup_figure(
        summary_df=summary_df,
        pairwise_df=pairwise_df,
        output_png=figure_png,
        output_pdf=figure_pdf,
    )

    print(f"[OK] Figure saved to: {figure_png}")
    print(f"[OK] Figure saved to: {figure_pdf}")
    print(f"[OK] Summary saved to: {summary_csv}")
    print(f"[OK] Pairwise p-values saved to: {pvalue_csv}")
    print(f"[OK] Global tests saved to: {global_csv}")
    print(f"[OK] Text report saved to: {report_txt}")


if __name__ == "__main__":
    main()

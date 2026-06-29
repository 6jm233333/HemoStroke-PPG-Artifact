from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cohort note/time distributions from an anchored note table.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", default="outputs/figures/cohort_distributions")
    parser.add_argument("--time-col", default="Extracted_Timestamp")
    parser.add_argument("--dataset-name", default="Cohort")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.time_col not in df.columns:
        raise KeyError(f"Missing time column: {args.time_col}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    times = pd.to_datetime(df[args.time_col], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    times.dt.hour.value_counts().sort_index().reindex(range(24), fill_value=0).plot(kind="bar", ax=ax)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Count")
    ax.set_title(f"{args.dataset_name}: documented onset anchors by hour")
    fig.tight_layout()
    fig.savefig(out_dir / "onset_hour_distribution.png", dpi=300)
    fig.savefig(out_dir / "onset_hour_distribution.pdf")
    plt.close(fig)

    summary = {
        "n_rows": int(len(df)),
        "n_valid_times": int(len(times)),
        "min_time": str(times.min()) if len(times) else "",
        "max_time": str(times.max()) if len(times) else "",
    }
    pd.DataFrame([summary]).to_csv(out_dir / "onset_time_summary.csv", index=False)


if __name__ == "__main__":
    main()

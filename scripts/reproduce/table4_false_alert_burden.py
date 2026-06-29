from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.operating_point import (  # noqa: E402
    apply_binary_threshold,
    max_consecutive_positives,
    validate_threshold,
)


def load_threshold(config_path: Path) -> float:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return validate_threshold(float(cfg["evaluation"]["threshold"]))


def summarize_false_alerts(
    predictions: pd.DataFrame,
    *,
    identifier_col: str,
    score_col: str,
    order_col: str,
    threshold: float,
    min_consecutive_windows: int,
    n_packaged: int | None = None,
    cohort: str | None = None,
    horizon_minutes: int | None = None,
    stroke_tpr: float | None = None,
    threshold_source: str = "config",
) -> pd.DataFrame:
    if min_consecutive_windows < 1:
        raise ValueError("min_consecutive_windows must be at least 1.")
    required = [identifier_col, score_col, order_col]
    missing = [col for col in required if col not in predictions.columns]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    work = predictions.copy()
    packaged_count = int(len(work) if n_packaged is None else n_packaged)
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work[work[identifier_col].notna() & np.isfinite(work[score_col])].copy()
    if work.empty:
        raise ValueError("No NaN-free inference windows remain.")

    work = work.sort_values([identifier_col, order_col], kind="mergesort")
    work["_positive"] = apply_binary_threshold(work[score_col].to_numpy(), threshold)

    id_rows: list[dict[str, Any]] = []
    for identifier, group in work.groupby(identifier_col, sort=False):
        longest_run = max_consecutive_positives(group["_positive"].tolist())
        id_rows.append(
            {
                "identifier": identifier,
                "max_consecutive_positive_windows": longest_run,
                "id_positive": int(longest_run >= min_consecutive_windows),
            }
        )
    id_df = pd.DataFrame(id_rows)

    row = {
        "horizon_minutes": horizon_minutes,
        "cohort": cohort,
        "n_packaged": packaged_count,
        "n_windows": int(len(work)),
        "n_identifiers": int(len(id_df)),
        "fpr": float(work["_positive"].mean()),
        "stroke_tpr": stroke_tpr,
        "id_positive_fraction": float(id_df["id_positive"].mean()),
        "threshold": validate_threshold(threshold),
        "threshold_source": threshold_source,
        "id_positive_rule": f">={min_consecutive_windows} consecutive positive windows",
    }
    return pd.DataFrame([row])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one Table IV false-alert row from high-risk non-stroke control windows."
    )
    parser.add_argument("--predictions", required=True, help="CSV with one row per packaged control window.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--identifier-col", default="file_id", help="File-level packaging-group column.")
    parser.add_argument("--score-col", default="y_prob")
    parser.add_argument("--order-col", required=True, help="Within-identifier window ordering column.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional explicit override; default reads config.")
    parser.add_argument("--min-consecutive-windows", type=int, default=5)
    parser.add_argument("--n-packaged", type=int, default=None, help="Optional pre-NaN-filter window count.")
    parser.add_argument("--cohort", default=None)
    parser.add_argument("--horizon-minutes", type=int, default=None)
    parser.add_argument("--stroke-tpr", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threshold is None:
        threshold = load_threshold(Path(args.config))
        threshold_source = str(args.config)
    else:
        threshold = validate_threshold(args.threshold)
        threshold_source = "explicit --threshold override"

    predictions = pd.read_csv(args.predictions, low_memory=False)
    summary = summarize_false_alerts(
        predictions,
        identifier_col=args.identifier_col,
        score_col=args.score_col,
        order_col=args.order_col,
        threshold=threshold,
        min_consecutive_windows=args.min_consecutive_windows,
        n_packaged=args.n_packaged,
        cohort=args.cohort,
        horizon_minutes=args.horizon_minutes,
        stroke_tpr=args.stroke_tpr,
        threshold_source=threshold_source,
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

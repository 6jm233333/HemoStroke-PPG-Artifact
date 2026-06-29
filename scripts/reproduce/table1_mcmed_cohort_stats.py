from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    counts = df[col].fillna("Unknown").astype(str).value_counts(dropna=False)
    total = max(len(df), 1)
    return pd.DataFrame(
        {
            "characteristic": col,
            "subgroup": counts.index,
            "n": counts.values,
            "percent": np.round(counts.values / total * 100, 2),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact MC-MED cohort table from a local metadata CSV.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", default="outputs/tables/table1_mcmed_cohort_stats.csv")
    parser.add_argument("--age-col", default="Age")
    parser.add_argument("--categorical-cols", nargs="*", default=["Gender", "Race"])
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    rows = []

    if args.age_col in df.columns:
        age = pd.to_numeric(df[args.age_col], errors="coerce").dropna()
        rows.append(
            pd.DataFrame(
                [
                    {
                        "characteristic": args.age_col,
                        "subgroup": "Mean (SD)",
                        "n": len(age),
                        "percent": "",
                        "value": f"{age.mean():.2f} ({age.std(ddof=1):.2f})",
                    },
                    {
                        "characteristic": args.age_col,
                        "subgroup": "Median [IQR]",
                        "n": len(age),
                        "percent": "",
                        "value": f"{age.median():.2f} [{age.quantile(0.25):.2f}-{age.quantile(0.75):.2f}]",
                    },
                ]
            )
        )

    for col in args.categorical_cols:
        if col in df.columns:
            part = summarize_categorical(df, col)
            part["value"] = ""
            rows.append(part)

    if not rows:
        raise ValueError("No requested columns were found in the input CSV.")

    out = pd.concat(rows, ignore_index=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"[table1_mcmed_cohort_stats] wrote {output_csv}")


if __name__ == "__main__":
    main()

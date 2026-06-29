#!/usr/bin/env python
"""
Summarize label coverage for per-file feature tables.

This script is intentionally placed under scripts/qc/ because it is a
quality-control utility rather than a core data-construction step.

What it does
------------
1. Scans a directory of CSV feature tables.
2. Reads the label column (default: "Label") plus a few optional metadata columns.
3. Reports file-level label coverage:
   - row count
   - per-label counts and ratios
   - observed label pattern
   - whether a file contains usable binary labels (0 and 1)
   - whether a file still contains ignore / detection labels (-1 / 2)
4. Exports dataset-level summaries for reproducibility and manuscript reporting.

Why this is useful
------------------
- Early-stage tables in this project may contain labels {-1, 0, 1, 2}.
- Final relabeled training tables usually contain {-1, 0, 1}.
- Total sample counts and class imbalance details support transparent reporting.
- This script gives a transparent QC trace without modifying data.

Example
-------
python scripts/qc/summarize_label_coverage.py \
    --input-dir data/processed/mimic/features_labeled \
    --output-dir outputs/qc/mimic_label_coverage

python scripts/qc/summarize_label_coverage.py \
    --input-dir data/processed/mcmed/features_labeled \
    --output-dir outputs/qc/mcmed_label_coverage \
    --expected-labels -1 0 1 2
"""

from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import pandas as pd

DEFAULT_IGNORE_PREFIXES = (
    "summary",
    "statistics",
    "boolean_summary",
    "list_with",
    "processing_summary",
    "preprocess_log",
)

DEFAULT_IGNORE_NAMES = (
    "processing_summary.csv",
    "preprocess_log.csv",
    "summary_all.csv",
)

DEFAULT_METADATA_COLS = (
    "Group_ID",
    "Wave_Type",
    "File_ID",
    "Is_Stroke_Subject",
)

LabelValue = Union[int, float, str]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize label coverage for a directory of CSV feature tables."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing per-file CSV feature tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for QC summaries. Defaults to <input-dir>/qc_label_coverage.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="Label",
        help="Name of the label column. Default: Label",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.csv",
        help='Glob pattern for files. Default: "*.csv"',
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Disable recursive directory traversal.",
    )
    parser.add_argument(
        "--expected-labels",
        nargs="+",
        default=["-1", "0", "1", "2"],
        help="Expected label values to report explicitly. Default: -1 0 1 2",
    )
    parser.add_argument(
        "--binary-negative-label",
        type=str,
        default="0",
        help="Negative class label for usable binary rows. Default: 0",
    )
    parser.add_argument(
        "--binary-positive-label",
        type=str,
        default="1",
        help="Positive class label for usable binary rows. Default: 1",
    )
    parser.add_argument(
        "--ignore-prefixes",
        nargs="*",
        default=list(DEFAULT_IGNORE_PREFIXES),
        help="Filename prefixes to ignore (case-insensitive).",
    )
    parser.add_argument(
        "--ignore-names",
        nargs="*",
        default=list(DEFAULT_IGNORE_NAMES),
        help="Exact filenames to ignore (case-insensitive).",
    )
    parser.add_argument(
        "--metadata-cols",
        nargs="*",
        default=list(DEFAULT_METADATA_COLS),
        help="Optional metadata columns to keep in file-level summary.",
    )
    parser.add_argument(
        "--strict-label-column",
        action="store_true",
        help="Treat missing label columns as hard errors in the summary.",
    )
    return parser.parse_args()


def normalize_expected_labels(values: Sequence[str]) -> List[LabelValue]:
    return [canonicalize_scalar_label(v) for v in values]


def canonicalize_scalar_label(value: object) -> Optional[LabelValue]:
    """
    Convert label-like values into a stable canonical form.

    Rules:
    - empty / NaN -> None
    - integral numeric values -> int
    - non-integral numeric values -> float
    - otherwise -> stripped string
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        num = float(s)
        if num.is_integer():
            return int(num)
        return float(num)
    except ValueError:
        return s

def canonicalize_label_series(series: pd.Series) -> pd.Series:
    return series.map(canonicalize_scalar_label)

def is_ignored_file(
    path: Path,
    ignore_prefixes: Sequence[str],
    ignore_names: Sequence[str],
) -> bool:
    name_lower = path.name.lower()
    if name_lower in {n.lower() for n in ignore_names}:
        return True
    return any(name_lower.startswith(prefix.lower()) for prefix in ignore_prefixes)

def discover_csv_files(
    input_dir: Path,
    glob_pattern: str,
    recursive: bool,
    ignore_prefixes: Sequence[str],
    ignore_names: Sequence[str],
) -> List[Path]:
    iterator = input_dir.rglob(glob_pattern) if recursive else input_dir.glob(glob_pattern)
    files = [
        p for p in iterator
        if p.is_file() and not is_ignored_file(p, ignore_prefixes, ignore_names)
    ]
    return sorted(files)

def first_non_null_value(series: pd.Series) -> Optional[str]:
    non_null = series.dropna()
    if non_null.empty:
        return None

    for value in non_null.tolist():
        s = str(value).strip()
        if s != "" and s.lower() != "nan":
            return s
    return None

def safe_ratio(num: int, den: int) -> Optional[float]:
    if den == 0:
        return None
    return float(num) / float(den)

def order_labels_for_display(
    observed: Iterable[LabelValue],
    expected: Sequence[LabelValue],
) -> List[LabelValue]:
    expected_set = set(expected)
    observed_list = list(observed)
    ordered = [x for x in expected if x in observed_list]
    extras = [x for x in observed_list if x not in expected_set]

    def _sort_key(v: LabelValue) -> Tuple[int, str]:
        if isinstance(v, (int, float)):
            return (0, str(v).zfill(8))
        return (1, str(v))

    extras = sorted(extras, key=_sort_key)
    return ordered + extras

def summarize_one_file(
    csv_path: Path,
    label_col: str,
    expected_labels: Sequence[LabelValue],
    binary_negative_label: LabelValue,
    binary_positive_label: LabelValue,
    metadata_cols: Sequence[str],
    strict_label_column: bool,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "file_name": csv_path.name,
        "file_path": str(csv_path),
        "status": "ok",
        "error": "",
    }

    try:
        header = pd.read_csv(csv_path, nrows=0)
        available_cols = list(header.columns)
    except Exception as exc:
        summary["status"] = "error"
        summary["error"] = f"HeaderReadError: {type(exc).__name__}: {exc}"
        return summary

    if label_col not in available_cols:
        summary["status"] = "error" if strict_label_column else "skip"
        summary["error"] = f"MissingLabelColumn: '{label_col}' not found."
        return summary

    keep_cols = [label_col] + [c for c in metadata_cols if c in available_cols]

    try:
        df = pd.read_csv(csv_path, usecols=keep_cols, low_memory=False)
    except Exception as exc:
        summary["status"] = "error"
        summary["error"] = f"DataReadError: {type(exc).__name__}: {exc}"
        return summary

    summary["n_rows"] = int(len(df))
    summary["n_columns_read"] = int(len(keep_cols))

    label_series = canonicalize_label_series(df[label_col])
    valid_labels = label_series.dropna()

    counts = Counter(valid_labels.tolist())
    observed_labels = order_labels_for_display(counts.keys(), expected_labels)

    summary["n_valid_labels"] = int(valid_labels.shape[0])
    summary["n_missing_or_invalid_labels"] = int(label_series.isna().sum())
    summary["n_unique_labels"] = int(len(counts))
    summary["observed_labels"] = "|".join(str(x) for x in observed_labels) if observed_labels else ""

    # Explicit counts / ratios for expected labels
    for label in expected_labels:
        count = int(counts.get(label, 0))
        summary[f"count_label_{label}"] = count
        summary[f"ratio_label_{label}"] = safe_ratio(count, int(len(df)))

    # Unexpected labels
    unexpected_labels = [x for x in observed_labels if x not in set(expected_labels)]
    summary["has_unexpected_labels"] = bool(unexpected_labels)
    summary["unexpected_labels"] = "|".join(str(x) for x in unexpected_labels) if unexpected_labels else ""

    # Presence booleans for expected labels
    for label in expected_labels:
        summary[f"has_label_{label}"] = bool(counts.get(label, 0) > 0)

    # Binary usability metrics
    neg_count = int(counts.get(binary_negative_label, 0))
    pos_count = int(counts.get(binary_positive_label, 0))
    usable_binary_rows = neg_count + pos_count

    summary["binary_negative_label"] = str(binary_negative_label)
    summary["binary_positive_label"] = str(binary_positive_label)
    summary["binary_rows"] = int(usable_binary_rows)
    summary["binary_positive_ratio"] = safe_ratio(pos_count, usable_binary_rows)
    summary["has_binary_negative"] = neg_count > 0
    summary["has_binary_positive"] = pos_count > 0
    summary["has_binary_01_pair"] = (neg_count > 0) and (pos_count > 0)

    # Common QC flags for this project
    summary["has_ignore_label_neg1"] = bool(counts.get(-1, 0) > 0)
    summary["has_detection_label_2"] = bool(counts.get(2, 0) > 0)
    summary["has_triplet_neg1_0_1"] = bool(counts.get(-1, 0) > 0 and counts.get(0, 0) > 0 and counts.get(1, 0) > 0)
    summary["all_rows_binary_0_1_only"] = bool(
        len(valid_labels) > 0 and set(valid_labels.tolist()).issubset({binary_negative_label, binary_positive_label})
    )

    # Store a compact label pattern for grouping
    pattern_labels = order_labels_for_display(counts.keys(), expected_labels)
    summary["label_pattern"] = "|".join(str(x) for x in pattern_labels) if pattern_labels else "EMPTY"

    # Optional metadata snapshot
    for col in metadata_cols:
        if col in df.columns:
            summary[col] = first_non_null_value(df[col])

    return summary


def build_overall_summary(
    file_df: pd.DataFrame,
    expected_labels: Sequence[LabelValue],
    binary_negative_label: LabelValue,
    binary_positive_label: LabelValue,
) -> pd.DataFrame:
    ok_df = file_df[file_df["status"] == "ok"].copy()

    summary: Dict[str, object] = {
        "n_files_discovered": int(len(file_df)),
        "n_files_ok": int((file_df["status"] == "ok").sum()),
        "n_files_skipped": int((file_df["status"] == "skip").sum()),
        "n_files_error": int((file_df["status"] == "error").sum()),
        "total_rows": int(ok_df["n_rows"].sum()) if not ok_df.empty else 0,
        "total_valid_labels": int(ok_df["n_valid_labels"].sum()) if not ok_df.empty else 0,
        "total_missing_or_invalid_labels": int(ok_df["n_missing_or_invalid_labels"].sum()) if not ok_df.empty else 0,
        "files_with_binary_pair": int(ok_df["has_binary_01_pair"].sum()) if "has_binary_01_pair" in ok_df.columns else 0,
        "files_with_triplet_neg1_0_1": int(ok_df["has_triplet_neg1_0_1"].sum()) if "has_triplet_neg1_0_1" in ok_df.columns else 0,
        "files_with_detection_label_2": int(ok_df["has_detection_label_2"].sum()) if "has_detection_label_2" in ok_df.columns else 0,
        "files_with_unexpected_labels": int(ok_df["has_unexpected_labels"].sum()) if "has_unexpected_labels" in ok_df.columns else 0,
        "binary_negative_label": str(binary_negative_label),
        "binary_positive_label": str(binary_positive_label),
    }

    neg_total = 0
    pos_total = 0

    for label in expected_labels:
        count_col = f"count_label_{label}"
        has_col = f"has_label_{label}"
        summary[f"total_count_label_{label}"] = int(ok_df[count_col].sum()) if count_col in ok_df.columns else 0
        summary[f"files_with_label_{label}"] = int(ok_df[has_col].sum()) if has_col in ok_df.columns else 0

        if label == binary_negative_label:
            neg_total = summary[f"total_count_label_{label}"]
        if label == binary_positive_label:
            pos_total = summary[f"total_count_label_{label}"]

    binary_total = neg_total + pos_total
    summary["binary_rows_total"] = int(binary_total)
    summary["binary_positive_ratio_global"] = safe_ratio(pos_total, binary_total)

    return pd.DataFrame([summary])


def build_pattern_summary(file_df: pd.DataFrame) -> pd.DataFrame:
    ok_df = file_df[file_df["status"] == "ok"].copy()
    if ok_df.empty:
        return pd.DataFrame(
            columns=["label_pattern", "n_files", "total_rows", "total_binary_rows"]
        )

    group = (
        ok_df.groupby("label_pattern", dropna=False)
        .agg(
            n_files=("file_name", "count"),
            total_rows=("n_rows", "sum"),
            total_binary_rows=("binary_rows", "sum"),
        )
        .reset_index()
        .sort_values(["n_files", "total_rows"], ascending=[False, False])
    )
    return group


def build_manifest(
    args: argparse.Namespace,
    discovered_files: Sequence[Path],
    overall_df: pd.DataFrame,
) -> Dict[str, object]:
    overall = overall_df.iloc[0].to_dict() if not overall_df.empty else {}
    return {
        "input_dir": str(Path(args.input_dir).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()) if args.output_dir else None,
        "glob": args.glob,
        "recursive": not args.non_recursive,
        "label_col": args.label_col,
        "expected_labels": [str(x) for x in args.expected_labels],
        "binary_negative_label": str(args.binary_negative_label),
        "binary_positive_label": str(args.binary_positive_label),
        "n_files_discovered": len(discovered_files),
        "overall": overall,
    }


def save_outputs(
    output_dir: Path,
    file_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    pattern_df: pd.DataFrame,
    manifest: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    file_df.to_csv(output_dir / "label_coverage_file_level.csv", index=False, encoding="utf-8-sig")
    overall_df.to_csv(output_dir / "label_coverage_overall.csv", index=False, encoding="utf-8-sig")
    pattern_df.to_csv(output_dir / "label_coverage_patterns.csv", index=False, encoding="utf-8-sig")

    if not file_df.empty:
        file_df[file_df["status"] == "error"].to_csv(
            output_dir / "label_coverage_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )

        ok_df = file_df[file_df["status"] == "ok"].copy()

        ok_df[ok_df["has_binary_01_pair"]].to_csv(
            output_dir / "files_with_binary_0_1.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if "has_triplet_neg1_0_1" in ok_df.columns:
            ok_df[ok_df["has_triplet_neg1_0_1"]].to_csv(
                output_dir / "files_with_neg1_0_1.csv",
                index=False,
                encoding="utf-8-sig",
            )

        if "has_detection_label_2" in ok_df.columns:
            ok_df[ok_df["has_detection_label_2"]].to_csv(
                output_dir / "files_with_label_2.csv",
                index=False,
                encoding="utf-8-sig",
            )

        ok_df[ok_df["has_unexpected_labels"]].to_csv(
            output_dir / "files_with_unexpected_labels.csv",
            index=False,
            encoding="utf-8-sig",
        )

    with open(output_dir / "label_coverage_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def print_console_summary(overall_df: pd.DataFrame, output_dir: Path, expected_labels: Sequence[LabelValue]) -> None:
    if overall_df.empty:
        print("No summary available.")
        return

    row = overall_df.iloc[0].to_dict()

    print("=" * 72)
    print("Label coverage QC summary")
    print("=" * 72)
    print(f"Files discovered : {row.get('n_files_discovered', 0)}")
    print(f"Files OK         : {row.get('n_files_ok', 0)}")
    print(f"Files skipped    : {row.get('n_files_skipped', 0)}")
    print(f"Files error      : {row.get('n_files_error', 0)}")
    print(f"Total rows       : {row.get('total_rows', 0)}")
    print(f"Valid labels     : {row.get('total_valid_labels', 0)}")
    print(f"Invalid labels   : {row.get('total_missing_or_invalid_labels', 0)}")

    print("-" * 72)
    for label in expected_labels:
        print(
            f"Label {label:>4}: "
            f"count={row.get(f'total_count_label_{label}', 0):>10} | "
            f"files={row.get(f'files_with_label_{label}', 0):>6}"
        )

    binary_ratio = row.get("binary_positive_ratio_global", None)
    binary_ratio_str = "NA" if binary_ratio is None or pd.isna(binary_ratio) else f"{binary_ratio:.4f}"

    print("-" * 72)
    print(f"Binary rows total             : {row.get('binary_rows_total', 0)}")
    print(f"Global positive ratio (0/1)   : {binary_ratio_str}")
    print(f"Files with usable 0/1 pair    : {row.get('files_with_binary_pair', 0)}")
    print(f"Files with -1/0/1 triplet     : {row.get('files_with_triplet_neg1_0_1', 0)}")
    print(f"Files with label 2            : {row.get('files_with_detection_label_2', 0)}")
    print(f"Files with unexpected labels  : {row.get('files_with_unexpected_labels', 0)}")
    print("-" * 72)
    print(f"Saved to: {output_dir}")
    print("=" * 72)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else (input_dir / "qc_label_coverage")

    expected_labels = normalize_expected_labels(args.expected_labels)
    binary_negative_label = canonicalize_scalar_label(args.binary_negative_label)
    binary_positive_label = canonicalize_scalar_label(args.binary_positive_label)

    if binary_negative_label is None or binary_positive_label is None:
        raise ValueError("Binary negative / positive labels must be valid non-empty values.")

    recursive = not args.non_recursive

    csv_files = discover_csv_files(
        input_dir=input_dir,
        glob_pattern=args.glob,
        recursive=recursive,
        ignore_prefixes=args.ignore_prefixes,
        ignore_names=args.ignore_names,
    )

    if not csv_files:
        output_dir.mkdir(parents=True, exist_ok=True)
        empty_file_df = pd.DataFrame(
            columns=["file_name", "file_path", "status", "error"]
        )
        empty_overall_df = pd.DataFrame([{
            "n_files_discovered": 0,
            "n_files_ok": 0,
            "n_files_skipped": 0,
            "n_files_error": 0,
            "total_rows": 0,
            "total_valid_labels": 0,
            "total_missing_or_invalid_labels": 0,
            "binary_rows_total": 0,
            "binary_positive_ratio_global": None,
        }])
        empty_pattern_df = pd.DataFrame(columns=["label_pattern", "n_files", "total_rows", "total_binary_rows"])
        manifest = build_manifest(args, [], empty_overall_df)
        save_outputs(output_dir, empty_file_df, empty_overall_df, empty_pattern_df, manifest)
        print(f"No CSV files found under: {input_dir}")
        print(f"Empty QC outputs were still written to: {output_dir}")
        return

    records: List[Dict[str, object]] = []
    for csv_path in csv_files:
        record = summarize_one_file(
            csv_path=csv_path,
            label_col=args.label_col,
            expected_labels=expected_labels,
            binary_negative_label=binary_negative_label,
            binary_positive_label=binary_positive_label,
            metadata_cols=args.metadata_cols,
            strict_label_column=args.strict_label_column,
        )
        records.append(record)

    file_df = pd.DataFrame(records)
    if not file_df.empty:
        file_df = file_df.sort_values(["status", "file_name"], ascending=[True, True]).reset_index(drop=True)

    overall_df = build_overall_summary(
        file_df=file_df,
        expected_labels=expected_labels,
        binary_negative_label=binary_negative_label,
        binary_positive_label=binary_positive_label,
    )
    pattern_df = build_pattern_summary(file_df)
    manifest = build_manifest(args, csv_files, overall_df)

    save_outputs(
        output_dir=output_dir,
        file_df=file_df,
        overall_df=overall_df,
        pattern_df=pattern_df,
        manifest=manifest,
    )
    print_console_summary(overall_df, output_dir, expected_labels)

if __name__ == "__main__":
    main()

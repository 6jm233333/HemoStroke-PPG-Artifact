from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_csv_with_fallback(
    path: str | Path,
    encodings: tuple[str, ...] = ("utf-8-sig", "utf-8", "gbk", "latin1"),
    **kwargs: Any,
) -> pd.DataFrame:
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Failed to read CSV: {path}\nLast error: {last_error}")


def normalize_mimic_id(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    try:
        return str(int(float(text)))
    except Exception:
        return text


def clean_note_text(text: Any, replace_double_quote: bool = True) -> str:
    if pd.isna(text):
        cleaned = ""
    else:
        cleaned = str(text)

    cleaned = (
        cleaned.replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
    )
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")

    cleaned = cleaned.strip()

    if replace_double_quote:
        cleaned = cleaned.replace('"', "'")

    return cleaned


def validate_required_columns(df: pd.DataFrame, required: list[str], file_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{file_name} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def save_csv(df: pd.DataFrame, path: str | Path, quote_all: bool = False) -> None:
    ensure_parent_dir(path)
    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL,
    )


def process_noteevents(
    left_df: pd.DataFrame,
    notes_csv_path: str | Path,
    subject_col: str,
    hadm_col: str,
    charttime_col: str,
    text_col: str,
    chunk_size: int,
    clean_text_flag: bool,
    replace_double_quote: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    header_df = read_csv_with_fallback(notes_csv_path, nrows=0)
    validate_required_columns(
        header_df,
        [subject_col, hadm_col, charttime_col, text_col],
        file_name=str(notes_csv_path),
    )

    target_hadm_ids = set(left_df[hadm_col].dropna().astype(str).unique())

    cleaned_note_chunks: list[pd.DataFrame] = []
    timed_note_chunks: list[pd.DataFrame] = []

    stats = {
        "chunks_processed": 0,
        "rows_scanned": 0,
        "rows_relevant": 0,
        "rows_cleaned_output": 0,
        "rows_timed_output": 0,
        "target_hadm_count": len(target_hadm_ids),
    }

    chunk_iter = read_csv_with_fallback(
        notes_csv_path,
        chunksize=chunk_size,
        low_memory=False,
    )

    for chunk in chunk_iter:
        stats["chunks_processed"] += 1
        stats["rows_scanned"] += len(chunk)

        if subject_col in chunk.columns:
            chunk[subject_col] = chunk[subject_col].apply(normalize_mimic_id)
        if hadm_col in chunk.columns:
            chunk[hadm_col] = chunk[hadm_col].apply(normalize_mimic_id)

        relevant = chunk[chunk[hadm_col].isin(target_hadm_ids)].copy()
        stats["rows_relevant"] += len(relevant)

        if relevant.empty:
            if stats["chunks_processed"] % 10 == 0:
                print(
                    f"[build_stroke_note_table] scanned={stats['rows_scanned']:,} "
                    f"relevant={stats['rows_relevant']:,} "
                    f"cleaned={stats['rows_cleaned_output']:,} "
                    f"timed={stats['rows_timed_output']:,}"
                )
            continue

        if clean_text_flag:
            relevant[text_col] = relevant[text_col].apply(
                lambda x: clean_note_text(x, replace_double_quote=replace_double_quote)
            )

        if "ROW_ID" in relevant.columns and "ROW_ID_NOTE" not in relevant.columns:
            relevant = relevant.rename(columns={"ROW_ID": "ROW_ID_NOTE"})

        cleaned_note_chunks.append(relevant)
        stats["rows_cleaned_output"] += len(relevant)

        timed = relevant.dropna(subset=[charttime_col]).copy()
        if not timed.empty:
            timed_note_chunks.append(timed)
            stats["rows_timed_output"] += len(timed)

        if stats["chunks_processed"] % 10 == 0:
            print(
                f"[build_stroke_note_table] scanned={stats['rows_scanned']:,} "
                f"relevant={stats['rows_relevant']:,} "
                f"cleaned={stats['rows_cleaned_output']:,} "
                f"timed={stats['rows_timed_output']:,}"
            )

    cleaned_notes_df = (
        pd.concat(cleaned_note_chunks, ignore_index=True)
        if cleaned_note_chunks
        else pd.DataFrame()
    )
    timed_notes_df = (
        pd.concat(timed_note_chunks, ignore_index=True)
        if timed_note_chunks
        else pd.DataFrame()
    )

    return cleaned_notes_df, timed_notes_df, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cleaned and timed stroke note tables from MIMIC-III.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configs/mimic_data.yaml",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    subject_col = cfg["id_columns"]["subject_id"]
    hadm_col = cfg["id_columns"]["hadm_id"]
    charttime_col = cfg["time_columns"]["note_charttime"]
    text_col = cfg["text_columns"]["note_text"]

    left_file_path = cfg["paths"]["icd9_icustays_merged_csv"]
    notes_file_path = cfg["paths"]["noteevents_csv"]

    cleaned_note_output = cfg["paths"]["cleaned_note_table_csv"]
    timed_note_output = cfg["paths"]["note_table_with_time_csv"]

    chunk_size = int(cfg["note_filter"].get("chunk_size", 100000))
    clean_text_flag = bool(cfg["note_filter"].get("clean_text", True))
    replace_double_quote = bool(cfg["note_filter"].get("replace_double_quote", True))
    save_processing_log = bool(cfg["quality_control"].get("save_processing_log", True))
    fail_if_no_notes_found = bool(cfg["quality_control"].get("fail_if_no_notes_found", True))

    print(f"[build_stroke_note_table] reading left table: {left_file_path}")
    left_df = read_csv_with_fallback(left_file_path, low_memory=False)
    validate_required_columns(left_df, [subject_col, hadm_col], str(left_file_path))

    left_df[subject_col] = left_df[subject_col].apply(normalize_mimic_id)
    left_df[hadm_col] = left_df[hadm_col].apply(normalize_mimic_id)

    print(
        f"[build_stroke_note_table] left rows={len(left_df):,}, "
        f"unique_hadm={left_df[hadm_col].nunique(dropna=True):,}"
    )

    cleaned_notes_df, timed_notes_df, stats = process_noteevents(
        left_df=left_df,
        notes_csv_path=notes_file_path,
        subject_col=subject_col,
        hadm_col=hadm_col,
        charttime_col=charttime_col,
        text_col=text_col,
        chunk_size=chunk_size,
        clean_text_flag=clean_text_flag,
        replace_double_quote=replace_double_quote,
    )

    if fail_if_no_notes_found and cleaned_notes_df.empty:
        raise RuntimeError(
            "No relevant notes were found after filtering NOTEEVENTS by HADM_ID."
        )

    print("[build_stroke_note_table] merging cleaned notes...")
    cleaned_merged = pd.merge(
        left_df,
        cleaned_notes_df,
        on=[subject_col, hadm_col],
        how="left",
    )

    print("[build_stroke_note_table] merging timed notes...")
    timed_merged = pd.merge(
        left_df,
        timed_notes_df,
        on=[subject_col, hadm_col],
        how="left",
    )

    print(f"[build_stroke_note_table] saving cleaned note table -> {cleaned_note_output}")
    save_csv(cleaned_merged, cleaned_note_output, quote_all=True)

    print(f"[build_stroke_note_table] saving timed note table -> {timed_note_output}")
    save_csv(timed_merged, timed_note_output, quote_all=False)

    if save_processing_log:
        log_path = Path(cfg["paths"]["interim_root"]) / "build_stroke_note_table_log.json"
        ensure_parent_dir(log_path)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "left_rows": int(len(left_df)),
                    "cleaned_note_rows": int(len(cleaned_notes_df)),
                    "timed_note_rows": int(len(timed_notes_df)),
                    "cleaned_merged_rows": int(len(cleaned_merged)),
                    "timed_merged_rows": int(len(timed_merged)),
                    "stats": stats,
                    "outputs": {
                        "cleaned_note_table_csv": str(cleaned_note_output),
                        "note_table_with_time_csv": str(timed_note_output),
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[build_stroke_note_table] log saved -> {log_path}")

    print("[build_stroke_note_table] done.")
    print(
        f"  cleaned_note_rows={len(cleaned_notes_df):,}, "
        f"timed_note_rows={len(timed_notes_df):,}, "
        f"cleaned_merged_rows={len(cleaned_merged):,}, "
        f"timed_merged_rows={len(timed_merged):,}"
    )


if __name__ == "__main__":
    main()

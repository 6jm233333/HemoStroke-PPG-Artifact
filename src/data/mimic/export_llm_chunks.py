from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


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


def choose_row_id_column(columns: list[str]) -> str:
    candidates = ["Row_ID", "ROW_ID", "ROW_ID_NOTE"]
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(
        f"Could not find a row-id column. Expected one of {candidates}, got: {columns}"
    )


def normalize_row_id(value: Any) -> int | str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return ""
    try:
        return int(float(text))
    except Exception:
        return text


def normalize_charttime(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def save_json_chunk(records: list[dict[str, Any]], output_path: str | Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LLM-ready JSON chunks from note table.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configs/mimic_data.yaml",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    input_csv = cfg["paths"]["llm_input_csv"]
    output_dir = cfg["paths"]["llm_chunk_dir"]

    rows_per_file = int(cfg["llm_chunking"].get("rows_per_file", 300))
    hard_limit_chars = int(cfg["llm_chunking"].get("hard_limit_chars", 200000))
    soft_limit_chars = int(cfg["llm_chunking"].get("soft_limit_chars", 200000))
    drop_empty_text = bool(cfg["llm_chunking"].get("drop_empty_text", True))

    charttime_col = cfg["time_columns"]["note_charttime"]
    text_col = cfg["text_columns"]["note_text"]

    ensure_dir(output_dir)

    header_df = read_csv_with_fallback(input_csv, nrows=0)
    row_id_col = choose_row_id_column(list(header_df.columns))

    required_cols = [row_id_col, charttime_col, text_col]
    missing = [c for c in required_cols if c not in header_df.columns]
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {missing}\n"
            f"Available columns: {list(header_df.columns)}"
        )

    print(f"[export_llm_chunks] reading input: {input_csv}")
    df = read_csv_with_fallback(
        input_csv,
        usecols=required_cols,
        low_memory=False,
    )

    df = df.rename(columns={row_id_col: "Row_ID"})
    df["Row_ID"] = df["Row_ID"].apply(normalize_row_id)
    df[charttime_col] = df[charttime_col].apply(normalize_charttime)
    df[text_col] = df[text_col].apply(normalize_text)

    if drop_empty_text:
        before = len(df)
        df = df[df[text_col].str.strip().ne("")].copy()
        print(f"[export_llm_chunks] dropped empty TEXT rows: {before - len(df):,}")

    current_chunk: list[dict[str, Any]] = []
    current_chars = 0
    file_count = 1
    manifest_rows: list[dict[str, Any]] = []

    def flush_chunk(chunk_records: list[dict[str, Any]], chunk_chars: int, chunk_idx: int) -> None:
        if not chunk_records:
            return

        save_path = Path(output_dir) / f"NLP_Group_{chunk_idx:03d}.json"
        save_json_chunk(chunk_records, save_path)

        manifest_rows.append(
            {
                "file_name": save_path.name,
                "file_path": str(save_path),
                "n_records": len(chunk_records),
                "n_chars": chunk_chars,
                "first_row_id": chunk_records[0]["Row_ID"],
                "last_row_id": chunk_records[-1]["Row_ID"],
            }
        )

    print("[export_llm_chunks] chunking records...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        record = {
            "Row_ID": row["Row_ID"],
            "CHARTTIME": row[charttime_col],
            "TEXT": row[text_col],
        }

        record_chars = len(json.dumps(record, ensure_ascii=False))

        should_flush = False
        if current_chunk:
            if len(current_chunk) >= rows_per_file:
                should_flush = True
            elif current_chars + record_chars > hard_limit_chars:
                should_flush = True
            elif current_chars >= soft_limit_chars:
                should_flush = True

        if should_flush:
            flush_chunk(current_chunk, current_chars, file_count)
            file_count += 1
            current_chunk = []
            current_chars = 0

        current_chunk.append(record)
        current_chars += record_chars

    if current_chunk:
        flush_chunk(current_chunk, current_chars, file_count)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = Path(output_dir) / "chunk_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print("[export_llm_chunks] done.")
    print(f"  total_records={len(df):,}")
    print(f"  total_files={len(manifest_df):,}")
    print(f"  output_dir={output_dir}")
    print(f"  manifest={manifest_path}")


if __name__ == "__main__":
    main()

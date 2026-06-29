from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv_with_fallback(path: str | Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception as exc:  # pragma: no cover - exercised only on fallback
            last_error = exc
    raise RuntimeError(f"Failed to read CSV: {path}\nLast error: {last_error}")


def validate_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"MC-MED radiology table is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def combine_text_fields(row: pd.Series, text_fields: list[str]) -> str:
    parts: list[str] = []
    for field in text_fields:
        value = row.get(field)
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            parts.append(f"{field}: {text}")
    return "\n".join(parts)


def build_canonical_llm_input(
    rads_df: pd.DataFrame,
    *,
    visit_id_col: str,
    timestamp_col: str,
    source_timestamp_col: str,
    text_fields: list[str],
    row_id_col: str,
) -> pd.DataFrame:
    validate_required_columns(
        rads_df,
        [visit_id_col, timestamp_col, source_timestamp_col, *text_fields],
    )

    out = rads_df.copy()
    out.columns = [str(column).strip() for column in out.columns]
    out["TEXT"] = out.apply(lambda row: combine_text_fields(row, text_fields), axis=1)
    out["CHARTTIME"] = out[timestamp_col]
    out = out[out["TEXT"].str.strip().ne("")].copy()
    out = out[out["CHARTTIME"].notna()].copy()
    out = out.reset_index(drop=True)
    out[row_id_col] = out.index + 1

    columns = [
        row_id_col,
        visit_id_col,
        "CHARTTIME",
        "TEXT",
        source_timestamp_col,
        timestamp_col,
        *text_fields,
    ]
    columns = list(dict.fromkeys(columns))
    return out[columns]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize credentialed MC-MED radiology records into the canonical LLM schema."
    )
    parser.add_argument("--config", required=True, help="Path to configs/mcmed_data.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg["paths"]
    anchor_cfg = cfg["anchor_generation"]
    visit_id_col = cfg["id_columns"]["visit_id"]

    rads_csv = paths["rads_csv"]
    output_csv = paths["llm_input_csv"]
    row_id_col = str(anchor_cfg.get("radiology_row_id", "Row_ID"))
    timestamp_col = str(anchor_cfg.get("canonical_note_timestamp", "Result_time"))
    source_timestamp_col = str(
        anchor_cfg.get(
            "retained_source_timestamp",
            anchor_cfg.get("retained_audit_timestamp", "Order_time"),
        )
    )
    text_fields = list(anchor_cfg.get("radiology_text_fields", ["Study", "Impression"]))

    print(f"[build_llm_input] reading: {rads_csv}")
    rads_df = read_csv_with_fallback(rads_csv)
    out = build_canonical_llm_input(
        rads_df,
        visit_id_col=visit_id_col,
        timestamp_col=timestamp_col,
        source_timestamp_col=source_timestamp_col,
        text_fields=text_fields,
        row_id_col=row_id_col,
    )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")

    log_path = output_path.with_name("build_llm_input_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_path": str(rads_csv),
                "output_path": str(output_path),
                "input_rows": int(len(rads_df)),
                "output_rows": int(len(out)),
                "canonical_fields": ["Row_ID", "CHARTTIME", "TEXT"],
                "text_fields": text_fields,
                "canonical_note_timestamp": timestamp_col,
                "retained_source_timestamp": source_timestamp_col,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[build_llm_input] done.")
    print(f"  output={output_path}")
    print(f"  rows={len(out):,}")
    print(f"  log={log_path}")


if __name__ == "__main__":
    main()

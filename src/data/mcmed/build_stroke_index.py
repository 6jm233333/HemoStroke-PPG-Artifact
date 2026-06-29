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


def validate_required_columns(
    df: pd.DataFrame,
    required: list[str],
    *,
    table_name: str,
) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def validate_unique_anchors(
    anchors: pd.DataFrame,
    *,
    visit_id_col: str,
    anchor_time_col: str,
) -> pd.DataFrame:
    clean = anchors[[visit_id_col, anchor_time_col]].copy()
    clean[anchor_time_col] = pd.to_datetime(clean[anchor_time_col], errors="coerce")
    clean = clean.dropna(subset=[visit_id_col, anchor_time_col]).copy()
    unique_counts = clean.groupby(visit_id_col)[anchor_time_col].nunique(dropna=True)
    ambiguous = unique_counts[unique_counts > 1]
    if not ambiguous.empty:
        example_ids = ambiguous.index.astype(str).tolist()[:5]
        raise ValueError(
            "Reviewed anchor table contains multiple distinct anchors for a visit. "
            "Resolve the local review table before continuing. "
            f"Example visit IDs: {example_ids}"
        )
    return clean.drop_duplicates(subset=[visit_id_col, anchor_time_col]).copy()


def build_stroke_index(
    anchors: pd.DataFrame,
    segments: pd.DataFrame,
    *,
    visit_id_col: str,
    anchor_time_col: str,
    waveform_type_col: str,
    waveform_path_col: str,
    waveform_start_col: str,
    waveform_end_col: str,
    pleth_value: str,
) -> pd.DataFrame:
    validate_required_columns(anchors, [visit_id_col, anchor_time_col], table_name="reviewed anchors")
    validate_required_columns(
        segments,
        [
            visit_id_col,
            waveform_type_col,
            waveform_path_col,
            waveform_start_col,
            waveform_end_col,
        ],
        table_name="Pleth segment manifest",
    )

    anchors = validate_unique_anchors(
        anchors,
        visit_id_col=visit_id_col,
        anchor_time_col=anchor_time_col,
    )
    segments = segments.copy()
    segments = segments[
        segments[waveform_type_col].astype(str).str.lower().eq(str(pleth_value).lower())
    ].copy()
    segments[waveform_start_col] = pd.to_datetime(segments[waveform_start_col], errors="coerce")
    segments[waveform_end_col] = pd.to_datetime(segments[waveform_end_col], errors="coerce")
    segments = segments.dropna(
        subset=[visit_id_col, waveform_path_col, waveform_start_col, waveform_end_col]
    ).copy()

    merged = segments.merge(anchors, on=visit_id_col, how="inner", validate="many_to_one")
    merged = merged.rename(
        columns={
            anchor_time_col: "Extracted_Timestamp",
            waveform_start_col: "WAVE_START",
            waveform_end_col: "WAVE_END",
            waveform_path_col: "WAVE_PATH",
            waveform_type_col: "Wave_Type",
        }
    )
    merged = merged.drop_duplicates(subset=["WAVE_PATH", "Extracted_Timestamp"], keep="first")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge reviewed MC-MED anchors with a local Pleth waveform-segment manifest."
    )
    parser.add_argument("--config", required=True, help="Path to configs/mcmed_data.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg["paths"]
    anchor_cfg = cfg["anchor_generation"]
    time_cfg = cfg["time_columns"]
    waveform_cfg = cfg["waveform_columns"]

    reviewed_anchor_csv = paths["reviewed_anchor_csv"]
    segment_manifest_csv = paths["pleth_segment_manifest_csv"]
    output_csv = paths["stroke_index_csv"]
    visit_id_col = cfg["id_columns"]["visit_id"]
    anchor_time_col = str(anchor_cfg.get("reviewed_anchor_time", "Extracted_Timestamp"))

    print(f"[build_stroke_index] reading reviewed anchors: {reviewed_anchor_csv}")
    anchors = read_csv_with_fallback(reviewed_anchor_csv)
    print(f"[build_stroke_index] reading Pleth segment manifest: {segment_manifest_csv}")
    segments = read_csv_with_fallback(segment_manifest_csv)

    out = build_stroke_index(
        anchors,
        segments,
        visit_id_col=visit_id_col,
        anchor_time_col=anchor_time_col,
        waveform_type_col=waveform_cfg["waveform_type"],
        waveform_path_col=waveform_cfg["waveform_path"],
        waveform_start_col=time_cfg["waveform_start"],
        waveform_end_col=time_cfg["waveform_end"],
        pleth_value=str(anchor_cfg.get("pleth_value", "Pleth")),
    )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")

    log_path = output_path.with_name("build_stroke_index_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "reviewed_anchor_path": str(reviewed_anchor_csv),
                "segment_manifest_path": str(segment_manifest_csv),
                "output_path": str(output_path),
                "reviewed_anchor_rows": int(len(anchors)),
                "segment_manifest_rows": int(len(segments)),
                "merged_index_rows": int(len(out)),
                "waveform_type": str(anchor_cfg.get("pleth_value", "Pleth")),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[build_stroke_index] done.")
    print(f"  output={output_path}")
    print(f"  rows={len(out):,}")
    print(f"  log={log_path}")


if __name__ == "__main__":
    main()

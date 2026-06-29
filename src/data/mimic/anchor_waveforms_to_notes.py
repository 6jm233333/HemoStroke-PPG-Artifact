from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm


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


def extract_subject_id_from_wave_path(path_value: Any) -> str | None:
    if pd.isna(path_value):
        return None
    text = str(path_value).strip()
    if not text:
        return None

    # Matches paths like p04/p044083/... or .../p044083-2112-...
    match = re.search(r"p0*(\d+)", text)
    if match:
        try:
            return str(int(match.group(1)))
        except Exception:
            return match.group(1)

    return None


def validate_required_columns(df: pd.DataFrame, required: list[str], file_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{file_name} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Anchor cleaned stroke notes to waveform time windows for MIMIC."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configs/mimic_data.yaml",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    waveform_index_csv = cfg["paths"]["waveform_index_csv"]
    clinical_notes_csv = cfg["paths"]["cleaned_note_table_csv"]
    output_csv = cfg["paths"]["waveform_anchored_csv"]

    subject_col = cfg["id_columns"]["subject_id"]
    charttime_col = cfg["time_columns"]["note_charttime"]
    waveform_start_col = cfg["time_columns"]["waveform_start"]
    waveform_end_col = cfg["time_columns"]["waveform_end"]

    output_wave_path_col = cfg["waveform_columns"]["waveform_path"]
    source_wave_path_col = cfg["waveform_columns"]["original_waveform_path"]

    buffer_hours_after_wave_end = int(cfg["waveform_anchor"].get("buffer_hours_after_wave_end", 4))
    require_note_in_window = bool(cfg["waveform_anchor"].get("require_note_in_window", True))
    save_processing_log = bool(cfg["quality_control"].get("save_processing_log", True))

    print(f"[anchor_waveforms_to_notes] reading waveform index: {waveform_index_csv}")
    df_wave = read_csv_with_fallback(waveform_index_csv, low_memory=False)
    validate_required_columns(
        df_wave,
        [source_wave_path_col, waveform_start_col, waveform_end_col],
        str(waveform_index_csv),
    )

    df_wave = df_wave.copy()
    df_wave[subject_col] = df_wave[source_wave_path_col].apply(extract_subject_id_from_wave_path)
    df_wave[waveform_start_col] = pd.to_datetime(df_wave[waveform_start_col], errors="coerce")
    df_wave[waveform_end_col] = pd.to_datetime(df_wave[waveform_end_col], errors="coerce")

    df_wave = df_wave.dropna(subset=[subject_col, waveform_start_col, waveform_end_col]).copy()
    if df_wave.empty:
        raise RuntimeError("Waveform index became empty after parsing subject_id/start/end time.")

    wave_dict: dict[str, pd.DataFrame] = {
        str(k): v.sort_values([waveform_start_col, waveform_end_col]).copy()
        for k, v in df_wave.groupby(subject_col, sort=False)
    }

    print(f"[anchor_waveforms_to_notes] reading clinical notes: {clinical_notes_csv}")
    df_clinical = read_csv_with_fallback(clinical_notes_csv, low_memory=False)
    validate_required_columns(
        df_clinical,
        [subject_col, charttime_col],
        str(clinical_notes_csv),
    )

    df_clinical = df_clinical.copy()
    df_clinical[subject_col] = df_clinical[subject_col].apply(normalize_mimic_id)
    df_clinical[charttime_col] = pd.to_datetime(df_clinical[charttime_col], errors="coerce")
    df_clinical = df_clinical.dropna(subset=[subject_col, charttime_col]).copy()

    if df_clinical.empty:
        raise RuntimeError("Clinical note table has no valid SUBJECT_ID + CHARTTIME rows.")

    clinical_dict: dict[str, pd.DataFrame] = {
        str(k): v.sort_values(charttime_col).copy()
        for k, v in df_clinical.groupby(subject_col, sort=False)
    }

    common_ids = sorted(set(clinical_dict.keys()).intersection(set(wave_dict.keys())))
    print(
        f"[anchor_waveforms_to_notes] waveform_subjects={len(wave_dict):,}, "
        f"note_subjects={len(clinical_dict):,}, "
        f"common_subjects={len(common_ids):,}"
    )

    matched_parts: list[pd.DataFrame] = []
    total_wave_segments = 0
    total_notes_matched = 0

    for sub_id in tqdm(common_ids, desc="Anchoring notes to waveforms"):
        sub_notes = clinical_dict[sub_id]
        sub_waves = wave_dict[sub_id]

        for _, wave_row in sub_waves.iterrows():
            total_wave_segments += 1
            wave_start = wave_row[waveform_start_col]
            wave_end = wave_row[waveform_end_col]
            wave_end_with_buffer = wave_end + pd.Timedelta(hours=buffer_hours_after_wave_end)

            mask = (
                (sub_notes[charttime_col] >= wave_start)
                & (sub_notes[charttime_col] <= wave_end_with_buffer)
            )
            current_matches = sub_notes.loc[mask].copy()

            if current_matches.empty:
                if require_note_in_window:
                    continue

                current_matches = pd.DataFrame(columns=sub_notes.columns)

            if not current_matches.empty:
                current_matches["WAVE_START"] = wave_start
                current_matches["WAVE_END"] = wave_end
                current_matches[output_wave_path_col] = wave_row[source_wave_path_col]
                matched_parts.append(current_matches)
                total_notes_matched += len(current_matches)

    if not matched_parts:
        raise RuntimeError(
            "No notes were anchored to waveform windows. "
            "Check waveform dates, note dates, and subject-id extraction."
        )

    final_df = pd.concat(matched_parts, ignore_index=True)
    final_df = final_df.sort_values([subject_col, "WAVE_START", charttime_col]).reset_index(drop=True)

    ensure_parent_dir(output_csv)
    final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("[anchor_waveforms_to_notes] done.")
    print(f"  output={output_csv}")
    print(f"  anchored_rows={len(final_df):,}")
    print(f"  total_wave_segments_scanned={total_wave_segments:,}")
    print(f"  total_note_matches={total_notes_matched:,}")

    if save_processing_log:
        log_path = Path(output_csv).with_name("anchor_waveforms_to_notes_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "waveform_index_csv": str(waveform_index_csv),
                    "clinical_notes_csv": str(clinical_notes_csv),
                    "output_csv": str(output_csv),
                    "buffer_hours_after_wave_end": buffer_hours_after_wave_end,
                    "waveform_subjects": len(wave_dict),
                    "note_subjects": len(clinical_dict),
                    "common_subjects": len(common_ids),
                    "total_wave_segments_scanned": total_wave_segments,
                    "anchored_rows": int(len(final_df)),
                    "total_note_matches": total_notes_matched,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"  log={log_path}")


if __name__ == "__main__":
    main()

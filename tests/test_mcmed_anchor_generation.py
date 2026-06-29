import pandas as pd
import pytest

from src.data.mcmed.build_llm_input import build_canonical_llm_input
from src.data.mcmed.build_stroke_index import build_stroke_index


def test_build_canonical_llm_input_uses_radiology_adapter():
    rads = pd.DataFrame(
        [
            {
                "CSN": "visit-1",
                "Order_time": "2025-01-01T09:00:00Z",
                "Result_time": "2025-01-01T10:00:00Z",
                "Study": "MR BRAIN",
                "Impression": "Acute infarct.",
            }
        ]
    )

    out = build_canonical_llm_input(
        rads,
        visit_id_col="CSN",
        timestamp_col="Result_time",
        source_timestamp_col="Order_time",
        text_fields=["Study", "Impression"],
        row_id_col="Row_ID",
    )

    assert out.loc[0, "CHARTTIME"] == "2025-01-01T10:00:00Z"
    assert "Study: MR BRAIN" in out.loc[0, "TEXT"]
    assert "Impression: Acute infarct." in out.loc[0, "TEXT"]


def test_build_stroke_index_keeps_pleth_only():
    anchors = pd.DataFrame(
        [{"CSN": "visit-1", "Extracted_Timestamp": "2025-01-01T10:30:00Z"}]
    )
    segments = pd.DataFrame(
        [
            {
                "CSN": "visit-1",
                "Wave_Type": "Pleth",
                "WAVE_PATH": "wave-1",
                "WAVE_START": "2025-01-01T09:00:00Z",
                "WAVE_END": "2025-01-01T10:00:00Z",
            },
            {
                "CSN": "visit-1",
                "Wave_Type": "II",
                "WAVE_PATH": "wave-2",
                "WAVE_START": "2025-01-01T09:00:00Z",
                "WAVE_END": "2025-01-01T10:00:00Z",
            },
        ]
    )

    out = build_stroke_index(
        anchors,
        segments,
        visit_id_col="CSN",
        anchor_time_col="Extracted_Timestamp",
        waveform_type_col="Wave_Type",
        waveform_path_col="WAVE_PATH",
        waveform_start_col="WAVE_START",
        waveform_end_col="WAVE_END",
        pleth_value="Pleth",
    )

    assert out["WAVE_PATH"].tolist() == ["wave-1"]
    assert out["Wave_Type"].tolist() == ["Pleth"]


def test_build_stroke_index_rejects_ambiguous_reviewed_anchors():
    anchors = pd.DataFrame(
        [
            {"CSN": "visit-1", "Extracted_Timestamp": "2025-01-01T10:30:00Z"},
            {"CSN": "visit-1", "Extracted_Timestamp": "2025-01-01T11:30:00Z"},
        ]
    )
    segments = pd.DataFrame(
        [
            {
                "CSN": "visit-1",
                "Wave_Type": "Pleth",
                "WAVE_PATH": "wave-1",
                "WAVE_START": "2025-01-01T09:00:00Z",
                "WAVE_END": "2025-01-01T10:00:00Z",
            }
        ]
    )

    with pytest.raises(ValueError, match="multiple distinct anchors"):
        build_stroke_index(
            anchors,
            segments,
            visit_id_col="CSN",
            anchor_time_col="Extracted_Timestamp",
            waveform_type_col="Wave_Type",
            waveform_path_col="WAVE_PATH",
            waveform_start_col="WAVE_START",
            waveform_end_col="WAVE_END",
            pleth_value="Pleth",
        )

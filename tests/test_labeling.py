import pandas as pd

from src.labels.relabel_time_windows import assign_label, rebuild_group_time_axis
from src.models.sensitivity import label_relative_minutes, shifted_relative_minutes


def test_relabel_time_window_boundaries_match_config():
    cfg = {
        "normal_range": {"start": -480, "end": -255},
        "positive_range": {"start": -225, "end": -15},
        "ignore_ranges": [[-255, -225], [-15, float("inf")]],
    }
    assert assign_label(-300, cfg) == 0
    assert assign_label(-240, cfg) == -1
    assert assign_label(-120, cfg) == 1
    assert assign_label(-5, cfg) == -1


def test_labeling_boundary_points_are_consistent():
    cfg = {
        "normal_range": {"start": -480, "end": -255},
        "positive_range": {"start": -225, "end": -15},
        "ignore_ranges": [[-255, -225], [-15, float("inf")]],
    }

    assert assign_label(-480, cfg) == 0
    assert assign_label(-255, cfg) == -1
    assert assign_label(-225, cfg) == 1
    assert assign_label(-15, cfg) == -1


def test_sensitivity_labeling_uses_horizon_and_shift():
    assert label_relative_minutes(-300, horizon_minutes=240) == 0
    assert label_relative_minutes(-240, horizon_minutes=240) == -1
    assert label_relative_minutes(-225, horizon_minutes=240) == 1
    assert label_relative_minutes(-120, horizon_minutes=240) == 1
    assert label_relative_minutes(-15, horizon_minutes=240) == -1

    shifted = shifted_relative_minutes(pd.Series([-120]), 15)
    assert float(shifted.iloc[0]) == -135.0


def test_relabeling_preserves_existing_beat_timestamps():
    cfg = {
        "normal_range": {"start": -60, "end": -30},
        "positive_range": {"start": -20, "end": -5},
        "ignore_ranges": [[-30, -20], [-5, float("inf")]],
    }
    df = pd.DataFrame(
        {
            "Beat_Idx": [0, 1, 2],
            "Absolute_Time": [
                "2025-01-01 00:00:00",
                "2025-01-01 00:10:00",
                "2025-01-01 00:40:00",
            ],
            "Wave_Start": ["2025-01-01 00:00:00"] * 3,
            "Wave_End": ["2025-01-01 01:00:00"] * 3,
            "Actual_Stroke_Time": ["2025-01-01 00:20:00"] * 3,
            "Is_Stroke_Subject": [1] * 3,
        }
    )
    out = rebuild_group_time_axis(
        g=df,
        beat_col="Beat_Idx",
        abs_time_col="Absolute_Time",
        wave_start_col="Wave_Start",
        wave_end_col="Wave_End",
        stroke_col="Actual_Stroke_Time",
        stroke_subject_col="Is_Stroke_Subject",
        labeling_cfg=cfg,
    )
    assert out is not None
    assert out["Time_Rel_Min"].tolist() == [-20.0, -10.0, 20.0]
    assert out["Label"].tolist() == [1, 1, -1]

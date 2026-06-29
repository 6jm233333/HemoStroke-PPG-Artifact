import numpy as np
import pandas as pd

from src.datasets.build_main_horizon_sets import (
    build_windows_for_dataframe,
    classify_relative_minutes,
    split_patient_ids,
)


def test_horizon_labels_expand_warning_and_shrink_reference_regions():
    rel = pd.Series([-470, -380, -370, -340, -20, -10])
    labels = classify_relative_minutes(
        rel,
        horizon_minutes=360,
        normal_start_minutes=-480,
        boundary_buffer_minutes=15,
        onset_buffer_minutes=15,
    )
    np.testing.assert_array_equal(labels, [0, 0, -1, 1, 1, -1])


def test_main_packaging_builds_non_overlapping_500_beat_windows():
    df = pd.DataFrame(
        {
            "Group_ID": ["patient-1"] * 1000,
            "Source_File": ["wave-1"] * 1000,
            "Time_Rel_Min": np.linspace(-220, -20, 1000),
            "feature": np.arange(1000, dtype=float),
        }
    )
    windows, labels, patient_ids, manifest, n_packaged, n_nan = build_windows_for_dataframe(
        df,
        fallback_file_id="unused",
        feature_cols=["feature"],
        horizon_minutes=240,
        normal_start_minutes=-480,
        boundary_buffer_minutes=15,
        onset_buffer_minutes=15,
        window_size=500,
        stride=500,
        max_gap_minutes=1.0,
    )
    assert len(windows) == 2
    assert labels == [1, 1]
    assert patient_ids == ["patient-1", "patient-1"]
    assert len(manifest) == 2
    assert n_packaged == 2
    assert n_nan == 0


def test_patient_split_map_is_disjoint_and_deterministic():
    patient_ids = [f"p{i}" for i in range(20)]
    split_a = split_patient_ids(patient_ids, seed=42, train_fraction=0.7, val_fraction=0.1)
    split_b = split_patient_ids(patient_ids, seed=42, train_fraction=0.7, val_fraction=0.1)
    assert split_a == split_b
    assert set(split_a) == set(patient_ids)
    assert {"train", "val", "test"} == set(split_a.values())


def test_main_packaging_prefers_subject_id_over_waveform_group_id():
    df = pd.DataFrame(
        {
            "SUBJECT_ID": ["patient-1"] * 500,
            "Group_ID": ["waveform-group-1"] * 500,
            "Source_File": ["wave-1"] * 500,
            "Time_Rel_Min": np.linspace(-220, -20, 500),
            "feature": np.arange(500, dtype=float),
        }
    )
    _, _, patient_ids, _, _, _ = build_windows_for_dataframe(
        df,
        fallback_file_id="unused",
        feature_cols=["feature"],
        horizon_minutes=240,
        normal_start_minutes=-480,
        boundary_buffer_minutes=15,
        onset_buffer_minutes=15,
        window_size=500,
        stride=500,
        max_gap_minutes=1.0,
    )
    assert patient_ids == ["patient-1"]

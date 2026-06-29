import numpy as np
import pandas as pd

from src.datasets.build_subgroup_sets import build_samples_for_one_file


def test_subgroup_packaging_preserves_frozen_upstream_feature_values(tmp_path):
    feature_values = np.linspace(10.0, 20.0, 500)
    input_csv = tmp_path / "control.csv"
    pd.DataFrame(
        {
            "feature": feature_values,
            "Label": np.zeros(500, dtype=int),
            "Time_Rel_Min": np.linspace(-400.0, -300.0, 500),
        }
    ).to_csv(input_csv, index=False)

    x_chunks, y_chunks, _, _ = build_samples_for_one_file(
        file_path=input_csv,
        feature_cols=["feature"],
        patient_id=1,
        window_size=500,
        stride=500,
        max_gap_minutes=1.0,
        warning_window_minutes=240.0,
    )
    assert y_chunks == [0]
    np.testing.assert_allclose(x_chunks[0][:, 0], feature_values)

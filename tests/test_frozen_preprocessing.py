import numpy as np
import pandas as pd

from src.features.engineer_features import engineer_kinematic_features


def test_relative_features_default_to_mean_baseline():
    df = pd.DataFrame({"x": [1.0, 2.0, 9.0]})
    out = engineer_kinematic_features(
        df,
        group_col=None,
        sort_cols=[],
        base_features=["x"],
    )
    np.testing.assert_allclose(out["x_Rel"], [-0.75, -0.5, 1.25])


def test_relative_features_use_absolute_baseline_denominator():
    df = pd.DataFrame({"x": [-2.0, -4.0]})
    out = engineer_kinematic_features(
        df,
        group_col=None,
        sort_cols=[],
        base_features=["x"],
        baseline_frac=1.0,
        baseline_min_rows=1,
    )
    np.testing.assert_allclose(out["x_Rel"], [1.0 / 3.0, -1.0 / 3.0])

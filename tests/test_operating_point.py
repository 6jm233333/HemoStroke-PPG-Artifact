import numpy as np
import pandas as pd

from scripts.reproduce.table4_false_alert_burden import summarize_false_alerts
from src.analysis.operating_point import apply_binary_threshold, max_consecutive_positives


def test_apply_binary_threshold_uses_configured_operating_point():
    preds = apply_binary_threshold(np.asarray([0.5, 0.7909, 0.7910, 0.95]), 0.7910)
    np.testing.assert_array_equal(preds, [0, 0, 1, 1])


def test_max_consecutive_positives_resets_after_negative():
    assert max_consecutive_positives([1, 1, 0, 1, 1, 1, 0]) == 3


def test_false_alert_summary_uses_five_window_id_rule_and_nan_filter():
    predictions = pd.DataFrame(
        {
            "pid": ["a"] * 6 + ["b"] * 5,
            "window_index": list(range(6)) + list(range(5)),
            "y_prob": [0.9] * 6 + [0.9, 0.9, 0.9, 0.9, np.nan],
        }
    )
    summary = summarize_false_alerts(
        predictions,
        identifier_col="pid",
        score_col="y_prob",
        order_col="window_index",
        threshold=0.7910,
        min_consecutive_windows=5,
    ).iloc[0]
    assert summary["n_packaged"] == 11
    assert summary["n_windows"] == 10
    assert summary["n_identifiers"] == 2
    assert summary["id_positive_fraction"] == 0.5

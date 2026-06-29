import numpy as np
import pytest

pytest.importorskip("torch")
from src.models.train import filter_labels


def test_filter_labels_keeps_patient_ids_aligned():
    x = np.arange(5 * 2 * 3, dtype=np.float32).reshape(5, 2, 3)
    y = np.array([0, 1, -1, 1, 2])
    pid = np.array(["a", "b", "c", "d", "e"])

    x_out, y_out, pid_out = filter_labels(
        x,
        y,
        pid,
        positive_label=1,
        negative_label=0,
        ignore_label_value=-1,
        drop_ignore_label=True,
    )

    assert y_out.tolist() == [0, 1, 1]
    assert pid_out.tolist() == ["a", "b", "d"]
    assert x_out.shape == (3, 2, 3)

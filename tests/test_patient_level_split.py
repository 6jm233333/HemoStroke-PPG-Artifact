import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def test_patient_level_split_prevents_window_leakage():
    patient_ids = np.repeat(np.arange(20), 5)
    labels = np.tile([0, 0, 1, 1, 0], 20)
    x = np.zeros((len(labels), 500, 17))

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in splitter.split(x, labels, groups=patient_ids):
        train_patients = set(patient_ids[train_idx])
        test_patients = set(patient_ids[test_idx])
        assert train_patients.isdisjoint(test_patients)

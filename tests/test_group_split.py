import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def test_stratified_group_kfold_keeps_groups_disjoint():
    groups = np.repeat(np.arange(12), 4)
    y = np.tile([0, 0, 1, 1], 12)
    x = np.zeros((len(y), 2))

    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    for train_idx, test_idx in splitter.split(x, y, groups=groups):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups)

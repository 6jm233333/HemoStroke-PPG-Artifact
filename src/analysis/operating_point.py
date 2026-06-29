from __future__ import annotations

from typing import Iterable

import numpy as np


def validate_threshold(threshold: float) -> float:
    value = float(threshold)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Operating threshold must be between 0 and 1, got {value}.")
    return value


def apply_binary_threshold(prob_pos: np.ndarray, threshold: float) -> np.ndarray:
    scores = np.asarray(prob_pos, dtype=float)
    value = validate_threshold(threshold)
    return (scores >= value).astype(np.int64)


def max_consecutive_positives(values: Iterable[int | bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest

from __future__ import annotations

import numpy as np


def confusion_from_predictions(truth: np.ndarray, prediction: np.ndarray, classes: int = 11) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.int64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)
    if truth.shape != prediction.shape:
        raise ValueError("Truth and prediction shapes differ")
    if truth.size == 0:
        raise ValueError("Cannot build a confusion matrix from zero pixels")
    if truth.min() < 0 or truth.max() >= classes or prediction.min() < 0 or prediction.max() >= classes:
        raise ValueError("Class index outside the configured range")
    return np.bincount(truth * classes + prediction, minlength=classes * classes).reshape(classes, classes)


def _maximum_assignment(score: np.ndarray) -> list[int]:
    """Exact deterministic row-to-column assignment using bitmask DP."""
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 2 or score.shape[0] != score.shape[1]:
        raise ValueError("Assignment score matrix must be square")
    n = int(score.shape[0])
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for row in range(n):
        updated: dict[int, tuple[float, tuple[int, ...]]] = {}
        for mask, (value, assignment) in states.items():
            for column in range(n):
                if mask & (1 << column):
                    continue
                new_mask = mask | (1 << column)
                candidate = (value + float(score[row, column]), assignment + (column,))
                current = updated.get(new_mask)
                if current is None or candidate[0] > current[0] + 1e-15 or (
                    abs(candidate[0] - current[0]) <= 1e-15 and candidate[1] < current[1]
                ):
                    updated[new_mask] = candidate
        states = updated
    return list(states[(1 << n) - 1][1])


def learn_class_permutation(confusion: np.ndarray) -> dict[str, object]:
    """Learn true-class -> raw-output-channel mapping from train-only pixels.

    Rows are normalized before assignment so common classes cannot dominate the
    mapping. The returned channel order can directly index raw probabilities to
    obtain probabilities in canonical true-class order.
    """
    confusion = np.asarray(confusion, dtype=np.int64)
    support = confusion.sum(axis=1)
    if np.any(support <= 0):
        missing = (np.flatnonzero(support <= 0) + 1).tolist()
        raise ValueError(f"Permutation panel is missing true classes: {missing}")
    normalized = confusion / support[:, None]
    true_to_output = _maximum_assignment(normalized)
    output_to_true = [0] * len(true_to_output)
    for true_class, output_channel in enumerate(true_to_output):
        output_to_true[output_channel] = true_class
    score = float(np.mean([normalized[row, column] for row, column in enumerate(true_to_output)]))
    return {
        "true_class_to_output_channel": true_to_output,
        "output_channel_to_true_class": output_to_true,
        "output_channel_to_class_code": [value + 1 for value in output_to_true],
        "train_panel_macro_recall": score,
        "train_panel_support": support.astype(int).tolist(),
        "train_panel_confusion_raw_channels": confusion.astype(int).tolist(),
        "assignment_method": "exact maximum row-normalized recall; train-only panel",
    }


def align_probabilities(probabilities: np.ndarray, permutation: dict[str, object]) -> np.ndarray:
    order = np.asarray(permutation["true_class_to_output_channel"], dtype=np.int64)
    if probabilities.shape[0] != len(order):
        raise ValueError("Probability channel count and permutation size differ")
    return probabilities[order]

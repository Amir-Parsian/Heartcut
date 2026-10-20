from __future__ import annotations

import numpy as np
import pytest

from heartcut.eval.metrics import (
    INVALID,
    accuracy,
    confused_pairs,
    expected_calibration_error,
    macro_f1,
    mean_std,
    per_class_report,
    reliability_table,
    score,
)


def test_accuracy_and_invalid_predictions() -> None:
    assert accuracy(["a", "b", "c", "a"], ["a", "b", INVALID, "b"]) == 0.5
    with pytest.raises(ValueError):
        accuracy([], [])


def test_macro_f1_counts_invalid_as_a_miss() -> None:
    labels = ["a", "b"]
    perfect = macro_f1(["a", "b"], ["a", "b"], labels)
    with_invalid = macro_f1(["a", "b"], ["a", INVALID], labels)
    assert perfect == 1.0
    assert with_invalid < 1.0


def test_ece_known_values() -> None:
    # Perfectly calibrated: confidence 1 and always right.
    assert expected_calibration_error([1.0, 1.0], [True, True]) == 0.0
    # Always 0.9 confident, right half the time: ECE = |0.5 - 0.9| = 0.4
    assert expected_calibration_error([0.9] * 4, [True, False, True, False]) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [True, False])


def test_score_bundle() -> None:
    s = score(["a", "b", "a"], ["a", "a", "a"], ["a", "b"], confidence=np.array([0.9, 0.6, 0.8]))
    assert s.n == 3
    assert s.accuracy == pytest.approx(2 / 3)
    assert s.ece is not None
    assert score(["a"], ["a"], ["a"]).ece is None
    assert set(s.to_dict()) == {"n", "accuracy", "macro_f1", "ece"}


def test_per_class_and_confusions() -> None:
    y_true = ["a", "a", "b", "b", "c"]
    y_pred = ["a", "b", "a", "a", "c"]
    report = per_class_report(y_true, y_pred, ["a", "b", "c"])
    assert report.set_index("label").loc["c", "f1"] == 1.0
    assert confused_pairs(y_true, y_pred) == [("b", "a", 2), ("a", "b", 1)]
    assert confused_pairs(y_true, y_pred, top_k=1) == [("b", "a", 2)]


def test_reliability_table_shape() -> None:
    table = reliability_table([0.05, 0.95, 0.96], [False, True, True], n_bins=10)
    assert len(table) == 10
    assert table["count"].sum() == 3


def test_mean_std() -> None:
    assert mean_std([1.0]) == (1.0, 0.0)
    mean, std = mean_std([1.0, 2.0, 3.0])
    assert mean == 2.0 and std == pytest.approx(1.0)
    with pytest.raises(ValueError):
        mean_std([])

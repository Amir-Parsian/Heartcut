from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from heartcut.config import TfidfStudentConfig
from heartcut.students import Student, TrainingData, build_student
from heartcut.students.base import expand_soft_targets, full_proba
from heartcut.students.registry import student_class
from tests.synthetic import INTENTS, generate

LABELS = sorted(INTENTS)


def _data(n: int = 20, soft: bool = False) -> TrainingData:
    rows = generate(n, seed=5)
    idx = {label: i for i, label in enumerate(LABELS)}
    targets = np.array([idx[r["intent"]] for r in rows])
    soft_targets = None
    if soft:
        soft_targets = np.full((len(rows), len(LABELS)), 0.02)
        soft_targets[np.arange(len(rows)), targets] = 1 - 0.02 * (len(LABELS) - 1)
    val = generate(3, seed=6)
    return TrainingData(
        texts=[r["text"] for r in rows],
        targets=targets,
        val_texts=[r["text"] for r in val],
        val_targets=np.array([idx[r["intent"]] for r in val]),
        soft_targets=soft_targets,
    )


@pytest.mark.parametrize("soft", [False, True])
def test_tfidf_learns_and_roundtrips(tmp_path: Path, soft: bool) -> None:
    student = build_student(TfidfStudentConfig(name="tfidf"), LABELS)
    info = student.fit(_data(soft=soft), seed=0)
    assert info["n_features"] > 0
    test = generate(5, seed=9)
    labels, confidence = student.predict([r["text"] for r in test])
    accuracy = np.mean([p == r["intent"] for p, r in zip(labels, test, strict=True)])
    assert accuracy > 0.8
    assert ((confidence > 0) & (confidence <= 1)).all()

    student.save(tmp_path / "model")
    loaded = Student.load(tmp_path / "model")
    np.testing.assert_allclose(
        loaded.predict_proba(["where is my card"]), student.predict_proba(["where is my card"])
    )


def test_missing_classes_get_zero_probability() -> None:
    data = _data()
    keep = data.targets != 0
    partial = TrainingData(
        texts=[t for t, k in zip(data.texts, keep, strict=True) if k],
        targets=data.targets[keep],
        val_texts=data.val_texts,
        val_targets=data.val_targets,
    )
    student = build_student(TfidfStudentConfig(name="tfidf"), LABELS)
    student.fit(partial, seed=0)
    proba = student.predict_proba(["anything"])
    assert proba.shape == (1, len(LABELS))
    assert proba[0, 0] == 0.0
    assert proba.sum() == pytest.approx(1.0)


def test_untrained_student_raises() -> None:
    with pytest.raises(RuntimeError):
        build_student(TfidfStudentConfig(name="t"), LABELS).predict_proba(["x"])


def test_soft_target_expansion_is_exact() -> None:
    soft = np.array([[0.75, 0.25, 0.0], [0.0, 0.0, 1.0]])
    rows, classes, weights = expand_soft_targets(soft)
    assert rows.tolist() == [0, 0, 1]
    assert classes.tolist() == [0, 1, 2]
    np.testing.assert_allclose(weights, [0.75, 0.25, 1.0])


def test_full_proba_places_columns() -> None:
    out = full_proba(np.array([[0.4, 0.6]]), np.array([0, 2]), 3)
    np.testing.assert_allclose(out, [[0.4, 0.0, 0.6]])


def test_training_data_validation() -> None:
    with pytest.raises(ValueError):
        TrainingData(["a"], np.array([0, 1]), [], np.array([]))
    with pytest.raises(ValueError):
        TrainingData(["a"], np.array([0]), [], np.array([]), soft_targets=np.zeros((2, 3)))


def test_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown student kind"):
        student_class("gbdt")

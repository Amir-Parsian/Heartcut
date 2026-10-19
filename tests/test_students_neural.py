"""SetFit-style and encoder students, trained on tiny local models (no downloads)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

import torch

from distillkit.config import EncoderStudentConfig, SetFitStudentConfig
from distillkit.students import Student, TrainingData, build_student
from distillkit.students.setfit import make_pairs
from distillkit.students.torch_utils import distillation_loss
from tests.synthetic import INTENTS, generate

LABELS = sorted(INTENTS)


def _data(soft: bool = False) -> TrainingData:
    idx = {label: i for i, label in enumerate(LABELS)}
    rows, val = generate(12, seed=5), generate(3, seed=6)
    targets = np.array([idx[r["intent"]] for r in rows])
    soft_targets = None
    if soft:
        soft_targets = np.full((len(rows), len(LABELS)), 0.1 / (len(LABELS) - 1))
        soft_targets[np.arange(len(rows)), targets] = 0.9
    return TrainingData(
        texts=[r["text"] for r in rows],
        targets=targets,
        val_texts=[r["text"] for r in val],
        val_targets=np.array([idx[r["intent"]] for r in val]),
        soft_targets=soft_targets,
    )


def test_make_pairs_is_balanced() -> None:
    targets = np.array([0, 0, 1, 1, 2])
    a, b, sim = make_pairs(targets, 100, np.random.default_rng(0))
    assert len(a) == 100
    assert sim.mean() == pytest.approx(0.5)
    assert (targets[a[sim == 1]] == targets[b[sim == 1]]).all()
    assert (targets[a[sim == 0]] != targets[b[sim == 0]]).all()


def test_make_pairs_without_positive_candidates() -> None:
    a, _, sim = make_pairs(np.array([0, 1, 2]), 10, np.random.default_rng(0))
    assert len(a) == 10 and sim.sum() == 0


def test_distillation_loss() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0]])
    hard = torch.tensor([0])
    ce = torch.nn.functional.cross_entropy(logits, hard)
    assert torch.isclose(distillation_loss(logits, hard, None, alpha=0.5, temperature=2), ce)
    # A teacher distribution equal to the student's own (tempered) prediction adds no KL.
    own = torch.softmax(logits, dim=-1)
    mixed = distillation_loss(logits, hard, own, alpha=0.5, temperature=1.0)
    assert torch.isclose(mixed, 0.5 * ce, atol=1e-5)
    disagree = distillation_loss(
        logits, hard, torch.tensor([[0.0, 0.0, 1.0]]), alpha=0.5, temperature=1.0
    )
    assert disagree > mixed


@pytest.mark.slow
@pytest.mark.parametrize("soft", [False, True])
def test_setfit_student_trains_saves_and_loads(tiny_st: Path, tmp_path: Path, soft: bool) -> None:
    config = SetFitStudentConfig(
        name="setfit",
        params={"model_name": str(tiny_st), "max_pairs": 64, "contrastive_batch_size": 16},
    )
    student = build_student(config, LABELS)
    info = student.fit(_data(soft=soft), seed=0)
    assert info["contrastive_steps"] > 0
    texts = ["where is my new card", "please cancel my transfer"]
    proba = student.predict_proba(texts)
    assert proba.shape == (2, len(LABELS))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)

    student.save(tmp_path / "setfit")
    loaded = Student.load(tmp_path / "setfit")
    np.testing.assert_allclose(loaded.predict_proba(texts), proba, rtol=1e-4, atol=1e-5)


@pytest.mark.slow
def test_setfit_without_contrastive_step(tiny_st: Path) -> None:
    config = SetFitStudentConfig(
        name="setfit", params={"model_name": str(tiny_st), "contrastive_epochs": 0}
    )
    info = build_student(config, LABELS).fit(_data(), seed=0)
    assert info["contrastive_steps"] == 0


@pytest.mark.slow
@pytest.mark.parametrize("soft", [False, True])
def test_encoder_student_trains_saves_and_loads(
    tiny_bert: Path, tmp_path: Path, soft: bool
) -> None:
    config = EncoderStudentConfig(
        name="encoder",
        use_soft_labels=soft,
        params={
            "model_name": str(tiny_bert),
            "epochs": 3,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "early_stopping_patience": 1,
        },
    )
    student = build_student(config, LABELS)
    info = student.fit(_data(soft=soft), seed=0)
    assert 1 <= info["epochs_run"] <= 3
    assert info["best_epoch"] >= 1
    texts = ["where is my new card", "please cancel my transfer"]
    proba = student.predict_proba(texts)
    assert proba.shape == (2, len(LABELS))

    student.save(tmp_path / "encoder")
    loaded = Student.load(tmp_path / "encoder")
    np.testing.assert_allclose(loaded.predict_proba(texts), proba, rtol=1e-4, atol=1e-5)
    assert loaded.predict_proba([]).shape == (0, len(LABELS))


@pytest.mark.slow
def test_encoder_student_actually_learns(tiny_bert: Path) -> None:
    config = EncoderStudentConfig(
        name="encoder",
        params={
            "model_name": str(tiny_bert),
            "epochs": 20,
            "batch_size": 16,
            "learning_rate": 2e-3,
            "early_stopping_patience": 20,
        },
    )
    student = build_student(config, LABELS)
    idx = {label: i for i, label in enumerate(LABELS)}
    rows = generate(25, seed=1)
    student.fit(
        TrainingData(
            [r["text"] for r in rows],
            np.array([idx[r["intent"]] for r in rows]),
            [r["text"] for r in rows[:20]],
            np.array([idx[r["intent"]] for r in rows[:20]]),
        ),
        seed=0,
    )
    test = generate(10, seed=3)
    predicted, _ = student.predict([r["text"] for r in test])
    accuracy = np.mean([p == r["intent"] for p, r in zip(predicted, test, strict=True)])
    assert accuracy > 0.8


def test_untrained_neural_students_raise() -> None:
    for config in (SetFitStudentConfig(name="s"), EncoderStudentConfig(name="e")):
        with pytest.raises(RuntimeError):
            build_student(config, LABELS).predict_proba(["x"])

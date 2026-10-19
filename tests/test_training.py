from __future__ import annotations

import pytest

import distillkit.data.prepare as prepare_module
import distillkit.eval.audit as audit_module
import distillkit.teachers.labeling as labeling_module
from distillkit.data.prepare import prepare_data
from distillkit.students import Student
from distillkit.students.training import (
    PREDICTIONS_FILE,
    load_trained_student,
    run_training,
    select_budget,
    student_run_dir,
)
from distillkit.teachers.labeling import load_teacher_labels, run_labeling


def _ready(cfg):  # type: ignore[no-untyped-def]
    prepare_data(cfg)
    run_labeling(cfg)
    return cfg


def test_training_never_reads_hidden_gold_labels(make_experiment, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg = _ready(make_experiment())

    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("training must not read the hidden human labels")

    for module in (prepare_module, audit_module, labeling_module):
        monkeypatch.setattr(module, "load_hidden_gold", forbidden)
    results = run_training(cfg)
    assert len(results) == 4  # 1 student x 2 budgets x 2 seeds


def test_training_records_metrics_and_predictions(make_experiment) -> None:  # type: ignore[no-untyped-def]
    cfg = _ready(make_experiment())
    results = run_training(cfg)
    by_budget = {(r["budget"], r["seed"]): r for r in results}
    small = by_budget[("60", 0)]
    assert small["budget_examples"] == 60
    assert small["train_examples"] <= 60
    assert small["labeling_cost_usd"] > 0
    full = by_budget[("full", 0)]
    assert full["budget_examples"] == 160
    assert 0 <= full["test"]["accuracy"] <= 1
    assert full["test"]["ece"] is not None
    directory = student_run_dir(cfg, "tfidf", "full", 0)
    assert (directory / PREDICTIONS_FILE).exists()
    student = load_trained_student(cfg, "tfidf", "full", 0)
    assert isinstance(student, Student)


def test_training_is_cached(make_experiment) -> None:  # type: ignore[no-untyped-def]
    cfg = _ready(make_experiment())
    first = run_training(cfg, budgets=["full"], seeds=[0])
    second = run_training(cfg, budgets=["full"], seeds=[0])
    assert first == second


def test_confidence_filter_reduces_training_set(make_experiment) -> None:  # type: ignore[no-untyped-def]
    cfg = _ready(make_experiment(label_filter={"min_confidence": 0.99}))
    result = run_training(cfg, budgets=["full"], seeds=[0])[0]
    assert result["dropped_examples"] > 0
    assert result["train_examples"] + result["dropped_examples"] == result["budget_examples"]


def test_soft_labels_are_used_when_available(make_experiment) -> None:  # type: ignore[no-untyped-def]
    students = [{"name": "soft", "kind": "tfidf", "use_soft_labels": True}]
    cfg = _ready(make_experiment(students=students))
    assert run_training(cfg, budgets=["full"], seeds=[0])[0]["soft_labels"]


def test_soft_labels_fall_back_with_single_sample(make_experiment) -> None:  # type: ignore[no-untyped-def]
    base = make_experiment()
    teacher = base.teacher.model_copy(update={"self_consistency": 1})
    students = [{"name": "soft", "kind": "tfidf", "use_soft_labels": True}]
    cfg = _ready(make_experiment(students=students).model_copy(update={"teacher": teacher}))
    assert not run_training(cfg, budgets=["full"], seeds=[0])[0]["soft_labels"]


def test_budgets_are_nested(make_experiment) -> None:  # type: ignore[no-untyped-def]
    cfg = _ready(make_experiment())
    train = load_teacher_labels(cfg, "train")
    small = set(select_budget(train, 30, seed=1)["id"])
    large = set(select_budget(train, 90, seed=1)["id"])
    assert small < large
    assert len(select_budget(train, "full", seed=1)) == len(train)
    assert len(select_budget(train, 10_000, seed=1)) == len(train)


def test_unknown_student_name(make_experiment) -> None:  # type: ignore[no-untyped-def]
    cfg = _ready(make_experiment())
    with pytest.raises(KeyError):
        run_training(cfg, students=["nope"])

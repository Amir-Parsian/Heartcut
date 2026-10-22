from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

import heartcut.config as config_module
from heartcut.config import (
    ExperimentConfig,
    TeacherConfig,
    apply_overrides,
    load_experiment,
    load_pricing_table,
    load_teacher,
)
from tests.conftest import PROMPT, REPO

EXPERIMENTS = sorted((REPO / "configs" / "experiments").glob("*.yaml"))
TEACHERS = sorted((REPO / "configs" / "teachers").glob("*.yaml"))


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_shipped_experiment_configs_validate(path: Path) -> None:
    cfg = load_experiment(path)
    assert cfg.students
    assert Path(cfg.teacher.prompt).exists()


@pytest.mark.parametrize("path", TEACHERS, ids=lambda p: p.stem)
def test_shipped_teacher_configs_validate(path: Path) -> None:
    teacher = load_teacher(path)
    assert Path(teacher.prompt).exists()


def test_banking77_references_are_inlined(banking77_config: ExperimentConfig) -> None:
    assert banking77_config.dataset.name == "banking77"
    assert [s.kind for s in banking77_config.students] == ["tfidf", "setfit", "encoder"]
    assert banking77_config.budgets == [500, 2000, 5000, "full"]


def test_overrides_reach_nested_values_and_named_list_items() -> None:
    cfg = load_experiment(
        REPO / "configs/experiments/banking77.yaml",
        [
            "dataset.max_examples.train=200",
            "students.tfidf-logreg.params.C=3",
            "seeds=[7]",
        ],
    )
    assert cfg.dataset.max_examples == {"train": 200}
    assert cfg.student("tfidf-logreg").params.C == 3  # type: ignore[union-attr]
    assert cfg.seeds == [7]


def test_override_can_swap_a_referenced_file() -> None:
    cfg = load_experiment(
        REPO / "configs/experiments/banking77.yaml",
        ["teacher=configs/teachers/simulated.yaml"],
    )
    assert cfg.teacher.backend == "simulated"


def test_override_can_edit_a_swapped_in_teacher() -> None:
    cfg = load_experiment(
        REPO / "configs/experiments/banking77.yaml",
        ["teacher=configs/teachers/claude-sonnet-5-5.yaml", "teacher.max_budget_usd=19"],
    )
    assert cfg.teacher.model == "claude-sonnet-5-5"
    assert cfg.teacher.max_budget_usd == 19
    # The prompt path still resolves relative to the swapped-in teacher file.
    assert Path(cfg.teacher.prompt).resolve() == PROMPT.resolve()


def test_override_can_edit_a_swapped_in_student_by_name() -> None:
    cfg = load_experiment(
        REPO / "configs/experiments/banking77.yaml",
        [
            "students=[configs/students/tfidf-logreg.yaml,"
            " configs/students/encoder-modernbert.yaml]",
            "students.encoder-modernbert.params.epochs=1",
        ],
    )
    assert [s.name for s in cfg.students] == ["tfidf-logreg", "encoder-modernbert"]
    params = cfg.student("encoder-modernbert").params
    assert params.epochs == 1  # type: ignore[union-attr]
    assert params.model_name == "answerdotai/ModernBERT-base"  # type: ignore[union-attr]


def test_a_later_file_swap_replaces_earlier_field_overrides() -> None:
    cfg = load_experiment(
        REPO / "configs/experiments/banking77.yaml",
        ["teacher.max_budget_usd=19", "teacher=configs/teachers/claude-sonnet-5-5.yaml"],
    )
    assert cfg.teacher.model == "claude-sonnet-5-5"
    assert cfg.teacher.max_budget_usd == 50


def test_override_syntax_errors() -> None:
    with pytest.raises(ValueError, match=r"key\.path=value"):
        apply_overrides({}, ["no-equals-sign"])
    with pytest.raises(KeyError):
        apply_overrides({"students": [{"name": "a"}]}, ["students.b.x=1"])


def test_unknown_keys_are_rejected(make_experiment) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValidationError):
        make_experiment(unexpected_key=1)


def test_duplicate_student_names_are_rejected(make_experiment) -> None:  # type: ignore[no-untyped-def]
    students = [{"name": "x", "kind": "tfidf"}, {"name": "x", "kind": "tfidf"}]
    with pytest.raises(ValidationError, match="duplicate"):
        make_experiment(students=students)


def test_budgets_must_be_positive(make_experiment) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValidationError):
        make_experiment(budgets=[0])


def test_teacher_identity_ignores_runtime_knobs() -> None:
    base = {"name": "t", "backend": "simulated", "model": "m", "prompt": "p.yaml"}
    a = TeacherConfig.model_validate(base)
    b = TeacherConfig.model_validate({**base, "concurrency": 64, "max_budget_usd": 5})
    c = TeacherConfig.model_validate({**base, "self_consistency": 3})
    assert a.identity() == b.identity()
    assert a.identity() != c.identity()


def test_unknown_student_lookup(banking77_config: ExperimentConfig) -> None:
    with pytest.raises(KeyError):
        banking77_config.student("nope")


def test_pricing_table_has_dates() -> None:
    table = load_pricing_table(REPO / "configs" / "pricing.yaml")
    assert "claude-haiku-5-5" in table
    assert table["claude-haiku-5-5"].as_of
    assert table["claude-opus-5-5"].output > table["claude-opus-5-5"].input


def test_installed_package_uses_its_packaged_price_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A wheel install: no configs/ in the working directory or above the package, only the
    # pricing.yaml that the wheel build copies next to config.py.
    package = tmp_path / "site-packages" / "heartcut"
    package.mkdir(parents=True)
    shutil.copy(REPO / "configs" / "pricing.yaml", package / "pricing.yaml")
    monkeypatch.setattr(config_module, "__file__", str(package / "config.py"))
    monkeypatch.chdir(tmp_path)
    assert load_pricing_table() == load_pricing_table(REPO / "configs" / "pricing.yaml")

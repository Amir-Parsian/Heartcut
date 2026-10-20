from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from tests.conftest import PROMPT, REPO
from tests.test_serve import LABELS, FixedPredictor, _router


def _demo() -> ModuleType:
    spec = importlib.util.spec_from_file_location("demo_app", REPO / "demo" / "app.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_without_teacher() -> None:
    demo = _demo()
    student, teacher = demo.compare("where is my card", FixedPredictor(), None, 0.002)
    assert LABELS[0] in student and "$0.000002" in student
    assert "disabled" in teacher
    assert demo.compare("   ", FixedPredictor(), None, None)[0] == "Type a message first."


def test_compare_with_teacher() -> None:
    demo = _demo()
    router = _router(threshold=0.5, answer=LABELS[0])
    student, teacher = demo.compare("x", FixedPredictor(), router.teacher, None)
    assert "no benchmark results" in student
    assert "agrees" in teacher and "250 ms" in teacher


def test_student_cost_from_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _demo()
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"models": [{"name": "fixed", "cost_per_1k_usd": 0.004}]}))
    monkeypatch.setenv("DISTILLKIT_RESULTS", str(results))
    assert demo.student_cost_per_1k(FixedPredictor()) == 0.004
    monkeypatch.setenv("DISTILLKIT_RESULTS", str(tmp_path / "missing.json"))
    assert demo.student_cost_per_1k(FixedPredictor()) is None


def test_teacher_is_disabled_without_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _demo()
    config = tmp_path / "teacher.yaml"
    config.write_text(
        f"name: t\nbackend: anthropic\nmodel: m\napi_key_env: DEMO_KEY\nprompt: {PROMPT}\n"
    )
    monkeypatch.delenv("DISTILLKIT_TEACHER_CONFIG", raising=False)
    assert demo.load_teacher_classifier(FixedPredictor()) is None
    monkeypatch.setenv("DISTILLKIT_TEACHER_CONFIG", str(config))
    monkeypatch.delenv("DEMO_KEY", raising=False)
    assert demo.load_teacher_classifier(FixedPredictor()) is None
    monkeypatch.setenv("DEMO_KEY", "k")
    assert demo.load_teacher_classifier(FixedPredictor()) is not None


def test_build_demo_constructs_the_ui() -> None:
    pytest.importorskip("gradio")
    demo = _demo()
    blocks = demo.build_demo(FixedPredictor(), None, None)
    assert blocks is not None


def test_load_student_needs_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _demo()
    monkeypatch.delenv("DISTILLKIT_MODEL_DIR", raising=False)
    monkeypatch.delenv("DISTILLKIT_MODEL_REPO", raising=False)
    with pytest.raises(RuntimeError):
        demo.load_student()

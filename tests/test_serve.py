from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from fastapi.testclient import TestClient

from heartcut.config import TeacherConfig, TfidfStudentConfig, TokenPricing
from heartcut.serve.app import Service, create_app, service_from_env
from heartcut.serve.export import export_student
from heartcut.serve.router import FallbackRouter, TeacherClassifier
from heartcut.students import TrainingData, build_student
from heartcut.teachers.base import Completion, FatalTeacherError, TeacherBackend
from heartcut.teachers.prompts import ChatPrompt
from tests.conftest import PROMPT
from tests.synthetic import INTENTS, generate

LABELS = sorted(INTENTS)


class FixedPredictor:
    """Predicts label 0 with a chosen confidence."""

    name = "fixed"
    meta: ClassVar[dict[str, Any]] = {"kind": "tfidf", "format": "sklearn", "task": "t"}

    def __init__(self, confidence: float = 0.9) -> None:
        self.labels = LABELS
        self.confidence = confidence

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        rest = (1 - self.confidence) / (len(LABELS) - 1)
        out = np.full((len(texts), len(LABELS)), rest)
        out[:, 0] = self.confidence
        return out


class EchoTeacher(TeacherBackend):
    def __init__(self, config: TeacherConfig, answer: str, fail: bool = False) -> None:
        super().__init__(config)
        self.answer = answer
        self.fail = fail
        self.calls = 0

    async def complete(
        self, prompt: ChatPrompt, schema: dict[str, Any] | None, sample_index: int = 0
    ) -> Completion:
        self.calls += 1
        if self.fail:
            raise FatalTeacherError("down")
        return Completion(f'{{"label": "{self.answer}"}}', 1000, 10, latency_ms=250.0)


def _teacher_config() -> TeacherConfig:
    return TeacherConfig.model_validate(
        {"name": "echo", "backend": "openai", "model": "m", "prompt": str(PROMPT)}
    )


def _router(threshold: float, answer: str = "lost_card", fail: bool = False) -> FallbackRouter:
    config = _teacher_config()
    backend = EchoTeacher(config, answer, fail)
    teacher = TeacherClassifier(
        config, LABELS, "t", backend=backend, pricing=TokenPricing(input=1.0, output=1.0)
    )
    return FallbackRouter(threshold, teacher)


def test_health_and_single_prediction() -> None:
    with TestClient(create_app(Service(FixedPredictor()))) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["labels"] == len(LABELS)
        assert health["fallback"] is None
        body = client.post("/predict", json={"text": "where is my card"}).json()
        assert body["label"] == LABELS[0]
        assert body["confidence"] == pytest.approx(0.9)
        assert body["latency_ms"] >= 0 and body["source"] == "student"


def test_batch_prediction() -> None:
    with TestClient(create_app(Service(FixedPredictor()))) as client:
        body = client.post("/predict", json={"texts": ["a", "b", "c"]}).json()
        assert len(body["predictions"]) == 3
        assert body["latency_ms"] >= 0


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": "a", "texts": ["b"]},
        {"text": ""},
        {"texts": []},
        {"texts": [""]},
        {"text": "x" * 5000},
        {"texts": ["a"] * 300},
    ],
)
def test_invalid_requests_are_rejected(payload: dict[str, Any]) -> None:
    with TestClient(create_app(Service(FixedPredictor()))) as client:
        assert client.post("/predict", json=payload).status_code == 422


def test_router_sends_low_confidence_to_teacher() -> None:
    router = _router(threshold=0.5)
    with TestClient(create_app(Service(FixedPredictor(confidence=0.3), router))) as client:
        assert client.get("/health").json()["fallback"] == {"threshold": 0.5, "teacher": "echo"}
        body = client.post("/predict", json={"text": "my card was stolen"}).json()
    assert body["source"] == "teacher"
    assert body["label"] == "lost_card" and body["student_label"] == LABELS[0]
    assert body["latency_ms"] >= 250.0
    assert body["teacher_cost_usd"] == pytest.approx(1010 / 1e6)


def test_router_keeps_confident_predictions() -> None:
    router = _router(threshold=0.5)
    with TestClient(create_app(Service(FixedPredictor(confidence=0.9), router))) as client:
        body = client.post("/predict", json={"text": "x"}).json()
    assert body["source"] == "student"
    assert router.teacher.backend.calls == 0  # type: ignore[attr-defined]


def test_router_falls_back_to_student_when_teacher_fails() -> None:
    router = _router(threshold=0.5, fail=True)
    with TestClient(create_app(Service(FixedPredictor(confidence=0.3), router))) as client:
        body = client.post("/predict", json={"text": "x"}).json()
    assert body["source"] == "student" and body["label"] == LABELS[0]


async def test_teacher_classifier_rejects_unknown_labels() -> None:
    router = _router(threshold=0.5, answer="not_a_label")
    answer = await router.teacher.classify("x")
    assert answer.label is None and answer.error


def test_service_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    idx = {label: i for i, label in enumerate(LABELS)}
    rows = generate(5, seed=1)
    student = build_student(TfidfStudentConfig(name="tfidf"), LABELS)
    student.fit(
        TrainingData(
            [r["text"] for r in rows], np.array([idx[r["intent"]] for r in rows]), [], np.array([])
        ),
        seed=0,
    )
    export_student(student, tmp_path / "model", ["x"], extra_meta={"task": "banking"})
    monkeypatch.delenv("HEARTCUT_FALLBACK_THRESHOLD", raising=False)
    monkeypatch.delenv("HEARTCUT_MODEL_DIR", raising=False)
    with pytest.raises(RuntimeError, match="HEARTCUT_MODEL_DIR"):
        service_from_env()
    monkeypatch.setenv("HEARTCUT_MODEL_DIR", str(tmp_path / "model"))
    monkeypatch.setenv("HEARTCUT_THREADS", "1")
    service = service_from_env()
    assert service.router is None and service.predictor.name == "tfidf"

    monkeypatch.setenv("HEARTCUT_FALLBACK_THRESHOLD", "0.4")
    with pytest.raises(RuntimeError, match="TEACHER_CONFIG"):
        service_from_env()
    teacher_file = tmp_path / "teacher.yaml"
    teacher_file.write_text(
        f"name: local\nbackend: openai\nmodel: m\nbase_url: http://localhost:9/v1\n"
        f"prompt: {PROMPT}\nis_local: true\n"
    )
    monkeypatch.setenv("HEARTCUT_TEACHER_CONFIG", str(teacher_file))
    service = service_from_env()
    assert service.router is not None and service.router.threshold == 0.4
    assert service.router.teacher.task == "banking"

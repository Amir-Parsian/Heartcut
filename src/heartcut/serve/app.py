"""FastAPI service for an exported student.

Endpoints:

* ``GET /health``   model name, kind, format, label count, fallback settings
* ``POST /predict`` ``{"text": "..."}`` -> one prediction, or
                    ``{"texts": ["...", ...]}`` -> a batch

Each prediction carries the label, the student's confidence and the latency. With the
fallback router enabled, predictions below the threshold are answered by the teacher and
marked ``"source": "teacher"``.

Configuration (environment variables, see ``.env.example``)::

    HEARTCUT_MODEL_DIR            exported model directory (required)
    HEARTCUT_THREADS              ONNX Runtime intra-op threads (default: all cores)
    HEARTCUT_FALLBACK_THRESHOLD   enable the router below this confidence
    HEARTCUT_TEACHER_CONFIG       teacher YAML used by the router
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator
from starlette.concurrency import run_in_threadpool

from heartcut import __version__
from heartcut.serve.router import FallbackRouter, TeacherClassifier
from heartcut.serve.runtime import Predictor, load_predictor

MAX_BATCH = 256
MAX_CHARS = 2000


class PredictRequest(BaseModel):
    text: str | None = Field(default=None, min_length=1, max_length=MAX_CHARS)
    texts: list[str] | None = Field(default=None, min_length=1, max_length=MAX_BATCH)

    @model_validator(mode="after")
    def _exactly_one(self) -> PredictRequest:
        if (self.text is None) == (self.texts is None):
            raise ValueError("send either 'text' or 'texts'")
        if self.texts is not None and any(not t or len(t) > MAX_CHARS for t in self.texts):
            raise ValueError(f"every text must have 1 to {MAX_CHARS} characters")
        return self


class PredictionOut(BaseModel):
    label: str | None
    confidence: float
    latency_ms: float
    source: Literal["student", "teacher"] = "student"
    student_label: str | None = None
    teacher_cost_usd: float | None = None


class BatchOut(BaseModel):
    predictions: list[PredictionOut]
    latency_ms: float


class Service:
    def __init__(self, predictor: Predictor, router: FallbackRouter | None = None) -> None:
        self.predictor = predictor
        self.router = router

    async def predict(self, texts: list[str]) -> tuple[list[PredictionOut], float]:
        start = time.perf_counter()
        proba = await run_in_threadpool(self.predictor.predict_proba, texts)
        student_ms = (time.perf_counter() - start) * 1000
        per_item_ms = student_ms / len(texts)
        out: list[PredictionOut] = []
        for text, row in zip(texts, proba, strict=True):
            idx = int(row.argmax())
            label, confidence = self.predictor.labels[idx], float(row[idx])
            if self.router is not None and self.router.should_route(confidence):
                answer = await self.router.teacher.classify(text)
                if answer.label is not None:
                    out.append(
                        PredictionOut(
                            label=answer.label,
                            confidence=confidence,
                            latency_ms=per_item_ms + answer.latency_ms,
                            source="teacher",
                            student_label=label,
                            teacher_cost_usd=answer.cost_usd,
                        )
                    )
                    continue
            out.append(PredictionOut(label=label, confidence=confidence, latency_ms=per_item_ms))
        return out, (time.perf_counter() - start) * 1000

    def health(self) -> dict[str, Any]:
        meta = self.predictor.meta
        return {
            "status": "ok",
            "version": __version__,
            "model": self.predictor.name,
            "kind": meta.get("kind"),
            "format": meta.get("format"),
            "labels": len(self.predictor.labels),
            "fallback": None
            if self.router is None
            else {"threshold": self.router.threshold, "teacher": self.router.teacher.config.name},
        }


def create_app(service: Service) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if service.router is not None:
            await service.router.teacher.aclose()

    app = FastAPI(
        title="Heartcut student",
        version=__version__,
        description="A distilled intent classifier served on CPU.",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return service.health()

    @app.post("/predict", response_model=PredictionOut | BatchOut)
    async def predict(request: PredictRequest) -> PredictionOut | BatchOut:
        if request.text is not None:
            predictions, _ = await service.predict([request.text])
            return predictions[0]
        assert request.texts is not None
        predictions, total_ms = await service.predict(request.texts)
        return BatchOut(predictions=predictions, latency_ms=total_ms)

    return app


def service_from_env() -> Service:
    model_dir = os.environ.get("HEARTCUT_MODEL_DIR")
    if not model_dir:
        raise RuntimeError("set HEARTCUT_MODEL_DIR to an exported model directory")
    threads = os.environ.get("HEARTCUT_THREADS")
    predictor = load_predictor(Path(model_dir), threads=int(threads) if threads else None)
    router = None
    threshold = os.environ.get("HEARTCUT_FALLBACK_THRESHOLD")
    if threshold:
        teacher_config = os.environ.get("HEARTCUT_TEACHER_CONFIG")
        if not teacher_config:
            raise RuntimeError("HEARTCUT_FALLBACK_THRESHOLD needs HEARTCUT_TEACHER_CONFIG")
        teacher = TeacherClassifier.from_file(
            teacher_config,
            predictor.labels,
            task=str(predictor.meta.get("task", "intent classification")),
            oos_label=predictor.meta.get("oos_label"),
        )
        router = FallbackRouter(float(threshold), teacher)
    return Service(predictor, router)


def app_from_env() -> FastAPI:
    """Factory for ``uvicorn heartcut.serve.app:app_from_env --factory``."""
    return create_app(service_from_env())

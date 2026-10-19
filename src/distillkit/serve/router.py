"""Optional fallback router: low-confidence student predictions go to the teacher.

The benchmark reports, for a range of thresholds, what share of traffic would be routed
and what the blended accuracy and cost would be. At serve time the same rule can be
switched on with a threshold and a teacher config.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from distillkit.config import TeacherConfig, TokenPricing, load_teacher
from distillkit.teachers.backends import build_backend
from distillkit.teachers.base import TeacherBackend
from distillkit.teachers.pricing import completion_cost, resolve_pricing
from distillkit.teachers.prompts import PromptTemplate, label_schema, parse_label
from distillkit.teachers.ratelimit import call_with_retries

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeacherAnswer:
    label: str | None
    latency_ms: float
    cost_usd: float | None
    error: str | None = None


class TeacherClassifier:
    """Classify one message with the teacher LLM (same prompt and parser as labeling)."""

    def __init__(
        self,
        config: TeacherConfig,
        labels: Sequence[str],
        task: str,
        oos_label: str | None = None,
        backend: TeacherBackend | None = None,
        pricing: TokenPricing | None = None,
    ) -> None:
        self.config = config
        self.labels = list(labels)
        self.task = task
        self.oos_label = oos_label
        self.template = PromptTemplate.from_file(config.prompt)
        self.schema = label_schema(self.labels) if config.structured_output != "none" else None
        self.backend = backend or build_backend(config, labels=self.labels)
        self.pricing = pricing if pricing is not None else resolve_pricing(config)

    @classmethod
    def from_file(
        cls, path: str | Path, labels: Sequence[str], task: str, oos_label: str | None = None
    ) -> TeacherClassifier:
        return cls(load_teacher(path), labels, task, oos_label)

    async def classify(self, text: str) -> TeacherAnswer:
        prompt = self.template.render(text, self.labels, self.task, self.oos_label)
        start = time.perf_counter()
        try:
            completion = await call_with_retries(
                lambda: self.backend.complete(prompt, self.schema),
                max_retries=min(self.config.max_retries, 2),
            )
        except Exception as exc:
            logger.warning("teacher fallback failed: %s", exc)
            return TeacherAnswer(None, (time.perf_counter() - start) * 1000, None, str(exc))
        parsed = parse_label(completion.text, self.labels)
        cost = completion_cost(completion, self.pricing) if self.pricing else None
        return TeacherAnswer(parsed.label, completion.latency_ms, cost, parsed.error)

    async def aclose(self) -> None:
        await self.backend.aclose()


@dataclass(frozen=True)
class FallbackRouter:
    threshold: float
    teacher: TeacherClassifier

    def should_route(self, confidence: float) -> bool:
        return confidence < self.threshold

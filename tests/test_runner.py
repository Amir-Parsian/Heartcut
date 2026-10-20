from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from heartcut.config import TeacherConfig, TokenPricing
from heartcut.teachers.backends import build_backend
from heartcut.teachers.base import Completion, FatalTeacherError, RetryableError, TeacherBackend
from heartcut.teachers.cache import CompletionCache
from heartcut.teachers.prompts import ChatPrompt, PromptTemplate
from heartcut.teachers.ratelimit import AsyncRateLimiter, backoff_delay, call_with_retries
from heartcut.teachers.runner import (
    BudgetExceeded,
    CacheMiss,
    LabelingRunner,
    aggregate_votes,
)
from tests.conftest import PROMPT

LABELS = ["a", "b", "c"]
PRICING = TokenPricing(input=1.0, output=1.0)


def _config(**kw: Any) -> TeacherConfig:
    return TeacherConfig.model_validate(
        {"name": "fake", "backend": "openai", "model": "fake", "prompt": str(PROMPT), **kw}
    )


class ScriptedBackend(TeacherBackend):
    """Answers from a per-text script; can raise errors a given number of times."""

    def __init__(
        self,
        config: TeacherConfig,
        answers: dict[str, list[str]],
        failures: int = 0,
        error: Exception | None = None,
    ) -> None:
        super().__init__(config)
        self.answers = answers
        self.failures = failures
        self.error = error or RetryableError("overloaded", retry_after=0.0)
        self.calls: list[tuple[str, int]] = []

    async def complete(
        self, prompt: ChatPrompt, schema: dict[str, Any] | None, sample_index: int = 0
    ) -> Completion:
        self.calls.append((prompt.text, sample_index))
        if self.failures:
            self.failures -= 1
            raise self.error
        answer = self.answers[prompt.text][sample_index % len(self.answers[prompt.text])]
        return Completion(
            json.dumps({"label": answer}) if answer != "?" else "no idea",
            input_tokens=100_000,
            output_tokens=0,
            latency_ms=12.0,
        )


async def _no_sleep(_: float) -> None:
    return None


def _runner(backend: TeacherBackend, tmp_path: Path, **kw: Any) -> LabelingRunner:
    return LabelingRunner(
        backend,
        PromptTemplate.from_file(PROMPT),
        LABELS,
        config=backend.config,
        task="test",
        cache=CompletionCache(tmp_path / "cache.sqlite"),
        pricing=kw.pop("pricing", PRICING),
        sleep=_no_sleep,
        **kw,
    )


def test_aggregate_votes() -> None:
    assert aggregate_votes(["a", "b", "a"]) == ("a", 2 / 3, {"a": 2, "b": 1})
    assert aggregate_votes(["b", "a"]) == ("b", 0.5, {"a": 1, "b": 1})  # tie -> first seen
    assert aggregate_votes(["a", None, None]) == ("a", 1 / 3, {"a": 1})
    assert aggregate_votes([None]) == (None, 0.0, {})


async def test_self_consistency_votes_and_costs(tmp_path: Path) -> None:
    backend = ScriptedBackend(
        _config(self_consistency=3), {"x": ["a", "b", "a"], "y": ["c", "?", "c"]}
    )
    examples, records = await _runner(backend, tmp_path).label(["1", "2"], ["x", "y"], "train")
    assert [e.label for e in examples] == ["a", "c"]
    assert examples[0].confidence == pytest.approx(2 / 3)
    assert examples[1].sample_labels == ["c", None, "c"]
    assert examples[1].n_valid == 2
    assert len(records) == 6
    assert records[4].error is not None
    # 100k input tokens at $1/M = $0.10 per call, three calls per example.
    assert examples[0].cost_usd == pytest.approx(0.3)


async def test_cache_makes_reruns_free(tmp_path: Path) -> None:
    backend = ScriptedBackend(_config(), {"x": ["a"], "y": ["b"]})
    runner = _runner(backend, tmp_path)
    await runner.label(["1", "2"], ["x", "y"], "train")
    assert len(backend.calls) == 2 and runner.spent_usd == pytest.approx(0.2)

    rerun = _runner(backend, tmp_path)
    examples, records = await rerun.label(["1", "2"], ["x", "y"], "train")
    assert len(backend.calls) == 2  # no new calls
    assert all(r.cache_hit for r in records)
    assert rerun.spent_usd == 0.0
    # Cached entries keep their original usage and latency.
    assert records[0].cost_usd == pytest.approx(0.1)
    assert records[0].latency_ms == 12.0
    assert [e.label for e in examples] == ["a", "b"]


async def test_offline_mode_refuses_to_call(tmp_path: Path) -> None:
    backend = ScriptedBackend(_config(), {"x": ["a"]})
    with pytest.raises(CacheMiss):
        await _runner(backend, tmp_path, offline=True).label(["1"], ["x"], "train")
    assert backend.calls == []


async def test_budget_cap_stops_spending(tmp_path: Path) -> None:
    texts = [f"t{i}" for i in range(20)]
    backend = ScriptedBackend(
        _config(max_budget_usd=0.25, concurrency=1), {t: ["a"] for t in texts}
    )
    runner = _runner(backend, tmp_path)
    with pytest.raises(BudgetExceeded):
        await runner.label([str(i) for i in range(20)], texts, "train")
    assert len(backend.calls) == 3  # $0.10 each: stops once $0.25 is passed


def test_budget_needs_pricing(tmp_path: Path) -> None:
    backend = ScriptedBackend(_config(max_budget_usd=1.0), {})
    with pytest.raises(ValueError, match="pricing"):
        _runner(backend, tmp_path, pricing=None)


async def test_transient_errors_are_retried(tmp_path: Path) -> None:
    backend = ScriptedBackend(_config(max_retries=3), {"x": ["b"]}, failures=2)
    examples, records = await _runner(backend, tmp_path).label(["1"], ["x"], "train")
    assert examples[0].label == "b"
    assert records[0].retries == 2


async def test_retries_are_bounded(tmp_path: Path) -> None:
    backend = ScriptedBackend(_config(max_retries=1), {"x": ["b"]}, failures=5)
    with pytest.raises(RetryableError):
        await _runner(backend, tmp_path).label(["1"], ["x"], "train")


async def test_fatal_errors_are_not_retried(tmp_path: Path) -> None:
    backend = ScriptedBackend(
        _config(max_retries=5), {"x": ["b"]}, failures=1, error=FatalTeacherError("401")
    )
    with pytest.raises(FatalTeacherError):
        await _runner(backend, tmp_path).label(["1"], ["x"], "train")
    assert len(backend.calls) == 1


async def test_unpriced_teacher_reports_unknown_cost(tmp_path: Path) -> None:
    backend = ScriptedBackend(_config(), {"x": ["a"]})
    examples, records = await _runner(backend, tmp_path, pricing=None).label(["1"], ["x"], "t")
    assert examples[0].cost_usd is None and records[0].cost_usd is None


async def test_refusals_are_flagged(tmp_path: Path) -> None:
    class Refuser(ScriptedBackend):
        async def complete(
            self, prompt: ChatPrompt, schema: dict[str, Any] | None, sample_index: int = 0
        ) -> Completion:
            return Completion("", input_tokens=1, output_tokens=0, stop_reason="refusal")

    examples, records = await _runner(Refuser(_config(), {}), tmp_path).label(["1"], ["x"], "t")
    assert examples[0].label is None
    assert records[0].error == "refusal"


async def test_mismatched_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        await _runner(ScriptedBackend(_config(), {}), tmp_path).label(["1"], [], "t")


async def test_rate_limiter_spaces_requests() -> None:
    now = [0.0]
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    limiter = AsyncRateLimiter(60, burst=1, clock=lambda: now[0], sleep=fake_sleep)
    for _ in range(3):
        await limiter.acquire()
    assert sum(slept) == pytest.approx(2.0)  # 1 request per second after the first
    with pytest.raises(ValueError):
        AsyncRateLimiter(0)


def test_backoff_grows_and_is_capped() -> None:
    assert backoff_delay(0, base=1, rng=lambda: 0.0) == 0.5
    assert backoff_delay(3, base=1, rng=lambda: 1.0) == 8.0
    assert backoff_delay(20, base=1, cap=30, rng=lambda: 1.0) == 30.0


async def test_retry_after_header_wins() -> None:
    delays: list[float] = []

    async def record(seconds: float) -> None:
        delays.append(seconds)

    attempts = iter([RetryableError("429", retry_after=7.0), None])

    async def flaky() -> str:
        err = next(attempts)
        if err:
            raise err
        return "ok"

    assert await call_with_retries(flaky, max_retries=2, sleep=record) == "ok"
    assert delays == [7.0]


async def test_simulated_backend_is_deterministic() -> None:
    config = TeacherConfig.model_validate(
        {
            "name": "s",
            "backend": "simulated",
            "model": "m",
            "prompt": str(PROMPT),
            "simulated": {"accuracy": 0.5, "seed": 1},
        }
    )
    oracle = {f"text {i}": LABELS[i % 3] for i in range(200)}
    backend = build_backend(config, labels=LABELS, oracle=oracle)
    prompts = [ChatPrompt("s", "u", t) for t in oracle]
    first = [await backend.complete(p, None) for p in prompts]
    second = [await backend.complete(p, None) for p in prompts]
    assert first == second
    hits = sum(
        json.loads(c.text)["label"] == oracle[p.text] for c, p in zip(first, prompts, strict=True)
    )
    assert 70 < hits < 130  # about half right at accuracy 0.5


def test_build_backend_requires_oracle_for_simulation() -> None:
    config = TeacherConfig.model_validate(
        {"name": "s", "backend": "simulated", "model": "m", "prompt": "p"}
    )
    with pytest.raises(ValueError, match="oracle"):
        build_backend(config)


def test_cache_roundtrip(tmp_path: Path) -> None:
    completion = Completion("x", 1, 2, 3, 4, 5.0, "end_turn", "m")
    with CompletionCache(tmp_path / "c.sqlite") as cache:
        assert cache.get("k") is None
        cache.put("k", completion)
        assert cache.get("k") == completion
        assert len(cache) == 1

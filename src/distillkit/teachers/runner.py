"""Asynchronous, cached, rate-limited, budget-capped teacher labeling."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from distillkit.config import TeacherConfig, TokenPricing
from distillkit.teachers.base import Completion, TeacherBackend, TeacherError
from distillkit.teachers.cache import CompletionCache, cache_key
from distillkit.teachers.pricing import completion_cost
from distillkit.teachers.prompts import ChatPrompt, PromptTemplate, label_schema, parse_label
from distillkit.teachers.ratelimit import AsyncRateLimiter, Sleep, call_with_retries

logger = logging.getLogger(__name__)


class BudgetExceeded(TeacherError):
    """The configured spending cap was reached; finished calls are kept in the cache."""


class CacheMiss(TeacherError):
    """A call was needed in offline mode (which guarantees zero spend)."""


@dataclass(frozen=True)
class CallRecord:
    """One teacher call (or cache lookup), as written to ``calls.jsonl``."""

    id: str
    split: str
    sample_index: int
    cache_hit: bool
    label: str | None
    error: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    latency_ms: float
    cost_usd: float | None
    stop_reason: str | None
    retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExampleLabel:
    id: str
    label: str | None
    confidence: float
    votes: dict[str, int]
    sample_labels: list[str | None]
    cost_usd: float | None

    @property
    def n_valid(self) -> int:
        return sum(label is not None for label in self.sample_labels)


def aggregate_votes(
    sample_labels: Sequence[str | None],
) -> tuple[str | None, float, dict[str, int]]:
    """Majority vote over self-consistency samples.

    Confidence is the share of *all* samples that agree with the winner, so invalid
    answers lower it. Ties go to the label that appeared first.
    """
    votes = Counter(label for label in sample_labels if label is not None)
    if not votes:
        return None, 0.0, {}
    top = max(votes.values())
    winner = next(lbl for lbl in sample_labels if lbl is not None and votes[lbl] == top)
    return winner, top / len(sample_labels), dict(votes)


@dataclass
class _Progress:
    total: int
    done: int = 0
    paid_calls: int = 0
    spent_usd: float = 0.0
    next_report: int = field(default=0)


class LabelingRunner:
    """Labels texts with a teacher backend.

    Responsibilities: render prompts, look up the cache, rate-limit and retry real calls,
    stop at the budget cap, parse and validate answers, aggregate self-consistency votes,
    and record tokens, latency and dollars for every call.
    """

    def __init__(
        self,
        backend: TeacherBackend,
        template: PromptTemplate,
        labels: Sequence[str],
        *,
        config: TeacherConfig,
        task: str,
        cache: CompletionCache,
        pricing: TokenPricing | None,
        oos_label: str | None = None,
        offline: bool = False,
        spent_usd: float = 0.0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.backend = backend
        self.template = template
        self.labels = list(labels)
        self.config = config
        self.task = task
        self.oos_label = oos_label
        self.cache = cache
        self.pricing = pricing
        self.offline = offline
        self.spent_usd = spent_usd
        self._sleep = sleep
        self.schema = label_schema(self.labels) if config.structured_output != "none" else None
        self.limiter = (
            AsyncRateLimiter(config.requests_per_minute, sleep=sleep)
            if config.requests_per_minute
            else None
        )
        if config.max_budget_usd is not None and pricing is None:
            raise ValueError("a budget cap needs pricing for the teacher model")

    # ------------------------------------------------------------------ helpers

    def prompt_for(self, text: str) -> ChatPrompt:
        return self.template.render(text, self.labels, self.task, self.oos_label)

    def key_for(self, prompt: ChatPrompt, sample_index: int) -> str:
        return cache_key(
            {
                "backend": self.config.backend,
                "model": self.config.model,
                "base_url": self.config.base_url,
                "system": prompt.system,
                "user": prompt.user,
                "schema": self.schema,
                "structured_output": self.config.structured_output,
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_output_tokens,
                "request_options": self.config.request_options,
                "sample_index": sample_index,
            }
        )

    def _cost(self, completion: Completion) -> float | None:
        return completion_cost(completion, self.pricing) if self.pricing else None

    def _budget_left(self) -> bool:
        cap = self.config.max_budget_usd
        return cap is None or self.spent_usd < cap

    # ------------------------------------------------------------------ calls

    async def _call(
        self,
        ex_id: str,
        text: str,
        split: str,
        sample_index: int,
        semaphore: asyncio.Semaphore,
        progress: _Progress,
    ) -> CallRecord:
        prompt = self.prompt_for(text)
        key = self.key_for(prompt, sample_index)
        completion = self.cache.get(key)
        cache_hit = completion is not None
        retries = 0
        if completion is None:
            if self.offline:
                raise CacheMiss(f"{split}/{ex_id} sample {sample_index} is not cached")
            async with semaphore:
                if not self._budget_left():
                    raise BudgetExceeded(
                        f"spent ${self.spent_usd:.2f} of the ${self.config.max_budget_usd:.2f} cap"
                    )
                if self.limiter is not None:
                    await self.limiter.acquire()

                def count_retry(*_: object) -> None:
                    nonlocal retries
                    retries += 1

                completion = await call_with_retries(
                    lambda: self.backend.complete(prompt, self.schema, sample_index),
                    max_retries=self.config.max_retries,
                    sleep=self._sleep,
                    on_retry=count_retry,
                )
            self.cache.put(key, completion)
        cost = self._cost(completion)
        if not cache_hit:
            progress.paid_calls += 1
            if cost is not None:
                self.spent_usd += cost
                progress.spent_usd += cost
        parsed = parse_label(completion.text, self.labels)
        error = parsed.error
        if completion.stop_reason == "refusal":
            error = "refusal"
        progress.done += 1
        if progress.done >= progress.next_report:
            logger.info(
                "%s: %d/%d calls (%d paid, $%.4f spent this run)",
                split,
                progress.done,
                progress.total,
                progress.paid_calls,
                progress.spent_usd,
            )
            progress.next_report += max(1, progress.total // 10)
        return CallRecord(
            id=ex_id,
            split=split,
            sample_index=sample_index,
            cache_hit=cache_hit,
            label=parsed.label,
            error=error,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cache_read_tokens=completion.cache_read_tokens,
            cache_write_tokens=completion.cache_write_tokens,
            latency_ms=completion.latency_ms,
            cost_usd=cost,
            stop_reason=completion.stop_reason,
            retries=retries,
        )

    async def label(
        self, ids: Sequence[str], texts: Sequence[str], split: str
    ) -> tuple[list[ExampleLabel], list[CallRecord]]:
        """Label every text ``config.self_consistency`` times and aggregate the votes."""
        if len(ids) != len(texts):
            raise ValueError("ids and texts differ in length")
        k = self.config.self_consistency
        semaphore = asyncio.Semaphore(self.config.concurrency)
        progress = _Progress(total=len(ids) * k)
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(self._call(ex_id, text, split, s, semaphore, progress))
                    for ex_id, text in zip(ids, texts, strict=True)
                    for s in range(k)
                ]
        except ExceptionGroup as group_error:
            # Surface the first teacher error (budget, cache miss, fatal API error) as is;
            # the remaining tasks were cancelled and everything finished is cached.
            first = group_error.exceptions[0]
            raise first from None
        records = [t.result() for t in tasks]

        examples: list[ExampleLabel] = []
        for i, ex_id in enumerate(ids):
            calls = records[i * k : (i + 1) * k]
            sample_labels = [c.label for c in calls]
            label, confidence, votes = aggregate_votes(sample_labels)
            costs = [c.cost_usd for c in calls]
            total = None if any(c is None for c in costs) else float(sum(c for c in costs if c))
            examples.append(ExampleLabel(ex_id, label, confidence, votes, sample_labels, total))
        return examples, records

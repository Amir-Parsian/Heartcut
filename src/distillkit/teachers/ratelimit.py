"""Client-side rate limiting and retries with exponential backoff."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from distillkit.teachers.base import RetryableError

logger = logging.getLogger(__name__)

T = TypeVar("T")
Sleep = Callable[[float], Awaitable[None]]


class AsyncRateLimiter:
    """Token bucket: at most ``rate_per_minute`` acquisitions per minute on average,
    with bursts of up to ``burst`` (default: one second's worth, at least one)."""

    def __init__(
        self,
        rate_per_minute: float,
        burst: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self.rate = rate_per_minute / 60.0
        self.capacity = burst if burst is not None else max(1.0, self.rate)
        self.tokens = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.rate)
        self._last = now

    async def acquire(self) -> None:
        async with self._lock:
            self._refill()
            while self.tokens < 1.0:
                await self._sleep((1.0 - self.tokens) / self.rate)
                self._refill()
            self.tokens -= 1.0


def backoff_delay(
    attempt: int, base: float = 1.0, cap: float = 60.0, rng: Callable[[], float] = random.random
) -> float:
    """Exponential backoff with "equal jitter": half fixed, half random."""
    ceiling = min(cap, base * 2.0**attempt)
    return float(ceiling / 2 + rng() * ceiling / 2)


async def call_with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    sleep: Sleep = asyncio.sleep,
    on_retry: Callable[[int, RetryableError, float], None] | None = None,
) -> T:
    """Await ``fn()``, retrying :class:`RetryableError` up to ``max_retries`` times.

    A server-provided ``retry-after`` wins over the computed backoff.
    """
    attempt = 0
    while True:
        try:
            return await fn()
        except RetryableError as exc:
            if attempt >= max_retries:
                raise
            delay = (
                exc.retry_after
                if exc.retry_after is not None
                else backoff_delay(attempt, base_delay, max_delay)
            )
            if on_retry is not None:
                on_retry(attempt + 1, exc, delay)
            logger.debug("retry %d after %.1fs: %s", attempt + 1, delay, exc)
            await sleep(delay)
            attempt += 1

"""Latency and throughput measurement on the local CPU."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

PredictFn = Callable[[Sequence[str]], Any]


def measure_latency(
    predict: PredictFn, texts: Sequence[str], *, warmup: int = 20, requests: int = 300
) -> dict[str, float]:
    """Single-request latency (batch of one), after ``warmup`` untimed calls."""
    if not texts:
        raise ValueError("need at least one text")
    for i in range(warmup):
        predict([texts[i % len(texts)]])
    timings = np.empty(requests)
    for i in range(requests):
        text = texts[i % len(texts)]
        start = time.perf_counter()
        predict([text])
        timings[i] = (time.perf_counter() - start) * 1000
    return {
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "mean_ms": float(timings.mean()),
        "requests": requests,
    }


def measure_throughput(
    predict: PredictFn, texts: Sequence[str], *, batch_size: int = 64, seconds: float = 5.0
) -> dict[str, float]:
    """Batched requests per second, running for at least ``seconds`` (one warm-up batch)."""
    if not texts:
        raise ValueError("need at least one text")
    batch = [texts[i % len(texts)] for i in range(batch_size)]
    predict(batch)
    done, start = 0, time.perf_counter()
    while True:
        predict(batch)
        done += batch_size
        elapsed = time.perf_counter() - start
        if elapsed >= seconds:
            break
    return {"requests_per_second": done / elapsed, "batch_size": batch_size, "seconds": elapsed}

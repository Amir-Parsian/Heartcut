"""Turn token usage into dollars using published per-token prices."""

from __future__ import annotations

import logging

from heartcut.config import TeacherConfig, TokenPricing, load_pricing_table
from heartcut.teachers.base import Completion

logger = logging.getLogger(__name__)

PER_MILLION = 1_000_000


def resolve_pricing(
    config: TeacherConfig, table: dict[str, TokenPricing] | None = None
) -> TokenPricing | None:
    """Teacher-specific pricing wins over the shared table; unknown models return None."""
    if config.pricing is not None:
        return config.pricing
    if table is None:
        try:
            table = load_pricing_table()
        except FileNotFoundError:
            table = {}
    return table.get(config.model)


def completion_cost(completion: Completion, pricing: TokenPricing) -> float:
    """Dollar cost of one call.

    Cache reads and writes fall back to the base input price when the table has no
    specific rate, which over- rather than under-states the teacher's cost.
    """
    cache_read = pricing.cache_read if pricing.cache_read is not None else pricing.input
    cache_write = pricing.cache_write if pricing.cache_write is not None else pricing.input
    dollars = (
        completion.input_tokens * pricing.input
        + completion.output_tokens * pricing.output
        + completion.cache_read_tokens * cache_read
        + completion.cache_write_tokens * cache_write
    )
    return dollars / PER_MILLION


def estimate_cost(
    n_calls: int, input_tokens_per_call: float, output_tokens_per_call: float, pricing: TokenPricing
) -> float:
    """Upper-bound estimate that ignores prompt caching."""
    per_call = input_tokens_per_call * pricing.input + output_tokens_per_call * pricing.output
    return n_calls * per_call / PER_MILLION

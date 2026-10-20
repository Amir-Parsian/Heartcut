from __future__ import annotations

import pytest

from heartcut.config import TeacherConfig, TokenPricing
from heartcut.teachers.base import Completion
from heartcut.teachers.pricing import completion_cost, estimate_cost, resolve_pricing


def _teacher(**kw: object) -> TeacherConfig:
    return TeacherConfig.model_validate(
        {"name": "t", "backend": "anthropic", "model": "claude-haiku-5-5", "prompt": "p", **kw}
    )


def test_completion_cost_uses_every_token_class() -> None:
    pricing = TokenPricing(input=1.0, output=5.0, cache_read=0.1, cache_write=1.25)
    completion = Completion(
        "x", input_tokens=1000, output_tokens=100, cache_read_tokens=10_000, cache_write_tokens=2000
    )
    expected = (1000 * 1.0 + 100 * 5.0 + 10_000 * 0.1 + 2000 * 1.25) / 1e6
    assert completion_cost(completion, pricing) == pytest.approx(expected)


def test_missing_cache_rates_fall_back_to_input_price() -> None:
    pricing = TokenPricing(input=2.0, output=4.0)
    completion = Completion("x", input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000)
    assert completion_cost(completion, pricing) == pytest.approx(2.0)


def test_resolve_pricing_prefers_teacher_override() -> None:
    table = {"claude-haiku-5-5": TokenPricing(input=0.1, output=0.5)}
    assert resolve_pricing(_teacher(), table) == table["claude-haiku-5-5"]
    override = _teacher(pricing={"input": 9, "output": 9})
    assert resolve_pricing(override, table) == TokenPricing(input=9, output=9)
    assert resolve_pricing(_teacher(model="unknown"), table) is None


def test_resolve_pricing_reads_shipped_table() -> None:
    pricing = resolve_pricing(_teacher())
    assert pricing is not None and pricing.input > 0


def test_estimate_cost() -> None:
    pricing = TokenPricing(input=1.0, output=10.0)
    assert estimate_cost(1000, 500, 10, pricing) == pytest.approx(1000 * (500 + 100) / 1e6)

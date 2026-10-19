"""Backend request/response handling against mocked HTTP transports (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2
import openai
import pytest

from distillkit.config import TeacherConfig
from distillkit.teachers.backends.anthropic_backend import AnthropicBackend
from distillkit.teachers.backends.openai_backend import OpenAICompatibleBackend
from distillkit.teachers.base import FatalTeacherError, RetryableError, deep_merge
from distillkit.teachers.prompts import ChatPrompt, label_schema

PROMPT = ChatPrompt(system="Pick a label.", user="<message>hi</message>", text="hi")
SCHEMA = label_schema(["greet", "leave"])

Handler = Callable[[httpx2.Request], httpx2.Response]


def _teacher(backend: str, **kw: Any) -> TeacherConfig:
    return TeacherConfig.model_validate(
        {"name": "t", "backend": backend, "model": "m-1", "prompt": "p", **kw}
    )


def _anthropic(handler: Handler, **kw: Any) -> AnthropicBackend:
    client = anthropic.AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return AnthropicBackend(_teacher("anthropic", **kw), client=client)


def _openai(handler: Handler, **kw: Any) -> OpenAICompatibleBackend:
    client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="http://localhost:8000/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return OpenAICompatibleBackend(_teacher("openai", **kw), client=client)


ANTHROPIC_OK = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "m-1",
    "content": [{"type": "text", "text": '{"label": "greet"}'}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 12,
        "output_tokens": 7,
        "cache_read_input_tokens": 300,
        "cache_creation_input_tokens": 0,
    },
}


async def test_anthropic_request_and_usage() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(json.loads(request.content))
        seen["headers"] = dict(request.headers)
        return httpx2.Response(200, json=ANTHROPIC_OK)

    backend = _anthropic(
        handler, request_options={"output_config": {"effort": "low"}}, temperature=0.5
    )
    completion = await backend.complete(PROMPT, SCHEMA)
    assert completion.text == '{"label": "greet"}'
    assert (completion.input_tokens, completion.output_tokens) == (12, 7)
    assert completion.cache_read_tokens == 300
    assert completion.stop_reason == "end_turn"
    assert seen["model"] == "m-1"
    assert seen["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert seen["messages"] == [{"role": "user", "content": PROMPT.user}]
    # Structured output and effort are merged into one output_config.
    assert seen["output_config"]["format"]["schema"] == SCHEMA
    assert seen["output_config"]["effort"] == "low"
    assert seen["temperature"] == 0.5
    assert seen["headers"]["x-api-key"] == "test-key"


async def test_anthropic_without_structured_output() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(json.loads(request.content))
        return httpx2.Response(200, json=ANTHROPIC_OK)

    await _anthropic(handler, structured_output="none").complete(PROMPT, SCHEMA)
    assert "output_config" not in seen
    assert "temperature" not in seen


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (429, RetryableError),
        (529, RetryableError),
        (500, RetryableError),
        (400, FatalTeacherError),
        (401, FatalTeacherError),
    ],
)
async def test_anthropic_error_mapping(status: int, error: type[Exception]) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = {"type": "error", "error": {"type": "x", "message": "nope"}}
        return httpx2.Response(status, json=body, headers={"retry-after": "3"})

    with pytest.raises(error) as info:
        await _anthropic(handler).complete(PROMPT, SCHEMA)
    if isinstance(info.value, RetryableError):
        assert info.value.retry_after == 3.0


async def test_anthropic_connection_errors_are_retryable() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("down", request=request)

    with pytest.raises(RetryableError):
        await _anthropic(handler).complete(PROMPT, SCHEMA)


OPENAI_OK = {
    "id": "c1",
    "object": "chat.completion",
    "created": 0,
    "model": "m-1",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": '{"label": "leave"}'},
        }
    ],
    "usage": {
        "prompt_tokens": 120,
        "completion_tokens": 6,
        "total_tokens": 126,
        "prompt_tokens_details": {"cached_tokens": 100},
    },
}


async def test_openai_compatible_request_and_usage() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(json.loads(request.content))
        seen["url"] = str(request.url)
        return httpx2.Response(200, json=OPENAI_OK)

    completion = await _openai(handler, temperature=0.7).complete(PROMPT, SCHEMA)
    assert completion.text == '{"label": "leave"}'
    # Cached tokens are split out of prompt_tokens.
    assert (completion.input_tokens, completion.cache_read_tokens) == (20, 100)
    assert completion.output_tokens == 6
    assert seen["url"].endswith("/chat/completions")
    assert seen["messages"][0] == {"role": "system", "content": PROMPT.system}
    assert seen["response_format"]["json_schema"]["schema"] == SCHEMA
    assert seen["max_tokens"] == 256
    assert seen["temperature"] == 0.7


async def test_openai_options_override_defaults() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(json.loads(request.content))
        return httpx2.Response(200, json=OPENAI_OK)

    backend = _openai(
        handler, structured_output="json_object", request_options={"max_completion_tokens": 512}
    )
    await backend.complete(PROMPT, SCHEMA)
    assert seen["response_format"] == {"type": "json_object"}
    assert "max_tokens" not in seen and seen["max_completion_tokens"] == 512


@pytest.mark.parametrize(
    ("status", "error"),
    [(429, RetryableError), (503, RetryableError), (404, FatalTeacherError)],
)
async def test_openai_error_mapping(status: int, error: type[Exception]) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json={"error": {"message": "nope"}})

    with pytest.raises(error):
        await _openai(handler).complete(PROMPT, SCHEMA)


async def test_openai_connection_errors_are_retryable() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("down", request=request)

    with pytest.raises(RetryableError):
        await _openai(handler).complete(PROMPT, SCHEMA)


async def test_backends_build_their_own_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_KEY", "abc")
    a = AnthropicBackend(_teacher("anthropic", api_key_env="MY_KEY"))
    o = OpenAICompatibleBackend(_teacher("openai", base_url="http://localhost:11434/v1"))
    assert a.client.max_retries == 0 and o.client.max_retries == 0
    await a.aclose()
    await o.aclose()


def test_deep_merge() -> None:
    assert deep_merge({"a": {"b": 1, "c": 2}, "d": 1}, {"a": {"c": 3}, "e": 4}) == {
        "a": {"b": 1, "c": 3},
        "d": 1,
        "e": 4,
    }

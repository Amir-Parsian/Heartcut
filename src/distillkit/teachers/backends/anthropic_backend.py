"""Hosted teacher: Anthropic's Messages API via the official ``anthropic`` SDK."""

from __future__ import annotations

import os
import time
from typing import Any

import anthropic

from distillkit.config import TeacherConfig
from distillkit.teachers.base import (
    Completion,
    FatalTeacherError,
    RetryableError,
    TeacherBackend,
    deep_merge,
    parse_retry_after,
)
from distillkit.teachers.prompts import ChatPrompt

# Request fields the SDK accepts as typed keyword arguments. Anything else in
# ``request_options`` is sent through ``extra_body`` unchanged.
_TYPED_FIELDS = {
    "metadata",
    "output_config",
    "service_tier",
    "stop_sequences",
    "system",
    "thinking",
    "inference_geo",
}
_RETRYABLE_STATUS = {408, 409, 429}


class AnthropicBackend(TeacherBackend):
    def __init__(
        self, config: TeacherConfig, client: anthropic.AsyncAnthropic | None = None
    ) -> None:
        super().__init__(config)
        if client is None:
            key_env = config.api_key_env or "ANTHROPIC_API_KEY"
            # Retries are handled by the shared runner so that every backend behaves the
            # same and every attempt is visible in the call log.
            client = anthropic.AsyncAnthropic(
                api_key=os.environ.get(key_env),
                base_url=config.base_url,
                max_retries=0,
                timeout=config.timeout_s,
            )
        self.client = client

    def build_request(self, prompt: ChatPrompt, schema: dict[str, Any] | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_output_tokens,
            # The system prompt (with the label list) is identical for every message, so
            # mark it cacheable: later requests read it at a fraction of the input price.
            "system": [
                {"type": "text", "text": prompt.system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": prompt.user}],
        }
        if schema is not None and self.config.structured_output == "json_schema":
            body["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        return deep_merge(body, self.config.request_options)

    async def complete(
        self, prompt: ChatPrompt, schema: dict[str, Any] | None, sample_index: int = 0
    ) -> Completion:
        body = self.build_request(prompt, schema)
        typed = {k: body.pop(k) for k in list(body) if k in _TYPED_FIELDS}
        model, max_tokens, messages = (
            body.pop("model"),
            body.pop("max_tokens"),
            body.pop("messages"),
        )
        start = time.perf_counter()
        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                extra_body=body or None,
                **typed,
            )
        except anthropic.RateLimitError as exc:
            raise RetryableError(str(exc), parse_retry_after(exc.response.headers)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code in _RETRYABLE_STATUS or exc.status_code >= 500:
                raise RetryableError(str(exc), parse_retry_after(exc.response.headers)) from exc
            raise FatalTeacherError(f"{exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:  # includes timeouts
            raise RetryableError(str(exc)) from exc
        latency_ms = (time.perf_counter() - start) * 1000

        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        return Completion(
            text=text,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
            latency_ms=latency_ms,
            stop_reason=response.stop_reason,
            model=response.model,
        )

    async def aclose(self) -> None:
        await self.client.close()

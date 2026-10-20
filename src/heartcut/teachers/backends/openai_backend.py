"""OpenAI-compatible Chat Completions backend.

One backend covers OpenAI's hosted API and the open-weights servers that expose the
same API locally: vLLM (``vllm serve <model>``, default ``http://localhost:8000/v1``) and
Ollama (``http://localhost:11434/v1``).
"""

from __future__ import annotations

import os
import time
from typing import Any

import openai

from heartcut.config import TeacherConfig
from heartcut.teachers.base import (
    Completion,
    FatalTeacherError,
    RetryableError,
    TeacherBackend,
    deep_merge,
    parse_retry_after,
)
from heartcut.teachers.prompts import ChatPrompt

_RETRYABLE_STATUS = {408, 409, 429}
_TOKEN_LIMIT_FIELDS = ("max_tokens", "max_completion_tokens")


class OpenAICompatibleBackend(TeacherBackend):
    def __init__(self, config: TeacherConfig, client: openai.AsyncOpenAI | None = None) -> None:
        super().__init__(config)
        if client is None:
            api_key = os.environ.get(config.api_key_env) if config.api_key_env else None
            client = openai.AsyncOpenAI(
                # Local servers ignore the key, but the SDK insists on having one.
                api_key=api_key or "not-needed",
                base_url=config.base_url,
                max_retries=0,
                timeout=config.timeout_s,
            )
        self.client = client

    def build_request(self, prompt: ChatPrompt, schema: dict[str, Any] | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
        }
        # Reasoning models want ``max_completion_tokens``; set it in request_options to
        # override the widely supported ``max_tokens``.
        if not any(f in self.config.request_options for f in _TOKEN_LIMIT_FIELDS):
            body["max_tokens"] = self.config.max_output_tokens
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        if self.config.structured_output == "json_schema" and schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "label", "strict": True, "schema": schema},
            }
        elif self.config.structured_output == "json_object":
            body["response_format"] = {"type": "json_object"}
        return deep_merge(body, self.config.request_options)

    async def complete(
        self, prompt: ChatPrompt, schema: dict[str, Any] | None, sample_index: int = 0
    ) -> Completion:
        body = self.build_request(prompt, schema)
        start = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(**body)
        except openai.RateLimitError as exc:
            raise RetryableError(str(exc), parse_retry_after(exc.response.headers)) from exc
        except openai.APIStatusError as exc:
            if exc.status_code in _RETRYABLE_STATUS or exc.status_code >= 500:
                raise RetryableError(str(exc), parse_retry_after(exc.response.headers)) from exc
            raise FatalTeacherError(f"{exc.status_code}: {exc.message}") from exc
        except openai.APIConnectionError as exc:  # includes timeouts
            raise RetryableError(str(exc)) from exc
        latency_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0] if response.choices else None
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        cached = 0
        if usage and usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens:
            cached = usage.prompt_tokens_details.cached_tokens
        return Completion(
            text=(choice.message.content or "") if choice else "",
            # OpenAI counts cached tokens inside prompt_tokens; split them out so pricing
            # can apply the cached-input rate.
            input_tokens=prompt_tokens - cached,
            output_tokens=usage.completion_tokens if usage else 0,
            cache_read_tokens=cached,
            latency_ms=latency_ms,
            stop_reason=choice.finish_reason if choice else None,
            model=response.model,
        )

    async def aclose(self) -> None:
        await self.client.close()

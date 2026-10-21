"""Backend-neutral teacher types.

A backend only knows how to send one prompt to one model and report what came back
(text, token usage, latency). Everything else (caching, retries, rate limiting,
self-consistency, cost accounting) lives in :mod:`heartcut.teachers.runner` and is
shared by every backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

from heartcut.config import TeacherConfig
from heartcut.teachers.prompts import ChatPrompt


@dataclass(frozen=True)
class Completion:
    text: str
    # Input tokens billed at the base input price (cache reads/writes are separate).
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: float = 0.0
    stop_reason: str | None = None
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Completion:
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in fields})


class TeacherError(Exception):
    """Base class for errors raised by teacher backends."""


class RetryableError(TeacherError):
    """Transient failure (rate limit, overload, timeout, 5xx): worth retrying."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class FatalTeacherError(TeacherError):
    """Permanent failure (bad request, auth, unknown model): retrying will not help."""


class NoUsableLabels(ValueError):
    """The teacher's answers leave nothing to train on, or nothing to check students with."""


def parse_retry_after(headers: Any) -> float | None:
    """Seconds from a ``retry-after`` header, if present and numeric."""
    if headers is None:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


class TeacherBackend(ABC):
    """Sends one prompt to one model."""

    def __init__(self, config: TeacherConfig) -> None:
        self.config = config

    @abstractmethod
    async def complete(
        self, prompt: ChatPrompt, schema: dict[str, Any] | None, sample_index: int = 0
    ) -> Completion:
        """Return the model's answer. Raise :class:`RetryableError` for transient errors."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network resources."""


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``extra`` into a copy of ``base`` (``extra`` wins)."""
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out

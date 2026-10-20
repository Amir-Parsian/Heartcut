"""Teacher backends. Imported lazily so that each SDK is only needed when used."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from heartcut.config import TeacherConfig
from heartcut.teachers.base import TeacherBackend


def build_backend(
    config: TeacherConfig,
    *,
    labels: Sequence[str] = (),
    oracle: Mapping[str, str] | None = None,
) -> TeacherBackend:
    """Create the backend named in ``config.backend``.

    ``oracle`` (text -> gold label) is only used by the simulated backend.
    """
    if config.backend == "anthropic":
        from heartcut.teachers.backends.anthropic_backend import AnthropicBackend

        return AnthropicBackend(config)
    if config.backend == "openai":
        from heartcut.teachers.backends.openai_backend import OpenAICompatibleBackend

        return OpenAICompatibleBackend(config)
    if config.backend == "simulated":
        from heartcut.teachers.backends.simulated import SimulatedBackend

        if oracle is None:
            raise ValueError("the simulated backend needs an oracle (text -> gold label)")
        return SimulatedBackend(config, labels=labels, oracle=oracle)
    raise ValueError(f"unknown backend {config.backend!r}")


__all__ = ["build_backend"]

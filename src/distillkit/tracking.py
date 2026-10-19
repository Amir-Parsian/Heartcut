"""Experiment tracking.

MLflow is the default because it needs no account: runs land in a local SQLite file
(``mlruns.db``) and ``mlflow ui --backend-store-uri sqlite:///mlruns.db`` shows them.
The ``none`` backend keeps CI and tests dependency-free.
Every training run logs its config, the fingerprints of the data and teacher labels it
was trained on, and its metrics.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from distillkit.config import TrackingConfig

logger = logging.getLogger(__name__)


class Run(Protocol):
    def log_params(self, params: Mapping[str, Any]) -> None: ...
    def log_metrics(self, metrics: Mapping[str, float]) -> None: ...
    def set_tags(self, tags: Mapping[str, str]) -> None: ...
    def log_artifact(self, path: Path) -> None: ...


class _NullRun:
    def log_params(self, params: Mapping[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        pass

    def set_tags(self, tags: Mapping[str, str]) -> None:
        pass

    def log_artifact(self, path: Path) -> None:
        pass


class _MlflowRun:
    def __init__(self, mlflow: Any) -> None:
        self._mlflow = mlflow

    def log_params(self, params: Mapping[str, Any]) -> None:
        # MLflow caps parameter values at a few hundred characters.
        self._mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        self._mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self._mlflow.set_tags(dict(tags))

    def log_artifact(self, path: Path) -> None:
        self._mlflow.log_artifact(str(path))


def flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """``{"a": {"b": 1}}`` -> ``{"a.b": 1}`` for parameter logging."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


@contextmanager
def start_run(config: TrackingConfig, experiment: str, run_name: str) -> Iterator[Run]:
    if config.backend == "none":
        yield _NullRun()
        return
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "tracking.backend is 'mlflow' but MLflow is not installed: "
            "install distillkit[tracking] or set tracking.backend: none"
        ) from exc
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name or experiment)
    with mlflow.start_run(run_name=run_name):
        yield _MlflowRun(mlflow)

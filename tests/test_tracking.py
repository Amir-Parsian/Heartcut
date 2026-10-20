from __future__ import annotations

from pathlib import Path

import pytest

from heartcut.config import TrackingConfig
from heartcut.tracking import flatten, start_run


def test_flatten() -> None:
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}


def test_null_tracker_accepts_everything(tmp_path: Path) -> None:
    with start_run(TrackingConfig(backend="none"), "exp", "run") as run:
        run.log_params({"a": 1})
        run.log_metrics({"m": 0.5})
        run.set_tags({"t": "x"})
        run.log_artifact(tmp_path)


def test_mlflow_tracker_logs_a_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'mlruns.db'}"
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}")
    monkeypatch.chdir(tmp_path)  # run artifacts land next to the database
    with start_run(TrackingConfig(backend="mlflow", tracking_uri=uri), "exp", "run-1") as run:
        run.log_params({"student.params.C": 10, "long": "x" * 1000})
        run.log_metrics({"test_accuracy": 0.9, "skipped": None})  # type: ignore[dict-item]
        run.set_tags({"student": "tfidf"})
        run.log_artifact(artifact)
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["exp"])
    assert len(runs) == 1
    assert runs.iloc[0]["metrics.test_accuracy"] == 0.9
    assert runs.iloc[0]["tags.student"] == "tfidf"

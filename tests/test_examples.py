"""The example custom task works with no changes to Heartcut's code."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from heartcut.config import load_experiment
from heartcut.data import loaders
from heartcut.data.prepare import load_prepared, prepare_data
from heartcut.serve.runtime import load_predictor, predict
from heartcut.students.training import read_metrics, student_run_dir
from heartcut.teachers.backends.simulated import SimulatedBackend
from heartcut.teachers.labeling import labels_stage, read_cost_report
from tests.conftest import REPO


def test_helpdesk_example_runs_the_data_stage(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(REPO)
    cfg = load_experiment(
        "examples/helpdesk/experiment.yaml",
        [f"artifacts_dir={tmp_path / 'artifacts'}", f"results_dir={tmp_path / 'results'}"],
    )
    prepared = prepare_data(cfg)
    assert len(prepared.labels) == 6
    assert len(prepared.train) + len(prepared.validation) + len(prepared.test) == 120
    assert prepared.train["label"].isna().all()
    assert ". " in prepared.test["text"].iloc[0]  # subject and body were joined


@pytest.mark.integration
def test_helpdesk_demo_distills_exports_and_replays_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO)
    cfg = load_experiment(
        REPO / "examples/helpdesk/demo.yaml",
        [
            f"artifacts_dir={tmp_path / 'artifacts'}",
            f"results_dir={tmp_path / 'results'}",
            "benchmark.latency.warmup=0",
            "benchmark.latency.requests=10",
            "benchmark.latency.throughput_seconds=0.01",
        ],
    )
    run_demo = runpy.run_path(str(REPO / "examples/helpdesk/demo.py"))["run_demo"]
    texts = ["The VPN shows a certificate error.", "Please reset my password."]
    summary = run_demo(cfg, texts=texts)

    data = load_prepared(cfg)
    assert data.train["label"].isna().all()
    assert data.validation["label"].isna().all()
    assert len(data.train) + len(data.validation) + len(data.test) == 120
    for student in cfg.students:
        for budget in cfg.budgets:
            directory = student_run_dir(cfg, student.name, budget, 0)
            record = read_metrics(directory)
            assert record["train_examples"] > 0
            assert record["soft_labels"] is student.use_soft_labels
            assert (directory / "model").is_dir()

    assert summary["simulated"] is True
    export_dir = Path(summary["export_dir"])
    results_dir = Path(summary["results_dir"])
    assert export_dir.is_absolute() and (export_dir / "export.json").is_file()
    assert results_dir == cfg.experiment_results_dir.resolve()
    assert json.loads((results_dir / "demo.json").read_text()) == summary
    for report in ("results.json", "results.md", "error_analysis.md", "MODEL_CARD.md"):
        assert (results_dir / report).is_file()
    results = json.loads((results_dir / "results.json").read_text())
    assert results["simulated"] is True
    assert {m["name"] for m in results["models"] if m["role"] == "student"} == {
        "tfidf-hard",
        "tfidf-soft",
    }

    predictor = load_predictor(export_dir)
    reloaded, _ = predict(predictor, texts)
    assert [p["text"] for p in summary["predictions"]] == texts
    assert [p["label"] for p in summary["predictions"]] == [p.label for p in reloaded]
    assert [p["confidence"] for p in summary["predictions"]] == pytest.approx(
        [p.confidence for p in reloaded]
    )
    assert all(p.label in data.labels and 0 <= p.confidence <= 1 for p in reloaded)

    async def unexpected_call(*args: object, **kwargs: object) -> None:
        pytest.fail("offline demo must reconstruct teacher labels from its cache")

    monkeypatch.setattr(SimulatedBackend, "complete", unexpected_call)
    # Remove generated labels so this checks cached replay rather than a stage skip.
    label_stage = labels_stage(cfg)
    label_stage.manifest_path.unlink()
    for split in ("train", "validation", "test"):
        (label_stage.path / f"{split}.parquet").unlink()
    offline_summary = run_demo(cfg, offline=True, texts=texts)
    cost = read_cost_report(labels_stage(cfg).path)["total"]
    assert cost["calls"] > 0 and cost["cache_hits"] == cost["calls"]
    assert cost["paid_calls_this_run"] == 0
    assert offline_summary["export_dir"] == summary["export_dir"]
    assert offline_summary["predictions"] == summary["predictions"]


def test_loader_from_file_errors(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(FileNotFoundError):
        loaders.resolve_loader(str(tmp_path / "missing.py") + ":load")
    module = tmp_path / "my_loader.py"
    module.write_text("VALUE = 1\n")
    with pytest.raises(ValueError, match="callable"):
        loaders.resolve_loader(f"{module}:VALUE")
    with pytest.raises(ModuleNotFoundError):
        loaders.resolve_loader("no_such_module_anywhere:load")

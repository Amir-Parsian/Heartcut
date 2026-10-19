from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from distillkit.data.prepare import prepare_data
from distillkit.eval import cost
from distillkit.eval.benchmark import RESULTS_JSON, load_results, routing_table, run_benchmark
from distillkit.eval.charts import pareto_frontier
from distillkit.eval.latency import measure_latency, measure_throughput
from distillkit.eval.report import (
    END,
    START,
    SimulatedResultsError,
    fmt_ms,
    fmt_pct,
    fmt_usd,
    render_readme_section,
    replace_section,
    update_readme,
)
from distillkit.teachers.labeling import run_labeling


@pytest.fixture
def results(make_experiment):  # type: ignore[no-untyped-def]
    cfg = make_experiment()
    prepare_data(cfg)
    run_labeling(cfg)
    return cfg, run_benchmark(cfg)


def test_benchmark_writes_every_output(results) -> None:  # type: ignore[no-untyped-def]
    cfg, res = results
    directory = cfg.experiment_results_dir
    for name in (
        "results.json",
        "results.md",
        "error_analysis.md",
        "pareto.png",
        "pareto-dark.png",
        "learning_curve.png",
        "learning_curve-dark.png",
    ):
        assert (directory / name).exists(), name
    assert load_results(directory) == json.loads((directory / RESULTS_JSON).read_text())
    assert res["simulated"] is True
    assert "SIMULATED TEACHER" in (directory / "results.md").read_text()
    card = (directory / "MODEL_CARD.md").read_text()
    assert "Known failure modes" in card and "Inherited teacher errors" not in card
    assert "no out-of-scope class" in card


def test_benchmark_rows(results) -> None:  # type: ignore[no-untyped-def]
    _, res = results
    roles = [m["role"] for m in res["models"]]
    assert roles == ["teacher", "teacher", "student"]  # single call, vote of 3, tfidf
    single, voted, student = res["models"]
    assert single["variant"] == "single call" and single["ece"]["mean"] is None
    assert voted["ece"]["mean"] is not None
    assert voted["cost_per_1k_usd"] == pytest.approx(3 * single["cost_per_1k_usd"], rel=0.05)
    assert student["accuracy"]["n"] == 2
    assert student["served"]["format"] == "sklearn"
    assert student["cost_per_1k_usd"] > 0
    assert student["latency"]["p95_ms"] >= student["latency"]["p50_ms"] > 0
    assert student["size_mb"] > 0
    assert "predictions" not in student  # per-example data stays out of results.json
    assert res["prices"]["teacher_as_of"] is not None or res["teacher"]["pricing"]
    assert res["hardware"]["cpu"]


def test_benchmark_analyses(results) -> None:  # type: ignore[no-untyped-def]
    _, res = results
    budgets = {(r["student"], r["budget"]) for r in res["learning_curve"]}
    assert budgets == {("tfidf", "60"), ("tfidf", "full")}
    assert len(res["routing"]) == 8
    shares = [r["routed_share"] for r in res["routing"]]
    assert shares == sorted(shares)  # higher thresholds route more traffic
    assert res["error_analysis"]["student"] == "tfidf"
    assert res["headline"]["text"].startswith("tfidf:")
    assert res["headline"]["cost_ratio"] is not None


def test_benchmark_without_latency(make_experiment) -> None:  # type: ignore[no-untyped-def]
    cfg = make_experiment()
    prepare_data(cfg)
    run_labeling(cfg)
    res = run_benchmark(cfg, measure=False)
    student = res["models"][-1]
    assert student["latency"] is None and student["cost_per_1k_usd"] is None
    assert "cost per 1,000" not in res["headline"]["text"]


def test_readme_section_refuses_simulated_results(results, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _, res = results
    with pytest.raises(SimulatedResultsError):
        render_readme_section(res, "results/support")
    real = copy.deepcopy(res)
    real["simulated"] = False
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{START}\nold\n{END}\n\nRest.\n")
    assert not update_readme(readme, real, "results/support", check=True)
    update_readme(readme, real, "results/support")
    text = readme.read_text()
    assert real["headline"]["text"] in text and "Rest." in text
    assert "results/support/pareto-dark.png" in text
    assert update_readme(readme, real, "results/support", check=True)


def test_replace_section_needs_markers() -> None:
    with pytest.raises(ValueError):
        replace_section("no markers here", "x")


def test_routing_table() -> None:
    student = {
        "confidence": [0.9, 0.2, 0.6],
        "predictions": ["a", "x", "c"],
        "cost_per_1k_usd": 0.01,
    }
    teacher = {"predictions": ["a", "b", "x"], "cost_per_1k_usd": 1.0}
    rows = routing_table(student, teacher, ["a", "b", "c"], [0.5, 0.7])
    assert rows[0] == {
        "threshold": 0.5,
        "routed_share": pytest.approx(1 / 3),
        "accuracy": 1.0,
        "cost_per_1k_usd": pytest.approx(0.01 + 1 / 3),
    }
    assert rows[1]["accuracy"] == pytest.approx(2 / 3)  # routing the 0.6 one hurts here


def test_cost_model() -> None:
    assert cost.student_cost_per_1k(3.6, 1000) == pytest.approx(0.001)
    assert cost.teacher_cost_per_1k([0.001, 0.003]) == pytest.approx(2.0)
    assert cost.blended_cost_per_1k(0.01, 1.0, 0.1) == pytest.approx(0.11)
    with pytest.raises(ValueError):
        cost.student_cost_per_1k(1.0, 0)
    with pytest.raises(ValueError):
        cost.teacher_cost_per_1k([])


def test_latency_helpers() -> None:
    calls: list[int] = []

    def predict(texts):  # type: ignore[no-untyped-def]
        calls.append(len(texts))

    timing = measure_latency(predict, ["a", "b"], warmup=3, requests=10)
    assert calls[:13] == [1] * 13
    assert timing["p50_ms"] >= 0 and timing["requests"] == 10
    throughput = measure_throughput(predict, ["a"], batch_size=8, seconds=0.01)
    assert throughput["requests_per_second"] > 0 and calls[-1] == 8
    with pytest.raises(ValueError):
        measure_latency(predict, [])
    with pytest.raises(ValueError):
        measure_throughput(predict, [])


def test_pareto_frontier() -> None:
    points = [(1.0, 0.8), (0.1, 0.7), (0.5, 0.6), (10.0, 0.95), (0.01, 0.5)]
    assert pareto_frontier(points) == [(0.01, 0.5), (0.1, 0.7), (1.0, 0.8), (10.0, 0.95)]


def test_formatters() -> None:
    assert fmt_pct({"mean": 0.9234, "std": 0.004}) == "92.3 ± 0.4%"
    assert fmt_pct({"mean": None}) == "n/a" and fmt_pct(None) == "n/a"
    assert fmt_usd(12.345) == "$12.35"
    assert fmt_usd(0.0123) == "$0.012"
    assert fmt_usd(0.0000061) == "$0.0000061"
    assert fmt_usd(None) == "n/a" and fmt_usd(0) == "$0"
    assert fmt_ms(0.42) == "0.42 ms" and fmt_ms(650) == "650 ms" and fmt_ms(None) == "n/a"


def test_unpriced_teacher_has_unknown_cost(make_experiment) -> None:  # type: ignore[no-untyped-def]
    base = make_experiment()
    teacher = base.teacher.model_copy(update={"pricing": None, "is_local": True})
    cfg = base.model_copy(update={"teacher": teacher})
    prepare_data(cfg)
    run_labeling(cfg)
    res = run_benchmark(cfg, measure=False)
    assert all(m["cost_per_1k_usd"] is None for m in res["models"] if m["role"] == "teacher")
    assert res["labeling_cost_per_example_usd"] is None

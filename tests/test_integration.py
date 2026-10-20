"""End to end: data -> label -> audit -> train -> benchmark -> export -> serve -> report.

Runs on the 200-example synthetic dataset with all three student tiers (tiny local
transformer checkpoints) and a *mocked* teacher: the real OpenAI-compatible backend,
runner, cache and cost accounting, with the HTTP layer replaced by a handler that
answers like a decent but imperfect LLM. No network, no API spend.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import httpx2
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from fastapi.testclient import TestClient

from heartcut.config import ExperimentConfig
from heartcut.data.prepare import prepare_data
from heartcut.eval.audit import run_audit
from heartcut.eval.benchmark import run_benchmark
from heartcut.eval.report import END, START, update_readme
from heartcut.serve.app import Service, create_app
from heartcut.serve.runtime import load_predictor
from heartcut.teachers.backends import openai_backend
from heartcut.teachers.labeling import run_labeling
from tests.conftest import PROMPT, experiment_dict
from tests.synthetic import INTENTS

MESSAGE = re.compile(r"<message>\s*(.*?)\s*</message>", re.DOTALL)


def mocked_llm(gold: dict[str, str], accuracy: float = 0.9) -> httpx2.MockTransport:
    labels = sorted(INTENTS)

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        text = MESSAGE.search(body["messages"][1]["content"]).group(1)  # type: ignore[union-attr]
        rng = random.Random(text)
        label = gold[text] if rng.random() < accuracy else rng.choice(labels)
        return httpx2.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps({"label": label})},
                    }
                ],
                "usage": {"prompt_tokens": 700, "completion_tokens": 9, "total_tokens": 709},
            },
        )

    return httpx2.MockTransport(handler)


@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline(
    tmp_path: Path,
    dataset_files: dict[str, Path],
    tiny_st: Path,
    tiny_bert: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gold: dict[str, str] = {}
    for path in dataset_files.values():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            gold[row["text"].strip()] = row["intent"]
    calls: list[int] = []
    transport = mocked_llm(gold)
    real_client = openai_backend.openai.AsyncOpenAI

    def client_factory(**kwargs: Any) -> Any:
        calls.append(1)
        return real_client(**kwargs, http_client=httpx2.AsyncClient(transport=transport))

    monkeypatch.setattr(openai_backend.openai, "AsyncOpenAI", client_factory)

    teacher = {
        "name": "mock-llm",
        "backend": "openai",
        "model": "mock-1",
        "base_url": "http://mock.invalid/v1",
        "prompt": str(PROMPT),
        "pricing": {"input": 0.5, "output": 1.5, "as_of": "2026-10-19"},
    }
    students = [
        {"name": "tfidf", "kind": "tfidf"},
        {
            "name": "setfit",
            "kind": "setfit",
            "params": {"model_name": str(tiny_st), "max_pairs": 200},
        },
        {
            "name": "encoder",
            "kind": "encoder",
            "params": {
                "model_name": str(tiny_bert),
                "epochs": 15,
                "learning_rate": 2e-3,
                "batch_size": 16,
                "early_stopping_patience": 15,
            },
        },
    ]
    cfg = ExperimentConfig.model_validate(
        experiment_dict(
            tmp_path,
            dataset_files,
            teacher=teacher,
            students=students,
            seeds=[0, 1],
            budgets=[80, "full"],
        )
    )

    prepare_data(cfg)
    labeling = run_labeling(cfg)
    total = labeling.cost_report["total"]
    assert total["calls"] == 280  # 160 train + 40 validation + 80 test messages
    # Duplicate messages render the same prompt, so they are usually served from cache.
    assert total["paid_calls_this_run"] + total["cache_hits"] == 280
    # List cost counts every call, cached or not: it is what labeling costs from scratch.
    assert total["list_cost_usd"] == pytest.approx(280 * (700 * 0.5 + 9 * 1.5) / 1e6)

    audit = run_audit(cfg)
    assert 0.8 < audit["pooled"]["accuracy"] < 1.0

    results = run_benchmark(cfg)
    names = [m["name"] for m in results["models"]]
    assert names == ["mock-llm", "tfidf", "setfit", "encoder"]
    formats = {m["name"]: m["served"]["format"] for m in results["models"] if "served" in m}
    assert formats == {"tfidf": "sklearn", "setfit": "onnx", "encoder": "onnx"}
    for m in results["models"][1:]:
        assert m["accuracy"]["n"] == 2 and m["cost_per_1k_usd"] > 0
        assert m["latency"]["p50_ms"] > 0 and m["size_mb"] > 0
    assert results["simulated"] is False
    assert len(results["learning_curve"]) == 6
    assert results["best_student"] in {"tfidf", "setfit", "encoder"}

    # Re-labeling and re-benchmarking hits the cache: no new HTTP clients do real work.
    again = run_labeling(cfg, force=True, offline=True)
    assert again.cost_report["total"]["paid_calls_this_run"] == 0

    best = next(m for m in results["models"] if m["name"] == results["best_student"])
    predictor = load_predictor(Path(best["served"]["export_dir"]), threads=1)
    with TestClient(create_app(Service(predictor))) as client:
        response = client.post("/predict", json={"texts": ["cancel my transfer please"]})
        assert response.status_code == 200
        assert response.json()["predictions"][0]["label"] in INTENTS

    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{START}\n{END}\n")
    update_readme(readme, results, "results/support")
    assert results["headline"]["text"] in readme.read_text()

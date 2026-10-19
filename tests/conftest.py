from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # tests never download models
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from distillkit.config import ExperimentConfig, load_experiment
from tests.synthetic import write_dataset

REPO = Path(__file__).resolve().parents[1]
PROMPT = REPO / "prompts" / "intent_classification.yaml"

ExperimentFactory = Callable[..., ExperimentConfig]


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test gets its own download and teacher cache."""
    monkeypatch.setenv("DISTILLKIT_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture
def dataset_files(tmp_path: Path) -> dict[str, Path]:
    return write_dataset(tmp_path / "raw")


def experiment_dict(tmp_path: Path, files: dict[str, Path], **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "support",
        "dataset": {
            "name": "support-intents",
            "task": "intent classification of customer-support messages",
            "sources": [
                {
                    "loader": "jsonl",
                    "args": {
                        "files": {k: str(v) for k, v in files.items()},
                        "label_column": "intent",
                    },
                }
            ],
            "validation_fraction": 0.2,
            "seed": 3,
        },
        "teacher": {
            "name": "simulated",
            "backend": "simulated",
            "model": "oracle",
            "prompt": str(PROMPT),
            "self_consistency": 3,
            "simulated": {"accuracy": 0.85, "invalid_rate": 0.02, "seed": 0},
            "pricing": {"input": 0.10, "output": 0.50},
        },
        "students": [{"name": "tfidf", "kind": "tfidf", "params": {"C": 5.0}}],
        "seeds": [0, 1],
        "budgets": [60, "full"],
        "artifacts_dir": str(tmp_path / "artifacts"),
        "results_dir": str(tmp_path / "results"),
        "benchmark": {
            "latency": {"warmup": 2, "requests": 20, "throughput_seconds": 0.2, "threads": 1},
        },
    }
    for key, value in overrides.items():
        data[key] = value
    return data


@pytest.fixture
def make_experiment(tmp_path: Path, dataset_files: dict[str, Path]) -> ExperimentFactory:
    def factory(**overrides: Any) -> ExperimentConfig:
        return ExperimentConfig.model_validate(
            experiment_dict(tmp_path, dataset_files, **overrides)
        )

    return factory


@pytest.fixture
def banking77_config() -> ExperimentConfig:
    return load_experiment(REPO / "configs" / "experiments" / "banking77.yaml")


@pytest.fixture(scope="session")
def tiny_st(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 2-layer sentence-transformers model built locally (no download)."""
    pytest.importorskip("sentence_transformers")
    from tests.tiny_models import build_tiny_sentence_transformer

    return build_tiny_sentence_transformer(tmp_path_factory.mktemp("models") / "st")


@pytest.fixture(scope="session")
def tiny_bert(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 2-layer BERT checkpoint built locally (no download)."""
    pytest.importorskip("transformers")
    from tests.tiny_models import build_tiny_bert

    return build_tiny_bert(tmp_path_factory.mktemp("models") / "bert")

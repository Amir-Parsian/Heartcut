from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from distillkit.cli import app
from tests.conftest import experiment_dict

runner = CliRunner()


def _config_file(tmp_path: Path, dataset_files: dict[str, Path], **overrides: object) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(experiment_dict(tmp_path, dataset_files, **overrides)))
    return path


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "distillkit" in result.output


def test_stage_commands(tmp_path: Path, dataset_files: dict[str, Path]) -> None:
    config = str(_config_file(tmp_path, dataset_files))
    result = runner.invoke(app, ["data", "-c", config])
    assert result.exit_code == 0, result.output
    assert '"labels": 8' in result.output

    result = runner.invoke(app, ["label", "-c", config, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["total"]["low_usd"] >= 0

    result = runner.invoke(app, ["label", "-c", config, "--offline"])
    assert result.exit_code == 3  # nothing cached yet, and offline never calls the teacher

    result = runner.invoke(app, ["label", "-c", config])
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["audit", "-c", config])
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app, ["train", "-c", config, "--budget", "full", "--seed", "0", "--set", "seeds=[0]"]
    )
    assert result.exit_code == 0, result.output
    assert "tfidf" in result.output and "budget=full" in result.output


def test_budget_cap_exit_code(tmp_path: Path, dataset_files: dict[str, Path]) -> None:
    config = str(_config_file(tmp_path, dataset_files))
    result = runner.invoke(
        app,
        [
            "label",
            "-c",
            config,
            "--set",
            "teacher.max_budget_usd=0.00001",
            "--set",
            "teacher.concurrency=1",
        ],
    )
    assert result.exit_code == 2
    assert "budget cap" in result.output

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from heartcut.cli import app
from tests.conftest import experiment_dict

runner = CliRunner()


def _config_file(tmp_path: Path, dataset_files: dict[str, Path], **overrides: object) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(experiment_dict(tmp_path, dataset_files, **overrides)))
    return path


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "heartcut" in result.output


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


def test_run_export_report_commands(tmp_path: Path, dataset_files: dict[str, Path]) -> None:
    config = str(_config_file(tmp_path, dataset_files, seeds=[0], budgets=["full"]))
    result = runner.invoke(app, ["run", "-c", config])
    assert result.exit_code == 0, result.output
    assert "tfidf:" in result.output

    result = runner.invoke(app, ["benchmark", "-c", config, "--skip-latency"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["export", "-c", config])
    assert result.exit_code == 0, result.output
    assert Path(result.output.strip().splitlines()[-1], "export.json").exists()

    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:start -->\n<!-- results:end -->\n")
    result = runner.invoke(app, ["report", "-c", config, "--readme", str(readme)])
    assert result.exit_code == 4  # simulated teacher results are never published
    assert "simulated" in result.output


def test_report_check_detects_stale_readme(tmp_path: Path, dataset_files: dict[str, Path]) -> None:
    import json

    config = str(_config_file(tmp_path, dataset_files, seeds=[0], budgets=["full"]))
    assert runner.invoke(app, ["run", "-c", config]).exit_code == 0
    results_file = tmp_path / "results" / "support" / "results.json"
    data = json.loads(results_file.read_text())
    data["simulated"] = False  # pretend a real teacher produced it
    results_file.write_text(json.dumps(data))
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:start -->\n<!-- results:end -->\n")

    result = runner.invoke(app, ["report", "-c", config, "--readme", str(readme), "--check"])
    assert result.exit_code == 1 and "out of date" in result.output
    assert runner.invoke(app, ["report", "-c", config, "--readme", str(readme)]).exit_code == 0
    result = runner.invoke(app, ["report", "-c", config, "--readme", str(readme), "--check"])
    assert result.exit_code == 0 and "up to date" in result.output

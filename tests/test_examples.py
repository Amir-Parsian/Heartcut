"""The example custom task works with no changes to Heartcut's code."""

from __future__ import annotations

import pytest

from heartcut.config import load_experiment
from heartcut.data import loaders
from heartcut.data.prepare import prepare_data
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


def test_loader_from_file_errors(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(FileNotFoundError):
        loaders.resolve_loader(str(tmp_path / "missing.py") + ":load")
    module = tmp_path / "my_loader.py"
    module.write_text("VALUE = 1\n")
    with pytest.raises(ValueError, match="callable"):
        loaders.resolve_loader(f"{module}:VALUE")
    with pytest.raises(ModuleNotFoundError):
        loaders.resolve_loader("no_such_module_anywhere:load")

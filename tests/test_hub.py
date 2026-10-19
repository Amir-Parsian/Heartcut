from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, ClassVar

import pytest

from distillkit.data.prepare import prepare_data
from distillkit.teachers import hub
from distillkit.teachers.labeling import labels_stage, run_labeling

huggingface_hub = pytest.importorskip("huggingface_hub")


class FakeApi:
    uploads: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, token: str | None = None) -> None:
        pass

    def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        self.uploads.append({"create": repo_id, **kwargs})

    def upload_folder(self, **kwargs: Any) -> None:
        self.uploads.append({"folder": kwargs})

    def upload_file(self, **kwargs: Any) -> None:
        self.uploads.append({"file": kwargs})


def test_publish_requires_complete_labels(make_experiment) -> None:  # type: ignore[no-untyped-def]
    cfg = make_experiment()
    prepare_data(cfg)
    with pytest.raises(FileNotFoundError):
        hub.publish_labels(cfg, "user/labels")


def test_publish_and_fetch_roundtrip(make_experiment, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg = make_experiment()
    prepare_data(cfg)
    run_labeling(cfg)
    FakeApi.uploads = []
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    url = hub.publish_labels(cfg, "user/labels")
    assert url == "https://huggingface.co/datasets/user/labels"
    folder = next(u["folder"] for u in FakeApi.uploads if "folder" in u)
    stage = labels_stage(cfg)
    assert folder["path_in_repo"] == f"support/{stage.path.name}"
    assert "*.parquet" in folder["allow_patterns"]
    card = next(u["file"] for u in FakeApi.uploads if "file" in u)["path_or_fileobj"].decode()
    assert "License" in card and stage.fingerprint in card
    assert FakeApi.uploads[0]["private"] is True

    # Simulate the Hub: a snapshot containing exactly what was uploaded.
    snapshot = tmp_path / "snapshot"
    shutil.copytree(stage.path, snapshot / "support" / stage.path.name)
    shutil.rmtree(stage.path)

    def fake_download(repo_id: str, **kwargs: Any) -> str:
        assert kwargs["allow_patterns"] == [f"support/{stage.path.name}/*"]
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    path = hub.fetch_labels(cfg, "user/labels")
    assert path == stage.path and stage.is_complete()
    # With the labels in place the label stage needs no teacher at all.
    assert run_labeling(cfg, offline=True).skipped


def test_fetch_reports_missing_fingerprint(make_experiment, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg = make_experiment()
    prepare_data(cfg)
    (tmp_path / "snap" / "support" / "simulated-0000000000").mkdir(parents=True)
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **k: str(tmp_path / "snap")
    )
    with pytest.raises(FileNotFoundError, match="simulated-0000000000"):
        hub.fetch_labels(cfg, "user/labels")

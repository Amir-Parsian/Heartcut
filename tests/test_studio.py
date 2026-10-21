"""Heartcut Studio: reading uploads, building projects, running them, and the web API."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from heartcut.cli import app as cli
from heartcut.config import ExperimentConfig
from heartcut.studio import catalog
from heartcut.studio.app import create_app
from heartcut.studio.projects import Settings, Workspace, learning_budgets
from heartcut.studio.runner import Busy, Runner, explain, progress_note, recover
from heartcut.studio.tables import (
    TableError,
    build_examples,
    guess_columns,
    profile,
    read_table,
)
from heartcut.teachers.base import FatalTeacherError
from heartcut.teachers.runner import BudgetExceeded

OCTET = {"content-type": "application/octet-stream"}


def _tickets() -> pd.DataFrame:
    raw = pd.read_json(catalog.resource(catalog.SAMPLE_DATA), lines=True)
    return pd.DataFrame(
        {
            "Ticket": raw["ticket_id"],
            "Title": raw["subject"],
            "Text": raw["body"],
            "Team": raw["queue"],
        }
    )


def _settings(upload_id: str, **overrides: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "name": "Helpdesk",
        "upload_id": upload_id,
        "text_columns": ["subject", "body"],
        "label_column": "queue",
        "task": catalog.SAMPLE_TASK,
        "teacher": "simulated",
        "students": ["tfidf-logreg"],
    }
    settings.update(overrides)
    return settings


# --------------------------------------------------------------------------- tables


def test_read_table_handles_the_formats_spreadsheets_export(tmp_path: Path) -> None:
    frame = _tickets()
    semicolons = frame.to_csv(sep=";", index=False).encode()
    assert list(read_table(semicolons, "t.csv").columns) == list(frame.columns)
    tabs = frame.to_csv(sep="\t", index=False).encode()
    assert len(read_table(tabs, "t.tsv")) == len(frame)
    lines = frame.to_json(orient="records", lines=True).encode()
    assert len(read_table(lines, "t.jsonl")) == len(frame)
    records = frame.to_json(orient="records").encode()
    assert len(read_table(records, "t.json")) == len(frame)
    assert len(read_table(lines, "t.json")) == len(frame)  # JSON lines named .json
    excel = io.BytesIO()
    frame.to_excel(excel, index=False)
    assert read_table(excel.getvalue(), "t.xlsx")["Team"].tolist() == frame["Team"].tolist()

    single = read_table(b"message\nreset my password\nprinter jammed, again\n", "new.csv")
    assert list(single.columns) == ["message"]
    assert single["message"].tolist() == ["reset my password", "printer jammed, again"]
    latin = read_table("text,label\ncaf\xe9,a\n".encode("cp1252"), "x.csv")
    assert latin["text"].iloc[0] == "caf\xe9"


def test_read_table_cleans_headers_and_blank_rows() -> None:
    frame = read_table(b"a,,a\n 1 ,x,\n,,\n2,y,z\n", "x.csv")
    assert list(frame.columns) == ["a", "Column 2", "a.1"]
    assert len(frame) == 2  # the all-empty row is dropped
    assert frame["a"].tolist() == ["1", "2"]  # whitespace stripped
    assert pd.isna(frame["a.1"].iloc[0])


@pytest.mark.parametrize(
    ("data", "name", "message"),
    [
        (b"x", "old.xls", "Old Excel files"),
        (b"x", "notes.pdf", "can't read"),
        (b"  ", "empty.csv", "empty"),
        (b"\x00\x01not a workbook", "bad.xlsx", "doesn't look like a valid Excel"),
        (b'{"a": [1, 2', "bad.json", "doesn't look like a valid JSON"),
        (b"a,b\n,\n", "blank.csv", "No rows"),
    ],
)
def test_read_table_explains_bad_files(data: bytes, name: str, message: str) -> None:
    with pytest.raises(TableError, match=message):
        read_table(data, name)


def test_read_table_caps_the_number_of_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("heartcut.studio.tables.MAX_ROWS", 3)
    with pytest.raises(TableError, match="takes up to 3"):
        read_table(b"t\n1\n2\n3\n4\n", "x.csv")


def test_guess_columns_finds_the_message_and_the_category() -> None:
    frame = read_table(_tickets().to_csv(index=False).encode(), "t.csv")
    assert guess_columns(frame) == {"text_columns": ["Text"], "label_column": "Team"}
    lone = read_table(b"only\n" + b"\n".join(f"m{i}".encode() for i in range(30)), "x.csv")
    assert guess_columns(lone) == {"text_columns": ["only"], "label_column": None}


def test_build_examples_joins_message_columns_and_drops_incomplete_rows() -> None:
    frame = read_table(b"s,b,l\nHi,there,a\n,only body,b\nno label,x,\n", "x.csv")
    examples = build_examples(frame, ["s", "b"], "l")
    assert examples.to_dict("records") == [
        {"text": "Hi. there", "label": "a"},
        {"text": "only body", "label": "b"},
    ]
    with pytest.raises(TableError, match="no column named 'nope'"):
        build_examples(frame, ["nope"], "l")


def test_profile_reports_problems_in_plain_words() -> None:
    frame = read_table(_tickets().to_csv(index=False).encode(), "t.csv")
    ready = profile(frame, ["Title", "Text"], "Team")
    assert ready["ready"] and ready["usable"] == 120 and len(ready["categories"]) == 6
    assert not ready["errors"] and not ready["warnings"]

    assert "Message" in profile(frame, [], "Team")["errors"][0]
    assert "Category" in profile(frame, ["Text"], None)["errors"][0]
    assert "both the message and the category" in profile(frame, ["Team"], "Team")["errors"][0]
    assert "no column named" in profile(frame, ["Gone"], "Team")["errors"][0]
    assert "its own category" in profile(frame, ["Text"], "Ticket")["errors"][0]

    tiny = frame.head(10)
    assert "at least 20" in profile(tiny, ["Text"], "Team")["errors"][0]
    same = frame.assign(Team="one")
    assert "same category" in profile(same, ["Text"], "Team")["errors"][0]

    rare = frame.copy()
    rare.loc[:2, "Team"] = "rare"
    rare.loc[3, "Team"] = pd.NA
    checked = profile(rare, ["Text"], "Team")
    assert checked["ready"]
    assert checked["skipped"] == 1
    assert any("1 rows are skipped" in w for w in checked["warnings"])
    assert any("1 category has fewer than 5" in w and "rare" in w for w in checked["warnings"])


# --------------------------------------------------------------------------- catalog


def test_catalog_choices_are_valid_configs() -> None:
    for teacher in catalog.TEACHERS:
        data = teacher.load()
        assert Path(data["prompt"]).is_absolute() and Path(data["prompt"]).exists()
        assert teacher.describe({})["title"] == teacher.title
    for student in catalog.STUDENTS:
        assert student.load()["name"] == student.id
    assert catalog.student("tfidf-logreg").describe()["available"]
    with pytest.raises(KeyError):
        catalog.teacher("nope")
    with pytest.raises(KeyError):
        catalog.student("nope")
    with pytest.raises(FileNotFoundError):
        catalog.resource("configs/teachers/missing.yaml")
    assert catalog.student_title("tfidf-logreg") == "Tiny (tfidf-logreg)"
    assert catalog.student_title("custom") == "custom"


def test_catalog_reports_api_keys_and_missing_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    haiku = catalog.teacher("claude-haiku-5-5")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not haiku.describe({})["has_key"]
    assert haiku.describe({"ANTHROPIC_API_KEY": "k"})["has_key"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert haiku.describe({})["has_key"]
    monkeypatch.setattr(catalog.importlib.util, "find_spec", lambda name: None)
    described = haiku.describe({})
    assert not described["available"] and "teachers" in described["install"]
    assert "students,export" in catalog.student("encoder-distilbert").describe()["install"]


# --------------------------------------------------------------------------- projects


def test_learning_budgets() -> None:
    assert learning_budgets(40) == ["full"]
    assert learning_budgets(1000) == [100, 250, 500, "full"]


def test_projects_are_ordinary_experiments(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    upload = ws.save_sample()
    with pytest.raises(TableError, match="Unknown choice"):
        ws.save_project(Settings(**_settings(upload.id, teacher="gpt-9")))
    with pytest.raises(TableError, match="only once"):
        ws.save_project(Settings(**_settings(upload.id, students=["tfidf-logreg"] * 2)))
    with pytest.raises(TableError, match="its own category"):
        ws.save_project(Settings(**_settings(upload.id, label_column="ticket_id")))
    assert not list(ws.projects_dir.iterdir())  # failed attempts leave nothing behind

    settings = Settings(
        **_settings(
            upload.id,
            teacher="claude-haiku-5-5",
            thorough=True,
            max_budget_usd=2.5,
            max_train_examples=60,
        )
    )
    project = ws.save_project(settings)
    cfg = project.config()
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.teacher.max_budget_usd == 2.5
    assert cfg.dataset.max_examples == {"train": 60}
    assert cfg.seeds == [0, 1, 2] and cfg.budgets == [25, "full"]
    assert cfg.experiment_dir == project.artifacts_dir / project.id
    assert project.experiment_path.read_text().startswith("# Written by Heartcut Studio")
    assert project.state()["status"] == "draft"

    project.set_budget(9.0)
    assert project.config().teacher.max_budget_usd == 9.0
    assert project.settings.max_budget_usd == 9.0
    assert "heartcut run -c" in project.experiment_path.read_text()

    local = ws.save_project(
        Settings(**_settings(upload.id, teacher="qwen2.5-7b-ollama", local_model="llama3:8b")),
    )
    assert local.config().teacher.model == "llama3:8b"
    assert local.config().benchmark.latency.requests == 100  # a quick run

    project.update(status="done")
    with pytest.raises(TableError, match="already run"):
        ws.save_project(settings, project.id)
    for bad in ("../etc", "UPPER", ""):
        with pytest.raises(KeyError):
            ws.project(bad)
    with pytest.raises(KeyError):
        ws.upload("missing-upload")
    with pytest.raises(KeyError):
        project.results_file("../project.json")


# --------------------------------------------------------------------------- runner


def test_progress_notes_translate_pipeline_logs() -> None:
    assert progress_note("validation: 6/12 calls (6 paid, $0.0001 spent this run)") == (
        "6 of 12 answers for the selection messages"
    )
    assert progress_note("training tfidf-logreg-b250-s2-0a1b2c3d4e on 250 teacher labels") == (
        "Training Tiny (tfidf-logreg) on 250 examples (run 3)"
    )
    assert progress_note("exported x (seed 0) to /a/b") == "Packaging x for download"
    assert progress_note("data stage up to date: /a/b") is None


def test_failures_are_explained(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    upload = ws.save_sample()
    paid = ws.save_project(Settings(**_settings(upload.id, teacher="claude-haiku-5-5")))
    local = ws.save_project(Settings(**_settings(upload.id, teacher="qwen2.5-7b-ollama")))
    assert explain(BudgetExceeded("cap"), paid)["code"] == "budget"
    assert explain(FatalTeacherError("401: invalid x-api-key"), paid)["code"] == "key"
    assert explain(FatalTeacherError("400: bad request"), paid)["code"] == "teacher"
    assert explain(ModuleNotFoundError("No module named 'torch'"), paid)["code"] == "install"
    assert explain(ConnectionError("Connection error."), local)["code"] == "local"
    assert explain(ValueError("boom"), paid) == {
        "code": "error",
        "title": "Something went wrong",
        "detail": "ValueError: boom",
    }


def test_runner_stops_between_stages_and_recovers(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    project = ws.save_project(Settings(**_settings(ws.save_sample().id)))
    runner = Runner(ws)
    runner.start(project)
    with pytest.raises(Busy):
        runner.start(project)
    assert runner.stop(project)
    runner.wait(120)
    state = project.state()
    assert state["status"] in ("stopped", "done")
    assert not runner.stop(project)  # nothing running any more

    project.update(status="running")  # as if the studio had been closed mid-run
    recover(ws)
    assert project.state()["status"] == "stopped"
    assert project.state()["error"]["code"] == "interrupted"


# --------------------------------------------------------------------------- web app


@pytest.fixture(scope="module")
def studio(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    app = create_app(tmp_path_factory.mktemp("studio"), allowed_hosts=["testserver"])
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def finished(studio: TestClient) -> dict[str, Any]:
    """The sample tickets, run once with the practice teacher and the Tiny student."""
    upload = studio.post("/api/uploads/sample").json()
    project = studio.post("/api/projects", json=_settings(upload["id"])).json()
    assert studio.post(f"/api/projects/{project['id']}/start", json={}).status_code == 200
    studio.app.state.runner.wait(180)  # type: ignore[attr-defined]
    done: dict[str, Any] = studio.get(f"/api/projects/{project['id']}").json()
    assert done["status"] == "done", done["error"]
    return done


def test_page_and_catalog(studio: TestClient) -> None:
    page = studio.get("/")
    assert page.status_code == 200 and "Heartcut Studio" in page.text
    assert studio.get("/static/studio.js").status_code == 200
    data = studio.get("/api/catalog").json()
    assert data["teachers"][0]["id"] == "simulated"
    assert data["limits"]["min_rows"] == 20
    assert data["active"] is None


def test_only_local_hosts_may_call_the_studio(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "ws"), base_url="http://evil.example")
    assert client.get("/api/catalog").status_code == 400
    local = TestClient(create_app(tmp_path / "ws"), base_url="http://127.0.0.1:8765")
    assert local.get("/api/catalog").status_code == 200


def test_upload_flow(studio: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    csv = _tickets().to_csv(index=False).encode()
    response = studio.post("/api/uploads?filename=tickets.csv", content=csv, headers=OCTET)
    assert response.status_code == 200, response.text
    upload = response.json()
    assert upload["rows"] == 120 and not upload["sample"]
    assert upload["guess"] == {"text_columns": ["Text"], "label_column": "Team"}
    assert upload["profile"]["ready"] and len(upload["preview"]) == 8

    choice = {"text_columns": ["Title", "Text"], "label_column": "Ticket"}
    checked = studio.post(f"/api/uploads/{upload['id']}/profile", json=choice).json()
    assert not checked["ready"]
    assert studio.post("/api/uploads/nope-123/profile", json=choice).status_code == 404

    bad = studio.post("/api/uploads?filename=a.xls", content=b"x", headers=OCTET)
    assert bad.status_code == 400 and "Old Excel" in bad.json()["detail"]
    form = studio.post(
        "/api/uploads?filename=a.csv", content=csv, headers={"content-type": "text/plain"}
    )
    assert form.status_code == 415  # what a cross-site form could send
    monkeypatch.setattr("heartcut.studio.app.MAX_UPLOAD_MB", 0.001)
    assert studio.post("/api/uploads?filename=a.csv", content=csv, headers=OCTET).status_code == 413


def test_drafts_can_change_until_they_run(
    studio: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload = studio.post("/api/uploads/sample").json()
    assert upload["sample"] and upload["task"] == catalog.SAMPLE_TASK
    bad = studio.post("/api/projects", json=_settings(upload["id"], label_column="ticket_id"))
    assert bad.status_code == 400
    invalid = studio.post("/api/projects", json=_settings(upload["id"], students=[]))
    assert invalid.status_code == 422

    draft = studio.post("/api/projects", json=_settings(upload["id"], teacher="claude-haiku-5-5"))
    assert draft.status_code == 200, draft.text
    project = draft.json()
    assert project["status"] == "draft"
    assert project["data"] == {
        "examples": 120,
        "categories": 6,
        "train": 84,
        "validation": 12,
        "test": 24,
    }
    assert project["estimate"]["total"]["high_usd"] > 0
    assert project["teacher_title"] == "Claude Haiku"

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    no_key = studio.post(f"/api/projects/{project['id']}/start", json={})
    assert no_key.status_code == 400 and "API key" in no_key.json()["detail"]

    updated = studio.put(
        f"/api/projects/{project['id']}", json=_settings(upload["id"], name="Renamed")
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "Renamed"
    assert updated.json()["created_at"] == project["created_at"]
    assert studio.get("/api/projects/missing-000000").status_code == 404
    assert (
        studio.post(f"/api/projects/{project['id']}/predict", json={"texts": ["x"]}).status_code
        == 409
    )
    assert studio.get(f"/api/projects/{project['id']}/download").status_code == 409
    assert studio.delete(f"/api/projects/{project['id']}").json() == {"deleted": project["id"]}


def test_missing_components_block_the_start(
    studio: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload = studio.post("/api/uploads/sample").json()
    project = studio.post("/api/projects", json=_settings(upload["id"])).json()
    monkeypatch.setattr(catalog.importlib.util, "find_spec", lambda name: None)
    response = studio.post(f"/api/projects/{project['id']}/start", json={})
    assert response.status_code == 400 and "isn't installed" in response.json()["detail"]


def test_finished_project_shows_results(finished: dict[str, Any]) -> None:
    summary = finished["summary"]
    assert summary["simulated"] is True
    assert summary["best"]["name"] == "tfidf-logreg"
    assert summary["best"]["title"] == "Tiny (tfidf-logreg)"
    assert summary["teacher"]["title"] == "Teacher: Practice run"
    assert 0 <= summary["best"]["accuracy"] <= 1
    assert summary["test_examples"] == 24 and summary["labels"] == 6
    assert {"pareto", "learning_curve"} <= set(summary["charts"])
    assert "results.md" in summary["reports"]
    assert finished["stage_index"] == len(finished["stages"])
    assert any("step 5/5" in line for line in finished["log"])
    assert Path(finished["experiment_path"]).exists()


def test_finished_project_predicts_and_packages(
    studio: TestClient, finished: dict[str, Any]
) -> None:
    url = f"/api/projects/{finished['id']}"
    result = studio.post(f"{url}/predict", json={"texts": ["I forgot my password", "VPN is down"]})
    predictions = result.json()["predictions"]
    assert [p["text"] for p in predictions] == ["I forgot my password", "VPN is down"]
    assert all(len(p["alternatives"]) == 2 for p in predictions)
    assert predictions[0]["confidence"] >= predictions[0]["alternatives"][0]["confidence"]
    assert studio.post(f"{url}/predict", json={"texts": ["  "]}).status_code == 400

    zipped = studio.get(f"{url}/download")
    assert zipped.status_code == 200
    with zipfile.ZipFile(io.BytesIO(zipped.content)) as archive:
        names = archive.namelist()
        readme = archive.read(next(n for n in names if n.endswith("README.txt"))).decode()
    assert any(n.endswith("export.json") for n in names)
    assert "heartcut serve --model" in readme and "serve,pipeline" in readme
    assert studio.get(f"{url}/download").content == zipped.content  # built once, then reused

    chart = studio.get(f"{url}/files/{finished['summary']['charts']['pareto']}")
    assert chart.status_code == 200 and chart.headers["content-type"] == "image/png"
    report = studio.get(f"{url}/files/results.md")
    assert report.headers["content-type"].startswith("text/plain")
    assert studio.get(f"{url}/files/..%2Fproject.json").status_code == 404
    assert studio.get(f"{url}/files/nothing.png").status_code == 404


def test_finished_project_labels_spreadsheets(studio: TestClient, finished: dict[str, Any]) -> None:
    url = f"/api/projects/{finished['id']}/label-file"
    single = studio.post(
        f"{url}?filename=new.csv", content=b"message\nreset my password\n\n", headers=OCTET
    )
    assert single.status_code == 200, single.text
    assert 'filename="new-labeled.csv"' in single.headers["content-disposition"]
    rows = pd.read_csv(io.StringIO(single.content.decode("utf-8-sig")))
    assert list(rows.columns) == ["message", "predicted_category", "confidence"]

    both = pd.DataFrame(
        {"subject": ["Printer", "VPN"], "body": ["", "keeps dropping"], "id": [1, 2]}
    )
    excel = io.BytesIO()
    both.to_excel(excel, index=False)
    labeled = studio.post(f"{url}?filename=more.xlsx", content=excel.getvalue(), headers=OCTET)
    rows = pd.read_csv(io.StringIO(labeled.content.decode("utf-8-sig")))
    assert rows["predicted_category"].notna().all() and len(rows) == 2

    wrong = studio.post(f"{url}?filename=x.csv", content=b"a,b\n1,2\n", headers=OCTET)
    assert wrong.status_code == 400 and "subject, body" in wrong.json()["detail"]


def test_project_list_stop_and_delete(studio: TestClient, finished: dict[str, Any]) -> None:
    listed = studio.get("/api/projects").json()
    assert finished["id"] in [p["id"] for p in listed]
    assert all("log" not in p for p in listed)
    url = f"/api/projects/{finished['id']}"
    assert studio.post(f"{url}/start", json={}).status_code == 409  # already done
    assert studio.post(f"{url}/stop").json()["status"] == "done"  # nothing to stop


def test_start_reports_a_busy_studio(studio: TestClient) -> None:
    upload = studio.post("/api/uploads/sample").json()
    first = studio.post("/api/projects", json=_settings(upload["id"], name="First")).json()
    second = studio.post("/api/projects", json=_settings(upload["id"], name="Second")).json()
    assert studio.post(f"/api/projects/{first['id']}/start", json={}).status_code == 200
    busy = studio.post(f"/api/projects/{second['id']}/start", json={})
    assert busy.status_code == 409
    assert studio.delete(f"/api/projects/{first['id']}").status_code == 409  # still running
    studio.app.state.runner.wait(180)  # type: ignore[attr-defined]
    assert studio.delete(f"/api/projects/{first['id']}").status_code == 200


def test_api_keys_and_budgets_are_applied_on_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[tuple[str, Any]] = []
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        Runner, "start", lambda self, project, keys=None: started.append((project.id, keys))
    )
    client = TestClient(create_app(tmp_path / "ws", allowed_hosts=["testserver"]))
    upload = client.post("/api/uploads/sample").json()
    project = client.post(
        "/api/projects", json=_settings(upload["id"], teacher="claude-opus-5-5")
    ).json()
    url = f"/api/projects/{project['id']}/start"
    assert client.post(url, json={"api_key": "sk-test", "max_budget_usd": 1.5}).status_code == 200
    assert started == [(project["id"], {"ANTHROPIC_API_KEY": "sk-test"})]
    experiment = yaml.safe_load(Path(project["experiment_path"]).read_text())
    assert experiment["teacher"]["max_budget_usd"] == 1.5
    assert "sk-test" not in json.dumps(client.get(f"/api/projects/{project['id']}").json())
    teachers = {t["id"]: t for t in client.get("/api/catalog").json()["teachers"]}
    assert teachers["claude-opus-5-5"]["has_key"]  # remembered for this session


# --------------------------------------------------------------------------- command line


def test_studio_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: calls.append({"app": app, **kwargs}))
    result = CliRunner().invoke(
        cli, ["studio", "--workspace", str(tmp_path / "ws"), "--no-browser", "--port", "9999"]
    )
    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:9999" in result.output
    assert calls[0]["host"] == "127.0.0.1" and calls[0]["port"] == 9999
    assert (tmp_path / "ws" / "projects").is_dir()

    shared = CliRunner().invoke(
        cli, ["studio", "--workspace", str(tmp_path / "ws"), "--no-browser", "--host", "0.0.0.0"]
    )
    assert shared.exit_code == 0 and "no login" in shared.output
    assert (
        TestClient(calls[1]["app"], base_url="http://192.168.1.5").get("/api/catalog").status_code
        == 200
    )

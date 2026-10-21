"""Studio uploads and projects on disk.

Layout of a studio workspace::

    <workspace>/
        uploads/<upload id>/        the file as it was dropped, plus upload.json
        projects/<project id>/
            project.json            what the person chose and how the run is going
            experiment.yaml         the same experiment as a config the CLI can re-run
            data/train.jsonl        their examples, split once into train and test
            data/test.jsonl
            artifacts/, results/    the usual stage outputs
            run.log
        cache/teacher-cache.sqlite  teacher answers, shared by every project

A project's settings can change while it is a draft; starting a run freezes them.
"""

from __future__ import annotations

import json
import re
import secrets
import shutil
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from heartcut.artifacts import utc_now
from heartcut.config import ExperimentConfig, load_experiment
from heartcut.data.splits import stratified_split
from heartcut.studio import catalog
from heartcut.studio.tables import (
    TableError,
    build_examples,
    describe_columns,
    guess_columns,
    preview,
    profile,
    read_table,
)

ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
TEST_FRACTION = 0.2
VALIDATION_FRACTION = 0.15
SEED = 13
THOROUGH_SEEDS = [0, 1, 2]
# Smaller labeling budgets for the learning curve of a thorough run.
LEARNING_BUDGETS = (25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000)
# A quick run measures latency with fewer requests; the numbers stay comparable.
QUICK_LATENCY = {"warmup": 5, "requests": 100, "throughput_seconds": 1.0}

STATE_FILE = "project.json"
UPLOAD_FILE = "upload.json"
EDITABLE = ("draft",)
STARTABLE = ("draft", "failed", "stopped")


class Settings(BaseModel):
    """Everything a person chooses in the studio's new-project wizard."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    upload_id: str
    text_columns: list[str] = Field(min_length=1)
    label_column: str
    task: str = Field(min_length=3, max_length=300)
    teacher: str
    students: list[str] = Field(min_length=1)
    thorough: bool = False
    max_budget_usd: float | None = Field(default=5.0, gt=0)
    max_train_examples: int | None = Field(default=None, ge=10)
    local_model: str | None = Field(default=None, max_length=200)
    local_url: str | None = Field(default=None, max_length=500)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40].strip("-")
    return slug or "project"


def _new_id(prefix: str) -> str:
    return f"{_slug(prefix)}-{secrets.token_hex(3)}"


def _check_id(value: str) -> str:
    if not ID_PATTERN.match(value):
        raise KeyError(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def learning_budgets(pool: int) -> list[int | str]:
    """Up to three smaller budgets below the training pool, then ``full``."""
    smaller = [b for b in LEARNING_BUDGETS if b <= 0.6 * pool]
    return [*smaller[-3:], "full"]


# --------------------------------------------------------------------------- uploads


@dataclass(frozen=True)
class Upload:
    id: str
    directory: Path
    filename: str
    sample: bool = False

    @property
    def path(self) -> Path:
        return self.directory / self.filename


class Workspace:
    """Every upload and project of one studio, under one folder."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.uploads_dir = self.root / "uploads"
        self.projects_dir = self.root / "projects"
        for directory in (self.uploads_dir, self.projects_dir, self.root / "cache"):
            directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._frames: OrderedDict[str, pd.DataFrame] = OrderedDict()

    @property
    def teacher_cache(self) -> Path:
        return self.root / "cache" / "teacher-cache.sqlite"

    # ------------------------------------------------------------------ uploads

    def save_upload(self, data: bytes, filename: str, *, sample: bool = False) -> Upload:
        """Check that the file parses, then keep it (bad files are never stored)."""
        name = Path(filename).name.strip() or "upload.csv"
        frame = read_table(data, name)
        upload_id = _new_id(Path(name).stem)
        directory = self.uploads_dir / upload_id
        directory.mkdir()
        safe = re.sub(r"[^\w.\- ]", "_", name)
        (directory / safe).write_bytes(data)
        meta = {"filename": safe, "sample": sample, "uploaded_at": utc_now()}
        _write_json(directory / UPLOAD_FILE, meta)
        upload = Upload(upload_id, directory, safe, sample)
        self._remember(upload_id, frame)
        return upload

    def save_sample(self) -> Upload:
        sample = catalog.resource(catalog.SAMPLE_DATA)
        return self.save_upload(sample.read_bytes(), "helpdesk-tickets.jsonl", sample=True)

    def upload(self, upload_id: str) -> Upload:
        directory = self.uploads_dir / _check_id(upload_id)
        if not (directory / UPLOAD_FILE).exists():
            raise KeyError(upload_id)
        meta = json.loads((directory / UPLOAD_FILE).read_text(encoding="utf-8"))
        return Upload(upload_id, directory, str(meta["filename"]), bool(meta.get("sample")))

    def frame(self, upload: Upload) -> pd.DataFrame:
        with self.lock:
            if upload.id in self._frames:
                self._frames.move_to_end(upload.id)
                return self._frames[upload.id]
        frame = read_table(upload.path.read_bytes(), upload.filename)
        self._remember(upload.id, frame)
        return frame

    def _remember(self, upload_id: str, frame: pd.DataFrame) -> None:
        with self.lock:
            self._frames[upload_id] = frame
            while len(self._frames) > 4:
                self._frames.popitem(last=False)

    def describe_upload(self, upload: Upload) -> dict[str, Any]:
        frame = self.frame(upload)
        guess: dict[str, Any] = (
            {
                "text_columns": list(catalog.SAMPLE_TEXT_COLUMNS),
                "label_column": catalog.SAMPLE_LABEL,
            }
            if upload.sample
            else guess_columns(frame)
        )
        return {
            "id": upload.id,
            "filename": upload.filename,
            "sample": upload.sample,
            "rows": len(frame),
            "columns": describe_columns(frame),
            "preview": preview(frame),
            "guess": guess,
            "task": catalog.SAMPLE_TASK if upload.sample else None,
            "profile": profile(frame, guess["text_columns"], guess["label_column"]),
        }

    # ------------------------------------------------------------------ projects

    def project(self, project_id: str) -> Project:
        directory = self.projects_dir / _check_id(project_id)
        if not (directory / STATE_FILE).exists():
            raise KeyError(project_id)
        return Project(directory, self.lock)

    def projects(self) -> list[Project]:
        found = [Project(p.parent, self.lock) for p in self.projects_dir.glob(f"*/{STATE_FILE}")]
        return sorted(found, key=lambda p: str(p.state().get("created_at", "")), reverse=True)

    def save_project(self, settings: Settings, project_id: str | None = None) -> Project:
        """Create a draft project, or rewrite a draft with new settings."""
        upload = self.upload(settings.upload_id)
        frame = self.frame(upload)
        check = profile(frame, settings.text_columns, settings.label_column)
        if not check["ready"]:
            raise TableError(check["errors"][0])
        try:
            catalog.teacher(settings.teacher)
            for name in settings.students:
                catalog.student(name)
        except KeyError as exc:
            raise TableError(f"Unknown choice: {exc.args[0]}") from exc
        if len(set(settings.students)) != len(settings.students):
            raise TableError("Each model can be in the line-up only once.")

        if project_id is None:
            project = Project(self.projects_dir / _new_id(settings.name), self.lock)
            project.directory.mkdir()
            try:
                return self._write_project(project, settings, upload, frame, utc_now())
            except Exception:
                shutil.rmtree(project.directory, ignore_errors=True)
                raise
        else:
            project = self.project(project_id)
            state = project.state()
            if state["status"] not in EDITABLE:
                raise TableError("This project has already run; start a new project instead.")
            for stale in (project.data_dir, project.artifacts_dir, project.results_dir):
                shutil.rmtree(stale, ignore_errors=True)
            return self._write_project(project, settings, upload, frame, state["created_at"])

    def _write_project(
        self,
        project: Project,
        settings: Settings,
        upload: Upload,
        frame: pd.DataFrame,
        created: str,
    ) -> Project:
        examples = build_examples(frame, settings.text_columns, settings.label_column)
        train, test = stratified_split(examples, TEST_FRACTION, SEED)
        project.data_dir.mkdir(parents=True)
        train.to_json(project.data_dir / "train.jsonl", orient="records", lines=True)
        test.to_json(project.data_dir / "test.jsonl", orient="records", lines=True)
        experiment = build_experiment(project, settings, upload.filename, len(train))
        project.experiment_path.write_text(
            "# Written by Heartcut Studio. Re-run it from the command line with:\n"
            f"#   heartcut run -c {project.experiment_path}\n"
            + yaml.safe_dump(experiment, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        project.write_state(
            {
                "id": project.id,
                "name": settings.name,
                "created_at": created,
                "updated_at": utc_now(),
                "source_file": upload.filename,
                "settings": settings.model_dump(mode="json"),
                "status": "draft",
                "stage": None,
                "stage_index": None,
                "stage_detail": None,
                "started_at": None,
                "finished_at": None,
                "error": None,
                "data": {
                    "examples": len(examples),
                    "categories": int(examples["label"].nunique()),
                },
                "estimate": None,
                "summary": None,
            }
        )
        return project

    def delete_project(self, project_id: str) -> None:
        shutil.rmtree(self.project(project_id).directory)


def build_experiment(
    project: Project, settings: Settings, filename: str, train_rows: int
) -> dict[str, Any]:
    """The experiment config for a project, validated before it is written."""
    choice = catalog.teacher(settings.teacher)
    teacher = choice.load()
    if choice.kind != "practice" and settings.max_budget_usd is not None:
        teacher["max_budget_usd"] = settings.max_budget_usd
    if choice.kind == "local":
        teacher["model"] = settings.local_model or teacher["model"]
        teacher["base_url"] = settings.local_url or teacher["base_url"]
    dataset: dict[str, Any] = {
        "name": project.id,
        "description": f"{settings.name}: {filename}, uploaded to Heartcut Studio",
        "task": settings.task,
        "sources": [
            {
                "loader": "jsonl",
                "args": {
                    "files": {
                        "train": str(project.data_dir / "train.jsonl"),
                        "test": str(project.data_dir / "test.jsonl"),
                    }
                },
            }
        ],
        "validation_fraction": VALIDATION_FRACTION,
        "seed": SEED,
    }
    pool = int(train_rows * (1 - VALIDATION_FRACTION))
    if settings.max_train_examples:
        dataset["max_examples"] = {"train": settings.max_train_examples}
        pool = min(pool, settings.max_train_examples)
    experiment: dict[str, Any] = {
        "name": project.id,
        "dataset": dataset,
        "teacher": teacher,
        "students": [catalog.student(name).load() for name in settings.students],
        "seeds": THOROUGH_SEEDS if settings.thorough else [0],
        "budgets": learning_budgets(pool) if settings.thorough else ["full"],
        "tracking": {"backend": "none"},
        "artifacts_dir": str(project.artifacts_dir),
        "results_dir": str(project.results_dir),
    }
    if not settings.thorough:
        experiment["benchmark"] = {"latency": dict(QUICK_LATENCY)}
    ExperimentConfig.model_validate(experiment)
    return experiment


class Project:
    def __init__(self, directory: Path, lock: threading.RLock) -> None:
        self.directory = directory
        self.id = directory.name
        self._lock = lock

    @property
    def state_path(self) -> Path:
        return self.directory / STATE_FILE

    @property
    def experiment_path(self) -> Path:
        return self.directory / "experiment.yaml"

    @property
    def log_path(self) -> Path:
        return self.directory / "run.log"

    @property
    def data_dir(self) -> Path:
        return self.directory / "data"

    @property
    def artifacts_dir(self) -> Path:
        return self.directory / "artifacts"

    @property
    def results_dir(self) -> Path:
        return self.directory / "results"

    def state(self) -> dict[str, Any]:
        with self._lock:
            state: dict[str, Any] = json.loads(self.state_path.read_text(encoding="utf-8"))
        return state

    def write_state(self, state: dict[str, Any]) -> None:
        with self._lock:
            _write_json(self.state_path, state)

    def update(self, **fields: Any) -> dict[str, Any]:
        with self._lock:
            state = self.state()
            state.update(fields, updated_at=utc_now())
            self.write_state(state)
        return state

    @property
    def settings(self) -> Settings:
        return Settings.model_validate(self.state()["settings"])

    def config(self) -> ExperimentConfig:
        return load_experiment(self.experiment_path)

    def set_budget(self, max_budget_usd: float) -> None:
        """Change the teacher's spending limit (a runtime knob: cached labels stay valid)."""
        settings = self.settings.model_copy(update={"max_budget_usd": max_budget_usd})
        text = self.experiment_path.read_text(encoding="utf-8")
        header = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))
        experiment = yaml.safe_load(text)
        if experiment["teacher"]["backend"] != "simulated":
            experiment["teacher"]["max_budget_usd"] = max_budget_usd
        ExperimentConfig.model_validate(experiment)
        self.experiment_path.write_text(
            header + yaml.safe_dump(experiment, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        self.update(settings=settings.model_dump(mode="json"))

    def log_tail(self, lines: int = 200) -> list[str]:
        if not self.log_path.exists():
            return []
        return self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]

    def results_file(self, name: str) -> Path:
        """A file the benchmark wrote (charts, reports); never anything outside results/."""
        if "/" in name or "\\" in name or name.startswith("."):
            raise KeyError(name)
        path = self.config().experiment_results_dir / name
        if not path.is_file():
            raise KeyError(name)
        return path

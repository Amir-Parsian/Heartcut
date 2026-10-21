"""Run a studio project's pipeline in the background and report progress.

One project runs at a time, in a worker thread, through the same stage functions as
``heartcut run``. Each stage's start is written to ``project.json`` so the browser can
draw a progress bar, every log line of the run goes to ``run.log``, and failures are
turned into a short explanation with a next step. Finished stages are cached by their
fingerprints, so pressing *Resume* after a failure or a stop redoes nothing that
already completed (and never pays the teacher twice).
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from heartcut.artifacts import utc_now
from heartcut.config import ExperimentConfig
from heartcut.studio import catalog
from heartcut.studio.catalog import student_title
from heartcut.studio.projects import STARTABLE, Project, Workspace
from heartcut.teachers.base import FatalTeacherError
from heartcut.teachers.runner import BudgetExceeded

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage:
    id: str
    title: str
    run: Callable[[ExperimentConfig, Workspace], Any]


def _prepare(cfg: ExperimentConfig, _: Workspace) -> None:
    from heartcut.data.prepare import prepare_data

    prepare_data(cfg)


def _label(cfg: ExperimentConfig, workspace: Workspace) -> None:
    from heartcut.teachers.labeling import run_labeling

    run_labeling(cfg, cache_path=workspace.teacher_cache)


def _audit(cfg: ExperimentConfig, _: Workspace) -> None:
    from heartcut.eval.audit import run_audit

    run_audit(cfg)


def _train(cfg: ExperimentConfig, _: Workspace) -> None:
    from heartcut.students.training import run_training

    run_training(cfg)


def _benchmark(cfg: ExperimentConfig, _: Workspace) -> None:
    from heartcut.eval.benchmark import run_benchmark

    run_benchmark(cfg)


STAGES = (
    Stage("prepare", "Getting your data ready", _prepare),
    Stage("label", "The teacher labels your messages", _label),
    Stage("audit", "Checking the teacher's answers against yours", _audit),
    Stage("train", "Training the small models", _train),
    Stage("benchmark", "Testing every model on messages it has never seen", _benchmark),
)


class Busy(RuntimeError):
    """Another project is already running."""


class _RunLog(logging.Handler):
    """Copies the worker thread's ``heartcut`` log lines into the project's run.log."""

    def __init__(self, project: Project, thread_id: int) -> None:
        super().__init__(logging.INFO)
        self.project = project
        self.thread_id = thread_id
        self.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self.thread_id:
            return
        try:
            with self.project.log_path.open("a", encoding="utf-8") as fh:
                fh.write(self.format(record) + "\n")
            detail = progress_note(record.getMessage())
            if detail:
                self.project.update(stage_detail=detail)
        except Exception:  # pragma: no cover - logging must never break a run
            self.handleError(record)


_SPLIT_NAMES = {"train": "learning", "validation": "selection", "test": "exam"}
_LABELING = re.compile(r"^(train|validation|test): (\d+)/(\d+) calls")
_TRAINING = re.compile(r"^training (.+)-b(\w+)-s(\d+)-\w+ on (\d+) teacher labels")
_EXPORTED = re.compile(r"^exported (\S+) ")


def progress_note(message: str) -> str | None:
    """A short progress line for the browser from a pipeline log message, if it has one."""
    if match := _LABELING.match(message):
        split, done, total = match.groups()
        return f"{int(done):,} of {int(total):,} answers for the {_SPLIT_NAMES[split]} messages"
    if match := _TRAINING.match(message):
        name, _, seed, examples = match.groups()
        return f"Training {student_title(name)} on {int(examples):,} examples (run {int(seed) + 1})"
    if match := _EXPORTED.match(message):
        return f"Packaging {student_title(match.group(1))} for download"
    return None


def explain(exc: BaseException, project: Project) -> dict[str, str]:
    """A failure in words for the person running the studio, with what to do next."""
    if isinstance(exc, BudgetExceeded):
        budget = project.settings.max_budget_usd or 0.0
        return {
            "code": "budget",
            "title": "Spending limit reached",
            "detail": (
                f"The teacher stopped after spending your limit of ${budget:,.2f} in this run. "
                "Every answer so far is saved and is never paid for again. Resume to label "
                "the rest: it may spend up to the limit again, which you can change here."
            ),
        }
    if isinstance(exc, FatalTeacherError):
        message = str(exc)
        if message.startswith(("401", "403")):
            return {
                "code": "key",
                "title": "The API key was not accepted",
                "detail": f"The teacher's provider said: {message}. Paste a valid key and resume.",
            }
        return {"code": "teacher", "title": "The teacher refused the request", "detail": message}
    if isinstance(exc, ImportError):
        return {
            "code": "install",
            "title": "A component is not installed",
            "detail": (
                f"{exc}. Install the missing parts with "
                "pip install 'heartcut[studio,teachers,students,export]' and restart the studio."
            ),
        }
    choice = catalog.teacher(project.settings.teacher)
    if choice.kind == "local" and "connect" in str(exc).lower():
        return {
            "code": "local",
            "title": "Could not reach the local model",
            "detail": f"Is Ollama running, with the model downloaded? ({exc})",
        }
    return {
        "code": "error",
        "title": "Something went wrong",
        "detail": f"{type(exc).__name__}: {exc}",
    }


class Runner:
    """Runs one project at a time in a background thread."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._active: str | None = None
        self._stop = threading.Event()

    @property
    def active(self) -> str | None:
        return self._active

    def start(self, project: Project, keys: Mapping[str, str] | None = None) -> None:
        """Start (or resume) a project; API keys are set in this process's environment."""
        with self._lock:
            if self._active is not None:
                raise Busy(f"{self._active} is still running")
            status = project.state()["status"]
            if status not in STARTABLE:
                raise ValueError(f"a {status} project cannot be started")
            for name, value in (keys or {}).items():
                os.environ[name] = value
            self._stop.clear()
            self._active = project.id
            project.update(
                status="running",
                stage=None,
                stage_index=None,
                stage_detail=None,
                error=None,
                started_at=utc_now(),
                finished_at=None,
            )
            self._thread = threading.Thread(
                target=self._run, args=(project,), name=f"studio-{project.id}", daemon=True
            )
            self._thread.start()

    def stop(self, project: Project) -> bool:
        """Ask the running project to stop after its current stage."""
        if self._active != project.id:
            return False
        self._stop.set()
        project.update(stage_detail="Stopping after this step...")
        return True

    def wait(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _run(self, project: Project) -> None:
        handler = _RunLog(project, threading.get_ident())
        package_logger = logging.getLogger("heartcut")
        if package_logger.getEffectiveLevel() > logging.INFO:
            package_logger.setLevel(logging.INFO)
        package_logger.addHandler(handler)
        try:
            cfg = project.config()
            for index, stage in enumerate(STAGES):
                if self._stop.is_set():
                    project.update(status="stopped", finished_at=utc_now(), stage_detail=None)
                    logger.info("stopped before: %s", stage.title)
                    return
                project.update(stage=stage.id, stage_index=index, stage_detail=None)
                logger.info("step %d/%d: %s", index + 1, len(STAGES), stage.title)
                stage.run(cfg, self.workspace)
            from heartcut.studio.summary import summarise

            project.update(
                status="done",
                summary=summarise(project),
                stage=None,
                stage_index=len(STAGES),
                stage_detail=None,
                finished_at=utc_now(),
            )
            logger.info("finished")
        except Exception as exc:
            logger.exception("run failed")
            project.update(status="failed", error=explain(exc, project), finished_at=utc_now())
        finally:
            package_logger.removeHandler(handler)
            with self._lock:
                self._active = None


def recover(workspace: Workspace) -> None:
    """Mark runs that were cut off by closing the studio as stopped, so they can resume."""
    for project in workspace.projects():
        if project.state()["status"] == "running":
            project.update(
                status="stopped",
                stage_detail=None,
                error={
                    "code": "interrupted",
                    "title": "The studio was closed during this run",
                    "detail": "Press Resume: finished steps are reused, nothing is paid twice.",
                },
            )

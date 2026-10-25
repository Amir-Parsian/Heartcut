"""The studio's web server: a JSON API and the single-page app in ``static/``.

The server is meant to run on the person's own computer (``heartcut studio`` binds to
127.0.0.1). Requests must name a local host, which stops other web sites from reaching
it through DNS rebinding, and every request that changes something needs a JSON or
``application/octet-stream`` body, which a cross-site form cannot send. API keys pasted
into the page are kept in this process's memory and environment only, never on disk.
"""

from __future__ import annotations

import os
import threading
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from heartcut import __version__
from heartcut.serve.runtime import Predictor, load_predictor
from heartcut.studio import catalog, hub
from heartcut.studio.projects import Project, Settings, Workspace
from heartcut.studio.runner import STAGES, Busy, Runner, recover
from heartcut.studio.tables import (
    FORMATS,
    MAX_ROWS,
    MIN_ROWS,
    TableError,
    join_columns,
    profile,
    read_table,
)

STATIC = Path(__file__).parent / "static"
LOOPBACK_HOSTS = ["127.0.0.1", "localhost", "::1", "[::1]"]
MAX_UPLOAD_MB = 50
MAX_TEXTS = 50
MAX_CHARS = 2000
PREDICT_BATCH = 256
TOP_K = 3


class ColumnChoice(BaseModel):
    text_columns: list[str] = Field(default_factory=list)
    label_column: str | None = None


class HubRequest(BaseModel):
    dataset: str = Field(min_length=1, max_length=200)
    config: str | None = Field(default=None, max_length=200)
    split: str | None = Field(default=None, max_length=200)
    revision: str | None = Field(default=None, max_length=200)
    max_rows: int = Field(default=hub.DEFAULT_ROWS, ge=MIN_ROWS, le=MAX_ROWS)
    token: str | None = Field(default=None, max_length=500)


class StartRequest(BaseModel):
    api_key: str | None = Field(default=None, max_length=500)
    max_budget_usd: float | None = Field(default=None, gt=0)


class PredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_TEXTS)


def create_app(workspace: Path, *, allowed_hosts: list[str] | None = None) -> FastAPI:
    """The studio for one workspace folder (created if missing)."""
    ws = Workspace(workspace)
    recover(ws)
    runner = Runner(ws)
    keys: dict[str, str] = {}  # API keys pasted into the page during this session
    predictors: dict[str, Predictor] = {}
    predictors_lock = threading.Lock()

    app = FastAPI(title="Heartcut Studio", version=__version__, docs_url=None, redoc_url=None)
    app.state.workspace = ws
    app.state.runner = runner
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts or LOOPBACK_HOSTS)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.exception_handler(TableError)
    async def _table_error(_: Request, exc: TableError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    def get_project(project_id: str) -> Project:
        try:
            return ws.project(project_id)
        except KeyError as exc:
            raise HTTPException(404, "This project doesn't exist (any more).") from exc

    def view(project: Project, *, log: bool = True) -> dict[str, Any]:
        state = project.state()
        state["running"] = runner.active == project.id
        state["stages"] = [{"id": s.id, "title": s.title} for s in STAGES]
        state["experiment_path"] = str(project.experiment_path)
        state["teacher_title"] = catalog.teacher(state["settings"]["teacher"]).title
        if log:
            state["log"] = project.log_tail()
        return state

    def predictor(project: Project) -> Predictor:
        summary = project.state().get("summary")
        if not summary:
            raise HTTPException(409, "This project hasn't finished training yet.")
        export_dir = str(summary["export_dir"])
        with predictors_lock:
            if export_dir not in predictors:
                predictors[export_dir] = load_predictor(Path(export_dir))
            return predictors[export_dir]

    async def read_upload(request: Request) -> bytes:
        if request.headers.get("content-type", "").split(";")[0] != "application/octet-stream":
            raise HTTPException(415, "Send the file as application/octet-stream.")
        limit = MAX_UPLOAD_MB * 1_000_000
        too_big = HTTPException(413, f"Files can be up to {MAX_UPLOAD_MB} MB.")
        if int(request.headers.get("content-length") or 0) > limit:
            raise too_big
        body = await request.body()
        if len(body) > limit:
            raise too_big
        return body

    def prepare(project: Project) -> None:
        """Run the data stage and the teacher's cost estimate for a draft."""
        from heartcut.data.prepare import data_manifest, prepare_data
        from heartcut.teachers.labeling import dry_run_estimate

        cfg = project.config()
        try:
            prepare_data(cfg)
        except Exception as exc:
            raise TableError(f"Heartcut couldn't prepare this data: {exc}") from exc
        state = project.state()
        project.update(
            data={**state["data"], **data_manifest(cfg)["counts"]},
            estimate=dry_run_estimate(cfg),
        )

    # ------------------------------------------------------------------ pages

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/catalog")
    def get_catalog() -> dict[str, Any]:
        return {
            "version": __version__,
            "workspace": str(ws.root),
            "teachers": [t.describe(keys) for t in catalog.TEACHERS],
            "students": [s.describe() for s in catalog.STUDENTS],
            "limits": {
                "max_upload_mb": MAX_UPLOAD_MB,
                "min_rows": MIN_ROWS,
                "max_rows": MAX_ROWS,
                "formats": sorted(FORMATS),
            },
            "hub": {
                "available": hub.available(),
                "install": hub.INSTALL,
                "has_token": bool(keys.get(hub.TOKEN_ENV) or hub.saved_token()),
                "default_rows": hub.DEFAULT_ROWS,
            },
            "active": runner.active,
        }

    # ------------------------------------------------------------------ uploads

    @app.post("/api/uploads")
    async def upload(request: Request, filename: str) -> dict[str, Any]:
        body = await read_upload(request)
        saved = await run_in_threadpool(ws.save_upload, body, filename)
        return await run_in_threadpool(ws.describe_upload, saved)

    @app.post("/api/uploads/sample")
    def upload_sample() -> dict[str, Any]:
        return ws.describe_upload(ws.save_sample())

    @app.post("/api/uploads/huggingface")
    def upload_from_hub(request: HubRequest) -> dict[str, Any]:
        if request.token and request.token.strip():
            keys[hub.TOKEN_ENV] = request.token.strip()
        options = request.model_dump(exclude={"dataset", "token"})
        saved = ws.save_hub_dataset(request.dataset, token=keys.get(hub.TOKEN_ENV), **options)
        return ws.describe_upload(saved)

    @app.post("/api/uploads/{upload_id}/profile")
    def profile_upload(upload_id: str, choice: ColumnChoice) -> dict[str, Any]:
        try:
            saved = ws.upload(upload_id)
        except KeyError as exc:
            raise HTTPException(404, "This upload doesn't exist; drop the file again.") from exc
        return profile(ws.frame(saved), choice.text_columns, choice.label_column)

    # ------------------------------------------------------------------ projects

    @app.get("/api/projects")
    def list_projects() -> list[dict[str, Any]]:
        return [view(p, log=False) for p in ws.projects()]

    @app.post("/api/projects")
    def create_project(settings: Settings) -> dict[str, Any]:
        project = ws.save_project(settings)
        try:
            prepare(project)
        except Exception:
            ws.delete_project(project.id)
            raise
        return view(project)

    @app.put("/api/projects/{project_id}")
    def update_project(project_id: str, settings: Settings) -> dict[str, Any]:
        project = ws.save_project(settings, get_project(project_id).id)
        prepare(project)
        return view(project)

    @app.get("/api/projects/{project_id}")
    def get_project_view(project_id: str) -> dict[str, Any]:
        return view(get_project(project_id))

    @app.post("/api/projects/{project_id}/start")
    def start(project_id: str, request: StartRequest) -> dict[str, Any]:
        project = get_project(project_id)
        settings = project.settings
        choice = catalog.teacher(settings.teacher)
        teacher = choice.describe(keys)
        key_env = teacher["api_key_env"]
        if key_env and request.api_key and request.api_key.strip():
            keys[key_env] = request.api_key.strip()
        if key_env and not (os.environ.get(key_env) or keys.get(key_env)):
            raise HTTPException(400, f"{choice.title} needs an API key: paste it in step 3.")
        missing = [teacher, *(catalog.student(s).describe() for s in settings.students)]
        for item in missing:
            if not item["available"]:
                raise HTTPException(
                    400, f"{item['title']} isn't installed. Install it with: {item['install']}"
                )
        if request.max_budget_usd is not None:
            project.set_budget(request.max_budget_usd)
        try:
            runner.start(project, {key_env: keys[key_env]} if key_env in keys else None)
        except Busy as exc:
            raise HTTPException(
                409, "Another project is training right now. Wait for it to finish."
            ) from exc
        except ValueError as exc:
            raise HTTPException(409, "This project has already finished.") from exc
        return view(project)

    @app.post("/api/projects/{project_id}/stop")
    def stop(project_id: str) -> dict[str, Any]:
        project = get_project(project_id)
        runner.stop(project)
        return view(project)

    @app.delete("/api/projects/{project_id}")
    def delete_project(project_id: str) -> dict[str, str]:
        project = get_project(project_id)
        if runner.active == project.id:
            raise HTTPException(409, "Stop this project before deleting it.")
        summary = project.state().get("summary") or {}
        with predictors_lock:
            predictors.pop(str(summary.get("export_dir")), None)
        ws.delete_project(project.id)
        return {"deleted": project.id}

    # ------------------------------------------------------------------ using the model

    @app.post("/api/projects/{project_id}/predict")
    def predict(project_id: str, request: PredictRequest) -> dict[str, Any]:
        model = predictor(get_project(project_id))
        texts = [t.strip()[:MAX_CHARS] for t in request.texts]
        if not all(texts):
            raise HTTPException(400, "Type a message first.")
        proba = model.predict_proba(texts)
        out = []
        for text, row in zip(texts, proba, strict=True):
            top = np.argsort(row)[::-1][:TOP_K]
            ranked = [{"label": model.labels[i], "confidence": float(row[i])} for i in top]
            out.append({"text": text, **ranked[0], "alternatives": ranked[1:]})
        return {"predictions": out}

    @app.post("/api/projects/{project_id}/label-file")
    async def label_file(project_id: str, request: Request, filename: str) -> Response:
        project = get_project(project_id)
        body = await read_upload(request)
        return await run_in_threadpool(_label_table, project, predictor(project), body, filename)

    @app.get("/api/projects/{project_id}/download")
    def download(project_id: str) -> FileResponse:
        project = get_project(project_id)
        state = project.state()
        if not state.get("summary"):
            raise HTTPException(409, "This project hasn't finished training yet.")
        path = _model_zip(project, state)
        return FileResponse(path, media_type="application/zip", filename=path.name)

    @app.get("/api/projects/{project_id}/files/{name}")
    def results_file(project_id: str, name: str) -> FileResponse:
        try:
            path = get_project(project_id).results_file(name)
        except KeyError as exc:
            raise HTTPException(404, "No such file.") from exc
        media = "text/plain; charset=utf-8" if path.suffix in (".md", ".json") else None
        return FileResponse(path, media_type=media)

    return app


def _label_table(project: Project, model: Predictor, body: bytes, filename: str) -> Response:
    """Add the model's category and confidence to every row of a spreadsheet, as CSV."""
    frame = read_table(body, filename)
    wanted = project.settings.text_columns
    columns = [c for c in wanted if c in frame.columns]
    if len(columns) != len(wanted):
        if len(frame.columns) != 1:
            raise TableError(
                f"This file needs the same message column(s) as your training file "
                f"({', '.join(wanted)}), or just one column of messages."
            )
        columns = list(frame.columns)
    texts = [t[:MAX_CHARS] for t in join_columns(frame, columns)]
    rows = [i for i, t in enumerate(texts) if t]
    labels: list[str | None] = [None] * len(texts)
    confidence: list[float | None] = [None] * len(texts)
    for start in range(0, len(rows), PREDICT_BATCH):
        batch = rows[start : start + PREDICT_BATCH]
        proba = model.predict_proba([texts[i] for i in batch])
        for i, row in zip(batch, proba, strict=True):
            labels[i] = model.labels[int(row.argmax())]
            confidence[i] = round(float(row.max()), 4)
    frame["predicted_category"] = labels
    frame["confidence"] = confidence
    # A byte-order mark makes Excel open the file as UTF-8.
    content = "﻿" + frame.to_csv(index=False)
    name = f"{Path(filename).stem or 'messages'}-labeled.csv"
    return Response(
        content.encode("utf-8"),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


def _model_zip(project: Project, state: dict[str, Any]) -> Path:
    """The best student's export folder as a zip, with a README for whoever gets it."""
    summary = state["summary"]
    export_dir = Path(summary["export_dir"])
    path = project.directory / "downloads" / f"{project.id}-{export_dir.name}.zip"
    if path.exists():
        return path
    path.parent.mkdir(exist_ok=True)
    folder = f"{project.id}-model"
    extras = "serve" if summary["best"]["format"] == "onnx" else "serve,pipeline"
    readme = (
        f"Heartcut model for '{state['name']}': {summary['best']['title']}\n"
        f"It sorts a message into one of {summary['labels']} categories.\n\n"
        "Run it as a web service on any computer with Python 3.11+:\n\n"
        f'    pip install "heartcut[{extras}]"\n'
        f"    heartcut serve --model {folder}\n\n"
        "Then send messages to it:\n\n"
        "    curl -s http://127.0.0.1:8000/predict -H 'content-type: application/json' \\\n"
        '         -d \'{"text": "your message here"}\'\n\n'
        "The answer has the category (label) and the model's confidence.\n"
    )
    tmp = path.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{folder}/README.txt", readme)
        for file in sorted(p for p in export_dir.rglob("*") if p.is_file()):
            archive.write(file, f"{folder}/{file.relative_to(export_dir).as_posix()}")
    tmp.replace(path)
    return path

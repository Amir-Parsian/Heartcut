"""``distillkit`` command-line interface.

Each pipeline stage is a separate command that reads an experiment config and writes
versioned outputs, so stages can be re-run independently::

    distillkit data      -c configs/experiments/banking77.yaml
    distillkit label     -c configs/experiments/banking77.yaml --dry-run
    distillkit label     -c configs/experiments/banking77.yaml
    distillkit audit     -c configs/experiments/banking77.yaml
    distillkit train     -c configs/experiments/banking77.yaml
    distillkit benchmark -c configs/experiments/banking77.yaml
    distillkit export    -c configs/experiments/banking77.yaml
    distillkit serve     --model artifacts/banking77/export/<student>-<fp>

Any config value can be overridden with ``--set key.path=value`` (repeatable), e.g.
``--set dataset.max_examples.train=200`` for a cheap development slice.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from distillkit import __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Distill an LLM classifier into a tiny CPU model and benchmark the trade-off.",
)

ConfigOpt = Annotated[
    Path,
    typer.Option("--config", "-c", exists=True, dir_okay=False, help="Experiment YAML file."),
]
SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", "-o", help="Override a config value: key.path=value (repeatable)."),
]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpx2", "urllib3", "filelock", "datasets", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _load(config: Path, overrides: list[str] | None, verbose: bool) -> Any:
    from distillkit.config import load_experiment

    _setup_logging(verbose)
    return load_experiment(config, overrides or [])


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"distillkit {__version__}")
        raise typer.Exit


labels_app = typer.Typer(no_args_is_help=True, help="Publish or fetch cached teacher labels.")
app.add_typer(labels_app, name="labels")

RepoOpt = Annotated[
    str, typer.Option(help="Hugging Face dataset repo, e.g. user/distillkit-labels.")
]


@labels_app.command("push")
def labels_push(
    config: ConfigOpt,
    repo_id: RepoOpt,
    set_: SetOpt = None,
    public: Annotated[bool, typer.Option(help="Create the repo as public.")] = False,
    verbose: VerboseOpt = False,
) -> None:
    """Upload teacher labels (never the hidden human labels) to the Hugging Face Hub."""
    from distillkit.teachers.hub import publish_labels

    cfg = _load(config, set_, verbose)
    typer.echo(publish_labels(cfg, repo_id, private=not public))


@labels_app.command("pull")
def labels_pull(
    config: ConfigOpt,
    repo_id: RepoOpt,
    set_: SetOpt = None,
    revision: str | None = None,
    verbose: VerboseOpt = False,
) -> None:
    """Download published teacher labels that match this config's fingerprint."""
    from distillkit.data.prepare import prepare_data
    from distillkit.teachers.hub import fetch_labels

    cfg = _load(config, set_, verbose)
    prepare_data(cfg)
    typer.echo(str(fetch_labels(cfg, repo_id, revision=revision)))


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = None,
) -> None:
    """DistillKit: teacher LLM -> labels -> tiny student -> benchmark -> CPU service."""


@app.command()
def data(
    config: ConfigOpt, set_: SetOpt = None, force: bool = False, verbose: VerboseOpt = False
) -> None:
    """Load the dataset into the common schema and hide the training labels."""
    from distillkit.data.prepare import data_manifest, prepare_data

    cfg = _load(config, set_, verbose)
    prepared = prepare_data(cfg, force=force)
    manifest = data_manifest(cfg)
    typer.echo(f"data: {prepared.directory}")
    _echo_json(
        {
            "counts": manifest["counts"],
            "labels": len(prepared.labels),
            "source": manifest["source_used"]["loader"],
            "overlap": manifest["overlap"],
        }
    )


@app.command()
def label(
    config: ConfigOpt,
    set_: SetOpt = None,
    split: Annotated[list[str] | None, typer.Option(help="Split(s) to label.")] = None,
    dry_run: Annotated[bool, typer.Option(help="Estimate the cost; make no calls.")] = False,
    offline: Annotated[
        bool, typer.Option(help="Fail on any cache miss instead of calling the teacher.")
    ] = False,
    force: bool = False,
    verbose: VerboseOpt = False,
) -> None:
    """Label the unlabeled pool (and the test split, for scoring) with the teacher LLM."""
    from distillkit.data.prepare import prepare_data
    from distillkit.data.schema import SPLITS
    from distillkit.teachers.labeling import dry_run_estimate, run_labeling
    from distillkit.teachers.runner import BudgetExceeded, CacheMiss

    cfg = _load(config, set_, verbose)
    prepare_data(cfg)
    splits = split or list(SPLITS)
    if dry_run:
        _echo_json(dry_run_estimate(cfg, splits))
        return
    try:
        outcome = run_labeling(cfg, splits=splits, offline=offline, force=force)
    except BudgetExceeded as exc:
        typer.echo(
            f"Stopped at the budget cap: {exc}. Finished calls are cached; "
            "raise teacher.max_budget_usd to continue.",
            err=True,
        )
        raise typer.Exit(code=2) from exc
    except CacheMiss as exc:
        typer.echo(f"Offline mode and a call is not cached: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    typer.echo(f"labels: {outcome.directory}{' (up to date)' if outcome.skipped else ''}")
    _echo_json(outcome.cost_report.get("total", {}))


@app.command()
def audit(config: ConfigOpt, set_: SetOpt = None, verbose: VerboseOpt = False) -> None:
    """Measure teacher label quality against the hidden human labels."""
    from distillkit.eval.audit import run_audit

    cfg = _load(config, set_, verbose)
    report = run_audit(cfg)
    pooled = report["pooled"]
    typer.echo(f"audit written to {cfg.experiment_results_dir / 'audit.md'}")
    _echo_json({k: pooled[k] for k in ("examples", "accuracy", "macro_f1", "invalid", "filter")})


@app.command()
def train(
    config: ConfigOpt,
    set_: SetOpt = None,
    student: Annotated[list[str] | None, typer.Option(help="Student name(s); default all.")] = None,
    seed: Annotated[list[int] | None, typer.Option(help="Seed(s); default from config.")] = None,
    budget: Annotated[
        list[str] | None, typer.Option(help="Budget(s): a number or 'full'; default from config.")
    ] = None,
    force: bool = False,
    verbose: VerboseOpt = False,
) -> None:
    """Train students on teacher labels (all students x budgets x seeds by default).

    Prints agreement with the teacher on the teacher-labeled validation set only; test-set
    scores belong to the benchmark, so nothing here invites tuning on them.
    """
    from distillkit.students.training import run_training

    cfg = _load(config, set_, verbose)
    budgets: list[Any] | None = [b if b == "full" else int(b) for b in budget] if budget else None
    results = run_training(cfg, students=student, seeds=seed, budgets=budgets, force=force)
    for r in results:
        typer.echo(
            f"{r['student']:<24} budget={r['budget']:<6} seed={r['seed']} "
            f"train_examples={r['train_examples']:<6} "
            f"val_agreement={r['validation_teacher_agreement']:.4f}"
        )


@app.command()
def benchmark(
    config: ConfigOpt,
    set_: SetOpt = None,
    skip_latency: Annotated[
        bool, typer.Option(help="Skip latency/throughput (no cost per 1k for students).")
    ] = False,
    verbose: VerboseOpt = False,
) -> None:
    """Score teacher and students on the human-labeled test set; write results and charts."""
    from distillkit.eval.benchmark import run_benchmark

    cfg = _load(config, set_, verbose)
    results = run_benchmark(cfg, measure=not skip_latency)
    typer.echo(f"results: {cfg.experiment_results_dir}")
    typer.echo(results["headline"]["text"])


@app.command()
def export(
    config: ConfigOpt,
    set_: SetOpt = None,
    student: Annotated[str | None, typer.Option(help="Student name; default: best.")] = None,
    force: bool = False,
    verbose: VerboseOpt = False,
) -> None:
    """Export a trained student (best full-budget seed) to ONNX int8 for CPU serving."""
    from distillkit.serve.export import run_export
    from distillkit.students.training import run_training

    cfg = _load(config, set_, verbose)
    records = run_training(cfg, budgets=["full"])
    path = run_export(cfg, records, student, force=force)
    typer.echo(str(path))


@app.command()
def report(
    config: ConfigOpt,
    set_: SetOpt = None,
    readme: Annotated[Path, typer.Option(help="README to update.")] = Path("README.md"),
    check: Annotated[
        bool, typer.Option(help="Only verify that the README matches results.json.")
    ] = False,
    verbose: VerboseOpt = False,
) -> None:
    """Write the README results section from results.json (refuses simulated results)."""
    import os

    from distillkit.eval.benchmark import load_results
    from distillkit.eval.report import SimulatedResultsError, update_readme

    cfg = _load(config, set_, verbose)
    results = load_results(cfg.experiment_results_dir)
    rel = Path(os.path.relpath(cfg.experiment_results_dir, readme.parent)).as_posix()
    try:
        in_sync = update_readme(readme, results, rel, check=check)
    except SimulatedResultsError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=4) from exc
    if check and not in_sync:
        typer.echo(f"{readme} is out of date; run `distillkit report -c {config}`", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"{readme} {'is up to date' if check else 'updated'}")


@app.command()
def serve(
    model: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="Exported model directory.")
    ],
    host: str = "127.0.0.1",
    port: int = 8000,
    threads: Annotated[int | None, typer.Option(help="ONNX Runtime intra-op threads.")] = None,
    fallback_threshold: Annotated[
        float | None, typer.Option(help="Route predictions below this confidence to the teacher.")
    ] = None,
    teacher_config: Annotated[
        Path | None, typer.Option(help="Teacher YAML for the fallback router.")
    ] = None,
) -> None:
    """Serve an exported student with FastAPI (POST /predict, GET /health)."""
    import os

    import uvicorn

    os.environ["DISTILLKIT_MODEL_DIR"] = str(model)
    if threads:
        os.environ["DISTILLKIT_THREADS"] = str(threads)
    if fallback_threshold is not None:
        os.environ["DISTILLKIT_FALLBACK_THRESHOLD"] = str(fallback_threshold)
    if teacher_config is not None:
        os.environ["DISTILLKIT_TEACHER_CONFIG"] = str(teacher_config)
    uvicorn.run("distillkit.serve.app:app_from_env", factory=True, host=host, port=port)


@app.command()
def run(
    config: ConfigOpt,
    set_: SetOpt = None,
    offline: Annotated[
        bool, typer.Option(help="Use cached teacher labels only (no API spend).")
    ] = False,
    verbose: VerboseOpt = False,
) -> None:
    """Run every stage: data, label, audit, train, benchmark."""
    from distillkit.data.prepare import prepare_data
    from distillkit.eval.audit import run_audit
    from distillkit.eval.benchmark import run_benchmark
    from distillkit.teachers.labeling import run_labeling

    cfg = _load(config, set_, verbose)
    prepare_data(cfg)
    run_labeling(cfg, offline=offline)
    run_audit(cfg)
    results = run_benchmark(cfg)
    typer.echo(results["headline"]["text"])


if __name__ == "__main__":  # pragma: no cover
    app()

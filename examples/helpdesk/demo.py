"""Run the full helpdesk distillation example from the repository root.

    uv run --extra pipeline python examples/helpdesk/demo.py

The default config needs only the pipeline extra. It uses a simulated teacher and
trains real TF-IDF students, then reloads the exported winner to route new tickets.
See README.md in this directory for real teachers, neural students and HTTP serving.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from heartcut.config import ExperimentConfig, load_experiment
from heartcut.data.prepare import prepare_data
from heartcut.eval.audit import run_audit
from heartcut.eval.benchmark import run_benchmark
from heartcut.serve.runtime import load_predictor, predict
from heartcut.students.training import run_training
from heartcut.teachers.labeling import dry_run_estimate, run_labeling

EXAMPLE_TEXTS = (
    "I forgot my password and cannot sign in. Please reset it.",
    "My VPN disconnects whenever I connect from home.",
    "Please give our new colleague access to the shared folder.",
)


def run_demo(
    cfg: ExperimentConfig,
    *,
    offline: bool = False,
    texts: Sequence[str] = EXAMPLE_TEXTS,
) -> dict[str, Any]:
    """Label, audit, train, benchmark/export, and predict with the saved student."""
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("provide at least one non-empty ticket")
    simulated = cfg.teacher.backend == "simulated"
    if simulated:
        print(
            "SIMULATED TEACHER: uses hidden labels; no API calls. "
            "Scores and teacher costs are illustrative.",
            flush=True,
        )

    print("1/6 Prepare data and hide training labels", flush=True)
    data = prepare_data(cfg)
    print(
        f"    {len(data.train)} train / {len(data.validation)} validation / "
        f"{len(data.test)} test tickets; {len(data.labels)} queues",
        flush=True,
    )

    print("2/6 Label with the teacher (reuse cached labels when available)", flush=True)
    run_labeling(cfg, offline=offline, cache_path=cfg.experiment_dir / "teacher-cache.sqlite")

    print("3/6 Audit teacher labels against hidden human labels", flush=True)
    run_audit(cfg)

    print("4/6 Train and save students on teacher labels", flush=True)
    records = run_training(cfg)
    for record in records:
        target = "soft vote distributions" if record["soft_labels"] else "hard labels"
        print(
            f"    {record['student']}: {record['train_examples']} examples, "
            f"seed={record['seed']}, {target}",
            flush=True,
        )

    print("5/6 Benchmark on held-out labels and export students", flush=True)
    results = run_benchmark(cfg)  # Reuses the students just trained; also checks exports.
    best = next(model for model in results["models"] if model["name"] == results["best_student"])
    export_dir = Path(best["served"]["export_dir"]).resolve()

    print("6/6 Reload the exported winner and route new tickets", flush=True)
    predictor = load_predictor(export_dir, threads=cfg.benchmark.latency.threads)
    predictions, elapsed_ms = predict(predictor, texts)
    examples = []
    for text, prediction in zip(texts, predictions, strict=True):
        print(f"    {prediction.label} ({prediction.confidence:.1%}): {text}", flush=True)
        examples.append(
            {"text": text, "label": prediction.label, "confidence": prediction.confidence}
        )

    summary = {
        "experiment": cfg.name,
        "teacher": cfg.teacher.name,
        "simulated": simulated,
        "training_runs": len(records),
        "best_student": predictor.name,
        "format": predictor.meta["format"],
        "export_dir": str(export_dir),
        "results_dir": str(cfg.experiment_results_dir.resolve()),
        "predictions": examples,
        "prediction_batch_latency_ms": elapsed_ms,
    }
    summary_path = cfg.experiment_results_dir / "demo.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nReports: {cfg.experiment_results_dir.resolve() / 'results.md'}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Export:  {export_dir} ({summary['format']})")
    print(
        "\nServe: uv run --extra pipeline --extra serve heartcut serve "
        f"--model {shlex.quote(str(export_dir))}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("demo.yaml"))
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--offline", action="store_true", help="Use cached teacher labels only.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Prepare data and estimate cost only."
    )
    parser.add_argument(
        "--text", action="append", help="Ticket to classify after training; repeatable."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_experiment(args.config, args.overrides)
    if args.dry_run:
        prepare_data(cfg)
        print(json.dumps(dry_run_estimate(cfg), indent=2))
        return
    run_demo(cfg, offline=args.offline, texts=args.text or EXAMPLE_TEXTS)


if __name__ == "__main__":
    main()

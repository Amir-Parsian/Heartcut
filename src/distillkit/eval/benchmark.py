"""The ``distillkit benchmark`` stage: one table, two charts, one error analysis.

Every model is scored on the same human-labeled test split:

* Teacher rows come from the labeling run on the test split (cached, so regenerating
  them is free): accuracy, macro-F1, logged dollars per call and logged API latency.
  With self-consistency there are two rows: a single call (sample 0, what you would
  deploy) and the majority vote (what the students were trained on).
* Student rows aggregate the full-budget runs over all seeds (mean and std). The
  best seed by *validation* agreement is exported, and the exported artifact (ONNX int8
  for transformer students) is what gets timed, sized and priced, and whose accuracy
  appears as "served accuracy".

Results go to ``results/<experiment>/``: ``results.json`` (every number plus hardware,
library versions and price dates), ``results.md``, ``error_analysis.md`` and the charts.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from distillkit.artifacts import environment_info, utc_now
from distillkit.config import ExperimentConfig
from distillkit.data.prepare import PreparedData, data_manifest, load_prepared
from distillkit.data.schema import ID, LABEL, TEXT
from distillkit.eval import charts
from distillkit.eval.audit import load_audit
from distillkit.eval.cost import blended_cost_per_1k, student_cost_per_1k, teacher_cost_per_1k
from distillkit.eval.latency import measure_latency, measure_throughput
from distillkit.eval.metrics import INVALID, confused_pairs, mean_std, score
from distillkit.eval.report import (
    render_error_analysis,
    render_model_card,
    render_results_markdown,
)
from distillkit.serve.export import best_seed, best_student_name, run_export
from distillkit.serve.runtime import artifact_size_mb, load_predictor
from distillkit.students.training import run_training
from distillkit.teachers.labeling import (
    CONFIDENCE,
    SAMPLE_LABELS,
    TEACHER_LABEL,
    labels_manifest,
    labels_stage,
    load_teacher_labels,
    read_calls,
    read_cost_report,
)

logger = logging.getLogger(__name__)

RESULTS_JSON = "results.json"
RESULTS_MD = "results.md"
ERRORS_MD = "error_analysis.md"
MODEL_CARD = "MODEL_CARD.md"
PREDICT_BATCH = 256


def _stat(values: Sequence[float | None]) -> dict[str, Any]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    mean, std = mean_std(clean)
    return {"mean": mean, "std": std if len(clean) > 1 else None, "n": len(clean)}


def _latency(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95))}


# --------------------------------------------------------------------------- teacher


def teacher_rows(cfg: ExperimentConfig, data: PreparedData) -> list[dict[str, Any]]:
    test = data.test[[ID, LABEL]].merge(load_teacher_labels(cfg, "test"), on=ID, how="left")
    calls = pd.DataFrame([c for c in read_calls(labels_stage(cfg).path) if c["split"] == "test"])
    k = cfg.teacher.self_consistency
    gold = test[LABEL].tolist()
    rows = []

    first = calls.loc[calls["sample_index"] == 0]
    single_pred = [s[0] if s[0] is not None else INVALID for s in test[SAMPLE_LABELS]]
    single = score(gold, single_pred, data.labels)
    single_costs = first["cost_usd"].tolist()
    rows.append(
        {
            "name": cfg.teacher.name,
            "label": f"LLM teacher ({cfg.teacher.name})",
            "role": "teacher",
            "kind": "teacher",
            "variant": "single call",
            "accuracy": _stat([single.accuracy]),
            "macro_f1": _stat([single.macro_f1]),
            "ece": _stat([None]),
            "cost_per_1k_usd": teacher_cost_per_1k(single_costs)
            if all(c is not None for c in single_costs)
            else None,
            "latency": _latency(first["latency_ms"].tolist()),
            "latency_source": "logged API round trip",
            "throughput_rps": None,
            "size_mb": None,
            "predictions": single_pred,
            "confidence": None,
        }
    )
    if k > 1:
        voted = test[TEACHER_LABEL].fillna(INVALID).tolist()
        conf = test[CONFIDENCE].to_numpy(dtype=float)
        majority = score(gold, voted, data.labels, conf, cfg.benchmark.ece_bins)
        per_example = calls.groupby("id")["latency_ms"].max()  # k calls in parallel
        costs = test["cost_usd"].tolist()
        rows.append(
            {
                "name": f"{cfg.teacher.name}-vote{k}",
                "label": f"LLM teacher, vote of {k}",
                "role": "teacher",
                "kind": "teacher",
                "variant": f"majority vote of {k} calls",
                "accuracy": _stat([majority.accuracy]),
                "macro_f1": _stat([majority.macro_f1]),
                "ece": _stat([majority.ece]),
                "cost_per_1k_usd": teacher_cost_per_1k(costs)
                if all(c is not None for c in costs)
                else None,
                "latency": _latency(per_example.tolist()),
                "latency_source": "logged API round trip (slowest of k parallel calls)",
                "throughput_rps": None,
                "size_mb": None,
                "predictions": voted,
                "confidence": conf.tolist(),
            }
        )
    return rows


# --------------------------------------------------------------------------- students


def _predict_all(predictor: Any, texts: Sequence[str]) -> np.ndarray:
    parts = [
        predictor.predict_proba(list(texts[i : i + PREDICT_BATCH]))
        for i in range(0, len(texts), PREDICT_BATCH)
    ]
    return np.concatenate(parts)


def student_row(
    cfg: ExperimentConfig,
    data: PreparedData,
    records: Sequence[dict[str, Any]],
    name: str,
    *,
    measure: bool,
) -> dict[str, Any]:
    runs = [r for r in records if r["student"] == name and r["budget"] == "full"]
    student_cfg = cfg.student(name)
    export_dir = run_export(cfg, records, name)
    lat = cfg.benchmark.latency
    predictor = load_predictor(export_dir, threads=lat.threads)
    texts = data.test[TEXT].tolist()
    proba = _predict_all(predictor, texts)
    pred = [predictor.labels[i] for i in proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    served = score(data.test[LABEL].tolist(), pred, data.labels, conf, cfg.benchmark.ece_bins)

    row: dict[str, Any] = {
        "name": name,
        "label": name,
        "role": "student",
        "kind": student_cfg.kind,
        "base_model": getattr(student_cfg.params, "model_name", None),
        "variant": "served: "
        + ("ONNX int8" if predictor.meta["format"] == "onnx" else "scikit-learn"),
        "accuracy": _stat([r["test"]["accuracy"] for r in runs]),
        "macro_f1": _stat([r["test"]["macro_f1"] for r in runs]),
        "ece": _stat([r["test"]["ece"] for r in runs]),
        "validation_teacher_agreement": _stat([r["validation_teacher_agreement"] for r in runs]),
        "served": {
            "seed": best_seed(records, name),
            "format": predictor.meta["format"],
            "accuracy": served.accuracy,
            "macro_f1": served.macro_f1,
            "ece": served.ece,
            "export_dir": str(export_dir),
            "verification": predictor.meta.get("verification"),
        },
        "size_mb": artifact_size_mb(export_dir),
        "latency": None,
        "latency_source": f"single request on local CPU, {lat.threads or 'all'} threads",
        "throughput_rps": None,
        "cost_per_1k_usd": None,
        "predictions": pred,
        "confidence": conf.tolist(),
    }
    if measure:
        sample = texts[: max(lat.requests, 1)]
        timing = measure_latency(
            predictor.predict_proba, sample, warmup=lat.warmup, requests=lat.requests
        )
        throughput = measure_throughput(
            predictor.predict_proba,
            sample,
            batch_size=lat.throughput_batch_size,
            seconds=lat.throughput_seconds,
        )
        rps = throughput["requests_per_second"]
        row.update(
            latency={"p50_ms": timing["p50_ms"], "p95_ms": timing["p95_ms"]},
            throughput_rps=rps,
            cost_per_1k_usd=student_cost_per_1k(cfg.benchmark.cpu.usd_per_hour, rps),
        )
    return row


# --------------------------------------------------------------------------- analyses


def learning_curve(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((r["student"], r["budget"]), []).append(r)
    out = []
    for (student, budget), runs in groups.items():
        costs = [r["labeling_cost_usd"] for r in runs]
        out.append(
            {
                "student": student,
                "budget": budget,
                "examples": float(np.mean([r["budget_examples"] for r in runs])),
                "train_examples": float(np.mean([r["train_examples"] for r in runs])),
                "accuracy": _stat([r["test"]["accuracy"] for r in runs]),
                "macro_f1": _stat([r["test"]["macro_f1"] for r in runs]),
                "labeling_cost_usd": None if None in costs else float(np.mean(costs)),
            }
        )
    return sorted(out, key=lambda r: (r["student"], r["examples"]))


def routing_table(
    student: dict[str, Any],
    teacher: dict[str, Any],
    gold: Sequence[str],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    """Send predictions below a confidence threshold to the teacher; what would it cost?"""
    conf = np.asarray(student["confidence"])
    s_pred = np.asarray(student["predictions"], dtype=object)
    t_pred = np.asarray(teacher["predictions"], dtype=object)
    truth = np.asarray(gold, dtype=object)
    rows = []
    for threshold in thresholds:
        routed = conf < threshold
        blended = np.where(routed, t_pred, s_pred)
        cost = None
        if student["cost_per_1k_usd"] is not None and teacher["cost_per_1k_usd"] is not None:
            cost = blended_cost_per_1k(
                student["cost_per_1k_usd"], teacher["cost_per_1k_usd"], float(routed.mean())
            )
        rows.append(
            {
                "threshold": threshold,
                "routed_share": float(routed.mean()),
                "accuracy": float(np.mean(blended == truth)),
                "cost_per_1k_usd": cost,
            }
        )
    return rows


def error_analysis(
    data: PreparedData, student: dict[str, Any], top_k: int, examples_per_pair: int
) -> dict[str, Any]:
    gold = data.test[LABEL].tolist()
    pred = student["predictions"]
    pairs = []
    for g, p, n in confused_pairs(gold, pred, top_k):
        mask = (data.test[LABEL] == g).to_numpy() & (np.asarray(pred, dtype=object) == p)
        texts = data.test.loc[mask, TEXT].head(examples_per_pair).tolist()
        pairs.append({"gold": g, "predicted": p, "count": n, "examples": texts})
    errors = int(sum(g != p for g, p in zip(gold, pred, strict=True)))
    return {
        "student": student["name"],
        "errors": errors,
        "test_examples": len(gold),
        "pairs": pairs,
    }


def headline(teacher: dict[str, Any], student: dict[str, Any], cpu: str) -> dict[str, Any]:
    """The README headline, computed from the numbers (never typed by hand)."""
    gap = 100 * (teacher["accuracy"]["mean"] - student["served"]["accuracy"])
    out: dict[str, Any] = {
        "student": student["name"],
        "teacher": teacher["name"],
        "accuracy_gap_points": gap,
        "cost_ratio": None,
        "speedup_p50": None,
    }
    parts = [
        f"within {gap:.1f} accuracy points of the LLM"
        if gap >= 0
        else f"{-gap:.1f} accuracy points above the LLM"
    ]
    if student["cost_per_1k_usd"] is not None and teacher["cost_per_1k_usd"]:
        ratio = student["cost_per_1k_usd"] / teacher["cost_per_1k_usd"]
        out["cost_ratio"] = ratio
        parts.append(f"at {100 * ratio:.2g}% of the cost per 1,000 predictions")
    if student["latency"] and teacher["latency"] and teacher["latency"]["p50_ms"] > 0:
        speedup = teacher["latency"]["p50_ms"] / student["latency"]["p50_ms"]
        out["speedup_p50"] = speedup
        parts.append(f"{speedup:,.0f}x faster per request (median latency; student on {cpu})")
    text = ", ".join(parts[:-1]) + (", and " if len(parts) > 1 else "") + parts[-1]
    out["text"] = f"{student['name']}: {text[0].upper()}{text[1:]}."
    return out


# --------------------------------------------------------------------------- stage


def run_benchmark(cfg: ExperimentConfig, *, measure: bool = True) -> dict[str, Any]:
    data = load_prepared(cfg)
    manifest = labels_manifest(cfg)
    records = run_training(cfg)  # trains anything missing; reuses cached runs
    teachers = teacher_rows(cfg, data)
    students = [student_row(cfg, data, records, s.name, measure=measure) for s in cfg.students]
    best = best_student_name(cfg, records)
    best_row = next(s for s in students if s["name"] == best)
    main_teacher = teachers[0]
    env = environment_info()
    cost_report = read_cost_report(labels_stage(cfg).path)
    train_cost = cost_report.get("splits", {}).get("train", {})

    results: dict[str, Any] = {
        "experiment": cfg.name,
        "generated_at": utc_now(),
        "simulated": bool(manifest.get("simulated", False)),
        "dataset": {
            "name": cfg.dataset.name,
            **data_manifest(cfg)["counts"],
            "labels": len(data.labels),
            "oos_label": cfg.dataset.oos_label,
        },
        "seeds": len(cfg.seeds),
        "teacher": {
            "name": cfg.teacher.name,
            "model": cfg.teacher.model,
            "backend": cfg.teacher.backend,
            "self_consistency": cfg.teacher.self_consistency,
            "pricing": manifest.get("pricing"),
        },
        "hardware": {
            "cpu": env["cpu"],
            "cpu_count": env["cpu_count"],
            "threads": cfg.benchmark.latency.threads,
            "platform": env["platform"],
        },
        "prices": {
            "teacher_as_of": (manifest.get("pricing") or {}).get("as_of"),
            "teacher_source": (manifest.get("pricing") or {}).get("source"),
            "cpu_instance": cfg.benchmark.cpu.model_dump(mode="json"),
        },
        "environment": env,
        "fingerprints": {
            "data": data.fingerprint,
            "labels": manifest.get("fingerprint"),
        },
        "best_student": best,
        "labeling_cost_per_example_usd": train_cost.get("cost_per_example_usd"),
        "labeling_cost_total_usd": cost_report.get("total", {}).get("list_cost_usd"),
        "models": [*teachers, *students],
        "learning_curve": learning_curve(records),
        "routing": routing_table(
            best_row, main_teacher, data.test[LABEL].tolist(), cfg.benchmark.routing.thresholds
        ),
        "error_analysis": error_analysis(
            data, best_row, cfg.benchmark.confused_pairs, cfg.benchmark.examples_per_pair
        ),
    }
    results["headline"] = headline(main_teacher, best_row, env["cpu"])
    return write_results(cfg, results)


def write_results(cfg: ExperimentConfig, results: dict[str, Any]) -> dict[str, Any]:
    directory = cfg.experiment_results_dir
    directory.mkdir(parents=True, exist_ok=True)
    results["charts"] = charts.render_all(results, directory)
    public = {
        **results,
        "models": [
            {k: v for k, v in m.items() if k not in ("predictions", "confidence")}
            for m in results["models"]
        ],
    }
    (directory / RESULTS_JSON).write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    (directory / RESULTS_MD).write_text(render_results_markdown(public), encoding="utf-8")
    (directory / ERRORS_MD).write_text(render_error_analysis(public), encoding="utf-8")
    audit = load_audit(labels_stage(cfg).path)
    (directory / MODEL_CARD).write_text(render_model_card(public, audit), encoding="utf-8")
    logger.info("benchmark results written to %s", directory)
    return public


def load_results(directory: Path) -> dict[str, Any]:
    results: dict[str, Any] = json.loads((directory / RESULTS_JSON).read_text(encoding="utf-8"))
    return results

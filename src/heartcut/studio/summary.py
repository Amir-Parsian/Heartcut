"""The benchmark results of a studio project, reduced to what its results page shows.

Every number comes from ``results.json`` and ``audit.json`` as the benchmark wrote
them; this module only picks fields and puts friendlier names on them.
"""

from __future__ import annotations

import json
from typing import Any

from heartcut.eval.audit import AUDIT_JSON
from heartcut.eval.benchmark import ERRORS_MD, MODEL_CARD, RESULTS_MD, load_results
from heartcut.studio.catalog import student_title
from heartcut.studio.catalog import teacher as teacher_choice
from heartcut.studio.projects import Project


def _row(model: dict[str, Any], best: str, teacher: str) -> dict[str, Any]:
    student = model["role"] == "student"
    latency = model.get("latency") or {}
    if student:
        title = student_title(model["name"])
    else:
        variant = model["variant"]
        title = f"Teacher: {teacher}" + ("" if variant == "single call" else f" ({variant})")
    return {
        "name": model["name"],
        "title": title,
        "role": model["role"],
        # Students: the exported artifact that is served; teachers: the logged answers.
        "accuracy": model["served"]["accuracy"] if student else model["accuracy"]["mean"],
        "macro_f1": model["served"]["macro_f1"] if student else model["macro_f1"]["mean"],
        "cost_per_1k_usd": model.get("cost_per_1k_usd"),
        "latency_p50_ms": latency.get("p50_ms"),
        "size_mb": model.get("size_mb"),
        "format": model["served"]["format"] if student else None,
        "best": model["name"] == best,
    }


def summarise(project: Project) -> dict[str, Any]:
    directory = project.config().experiment_results_dir
    results = load_results(directory)
    audit = json.loads((directory / AUDIT_JSON).read_text(encoding="utf-8"))
    best_name = results["best_student"]
    teacher = teacher_choice(project.settings.teacher).title
    rows = [_row(m, best_name, teacher) for m in results["models"]]
    best_model = next(m for m in results["models"] if m["name"] == best_name)
    analysis = results["error_analysis"]
    return {
        "simulated": results["simulated"],
        "headline": results["headline"]["text"],
        "teacher": rows[0],
        "best": next(r for r in rows if r["best"]),
        "models": rows,
        "teacher_label_accuracy": audit["pooled"]["accuracy"],
        "teacher_labeled": audit["pooled"]["examples"],
        "test_examples": results["dataset"]["test"],
        "labels": results["dataset"]["labels"],
        "labeling_cost_usd": results.get("labeling_cost_total_usd"),
        "errors": analysis["errors"],
        "confusions": analysis["pairs"][:5],
        "charts": results.get("charts", {}),
        "reports": [n for n in (RESULTS_MD, ERRORS_MD, MODEL_CARD) if (directory / n).exists()],
        "export_dir": best_model["served"]["export_dir"],
        "hardware": results["hardware"]["cpu"],
    }

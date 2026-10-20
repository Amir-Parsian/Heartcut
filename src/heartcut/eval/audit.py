"""The ``heartcut audit`` stage: how good are the teacher's labels?

Compares teacher labels on the train and validation pools with the human labels that
were hidden from training. This is the *only* place those labels are read, and the
result is a report: nothing here feeds back into training, so the audit cannot leak
human labels into the students. The low-confidence filter threshold is set in the config
up front; the audit only reports what that filter does.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from heartcut.config import ExperimentConfig
from heartcut.data.prepare import load_hidden_gold, load_prepared
from heartcut.data.schema import ID, LABEL, SPLIT, UNLABELED_SPLITS
from heartcut.eval.metrics import (
    INVALID,
    confused_pairs,
    per_class_report,
    reliability_table,
    score,
)
from heartcut.teachers.labeling import (
    CONFIDENCE,
    TEACHER_LABEL,
    labels_manifest,
    labels_stage,
    load_teacher_labels,
)

AUDIT_JSON = "audit.json"
AUDIT_MD = "audit.md"


def filter_by_confidence(
    frame: pd.DataFrame, min_confidence: float | None
) -> tuple[pd.DataFrame, int]:
    """Drop rows without a valid label or below ``min_confidence``; return (kept, dropped)."""
    keep = frame[TEACHER_LABEL].notna()
    if min_confidence is not None:
        keep &= frame[CONFIDENCE] >= min_confidence
    return frame.loc[keep].reset_index(drop=True), int((~keep).sum())


def audit_split(
    merged: pd.DataFrame, labels: list[str], min_confidence: float | None, top_k: int = 10
) -> dict[str, Any]:
    y_true = merged[LABEL].tolist()
    y_pred = merged[TEACHER_LABEL].fillna(INVALID).tolist()
    scores = score(y_true, y_pred, labels, merged[CONFIDENCE].to_numpy())
    per_class = per_class_report(y_true, y_pred, labels)
    worst = per_class.sort_values(["f1", "label"]).head(top_k)
    correct = merged[LABEL] == merged[TEACHER_LABEL]

    out: dict[str, Any] = {
        "examples": len(merged),
        "accuracy": scores.accuracy,
        "macro_f1": scores.macro_f1,
        "agreement_ece": scores.ece,
        "invalid": int(merged[TEACHER_LABEL].isna().sum()),
        "worst_classes": worst.to_dict(orient="records"),
        "confused_pairs": [
            {"gold": g, "teacher": p, "count": n}
            for g, p, n in confused_pairs(y_true, y_pred, top_k)
        ],
        "confidence_bins": reliability_table(
            merged[CONFIDENCE].to_numpy(), correct.to_numpy(), n_bins=5
        ).to_dict(orient="records"),
    }
    kept, dropped = filter_by_confidence(merged, min_confidence)
    kept_mask = merged[ID].isin(kept[ID])
    out["filter"] = {
        "min_confidence": min_confidence,
        "kept": len(kept),
        "dropped": dropped,
        "kept_accuracy": float(correct[kept_mask].mean()) if kept_mask.any() else None,
        "dropped_accuracy": float(correct[~kept_mask].mean()) if (~kept_mask).any() else None,
    }
    return out


def run_audit(cfg: ExperimentConfig) -> dict[str, Any]:
    data = load_prepared(cfg)
    gold = load_hidden_gold(cfg)
    manifest = labels_manifest(cfg)
    report: dict[str, Any] = {
        "experiment": cfg.name,
        "teacher": cfg.teacher.name,
        "model": cfg.teacher.model,
        "self_consistency": cfg.teacher.self_consistency,
        "simulated": bool(manifest.get("simulated", False)),
        "labels_fingerprint": manifest.get("fingerprint"),
        "splits": {},
    }
    pooled = []
    for split in UNLABELED_SPLITS:
        teacher = load_teacher_labels(cfg, split)
        merged = teacher.merge(gold.loc[gold[SPLIT] == split, [ID, LABEL]], on=ID, how="inner")
        if len(merged) != len(teacher):
            raise ValueError(f"{split}: teacher labels and hidden gold labels do not line up")
        report["splits"][split] = audit_split(merged, data.labels, cfg.label_filter.min_confidence)
        pooled.append(merged)
    report["pooled"] = audit_split(
        pd.concat(pooled, ignore_index=True), data.labels, cfg.label_filter.min_confidence
    )

    directory = labels_stage(cfg).path
    (directory / AUDIT_JSON).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (directory / AUDIT_MD).write_text(render_audit_markdown(report), encoding="utf-8")
    results = cfg.experiment_results_dir
    results.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(directory / AUDIT_MD, results / AUDIT_MD)
    shutil.copyfile(directory / AUDIT_JSON, results / AUDIT_JSON)
    return report


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def render_audit_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Teacher label audit: {report['teacher']} on {report['experiment']}",
        "",
    ]
    if report["simulated"]:
        lines += ["> **SIMULATED TEACHER** - these numbers test the pipeline, nothing else.", ""]
    lines += [
        "Teacher labels compared with the human labels that were hidden from training.",
        f"Self-consistency samples per message: {report['self_consistency']}.",
        "",
        "| Pool | Examples | Accuracy | Macro-F1 | Invalid answers |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in [*report["splits"].items(), ("train + validation", report["pooled"])]:
        lines.append(
            f"| {name} | {s['examples']:,} | {_pct(s['accuracy'])} | {_pct(s['macro_f1'])} "
            f"| {s['invalid']:,} |"
        )
    pooled = report["pooled"]
    lines += [
        "",
        "## Classes the teacher gets most wrong",
        "",
        "| Intent | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in pooled["worst_classes"]:
        lines.append(
            f"| `{row['label']}` | {_pct(row['precision'])} | {_pct(row['recall'])} "
            f"| {_pct(row['f1'])} | {row['support']} |"
        )
    lines += ["", "## Most frequent confusions (human label -> teacher label)", ""]
    lines += ["| Human label | Teacher label | Count |", "|---|---|---:|"]
    for pair in pooled["confused_pairs"]:
        lines.append(f"| `{pair['gold']}` | `{pair['teacher']}` | {pair['count']} |")
    lines += ["", "## Agreement score vs. correctness", ""]
    if report["self_consistency"] > 1:
        lines += ["| Agreement | Examples | Teacher accuracy |", "|---|---:|---:|"]
        for b in pooled["confidence_bins"]:
            if b["count"]:
                lines.append(
                    f"| {b['bin_low']:.1f}-{b['bin_high']:.1f} | {b['count']:,} "
                    f"| {_pct(b['accuracy'])} |"
                )
    else:
        lines.append("Single sample per message, so there is no agreement score to analyse.")
    f = pooled["filter"]
    lines += ["", "## Low-confidence filter", ""]
    if f["min_confidence"] is None:
        lines.append(
            f"No confidence filter configured; {f['dropped']:,} examples without a valid "
            "label are dropped from training."
        )
    else:
        lines.append(
            f"With `min_confidence = {f['min_confidence']}`: {f['kept']:,} kept "
            f"(teacher accuracy {_pct(f['kept_accuracy'])}), {f['dropped']:,} dropped "
            f"(teacher accuracy {_pct(f['dropped_accuracy'])})."
        )
    lines.append("")
    return "\n".join(lines)


def load_audit(directory: Path) -> dict[str, Any] | None:
    path = directory / AUDIT_JSON
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

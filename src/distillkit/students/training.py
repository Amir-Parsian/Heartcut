"""The ``distillkit train`` stage: train students on teacher labels.

For every (student, labeling budget, seed) the stage

1. takes the first ``budget`` examples of a seeded permutation of the unlabeled pool
   (nested subsets: the 500-example run sees a subset of the 2,000-example run),
2. drops examples the teacher could not label, and those below the configured
   confidence threshold,
3. trains the student on teacher labels only (hard majority labels, or vote
   distributions when ``use_soft_labels`` is set and the teacher sampled more than once),
4. scores agreement with the teacher on the teacher-labeled validation pool (used for
   model selection) and records test-set predictions for the benchmark.

Test labels are never used here for any decision; they are scored and stored so the
benchmark can aggregate them later.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from distillkit.artifacts import StageDir, fingerprint, stage_dir, write_manifest
from distillkit.config import Budget, ExperimentConfig
from distillkit.data.prepare import PreparedData, load_prepared
from distillkit.data.schema import ID, LABEL, TEXT
from distillkit.data.splits import nested_prefix
from distillkit.eval.audit import filter_by_confidence
from distillkit.eval.metrics import score
from distillkit.students.base import AnyStudentConfig, Student, TrainingData, set_seed
from distillkit.students.registry import build_student
from distillkit.teachers.labeling import (
    COST,
    TEACHER_LABEL,
    VOTES,
    labels_stage,
    load_teacher_labels,
    soft_label_matrix,
)
from distillkit.tracking import flatten, start_run

logger = logging.getLogger(__name__)

MODEL_DIR = "model"
METRICS_FILE = "metrics.json"
PREDICTIONS_FILE = "test_predictions.parquet"


def budget_tag(budget: Budget) -> str:
    return "full" if budget == "full" else str(budget)


def student_stage(
    cfg: ExperimentConfig, student: AnyStudentConfig, budget: Budget, seed: int, labels_fp: str
) -> StageDir:
    fp = fingerprint(
        "student",
        labels_fp,
        student.model_dump(mode="json"),
        budget_tag(budget),
        seed,
        cfg.label_filter.model_dump(mode="json"),
    )
    name = f"{student.name}-b{budget_tag(budget)}-s{seed}"
    return stage_dir(cfg.experiment_dir, "students", name, fp)


@dataclass(frozen=True)
class PreparedTraining:
    data: TrainingData
    budget_examples: int
    kept_examples: int
    dropped_examples: int
    labeling_cost_usd: float | None


def select_budget(train: pd.DataFrame, budget: Budget, seed: int) -> pd.DataFrame:
    if budget == "full" or budget >= len(train):
        return train.reset_index(drop=True)
    order = nested_prefix(len(train), seed)[:budget]
    return train.iloc[np.sort(order)].reset_index(drop=True)


def prepare_training(
    cfg: ExperimentConfig,
    student: AnyStudentConfig,
    labels: Sequence[str],
    teacher_train: pd.DataFrame,
    teacher_validation: pd.DataFrame,
    budget: Budget,
    seed: int,
) -> PreparedTraining:
    index = {label: i for i, label in enumerate(labels)}
    pool = select_budget(teacher_train, budget, seed)
    costs = pool[COST]
    labeling_cost = None if costs.isna().any() else float(costs.sum())
    kept, dropped = filter_by_confidence(pool, cfg.label_filter.min_confidence)
    if kept.empty:
        raise ValueError(f"no usable teacher labels for budget {budget}")
    val = teacher_validation.loc[teacher_validation[TEACHER_LABEL].notna()]
    soft = None
    if student.use_soft_labels:
        if cfg.teacher.self_consistency > 1:
            soft = soft_label_matrix(kept[VOTES].tolist(), labels)
        else:
            logger.warning(
                "%s asks for soft labels but the teacher sampled once; using hard labels",
                student.name,
            )
    data = TrainingData(
        texts=kept[TEXT].tolist(),
        targets=np.array([index[label] for label in kept[TEACHER_LABEL]], dtype=np.int64),
        val_texts=val[TEXT].tolist(),
        val_targets=np.array([index[label] for label in val[TEACHER_LABEL]], dtype=np.int64),
        soft_targets=soft,
    )
    return PreparedTraining(data, len(pool), len(kept), dropped, labeling_cost)


def evaluate_student(
    student: Student,
    data: PreparedData,
    val_texts: Sequence[str],
    val_teacher: Sequence[str],
    n_bins: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    val_pred, _ = student.predict(val_texts)
    val_scores = score(list(val_teacher), val_pred, data.labels)
    test_pred, test_conf = student.predict(data.test[TEXT].tolist())
    test_scores = score(data.test[LABEL].tolist(), test_pred, data.labels, test_conf, n_bins)
    predictions = pd.DataFrame(
        {ID: data.test[ID], "prediction": test_pred, "confidence": test_conf.astype(float)}
    )
    metrics = {
        "validation_teacher_agreement": val_scores.accuracy,
        "validation_teacher_macro_f1": val_scores.macro_f1,
        "test": test_scores.to_dict(),
    }
    return metrics, predictions


def train_one(
    cfg: ExperimentConfig,
    student_cfg: AnyStudentConfig,
    budget: Budget,
    seed: int,
    data: PreparedData,
    teacher_train: pd.DataFrame,
    teacher_validation: pd.DataFrame,
    labels_fp: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    stage = student_stage(cfg, student_cfg, budget, seed, labels_fp)
    if stage.is_complete() and not force:
        return read_metrics(stage.path)

    prepared = prepare_training(
        cfg, student_cfg, data.labels, teacher_train, teacher_validation, budget, seed
    )
    run_name = stage.path.name
    logger.info("training %s on %d teacher labels", run_name, prepared.kept_examples)
    with start_run(cfg.tracking, cfg.name, run_name) as run:
        run.set_tags(
            {
                "student": student_cfg.name,
                "kind": student_cfg.kind,
                "budget": budget_tag(budget),
                "seed": str(seed),
                "data_fingerprint": data.fingerprint,
                "labels_fingerprint": labels_fp,
                "teacher": cfg.teacher.name,
            }
        )
        run.log_params(flatten(student_cfg.model_dump(mode="json"), "student."))
        run.log_params({"label_filter.min_confidence": cfg.label_filter.min_confidence})

        student = build_student(student_cfg, data.labels)  # imports torch if needed
        set_seed(seed)
        fit_info = student.fit(prepared.data, seed)
        val_teacher = [data.labels[i] for i in prepared.data.val_targets]
        metrics, predictions = evaluate_student(
            student, data, prepared.data.val_texts, val_teacher, cfg.benchmark.ece_bins
        )
        record = {
            "student": student_cfg.name,
            "kind": student_cfg.kind,
            "budget": budget_tag(budget),
            "seed": seed,
            "budget_examples": prepared.budget_examples,
            "train_examples": prepared.kept_examples,
            "dropped_examples": prepared.dropped_examples,
            "labeling_cost_usd": prepared.labeling_cost_usd,
            "soft_labels": prepared.data.soft_targets is not None,
            "fit": fit_info,
            **metrics,
        }
        run.log_metrics(
            {
                "validation_teacher_agreement": metrics["validation_teacher_agreement"],
                "test_accuracy": metrics["test"]["accuracy"],
                "test_macro_f1": metrics["test"]["macro_f1"],
                "test_ece": metrics["test"]["ece"],
                "train_examples": prepared.kept_examples,
            }
        )

        directory = stage.ensure()
        student.save(directory / MODEL_DIR)
        predictions.to_parquet(directory / PREDICTIONS_FILE, index=False)
        (directory / METRICS_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        write_manifest(
            directory,
            {
                "stage": "students",
                "fingerprint": stage.fingerprint,
                "data_fingerprint": data.fingerprint,
                "labels_fingerprint": labels_fp,
                "student": student_cfg.model_dump(mode="json"),
                "budget": budget_tag(budget),
                "seed": seed,
                "label_filter": cfg.label_filter.model_dump(mode="json"),
            },
        )
        run.log_artifact(directory / METRICS_FILE)
    return record


def run_training(
    cfg: ExperimentConfig,
    *,
    students: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    budgets: Sequence[Budget] | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    data = load_prepared(cfg)
    labels_fp = labels_stage(cfg, data.fingerprint).fingerprint
    teacher_train = load_teacher_labels(cfg, "train")
    teacher_validation = load_teacher_labels(cfg, "validation")
    chosen = [cfg.student(n) for n in students] if students else list(cfg.students)
    results = []
    for student_cfg in chosen:
        for budget in budgets or cfg.budgets:
            for seed in seeds or cfg.seeds:
                results.append(
                    train_one(
                        cfg,
                        student_cfg,
                        budget,
                        seed,
                        data,
                        teacher_train,
                        teacher_validation,
                        labels_fp,
                        force=force,
                    )
                )
    return results


def read_metrics(directory: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = json.loads((directory / METRICS_FILE).read_text(encoding="utf-8"))
    return metrics


def student_run_dir(cfg: ExperimentConfig, student_name: str, budget: Budget, seed: int) -> Path:
    labels_fp = labels_stage(cfg).fingerprint
    return student_stage(cfg, cfg.student(student_name), budget, seed, labels_fp).path


def load_trained_student(
    cfg: ExperimentConfig, student_name: str, budget: Budget, seed: int
) -> Student:
    return Student.load(student_run_dir(cfg, student_name, budget, seed) / MODEL_DIR)

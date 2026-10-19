"""The student interface shared by every model tier."""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from distillkit.config import EncoderStudentConfig, SetFitStudentConfig, TfidfStudentConfig

AnyStudentConfig = TfidfStudentConfig | SetFitStudentConfig | EncoderStudentConfig

STUDENT_META = "student.json"


@dataclass(frozen=True)
class TrainingData:
    """Teacher-labeled training data, already mapped to label ids.

    ``targets`` are the teacher's majority labels. ``soft_targets`` (n x n_labels) are the
    teacher's vote distributions and are only set when the student asked for soft labels
    and the teacher was sampled more than once. Validation labels also come from the
    teacher: human labels are never available at training time.
    """

    texts: list[str]
    targets: np.ndarray
    val_texts: list[str]
    val_targets: np.ndarray
    soft_targets: np.ndarray | None = None

    def __post_init__(self) -> None:
        if len(self.texts) != len(self.targets):
            raise ValueError("texts and targets differ in length")
        if self.soft_targets is not None and self.soft_targets.shape[0] != len(self.texts):
            raise ValueError("soft_targets must have one row per text")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - also seeds libraries that use the legacy global RNG
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def expand_soft_targets(
    soft: np.ndarray, min_weight: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rewrite soft targets as weighted hard examples.

    Each example ``i`` with distribution ``p_i`` becomes one row per class ``c`` with
    ``p_ic > 0`` and sample weight ``p_ic``. For any model trained with log loss (logistic
    regression included) the weighted loss on the expanded rows equals the cross-entropy
    against the soft targets, so this is exact distillation for linear heads, not an
    approximation.

    Returns ``(row_index, class_index, weight)``.
    """
    rows, classes = np.nonzero(soft > min_weight)
    return rows, classes, soft[rows, classes]


def full_proba(partial: np.ndarray, seen_classes: np.ndarray, n_labels: int) -> np.ndarray:
    """Place the columns of a model trained on a subset of classes into the full label space.

    With small labeling budgets some intents may have no training examples at all; such
    classes get probability 0.
    """
    out = np.zeros((partial.shape[0], n_labels), dtype=np.float64)
    out[:, seen_classes] = partial
    return out


class Student(ABC):
    """A trainable text classifier over a fixed, ordered label set."""

    kind: ClassVar[str]

    def __init__(self, config: AnyStudentConfig, labels: Sequence[str]) -> None:
        self.config = config
        self.labels = list(labels)

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def fit(self, data: TrainingData, seed: int) -> dict[str, Any]:
        """Train in place and return training diagnostics (epochs run, timings, ...)."""

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, n_labels)`` array of class probabilities."""

    @abstractmethod
    def _save_weights(self, directory: Path) -> None: ...

    @abstractmethod
    def _load_weights(self, directory: Path) -> None: ...

    def predict(self, texts: Sequence[str]) -> tuple[list[str], np.ndarray]:
        """Labels and confidences (the top-class probability)."""
        proba = self.predict_proba(texts)
        idx = proba.argmax(axis=1)
        return [self.labels[i] for i in idx], proba.max(axis=1)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "kind": self.kind,
            "name": self.name,
            "labels": self.labels,
            "config": self.config.model_dump(mode="json"),
        }
        (directory / STUDENT_META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        self._save_weights(directory)

    @classmethod
    def load(cls, directory: Path) -> Student:
        from distillkit.students.registry import student_class

        meta = json.loads((directory / STUDENT_META).read_text(encoding="utf-8"))
        klass = student_class(meta["kind"])
        config = klass.config_model().model_validate(meta["config"])
        student = klass(config, meta["labels"])
        student._load_weights(directory)
        return student

    @classmethod
    @abstractmethod
    def config_model(cls) -> type[AnyStudentConfig]: ...

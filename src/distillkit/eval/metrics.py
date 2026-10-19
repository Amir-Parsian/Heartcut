"""Classification metrics used by the audit and the benchmark."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

# Placeholder prediction for a teacher answer that could not be parsed into a label.
# It is scored as wrong and never matches any gold label.
INVALID = "__invalid__"


@dataclass(frozen=True)
class Scores:
    n: int
    accuracy: float
    macro_f1: float
    ece: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if len(y_true) == 0:
        raise ValueError("cannot score an empty set")
    return float(np.mean(np.asarray(y_true, dtype=object) == np.asarray(y_pred, dtype=object)))


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    """Macro-F1 over the gold label set; invalid predictions count as misses."""
    return float(
        f1_score(list(y_true), list(y_pred), labels=list(labels), average="macro", zero_division=0)
    )


def expected_calibration_error(
    confidence: Sequence[float] | np.ndarray, correct: Sequence[bool] | np.ndarray, n_bins: int = 15
) -> float:
    """Expected calibration error with equal-width confidence bins.

    ECE = sum_b |B_b|/n * |acc(B_b) - conf(B_b)|. Confidence 1.0 falls in the last bin.
    """
    conf = np.asarray(confidence, dtype=float)
    hit = np.asarray(correct, dtype=float)
    if conf.shape != hit.shape or conf.size == 0:
        raise ValueError("confidence and correct must be non-empty and the same length")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(hit[mask].mean() - conf[mask].mean())
    return float(ece)


def reliability_table(
    confidence: Sequence[float] | np.ndarray, correct: Sequence[bool] | np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    conf = np.asarray(confidence, dtype=float)
    hit = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        rows.append(
            {
                "bin_low": edges[b],
                "bin_high": edges[b + 1],
                "count": int(mask.sum()),
                "accuracy": float(hit[mask].mean()) if mask.any() else None,
                "confidence": float(conf[mask].mean()) if mask.any() else None,
            }
        )
    return pd.DataFrame(rows)


def score(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    confidence: Sequence[float] | np.ndarray | None = None,
    n_bins: int = 15,
) -> Scores:
    correct = np.asarray(y_true, dtype=object) == np.asarray(y_pred, dtype=object)
    ece = None
    if confidence is not None:
        ece = expected_calibration_error(confidence, correct, n_bins)
    return Scores(
        n=len(y_true),
        accuracy=accuracy(y_true, y_pred),
        macro_f1=macro_f1(y_true, y_pred, labels),
        ece=ece,
    )


def per_class_report(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        list(y_true), list(y_pred), labels=list(labels), zero_division=0
    )
    return pd.DataFrame(
        {
            "label": list(labels),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def confused_pairs(
    y_true: Sequence[str], y_pred: Sequence[str], top_k: int = 10
) -> list[tuple[str, str, int]]:
    """Most frequent (gold, predicted) error pairs, most common first."""
    errors = Counter((t, p) for t, p in zip(y_true, y_pred, strict=True) if t != p)
    ranked = sorted(errors.items(), key=lambda item: (-item[1], item[0]))
    return [(t, p, n) for (t, p), n in ranked[:top_k]]


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (0 for a single value)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("no values")
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std

"""CPU inference runtime for exported students.

Kept free of PyTorch, pandas and transformers on purpose: the serving image only needs
``onnxruntime``, ``tokenizers`` and ``numpy``. The TF-IDF baseline is exported in its
scikit-learn form (its char n-gram vectoriser has no ONNX converter) and needs
scikit-learn at serve time.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

EXPORT_META = "export.json"


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class Predictor(Protocol):
    name: str
    labels: list[str]
    meta: dict[str, Any]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    out: np.ndarray = exp / exp.sum(axis=-1, keepdims=True)
    return out


def read_export_meta(directory: Path) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads((directory / EXPORT_META).read_text(encoding="utf-8"))
    return meta


class OnnxPredictor:
    """Tokenizer (``tokenizers``) + ONNX Runtime session on the CPU provider."""

    def __init__(self, directory: Path, *, threads: int | None = None, int8: bool = True) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.directory = directory
        self.meta = read_export_meta(directory)
        self.name = str(self.meta["name"])
        self.labels = list(self.meta["labels"])
        files = self.meta["files"]
        model_file = files.get("int8") if int8 and files.get("int8") else files["fp32"]
        self.model_path = directory / model_file

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}

        self.tokenizer = Tokenizer.from_file(str(directory / files["tokenizer"]))
        self.tokenizer.enable_truncation(max_length=int(self.meta["max_length"]))
        self.tokenizer.enable_padding(
            pad_id=int(self.meta["pad_token_id"]), pad_token=str(self.meta["pad_token"])
        )

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.labels)))
        encodings = self.tokenizer.encode_batch(list(texts))
        feeds = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
        (logits,) = self.session.run(["logits"], feeds)
        return softmax(logits.astype(np.float64))


class SklearnPredictor:
    """The exported TF-IDF baseline (needs scikit-learn)."""

    def __init__(self, directory: Path) -> None:
        from heartcut.students.base import Student

        self.directory = directory
        self.meta = read_export_meta(directory)
        self.student = Student.load(directory / self.meta["files"]["student"])
        self.name = str(self.meta["name"])
        self.labels = list(self.student.labels)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.student.predict_proba(list(texts))


def load_predictor(directory: str | Path, *, threads: int | None = None) -> Predictor:
    directory = Path(directory)
    meta = read_export_meta(directory)
    if meta["format"] == "onnx":
        return OnnxPredictor(directory, threads=threads)
    if meta["format"] == "sklearn":
        return SklearnPredictor(directory)
    raise ValueError(f"unknown export format {meta['format']!r}")


def predict(predictor: Predictor, texts: Sequence[str]) -> tuple[list[Prediction], float]:
    """Predictions plus the wall-clock time of the batch in milliseconds."""
    start = time.perf_counter()
    proba = predictor.predict_proba(texts)
    elapsed_ms = (time.perf_counter() - start) * 1000
    idx = proba.argmax(axis=1)
    return [
        Prediction(predictor.labels[i], float(proba[row, i])) for row, i in enumerate(idx)
    ], elapsed_ms


def artifact_size_mb(directory: Path) -> float:
    """Size of the files the runtime actually loads."""
    meta = read_export_meta(directory)
    files = meta["files"]
    if meta["format"] == "onnx":
        model = files.get("int8") or files["fp32"]
        paths = [directory / model, directory / files["tokenizer"]]
    else:
        paths = [p for p in (directory / files["student"]).rglob("*") if p.is_file()]
    return float(sum(p.stat().st_size for p in paths)) / 1_000_000

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from heartcut.config import (
    EncoderStudentConfig,
    ExperimentConfig,
    SetFitStudentConfig,
    TfidfStudentConfig,
)
from heartcut.serve import export as export_module
from heartcut.serve.export import best_seed, best_student_name, export_student
from heartcut.serve.runtime import (
    OnnxPredictor,
    artifact_size_mb,
    load_predictor,
    predict,
    read_export_meta,
    softmax,
)
from heartcut.students import Student, TrainingData, build_student
from tests.synthetic import INTENTS, generate

LABELS = sorted(INTENTS)
TEXTS = ["where is my new card", "please cancel my transfer", "close my account"]


def _data() -> TrainingData:
    idx = {label: i for i, label in enumerate(LABELS)}
    rows, val = generate(10, seed=5), generate(2, seed=6)
    return TrainingData(
        texts=[r["text"] for r in rows],
        targets=np.array([idx[r["intent"]] for r in rows]),
        val_texts=[r["text"] for r in val],
        val_targets=np.array([idx[r["intent"]] for r in val]),
    )


def _trained(config) -> Student:  # type: ignore[no-untyped-def]
    student = build_student(config, LABELS)
    student.fit(_data(), seed=0)
    return student


def test_tfidf_exports_as_sklearn(tmp_path: Path) -> None:
    student = _trained(TfidfStudentConfig(name="tfidf"))
    meta = export_student(student, tmp_path / "export", TEXTS)
    assert meta["format"] == "sklearn"
    predictor = load_predictor(tmp_path / "export")
    np.testing.assert_allclose(predictor.predict_proba(TEXTS), student.predict_proba(TEXTS))
    predictions, elapsed = predict(predictor, TEXTS)
    assert len(predictions) == 3 and elapsed >= 0
    assert artifact_size_mb(tmp_path / "export") > 0


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["setfit", "encoder"])
def test_transformer_students_export_to_onnx(
    kind: str, tmp_path: Path, tiny_st: Path, tiny_bert: Path
) -> None:
    pytest.importorskip("onnxruntime")
    if kind == "setfit":
        config = SetFitStudentConfig(name="s", params={"model_name": str(tiny_st), "max_pairs": 32})
    else:
        config = EncoderStudentConfig(name="e", params={"model_name": str(tiny_bert), "epochs": 1})
    student = _trained(config)
    directory = tmp_path / "export"
    meta = export_student(student, directory, TEXTS * 4, quantize=True)
    assert meta["format"] == "onnx"
    check = meta["verification"]
    assert check["fp32_max_abs_diff"] < 1e-3 and check["fp32_argmax_agreement"] == 1.0
    assert "int8_argmax_agreement_with_fp32" in check
    assert (directory / "model.int8.onnx").exists() and (directory / "tokenizer.json").exists()
    assert read_export_meta(directory)["labels"] == LABELS

    fp32 = OnnxPredictor(directory, int8=False, threads=1)
    np.testing.assert_allclose(fp32.predict_proba(TEXTS), student.predict_proba(TEXTS), atol=1e-4)
    int8 = load_predictor(directory, threads=1)
    assert int8.predict_proba(TEXTS).shape == (3, len(LABELS))
    assert int8.predict_proba([]).shape == (0, len(LABELS))


@pytest.mark.slow
def test_export_rejects_mismatching_graphs(
    tmp_path: Path, tiny_bert: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("onnxruntime")
    student = _trained(
        EncoderStudentConfig(name="e", params={"model_name": str(tiny_bert), "epochs": 1})
    )
    monkeypatch.setattr(export_module, "PARITY_TOLERANCE", -1.0)
    with pytest.raises(RuntimeError, match="does not match"):
        export_student(student, tmp_path / "export", TEXTS)


def test_unsupported_formats(tmp_path: Path) -> None:
    (tmp_path / "export.json").write_text('{"format": "tflite"}')
    with pytest.raises(ValueError, match="unknown export format"):
        load_predictor(tmp_path)


def test_softmax_is_stable() -> None:
    out = softmax(np.array([[1000.0, 1000.0], [0.0, -1000.0]]))
    np.testing.assert_allclose(out, [[0.5, 0.5], [1.0, 0.0]])


def test_best_student_selection_uses_validation_only(make_experiment) -> None:  # type: ignore[no-untyped-def]
    cfg: ExperimentConfig = make_experiment(
        students=[{"name": "a", "kind": "tfidf"}, {"name": "b", "kind": "tfidf"}]
    )
    records = [
        # "a" is better on test but worse on validation: validation decides.
        {
            "student": "a",
            "budget": "full",
            "seed": 0,
            "validation_teacher_agreement": 0.80,
            "test": {"accuracy": 0.99},
        },
        {
            "student": "b",
            "budget": "full",
            "seed": 0,
            "validation_teacher_agreement": 0.85,
            "test": {"accuracy": 0.50},
        },
        {
            "student": "b",
            "budget": "full",
            "seed": 1,
            "validation_teacher_agreement": 0.90,
            "test": {"accuracy": 0.40},
        },
        {
            "student": "a",
            "budget": "500",
            "seed": 0,
            "validation_teacher_agreement": 1.0,
            "test": {"accuracy": 1.0},
        },
    ]
    assert best_student_name(cfg, records) == "b"
    assert best_seed(records, "b") == 1
    with pytest.raises(ValueError):
        best_student_name(cfg, [])
    with pytest.raises(ValueError):
        best_seed(records, "missing")


@pytest.mark.slow
def test_distilbert_architecture_exports(tmp_path: Path) -> None:
    """distilbert-base-uncased has no token_type_ids and its own attention code."""
    pytest.importorskip("onnxruntime")
    from tests.tiny_models import build_tiny_distilbert

    model_dir = build_tiny_distilbert(tmp_path / "distilbert")
    config = EncoderStudentConfig(name="e", params={"model_name": str(model_dir), "epochs": 1})
    student = _trained(config)
    meta = export_student(student, tmp_path / "export", TEXTS * 4)
    assert meta["verification"]["fp32_max_abs_diff"] < 1e-3
    predictor = load_predictor(tmp_path / "export", threads=1)
    assert "token_type_ids" not in predictor.input_names  # type: ignore[attr-defined]


@pytest.mark.slow
def test_normalized_sentence_transformer_exports(tmp_path: Path) -> None:
    """all-MiniLM-L6-v2 ends with a Normalize module, which the export must fuse."""
    pytest.importorskip("onnxruntime")
    from tests.tiny_models import build_tiny_normalized_sentence_transformer

    model_dir = build_tiny_normalized_sentence_transformer(tmp_path / "st-norm")
    config = SetFitStudentConfig(
        name="s",
        params={"model_name": str(model_dir), "max_pairs": 32, "normalize_embeddings": False},
    )
    student = _trained(config)
    meta = export_student(student, tmp_path / "export", TEXTS * 4)
    assert meta["verification"]["fp32_max_abs_diff"] < 1e-3

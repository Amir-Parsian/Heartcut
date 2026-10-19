"""The ``distillkit export`` stage: trained student -> CPU-servable artifact.

Transformer students become one ONNX graph that maps ``input_ids``/``attention_mask``
to class logits; for the SetFit-style student the embedding body, pooling,
normalisation and logistic-regression head are fused into that graph. Every export is
checked against the original model on sample texts before it is accepted, then
dynamically quantised to int8 (weights int8, activations quantised on the fly), and the
int8 model's agreement with the fp32 model is recorded. The TF-IDF baseline is copied
in its scikit-learn form.

Output directory::

    export.json          format, labels, tokenizer settings, parity checks, sizes
    model.onnx           fp32 graph
    model.int8.onnx      dynamically quantised graph (served by default)
    tokenizer.json       Hugging Face ``tokenizers`` file
"""

from __future__ import annotations

import json
import logging
import shutil
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from distillkit.artifacts import fingerprint, stage_dir, write_manifest
from distillkit.config import ExperimentConfig
from distillkit.serve.runtime import EXPORT_META, OnnxPredictor, artifact_size_mb
from distillkit.students.base import Student

logger = logging.getLogger(__name__)

FP32 = "model.onnx"
INT8 = "model.int8.onnx"
TOKENIZER = "tokenizer.json"
# Logit bias for classes a linear head never saw in training (probability ~ 0).
_UNSEEN_BIAS = -1e4
# fp32 ONNX must reproduce the PyTorch probabilities to this tolerance.
PARITY_TOLERANCE = 1e-3


def _write_meta(directory: Path, meta: dict[str, Any]) -> None:
    (directory / EXPORT_META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _export_sklearn(student: Student, directory: Path) -> dict[str, Any]:
    student.save(directory / "student")
    return {"format": "sklearn", "files": {"student": "student"}}


def _logits_module(student: Student) -> tuple[Any, Any, int]:
    """Return ``(torch module, HF tokenizer, max_length)`` producing class logits."""
    import torch

    if student.kind == "encoder":
        from distillkit.students.encoder import EncoderStudent

        assert isinstance(student, EncoderStudent) and student.model is not None
        model = student.model.eval()

        class EncoderLogits(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model

            def forward(
                self, input_ids: torch.Tensor, attention_mask: torch.Tensor
            ) -> torch.Tensor:
                logits: torch.Tensor = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
                return logits

        return EncoderLogits().eval(), student.tokenizer, student.config.params.max_length

    if student.kind == "setfit":
        from distillkit.students.setfit import SetFitStudent

        assert isinstance(student, SetFitStudent) and student.body is not None
        assert student.head is not None
        modules: list[Any] = list(student.body)
        names = [type(m).__name__ for m in modules]
        if names[:2] != ["Transformer", "Pooling"] or set(names[2:]) - {"Normalize"}:
            raise ValueError(f"unsupported sentence-transformers pipeline for export: {names}")
        transformer = modules[0].auto_model.eval()
        pooling = getattr(modules[1], "pooling_mode", None) or modules[1].get_config_dict().get(
            "pooling_mode"
        )
        if pooling not in ("mean", "cls"):
            raise ValueError(f"unsupported pooling mode for export: {pooling!r}")
        normalize = "Normalize" in names or student.config.params.normalize_embeddings

        n_labels = len(student.labels)
        weight = np.zeros((n_labels, student.head.coef_.shape[1]), dtype=np.float32)
        bias = np.full(n_labels, _UNSEEN_BIAS, dtype=np.float32)
        seen = student.head.classes_
        coef, intercept = student.head.coef_, student.head.intercept_
        if len(seen) == 2 and coef.shape[0] == 1:
            # Binary logistic regression stores one row; softmax over [-z/2, z/2] matches it.
            coef = np.vstack([-coef[0] / 2, coef[0] / 2])
            intercept = np.array([-intercept[0] / 2, intercept[0] / 2])
        weight[seen] = coef
        bias[seen] = intercept
        head_w = torch.tensor(weight)
        head_b = torch.tensor(bias)

        class SetFitLogits(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer = transformer
                self.register_buffer("head_w", head_w)
                self.register_buffer("head_b", head_b)

            def forward(
                self, input_ids: torch.Tensor, attention_mask: torch.Tensor
            ) -> torch.Tensor:
                tokens = self.transformer(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
                if pooling == "cls":
                    embedding = tokens[:, 0]
                else:
                    mask = attention_mask.unsqueeze(-1).to(tokens.dtype)
                    embedding = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
                if normalize:
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                logits: torch.Tensor = embedding @ self.head_w.T + self.head_b
                return logits

        max_length = student.body.max_seq_length or student.config.params.max_length
        return SetFitLogits().eval(), student.body.tokenizer, int(max_length)

    raise ValueError(f"no ONNX export for student kind {student.kind!r}")


def _export_onnx(
    student: Student, directory: Path, sample_texts: Sequence[str], *, quantize: bool, opset: int
) -> dict[str, Any]:
    import torch

    module, tokenizer, max_length = _logits_module(student)
    tokenizer.backend_tokenizer.save(str(directory / TOKENIZER))
    example = tokenizer(
        list(sample_texts[:2]),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with warnings.catch_warnings():
        # The TorchScript exporter is stable for BERT-family encoders; silence its
        # deprecation and tracer chatter.
        warnings.simplefilter("ignore")
        torch.onnx.export(
            module,
            (example["input_ids"], example["attention_mask"]),
            str(directory / FP32),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    files: dict[str, str | None] = {"fp32": FP32, "int8": None, "tokenizer": TOKENIZER}
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        logging.getLogger("root").setLevel(logging.ERROR)  # pre-processing hint is noise here
        quantize_dynamic(str(directory / FP32), str(directory / INT8), weight_type=QuantType.QInt8)
        files["int8"] = INT8
    pad = tokenizer.pad_token or "[PAD]"
    return {
        "format": "onnx",
        "files": files,
        "max_length": max_length,
        "pad_token": pad,
        "pad_token_id": int(tokenizer.pad_token_id or 0),
    }


def verify_export(student: Student, directory: Path, sample_texts: Sequence[str]) -> dict[str, Any]:
    """Compare the exported graphs with the original student on ``sample_texts``."""
    reference = student.predict_proba(list(sample_texts))
    fp32 = OnnxPredictor(directory, int8=False).predict_proba(sample_texts)
    report: dict[str, Any] = {
        "samples": len(sample_texts),
        "fp32_max_abs_diff": float(np.abs(fp32 - reference).max()),
        "fp32_argmax_agreement": float(np.mean(fp32.argmax(1) == reference.argmax(1))),
    }
    if report["fp32_max_abs_diff"] > PARITY_TOLERANCE:
        raise RuntimeError(
            f"ONNX export does not match the student (max |diff| "
            f"{report['fp32_max_abs_diff']:.2e} > {PARITY_TOLERANCE})"
        )
    meta = json.loads((directory / EXPORT_META).read_text(encoding="utf-8"))
    if meta["files"].get("int8"):
        int8 = OnnxPredictor(directory, int8=True).predict_proba(sample_texts)
        report["int8_argmax_agreement_with_fp32"] = float(np.mean(int8.argmax(1) == fp32.argmax(1)))
        report["int8_max_abs_diff"] = float(np.abs(int8 - fp32).max())
    return report


def export_student(
    student: Student,
    directory: Path,
    sample_texts: Sequence[str],
    *,
    quantize: bool = True,
    opset: int = 17,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export ``student`` into ``directory`` and return the export metadata.

    ``extra_meta`` (for example the task description used by the fallback router) is
    stored in ``export.json`` alongside the labels.
    """
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    if student.kind == "tfidf":
        meta = _export_sklearn(student, directory)
    else:
        meta = _export_onnx(student, directory, sample_texts, quantize=quantize, opset=opset)
    meta.update({"name": student.name, "kind": student.kind, "labels": student.labels})
    meta.update(extra_meta or {})
    _write_meta(directory, meta)
    if meta["format"] == "onnx":
        meta["verification"] = verify_export(student, directory, sample_texts)
    meta["size_mb"] = artifact_size_mb(directory)
    _write_meta(directory, meta)
    return meta


# --------------------------------------------------------------------------- stage


def best_student_name(cfg: ExperimentConfig, records: Sequence[dict[str, Any]]) -> str:
    """Full-budget student with the highest mean validation agreement with the teacher.

    Uses teacher-labeled validation data only, never the test set. Ties go to the student
    listed first in the config (the cheaper tiers are listed first).
    """
    order = [s.name for s in cfg.students]
    scores: dict[str, list[float]] = {}
    for r in records:
        if r["budget"] == "full":
            scores.setdefault(r["student"], []).append(r["validation_teacher_agreement"])
    if not scores:
        raise ValueError("no full-budget training runs to choose from")
    return max(scores, key=lambda name: (float(np.mean(scores[name])), -order.index(name)))


def best_seed(records: Sequence[dict[str, Any]], student: str) -> int:
    runs = [r for r in records if r["student"] == student and r["budget"] == "full"]
    if not runs:
        raise ValueError(f"no full-budget runs for {student}")
    return int(max(runs, key=lambda r: (r["validation_teacher_agreement"], -r["seed"]))["seed"])


def run_export(
    cfg: ExperimentConfig,
    records: Sequence[dict[str, Any]],
    student_name: str | None = None,
    *,
    force: bool = False,
) -> Path:
    """Export the chosen (or best) student's best full-budget seed; reuse if up to date."""
    from distillkit.data.prepare import load_prepared
    from distillkit.students.training import MODEL_DIR, student_run_dir

    name = student_name or (
        cfg.export.student if cfg.export.student != "best" else best_student_name(cfg, records)
    )
    seed = best_seed(records, name)
    source = student_run_dir(cfg, name, "full", seed)
    fp = fingerprint("export", source.name, cfg.export.model_dump(mode="json"))
    stage = stage_dir(cfg.experiment_dir, "export", name, fp)
    if stage.is_complete() and not force:
        return stage.path

    data = load_prepared(cfg)
    samples = data.validation["text"].head(64).tolist()
    student = Student.load(source / MODEL_DIR)
    meta = export_student(
        student,
        stage.path,
        samples,
        quantize=cfg.export.quantize,
        opset=cfg.export.opset,
        extra_meta={
            "task": cfg.dataset.task,
            "oos_label": cfg.dataset.oos_label,
            "experiment": cfg.name,
            "seed": seed,
        },
    )
    write_manifest(
        stage.path,
        {
            "stage": "export",
            "fingerprint": fp,
            "student": name,
            "seed": seed,
            "source": str(source),
            "export": {k: v for k, v in meta.items() if k != "labels"},
        },
    )
    logger.info("exported %s (seed %d) to %s", name, seed, stage.path)
    return stage.path

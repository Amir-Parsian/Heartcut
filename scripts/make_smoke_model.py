"""Build a tiny exported model for smoke-testing the serving image (CI and local).

It trains a 2-layer, randomly initialised BERT on the synthetic test dataset for a few
epochs and exports it exactly like a real student (ONNX + int8 + tokenizer.json). The
model is only good enough to prove that the image loads a model and answers requests.

    uv run python scripts/make_smoke_model.py build/smoke-model
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.synthetic import INTENTS, generate
from tests.tiny_models import build_tiny_bert

from distillkit.config import EncoderStudentConfig
from distillkit.serve.export import export_student
from distillkit.students import TrainingData, build_student


def main(out: Path) -> None:
    labels = sorted(INTENTS)
    index = {label: i for i, label in enumerate(labels)}
    train, val = generate(25, seed=1), generate(5, seed=2)
    data = TrainingData(
        texts=[r["text"] for r in train],
        targets=np.array([index[r["intent"]] for r in train]),
        val_texts=[r["text"] for r in val],
        val_targets=np.array([index[r["intent"]] for r in val]),
    )
    with tempfile.TemporaryDirectory() as tmp:
        bert = build_tiny_bert(Path(tmp) / "bert")
        config = EncoderStudentConfig(
            name="smoke-encoder",
            params={
                "model_name": str(bert),
                "epochs": 20,
                "learning_rate": 2e-3,
                "batch_size": 16,
                "early_stopping_patience": 20,
            },
        )
        student = build_student(config, labels)
        student.fit(data, seed=0)
        meta = export_student(
            student, out, [r["text"] for r in val], extra_meta={"task": "smoke test"}
        )
    print(f"exported {meta['name']} to {out} ({meta['size_mb']:.2f} MB)")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "build/smoke-model"))

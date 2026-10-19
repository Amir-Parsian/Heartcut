"""Medium student: a small pre-trained encoder fine-tuned end to end.

A deliberately plain PyTorch training loop (AdamW, linear warm-up/decay, optional
mixed precision on CUDA, early stopping on validation agreement with the teacher) so
the distillation loss and model selection are explicit rather than hidden in a trainer.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from distillkit.config import EncoderStudentConfig
from distillkit.students.base import Student, TrainingData
from distillkit.students.torch_utils import (
    best_device,
    distillation_loss,
    seeded_generator,
    to_device,
)

logger = logging.getLogger(__name__)

HF_DIR = "hf"
_NO_DECAY = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")


class EncoderStudent(Student):
    kind = "encoder"
    config: EncoderStudentConfig

    def __init__(self, config: EncoderStudentConfig, labels: Sequence[str]) -> None:
        super().__init__(config, labels)
        # Typed as Any: transformers' generated model classes are only partially typed.
        self.model: Any = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.device = torch.device("cpu")

    @classmethod
    def config_model(cls) -> type[EncoderStudentConfig]:
        return EncoderStudentConfig

    def _tokenize(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        assert self.tokenizer is not None
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.config.params.max_length,
            return_tensors="pt",
        )
        return dict(encoded)

    def _init_model(self) -> None:
        p = self.config.params
        self.tokenizer = AutoTokenizer.from_pretrained(p.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            p.model_name,
            num_labels=len(self.labels),
            id2label=dict(enumerate(self.labels)),
            label2id={label: i for i, label in enumerate(self.labels)},
        )

    def _optimizer(self) -> torch.optim.Optimizer:
        assert self.model is not None
        p = self.config.params
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for name, param in self.model.named_parameters():
            (no_decay if name.endswith(_NO_DECAY) else decay).append(param)
        groups = [
            {"params": decay, "weight_decay": p.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=p.learning_rate)

    def fit(self, data: TrainingData, seed: int) -> dict[str, Any]:
        p = self.config.params
        start = time.perf_counter()
        self._init_model()
        assert self.model is not None
        self.device = best_device()
        self.model.to(self.device)

        targets = torch.as_tensor(data.targets, dtype=torch.long)
        soft = None
        if data.soft_targets is not None:
            soft = torch.as_tensor(data.soft_targets, dtype=torch.float32)
        n = len(data.texts)
        steps_per_epoch = math.ceil(n / p.batch_size)
        optimizer = self._optimizer()
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(p.warmup_ratio * steps_per_epoch * p.epochs),
            num_training_steps=steps_per_epoch * p.epochs,
        )
        use_amp = p.fp16 and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        generator = seeded_generator(seed)

        best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
        history = []
        for epoch in range(1, p.epochs + 1):
            self.model.train()
            order = torch.randperm(n, generator=generator)
            total_loss = 0.0
            for start_idx in range(0, n, p.batch_size):
                idx = order[start_idx : start_idx + p.batch_size]
                batch = to_device(self._tokenize([data.texts[i] for i in idx]), self.device)
                with torch.autocast(device_type=self.device.type, enabled=use_amp):
                    logits = self.model(**batch).logits
                loss = distillation_loss(
                    logits.float(),
                    targets[idx].to(self.device),
                    soft[idx].to(self.device) if soft is not None else None,
                    alpha=p.kd_alpha,
                    temperature=p.kd_temperature,
                    label_smoothing=p.label_smoothing,
                )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                total_loss += float(loss.detach()) * len(idx)

            agreement = self._agreement(data.val_texts, data.val_targets)
            history.append({"epoch": epoch, "loss": total_loss / n, "val_agreement": agreement})
            logger.info("epoch %d: loss %.4f, val agreement %.4f", epoch, total_loss / n, agreement)
            if agreement > best_score:
                best_score, best_epoch, stale = agreement, epoch, 0
                best_state = copy.deepcopy({k: v.cpu() for k, v in self.model.state_dict().items()})
            else:
                stale += 1
                if stale >= p.early_stopping_patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to(torch.device("cpu"))
        self.device = torch.device("cpu")
        self.model.eval()
        return {
            "train_seconds": time.perf_counter() - start,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "history": history,
            "device": str(best_device()),
        }

    def _agreement(self, texts: Sequence[str], targets: np.ndarray) -> float:
        if not texts:
            return 0.0
        predictions = self.predict_proba(texts).argmax(axis=1)
        return float(np.mean(predictions == targets))

    @torch.inference_mode()
    def predict_proba(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("student is not trained")
        self.model.eval()
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = to_device(self._tokenize(texts[start : start + batch_size]), self.device)
            logits = self.model(**batch).logits.float()
            outputs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        if not outputs:
            return np.zeros((0, len(self.labels)))
        return np.concatenate(outputs).astype(np.float64)

    def _save_weights(self, directory: Path) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("student is not trained")
        self.model.save_pretrained(directory / HF_DIR)
        self.tokenizer.save_pretrained(directory / HF_DIR)

    def _load_weights(self, directory: Path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(directory / HF_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(directory / HF_DIR)
        self.model.eval()
        self.device = torch.device("cpu")

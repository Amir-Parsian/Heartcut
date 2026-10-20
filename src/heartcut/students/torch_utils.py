"""Small PyTorch helpers shared by the neural students."""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seeded_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def distillation_loss(
    logits: torch.Tensor,
    hard_targets: torch.Tensor,
    soft_targets: torch.Tensor | None,
    *,
    alpha: float,
    temperature: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy on hard labels, optionally mixed with a distillation term.

    With teacher distributions ``p`` (self-consistency vote shares), the soft term is
    ``T^2 * KL(softmax(log p / T) || softmax(z / T))``: the teacher distribution is
    tempered the same way as the student logits, and the ``T^2`` factor keeps gradient
    magnitudes comparable across temperatures (Hinton et al., 2015).
    """
    hard = torch.nn.functional.cross_entropy(logits, hard_targets, label_smoothing=label_smoothing)
    if soft_targets is None or alpha == 0.0:
        return hard
    teacher_logits = torch.log(soft_targets.clamp_min(1e-8)) / temperature
    teacher = torch.softmax(teacher_logits, dim=-1)
    student = torch.log_softmax(logits / temperature, dim=-1)
    soft = torch.nn.functional.kl_div(student, teacher, reduction="batchmean") * temperature**2
    return alpha * soft + (1.0 - alpha) * hard

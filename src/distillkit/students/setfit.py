"""Small student: a SetFit-style sentence-embedding classifier.

SetFit (Tunstall et al., 2022) trains in two steps:

1. Contrastive fine-tuning of a sentence-embedding model on pairs of labeled examples:
   pairs with the same label are pulled together (cosine target 1), pairs with
   different labels pushed apart (target 0).
2. A logistic-regression head on the (fine-tuned) embeddings.

This is a compact re-implementation of that recipe on top of sentence-transformers, so
every step is visible and the training loop does not depend on the ``setfit`` package's
release cadence.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from distillkit.config import SetFitStudentConfig
from distillkit.students.base import Student, TrainingData, expand_soft_targets, full_proba
from distillkit.students.torch_utils import best_device, to_device

BODY_DIR = "body"
HEAD_FILE = "head.joblib"


def make_pairs(
    targets: np.ndarray, n_pairs: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample ``n_pairs`` (anchor, other, similarity) triples, half positive, half negative.

    Classes with a single example cannot form positive pairs; their anchors are only used
    for negatives.
    """
    by_class: dict[int, np.ndarray] = {
        int(c): np.flatnonzero(targets == c) for c in np.unique(targets)
    }
    multi = [c for c, idx in by_class.items() if len(idx) > 1]
    anchors, others, sims = [], [], []
    for k in range(n_pairs):
        positive = k % 2 == 0 and bool(multi)
        if positive:
            cls = multi[rng.integers(len(multi))]
            a, b = rng.choice(by_class[cls], size=2, replace=False)
        else:
            a = int(rng.integers(len(targets)))
            candidates = np.flatnonzero(targets != targets[a])
            if candidates.size == 0:
                continue
            b = int(rng.choice(candidates))
        anchors.append(int(a))
        others.append(int(b))
        sims.append(1.0 if positive else 0.0)
    return np.array(anchors), np.array(others), np.array(sims, dtype=np.float32)


class SetFitStudent(Student):
    kind = "setfit"
    config: SetFitStudentConfig

    def __init__(self, config: SetFitStudentConfig, labels: Sequence[str]) -> None:
        super().__init__(config, labels)
        self.body: SentenceTransformer | None = None
        self.head: LogisticRegression | None = None

    @classmethod
    def config_model(cls) -> type[SetFitStudentConfig]:
        return SetFitStudentConfig

    # ------------------------------------------------------------------ training

    def _contrastive_finetune(self, texts: Sequence[str], targets: np.ndarray, seed: int) -> int:
        assert self.body is not None
        p = self.config.params
        rng = np.random.default_rng(seed)
        n_pairs = min(p.max_pairs, 20 * len(texts))
        anchors, others, sims = make_pairs(targets, n_pairs, rng)
        if anchors.size == 0:
            return 0
        device = self.body.device
        optimizer = torch.optim.AdamW(self.body.parameters(), lr=p.learning_rate)
        steps = 0
        self.body.train()
        for _ in range(p.contrastive_epochs):
            order = rng.permutation(len(anchors))
            for start in range(0, len(order), p.contrastive_batch_size):
                idx = order[start : start + p.contrastive_batch_size]
                left = to_device(self._features([texts[i] for i in anchors[idx]]), device)
                right = to_device(self._features([texts[i] for i in others[idx]]), device)
                u = self.body(left)["sentence_embedding"]
                v = self.body(right)["sentence_embedding"]
                cosine = torch.nn.functional.cosine_similarity(u, v)
                target = torch.as_tensor(sims[idx], device=device)
                loss = torch.nn.functional.mse_loss(cosine, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                steps += 1
        self.body.eval()
        return steps

    def _features(self, texts: list[str]) -> dict[str, Any]:
        assert self.body is not None
        # sentence-transformers 5 renamed tokenize() to preprocess().
        prepare = getattr(self.body, "preprocess", None) or self.body.tokenize
        features: dict[str, Any] = prepare(texts)
        return features

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self.body is None:
            raise RuntimeError("student is not trained")
        p = self.config.params
        embeddings = self.body.encode(
            list(texts),
            batch_size=p.embed_batch_size,
            normalize_embeddings=p.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def fit(self, data: TrainingData, seed: int) -> dict[str, Any]:
        p = self.config.params
        start = time.perf_counter()
        self.body = SentenceTransformer(p.model_name, device=str(best_device()))
        self.body.max_seq_length = p.max_length
        steps = 0
        if p.contrastive_epochs > 0:
            steps = self._contrastive_finetune(data.texts, data.targets, seed)
        contrastive_seconds = time.perf_counter() - start

        features = self.embed(data.texts)
        self.head = LogisticRegression(C=p.head_C, max_iter=p.head_max_iter, random_state=seed)
        if data.soft_targets is not None:
            rows, classes, weights = expand_soft_targets(data.soft_targets)
            self.head.fit(features[rows], classes, sample_weight=weights)
        else:
            self.head.fit(features, data.targets)
        return {
            "train_seconds": time.perf_counter() - start,
            "contrastive_seconds": contrastive_seconds,
            "contrastive_steps": steps,
            "embedding_dim": int(features.shape[1]),
            "device": str(self.body.device),
        }

    # ------------------------------------------------------------------ inference

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self.head is None:
            raise RuntimeError("student is not trained")
        partial = self.head.predict_proba(self.embed(texts))
        return full_proba(partial, self.head.classes_, len(self.labels))

    def _save_weights(self, directory: Path) -> None:
        if self.body is None or self.head is None:
            raise RuntimeError("student is not trained")
        self.body.save(str(directory / BODY_DIR))
        joblib.dump(self.head, directory / HEAD_FILE)

    def _load_weights(self, directory: Path) -> None:
        self.body = SentenceTransformer(str(directory / BODY_DIR), device="cpu")
        self.body.max_seq_length = self.config.params.max_length
        self.head = joblib.load(directory / HEAD_FILE)

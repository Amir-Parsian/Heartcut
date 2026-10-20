"""Baseline student: TF-IDF word and character n-grams + multinomial logistic regression."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from heartcut.config import TfidfStudentConfig
from heartcut.students.base import Student, TrainingData, expand_soft_targets, full_proba

MODEL_FILE = "tfidf.joblib"


class TfidfStudent(Student):
    """Word 1-2 grams catch keywords; character 2-5 grams (within word boundaries) make
    the model robust to typos and inflections, which are common in support messages."""

    kind = "tfidf"
    config: TfidfStudentConfig

    def __init__(self, config: TfidfStudentConfig, labels: Sequence[str]) -> None:
        super().__init__(config, labels)
        self.vectorizer: FeatureUnion | None = None
        self.classifier: LogisticRegression | None = None

    @classmethod
    def config_model(cls) -> type[TfidfStudentConfig]:
        return TfidfStudentConfig

    def _build_vectorizer(self) -> FeatureUnion:
        p = self.config.params
        parts: list[tuple[str, TfidfVectorizer]] = [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=p.word_ngram_range,
                    min_df=p.min_df,
                    max_features=p.max_features,
                    sublinear_tf=p.sublinear_tf,
                    lowercase=True,
                ),
            )
        ]
        if p.char_ngram_range is not None:
            parts.append(
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=p.char_ngram_range,
                        min_df=p.min_df,
                        max_features=p.max_features,
                        sublinear_tf=p.sublinear_tf,
                        lowercase=True,
                    ),
                )
            )
        return FeatureUnion(parts)

    def fit(self, data: TrainingData, seed: int) -> dict[str, Any]:
        start = time.perf_counter()
        p = self.config.params
        self.vectorizer = self._build_vectorizer()
        features = self.vectorizer.fit_transform(data.texts)
        self.classifier = LogisticRegression(C=p.C, max_iter=p.max_iter, random_state=seed)
        if data.soft_targets is not None:
            rows, classes, weights = expand_soft_targets(data.soft_targets)
            self.classifier.fit(features[rows], classes, sample_weight=weights)
        else:
            self.classifier.fit(features, data.targets)
        return {
            "train_seconds": time.perf_counter() - start,
            "n_features": int(features.shape[1]),
            "n_iter": int(np.max(self.classifier.n_iter_)),
        }

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self.vectorizer is None or self.classifier is None:
            raise RuntimeError("student is not trained")
        partial = self.classifier.predict_proba(self.vectorizer.transform(list(texts)))
        return full_proba(partial, self.classifier.classes_, len(self.labels))

    def _save_weights(self, directory: Path) -> None:
        joblib.dump(
            {"vectorizer": self.vectorizer, "classifier": self.classifier}, directory / MODEL_FILE
        )

    def _load_weights(self, directory: Path) -> None:
        state = joblib.load(directory / MODEL_FILE)
        self.vectorizer = state["vectorizer"]
        self.classifier = state["classifier"]

"""Map student ``kind`` values to classes. Neural students are imported lazily so the
TF-IDF baseline works without PyTorch installed."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module

from heartcut.students.base import AnyStudentConfig, Student

_CLASSES = {
    "tfidf": "heartcut.students.tfidf:TfidfStudent",
    "setfit": "heartcut.students.setfit:SetFitStudent",
    "encoder": "heartcut.students.encoder:EncoderStudent",
}


def student_class(kind: str) -> type[Student]:
    try:
        target = _CLASSES[kind]
    except KeyError:
        raise ValueError(f"unknown student kind {kind!r}; known: {sorted(_CLASSES)}") from None
    module_name, _, attr = target.partition(":")
    klass: type[Student] = getattr(import_module(module_name), attr)
    return klass


def build_student(config: AnyStudentConfig, labels: Sequence[str]) -> Student:
    return student_class(config.kind)(config, labels)

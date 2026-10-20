"""Student models: TF-IDF baseline, SetFit-style sentence-embedding model, fine-tuned encoder."""

from heartcut.students.base import Student, TrainingData
from heartcut.students.registry import build_student, student_class

__all__ = ["Student", "TrainingData", "build_student", "student_class"]

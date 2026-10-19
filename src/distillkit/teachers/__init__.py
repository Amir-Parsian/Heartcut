"""Teacher LLMs: prompts, backends, caching and the labeling stage."""

from distillkit.teachers.base import Completion, TeacherBackend
from distillkit.teachers.prompts import ChatPrompt, PromptTemplate, parse_label

__all__ = ["ChatPrompt", "Completion", "PromptTemplate", "TeacherBackend", "parse_label"]

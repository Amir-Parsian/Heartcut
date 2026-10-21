"""What the studio offers: teachers, students and a sample dataset.

The choices are the repository's own config files, so the studio and the CLI never
disagree about a model's settings. A source checkout reads them from the repository; an
installed wheel reads the copies packaged inside ``heartcut/`` (see ``force-include`` in
pyproject.toml), which keep the same relative layout so the teachers' relative
``prompt`` paths still resolve.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from heartcut.config import TeacherConfig

_HERE = Path(__file__).resolve()
_SEARCH_ROOTS = (_HERE.parents[3], _HERE.parents[1])  # source checkout, installed package

SAMPLE_DATA = "examples/helpdesk/tickets.jsonl"
SAMPLE_TEXT_COLUMNS = ("subject", "body")
SAMPLE_LABEL = "queue"
SAMPLE_TASK = "routing IT-helpdesk tickets to the right support queue"


def resource(relative: str) -> Path:
    """A file shipped with Heartcut, from the source checkout or the installed package."""
    for root in _SEARCH_ROOTS:
        candidate = root / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{relative} is missing from this Heartcut installation")


def _installed(*modules: str) -> list[str]:
    """The modules in ``modules`` that cannot be imported."""
    return [m for m in modules if importlib.util.find_spec(m) is None]


@dataclass(frozen=True)
class TeacherChoice:
    id: str
    title: str
    blurb: str
    kind: Literal["practice", "cloud", "local"]
    price: str
    config: str
    modules: tuple[str, ...] = ()
    extra: str = "teachers"

    def load(self) -> dict[str, Any]:
        """The teacher config as a mapping, with the prompt path made absolute."""
        path = resource(self.config)
        data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        data["prompt"] = str((path.parent / data["prompt"]).resolve())
        TeacherConfig.model_validate(data)
        return data

    def describe(self, session_keys: Mapping[str, str]) -> dict[str, Any]:
        data = self.load()
        key_env = data.get("api_key_env")
        missing = _installed(*self.modules)
        return {
            "id": self.id,
            "title": self.title,
            "blurb": self.blurb,
            "kind": self.kind,
            "price": self.price,
            "model": data["model"],
            "api_key_env": key_env,
            "has_key": bool(key_env and (os.environ.get(key_env) or session_keys.get(key_env))),
            "base_url": data.get("base_url"),
            "available": not missing,
            "install": f"pip install 'heartcut[studio,{self.extra}]'" if missing else None,
        }


@dataclass(frozen=True)
class StudentChoice:
    id: str
    title: str
    blurb: str
    size: str
    speed: str
    training: str
    config: str
    modules: tuple[str, ...] = ()

    def load(self) -> dict[str, Any]:
        data: dict[str, Any] = yaml.safe_load(resource(self.config).read_text(encoding="utf-8"))
        return data

    def describe(self) -> dict[str, Any]:
        missing = _installed(*self.modules)
        return {
            "id": self.id,
            "title": self.title,
            "blurb": self.blurb,
            "size": self.size,
            "speed": self.speed,
            "training": self.training,
            "available": not missing,
            "install": "pip install 'heartcut[studio,students,export]'" if missing else None,
        }


TEACHERS = (
    TeacherChoice(
        id="simulated",
        title="Practice run",
        blurb=(
            "Free and instant. No AI is called: a stand-in teacher copies your own "
            "categories and makes deliberate mistakes, so you can try every step before "
            "paying for anything. The scores are for practice only."
        ),
        kind="practice",
        price="Free",
        config="configs/teachers/simulated.yaml",
    ),
    TeacherChoice(
        id="claude-haiku-5-5",
        title="Claude Haiku",
        blurb="The fastest and cheapest Claude. A good first choice for most sorting tasks.",
        kind="cloud",
        price="$",
        config="configs/teachers/claude-haiku-5-5.yaml",
        modules=("anthropic",),
    ),
    TeacherChoice(
        id="claude-sonnet-5-5",
        title="Claude Sonnet",
        blurb="Stronger on subtle or overlapping categories. Costs more per message.",
        kind="cloud",
        price="$$",
        config="configs/teachers/claude-sonnet-5-5.yaml",
        modules=("anthropic",),
    ),
    TeacherChoice(
        id="claude-opus-5-5",
        title="Claude Opus",
        blurb="The most capable Claude, for the hardest tasks. The most expensive per message.",
        kind="cloud",
        price="$$$",
        config="configs/teachers/claude-opus-5-5.yaml",
        modules=("anthropic",),
    ),
    TeacherChoice(
        id="qwen2.5-7b-ollama",
        title="Local model (Ollama)",
        blurb=(
            "Runs on your own computer with Ollama, so your messages never leave it. "
            "Needs Ollama running with the model downloaded; slow without a graphics card."
        ),
        kind="local",
        price="No fee",
        config="configs/teachers/qwen2.5-7b-ollama.yaml",
        modules=("openai",),
    ),
)

_NEURAL = ("torch", "transformers", "onnx", "onnxruntime", "onnxscript")

STUDENTS = (
    StudentChoice(
        id="tfidf-logreg",
        title="Tiny",
        blurb=(
            "Learns which words and word pieces point to each category. Trains in seconds "
            "on any computer and is a strong baseline."
        ),
        size="a few MB",
        speed="under 1 ms",
        training="seconds",
        config="configs/students/tfidf-logreg.yaml",
        modules=("sklearn",),
    ),
    StudentChoice(
        id="setfit-minilm",
        title="Small",
        blurb=(
            "A compact language model (22M parameters) that understands meaning, not just "
            "words, so it copes better with rephrasings."
        ),
        size="about 25 MB",
        speed="a few ms",
        training="minutes",
        config="configs/students/setfit-minilm.yaml",
        modules=(*_NEURAL, "sentence_transformers"),
    ),
    StudentChoice(
        id="encoder-distilbert",
        title="Medium",
        blurb=(
            "DistilBERT (66M parameters), usually the most accurate of the three. Training "
            "is slow without a graphics card (GPU)."
        ),
        size="about 70 MB",
        speed="5-20 ms",
        training="minutes on a GPU, longer on a laptop",
        config="configs/students/encoder-distilbert.yaml",
        modules=_NEURAL,
    ),
)


def teacher(choice_id: str) -> TeacherChoice:
    for choice in TEACHERS:
        if choice.id == choice_id:
            return choice
    raise KeyError(f"unknown teacher {choice_id!r}")


def student(choice_id: str) -> StudentChoice:
    for choice in STUDENTS:
        if choice.id == choice_id:
            return choice
    raise KeyError(f"unknown student {choice_id!r}")


def student_title(name: str) -> str:
    """``Tiny (tfidf-logreg)`` for a catalog student, the name itself otherwise."""
    for choice in STUDENTS:
        if choice.id == name:
            return f"{choice.title} ({name})"
    return name

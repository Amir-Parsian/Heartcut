"""Prompt templates and parsing of the teacher's structured answer.

Templates are YAML files under ``prompts/`` with a ``system`` and a ``user`` part, using
``string.Template`` placeholders:

``$task``             task description from the dataset config
``$labels``           the label list, one per line
``$oos_instruction``  a rule for the out-of-scope label, or an empty string
``$text``             the message to classify (user part only)

The label list lives in the system prompt, so it forms an identical prefix for every
request and can be served from the provider's prompt cache.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

import yaml


@dataclass(frozen=True)
class ChatPrompt:
    system: str
    user: str
    # The raw message being classified. Not sent separately; kept for logging and for
    # the simulated backend.
    text: str = ""


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    system: str
    user: str
    digest: str

    @classmethod
    def from_file(cls, path: str | Path) -> PromptTemplate:
        path = Path(path)
        raw = path.read_bytes()
        data = yaml.safe_load(raw)
        if not isinstance(data, dict) or "system" not in data or "user" not in data:
            raise ValueError(f"{path}: a prompt file needs 'system' and 'user' entries")
        return cls(
            name=str(data.get("name", path.stem)),
            system=str(data["system"]),
            user=str(data["user"]),
            digest=hashlib.sha256(raw).hexdigest()[:10],
        )

    def render_system(self, labels: Sequence[str], task: str, oos_label: str | None = None) -> str:
        system = self.system
        oos_instruction = ""
        if oos_label:
            oos_instruction = (
                f'- If the message does not match any other label, answer "{oos_label}".'
            )
        else:
            # Drop the placeholder's whole line rather than leaving a blank rule.
            system = re.sub(r"^[ \t]*\$oos_instruction[ \t]*\n", "", system, flags=re.MULTILINE)
        return (
            Template(system)
            .substitute(
                task=task,
                labels="\n".join(f"- {label}" for label in labels),
                oos_instruction=oos_instruction,
            )
            .strip()
        )

    def render(
        self, text: str, labels: Sequence[str], task: str, oos_label: str | None = None
    ) -> ChatPrompt:
        return ChatPrompt(
            system=self.render_system(labels, task, oos_label),
            user=Template(self.user).substitute(text=text).strip(),
            text=text,
        )


def label_schema(labels: Sequence[str]) -> dict[str, Any]:
    """JSON schema that restricts the answer to exactly one known label."""
    return {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": list(labels)}},
        "required": ["label"],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class ParsedLabel:
    label: str | None
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.label is not None


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


def _normalise(label: str) -> str:
    return re.sub(r"[\s\-]+", "_", label.strip().strip("\"'").lower())


def parse_label(raw: str, labels: Sequence[str]) -> ParsedLabel:
    """Extract a label from the teacher's answer and validate it against the label set.

    Accepts a JSON object with a ``label`` key (optionally wrapped in a code fence or
    surrounded by prose), or a bare label. Matching ignores case, surrounding quotes and
    space/hyphen vs. underscore differences, but never guesses: anything that is not one
    of the labels is rejected and counted as invalid.
    """
    lookup = {_normalise(label): label for label in labels}
    text = _FENCE.sub("", raw.strip())
    if not text:
        return ParsedLabel(None, "empty response")

    candidate: str | None = None
    for match in [text, *_OBJECT.findall(text)]:
        try:
            obj = json.loads(match)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            value = obj.get("label", obj.get("intent"))
            if isinstance(value, str):
                candidate = value
                break
        elif isinstance(obj, str):
            candidate = obj
            break
    if candidate is None:
        candidate = text

    label = lookup.get(_normalise(candidate))
    if label is None:
        return ParsedLabel(None, f"not a known label: {candidate[:80]!r}")
    return ParsedLabel(label)

from __future__ import annotations

from pathlib import Path

import pytest

from distillkit.teachers.prompts import PromptTemplate, label_schema, parse_label
from tests.conftest import PROMPT

LABELS = ["card_arrival", "lost_or_stolen_card", "Refund_not_showing_up", "oos"]


def test_render_injects_labels_task_and_text() -> None:
    template = PromptTemplate.from_file(PROMPT)
    prompt = template.render("Where is my card?", LABELS, task="banking intents")
    assert "- card_arrival\n- lost_or_stolen_card" in prompt.system
    assert "banking intents" in prompt.system
    assert "$" not in prompt.system
    assert "Where is my card?" in prompt.user
    assert prompt.text == "Where is my card?"
    assert "out-of-scope" not in prompt.system and 'answer "oos"' not in prompt.system


def test_system_prompt_is_identical_across_messages() -> None:
    template = PromptTemplate.from_file(PROMPT)
    a = template.render("first", LABELS, task="t")
    b = template.render("second", LABELS, task="t")
    assert a.system == b.system  # a stable prefix is what makes prompt caching work


def test_oos_instruction_is_added_when_configured() -> None:
    template = PromptTemplate.from_file(PROMPT)
    system = template.render_system(LABELS, task="t", oos_label="oos")
    assert 'answer "oos"' in system


def test_template_file_validation(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("system: hi\n")
    with pytest.raises(ValueError, match="'system' and 'user'"):
        PromptTemplate.from_file(bad)


def test_digest_changes_with_content(tmp_path: Path) -> None:
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("system: one $labels\nuser: $text\n")
    b.write_text("system: two $labels\nuser: $text\n")
    assert PromptTemplate.from_file(a).digest != PromptTemplate.from_file(b).digest


def test_label_schema_is_an_enum() -> None:
    schema = label_schema(LABELS)
    assert schema["properties"]["label"]["enum"] == LABELS
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"label": "card_arrival"}', "card_arrival"),
        ('```json\n{"label": "card_arrival"}\n```', "card_arrival"),
        ('Sure! {"label": "lost_or_stolen_card"} is my answer.', "lost_or_stolen_card"),
        ('{"intent": "oos"}', "oos"),
        ("card_arrival", "card_arrival"),
        ('"card_arrival"', "card_arrival"),
        ('{"label": "Card Arrival"}', "card_arrival"),
        ('{"label": "refund-not-showing-up"}', "Refund_not_showing_up"),
    ],
)
def test_parse_label_accepts_valid_answers(raw: str, expected: str) -> None:
    parsed = parse_label(raw, LABELS)
    assert parsed.valid
    assert parsed.label == expected


@pytest.mark.parametrize(
    "raw",
    ["", "   ", '{"label": "card"}', "I think it is about cards", '{"label": 3}', "{not json}"],
)
def test_parse_label_rejects_everything_else(raw: str) -> None:
    parsed = parse_label(raw, LABELS)
    assert not parsed.valid
    assert parsed.error

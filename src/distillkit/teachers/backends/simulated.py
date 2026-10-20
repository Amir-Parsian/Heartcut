"""A deterministic stand-in teacher for tests, CI and dry runs.

It answers with the *gold* label with probability ``accuracy`` and with a random other
label otherwise. Because it reads the hidden human labels, anything labeled with it is
marked ``simulated`` in every manifest and results file, and the README generator
refuses to publish such results.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from typing import Any

from distillkit.config import SimulatedTeacherConfig, TeacherConfig
from distillkit.teachers.base import Completion, TeacherBackend
from distillkit.teachers.prompts import ChatPrompt


class SimulatedBackend(TeacherBackend):
    def __init__(
        self, config: TeacherConfig, *, labels: Sequence[str], oracle: Mapping[str, str]
    ) -> None:
        super().__init__(config)
        self.settings = config.simulated or SimulatedTeacherConfig()
        self.labels = list(labels)
        self.oracle = dict(oracle)

    def _rng(self, text: str, sample_index: int) -> random.Random:
        seed = hashlib.sha256(f"{self.settings.seed}|{sample_index}|{text}".encode()).hexdigest()
        return random.Random(int(seed[:16], 16))

    async def complete(
        self, prompt: ChatPrompt, schema: dict[str, Any] | None, sample_index: int = 0
    ) -> Completion:
        rng = self._rng(prompt.text, sample_index)
        gold = self.oracle.get(prompt.text)
        if rng.random() < self.settings.invalid_rate:
            text = "I'm not sure which intent this is."
        else:
            if gold is not None and rng.random() < self.settings.accuracy:
                label = gold
            else:
                label = rng.choice([lbl for lbl in self.labels if lbl != gold] or self.labels)
            text = json.dumps({"label": label})
        if self.settings.latency_ms:
            await asyncio.sleep(self.settings.latency_ms / 1000)
        return Completion(
            text=text,
            # Rough token counts (about four characters per token) so cost code runs.
            input_tokens=(len(prompt.system) + len(prompt.user)) // 4,
            output_tokens=max(1, len(text) // 4),
            latency_ms=self.settings.latency_ms,
            stop_reason="end_turn",
            model=f"simulated:{self.config.model}",
        )

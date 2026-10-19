"""Custom loader for the example IT-helpdesk dataset.

The raw data has its own schema (``subject``, ``body``, ``queue``) and no official
test split. The loader turns it into what DistillKit expects: ``{"train": df, "test":
df}`` with ``text`` and ``label`` columns. DistillKit carves the validation pool out of
train, hides the train/validation labels, and does everything else.

Referenced from ``dataset.yaml`` as ``examples/helpdesk/loader.py:load_tickets``.
"""

from __future__ import annotations

import pandas as pd

from distillkit.data.splits import stratified_split


def load_tickets(path: str, test_fraction: float = 0.25, seed: int = 0) -> dict[str, pd.DataFrame]:
    raw = pd.read_json(path, lines=True)
    frame = pd.DataFrame(
        {
            # The subject line carries a lot of signal, so the model sees both.
            "text": raw["subject"].str.strip() + ". " + raw["body"].str.strip(),
            "label": raw["queue"],
        }
    )
    train, test = stratified_split(frame, test_fraction, seed)
    return {"train": train, "test": test}

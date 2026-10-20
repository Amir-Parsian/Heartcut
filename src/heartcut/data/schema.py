"""Common in-memory schema shared by every dataset.

Every split is a :class:`pandas.DataFrame` with the columns ``id``, ``text`` and
``label``. For the splits treated as unlabeled (train and validation) the ``label``
column is ``None``: the human labels are moved to a separate *hidden* table that only
the label-quality audit is allowed to read.
"""

from __future__ import annotations

import pandas as pd

ID = "id"
TEXT = "text"
LABEL = "label"
SPLIT = "split"

SPLITS = ("train", "validation", "test")
UNLABELED_SPLITS = ("train", "validation")
COLUMNS = [ID, TEXT, LABEL]


def validate_frame(df: pd.DataFrame, *, split: str, labeled: bool) -> None:
    """Raise ``ValueError`` if ``df`` does not follow the common schema."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{split}: missing columns {missing}")
    if df[ID].duplicated().any():
        raise ValueError(f"{split}: duplicate ids")
    if df[TEXT].isna().any() or (df[TEXT].str.strip() == "").any():
        raise ValueError(f"{split}: empty texts")
    if labeled and df[LABEL].isna().any():
        raise ValueError(f"{split}: missing labels in a labeled split")
    if not labeled and df[LABEL].notna().any():
        raise ValueError(f"{split}: labels must be hidden in an unlabeled split")

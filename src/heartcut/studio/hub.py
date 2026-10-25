"""Fetch a dataset from the Hugging Face Hub as a table, for the studio's data step.

The person types a dataset id such as ``PolyAI/banking77``. The dataset is downloaded
once into the usual Hugging Face cache (``HF_HOME``), every split they asked for is
joined into one table, and class-label ids become their names. The studio then treats
the table like a dropped spreadsheet: it does its own train/test split, so the Hub's
splits only decide which rows come in.

Public datasets need no account. Gated and private ones need a token: ``HF_TOKEN``,
a ``huggingface-cli login`` on this computer, or one pasted into the page. As for every
Hugging Face tool, ``HF_ENDPOINT`` points at a mirror and ``HF_HUB_OFFLINE=1`` reads
only what is already cached.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

import pandas as pd

from heartcut.studio.tables import MAX_ROWS, TableError

INSTALL = "pip install 'heartcut[studio]'"
DEFAULT_ROWS = 10_000
SAMPLE_SEED = 13
TOKEN_ENV = "HF_TOKEN"
PARQUET_BRANCH = "refs/convert/parquet"
# owner/name, or a bare name for the few datasets without an owner (e.g. "imdb"), with
# the Hub's rules for names: letters, digits, "-", "_" and ".", no ".." and no leading dot.
DATASET_ID = re.compile(r"^(?:[A-Za-z0-9][\w.-]*/)?[A-Za-z0-9][\w.-]*$")
_SCALARS = ("string", "large_string", "bool", "int", "uint", "float", "double", "decimal")


def available() -> bool:
    """Whether the ``datasets`` package is installed."""
    return importlib.util.find_spec("datasets") is not None


def saved_token() -> str | None:
    """The token ``huggingface_hub`` would use on its own (``HF_TOKEN`` or a login)."""
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    token: str | None = get_token()
    return token


def fetch_table(
    dataset: str,
    *,
    config: str | None = None,
    split: str | None = None,
    revision: str | None = None,
    max_rows: int = DEFAULT_ROWS,
    token: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The dataset's rows (a seeded random sample if there are more than ``max_rows``).

    ``split`` is one split name, or ``None`` for every split together. Columns that hold
    lists, images or audio are left out; class labels are turned into their names, and
    a label outside the class list (``-1`` marks "no label" in many datasets) becomes an
    empty cell. Returns the table and a description of where it came from.
    """
    dataset = dataset.strip().removeprefix("https://huggingface.co/datasets/").strip("/")
    if not DATASET_ID.match(dataset) or ".." in dataset:
        raise TableError(
            f"'{dataset}' isn't a Hugging Face dataset name. Copy it from the dataset's page, "
            "for example PolyAI/banking77."
        )
    if Path(dataset).exists():
        # load_dataset would read the local folder or file of that name instead.
        raise TableError(
            f"There is a file or folder named '{dataset}' where the studio was started, so "
            "the dataset can't be fetched by that name. Start the studio from another folder."
        )
    if not available():
        raise TableError(f"Fetching from Hugging Face needs the datasets package: {INSTALL}")
    max_rows = max(1, min(max_rows, MAX_ROWS))
    from datasets import ClassLabel, DatasetDict, concatenate_datasets, load_dataset

    def load(at: str | None) -> Any:
        return load_dataset(dataset, config or None, split=split or None, revision=at, token=token)

    try:
        try:
            loaded = load(revision or None)
        except Exception as exc:
            # Recent versions of datasets no longer run loading scripts; the Hub keeps a
            # Parquet copy of many such datasets on a branch of their own.
            if revision or "script" not in str(exc).lower():
                raise
            loaded = load(PARQUET_BRANCH)
            revision = PARQUET_BRANCH
    except Exception as exc:
        raise TableError(_explain(dataset, exc)) from exc
    parts = list(loaded.values()) if isinstance(loaded, DatasetDict) else [loaded]
    splits = list(loaded) if isinstance(loaded, DatasetDict) else [split]
    try:
        data = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    except ValueError as exc:
        raise TableError(
            f"The splits of {dataset} have different columns, so they can't be joined. "
            f"Fetch one of them: {', '.join(map(str, splits))}."
        ) from exc
    total = len(data)
    if total > max_rows:
        data = data.shuffle(seed=SAMPLE_SEED).select(range(max_rows))

    labels = {n: f for n, f in data.features.items() if isinstance(f, ClassLabel)}
    keep = [
        name
        for name, feature in data.features.items()
        if name in labels or str(getattr(feature, "dtype", "")).startswith(_SCALARS)
    ]
    if not keep:
        raise TableError(
            f"{dataset} has no text or category columns that Heartcut can read (only lists, "
            "images or audio)."
        )
    frame: pd.DataFrame = data.select_columns(keep).to_pandas()
    for name, feature in labels.items():
        names = feature.names
        frame[name] = [
            names[int(v)] if pd.notna(v) and 0 <= int(v) < len(names) else None for v in frame[name]
        ]
    source = {
        "kind": "huggingface",
        "dataset": dataset,
        "config": config or None,
        "splits": splits,
        "revision": revision or None,
        "rows": len(data),
        "total_rows": total,
        "url": f"https://huggingface.co/datasets/{dataset}",
    }
    return frame, source


def _explain(dataset: str, exc: Exception) -> str:
    """An error from ``load_dataset``, worded for the person who typed the name."""
    message = str(exc).strip()
    lowered = message.lower()
    kind = type(exc).__name__
    if "script" in lowered:
        return (
            f"{dataset} is published as a loading script, which current versions of the "
            "datasets library no longer run, and the Hub has no Parquet copy of it. Search "
            "the Hub for another copy of the same data, or download it as CSV and drop the file."
        )
    if "offline" in lowered:
        return (
            f"{dataset} isn't in this computer's Hugging Face cache, and HF_HUB_OFFLINE is "
            "set, so Heartcut can't download it. Unset HF_HUB_OFFLINE and try again."
        )
    if kind in ("ConnectionError", "ConnectTimeout", "ReadTimeout", "Timeout") or (
        "connection" in lowered and "not found" not in lowered
    ):
        return (
            "Heartcut can't reach the Hugging Face Hub. Check your internet connection (and "
            "any proxy), or set HF_ENDPOINT if you use a mirror."
        )
    if "gated" in lowered:
        return (
            f"{dataset} is gated: accept its terms on huggingface.co/datasets/{dataset} with "
            "your account, then add a Hugging Face access token below."
        )
    if "config name is missing" in lowered or "builderconfig" in lowered:
        return f"{dataset} has several subsets. Type the one you want in Subset. ({message})"
    if "unknown split" in lowered or "bad split" in lowered:
        return f"{dataset} has no such split. ({message})"
    if kind in ("DatasetNotFoundError", "RepositoryNotFoundError", "FileNotFoundError") or (
        "doesn't exist" in lowered or "not found" in lowered
    ):
        return (
            f"Heartcut couldn't find the dataset {dataset} on the Hugging Face Hub. Check the "
            "name; if it is private, add a Hugging Face access token below."
        )
    return f"Heartcut couldn't fetch {dataset} from the Hugging Face Hub: {message}"

"""Dataset loaders.

A loader is a function that returns ``{split: DataFrame}`` where every frame has the
columns ``text`` and ``label`` (label *names*, not integer ids). Split names are
normalised to ``train``, ``validation`` and ``test``.

Built-in loaders are registered by name. A dataset config can also point at any
function with an import path (``my_package.loaders:load_tickets``), so adding a new
dataset never requires editing this module.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import shutil
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from distillkit.artifacts import cache_root
from distillkit.data.schema import LABEL, TEXT

logger = logging.getLogger(__name__)

RawSplits = dict[str, pd.DataFrame]
Loader = Callable[..., RawSplits]

_REGISTRY: dict[str, Loader] = {}

_SPLIT_ALIASES = {"val": "validation", "valid": "validation", "dev": "validation"}


def register_loader(name: str) -> Callable[[Loader], Loader]:
    """Decorator that makes a loader available under ``name`` in dataset configs."""

    def decorator(fn: Loader) -> Loader:
        if name in _REGISTRY and _REGISTRY[name] is not fn:
            raise ValueError(f"loader {name!r} is already registered")
        _REGISTRY[name] = fn
        return fn

    return decorator


def available_loaders() -> list[str]:
    return sorted(_REGISTRY)


def resolve_loader(spec: str) -> Loader:
    """Return the loader registered as ``spec`` or importable as ``module:function``."""
    if spec in _REGISTRY:
        return _REGISTRY[spec]
    if ":" in spec:
        module_name, _, attr = spec.partition(":")
        module = importlib.import_module(module_name)
        fn = getattr(module, attr, None)
        if not callable(fn):
            raise ValueError(f"{spec!r} does not name a callable")
        return fn  # type: ignore[no-any-return]
    raise ValueError(f"unknown loader {spec!r}; built-ins: {available_loaders()}")


def normalise_split_name(name: str) -> str:
    return _SPLIT_ALIASES.get(name, name)


def _frame(texts: Any, labels: Any) -> pd.DataFrame:
    df = pd.DataFrame({TEXT: list(texts), LABEL: list(labels)})
    df[TEXT] = df[TEXT].astype(str).str.strip()
    df[LABEL] = df[LABEL].astype(str)
    return df


# --------------------------------------------------------------------------- downloads


def fetch(location: str | Path) -> Path:
    """Return a local path for ``location``, downloading URLs once into the cache."""
    location = str(location)
    if not location.startswith(("http://", "https://")):
        path = Path(location)
        if not path.exists():
            raise FileNotFoundError(location)
        return path
    name = location.rsplit("/", 1)[-1] or "download"
    digest = hashlib.sha256(location.encode()).hexdigest()[:16]
    target = cache_root() / "downloads" / f"{digest}-{name}"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s", location)
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(location, timeout=60) as response, tmp.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    tmp.replace(target)
    return target


# --------------------------------------------------------------------------- built-ins


@register_loader("huggingface")
def load_huggingface(
    path: str,
    name: str | None = None,
    revision: str | None = None,
    text_column: str = "text",
    label_column: str = "label",
    splits: Mapping[str, str] | None = None,
) -> RawSplits:
    """Load a dataset from the Hugging Face Hub.

    ``splits`` maps canonical split names to the dataset's own split names, for example
    ``{"train": "train", "test": "test"}``. Integer ``ClassLabel`` columns are converted
    to their string names.
    """
    from datasets import ClassLabel, load_dataset

    dataset = load_dataset(path, name, revision=revision)
    mapping = dict(splits) if splits else {normalise_split_name(s): s for s in dataset}
    out: RawSplits = {}
    for canonical, source in mapping.items():
        part = dataset[source]
        feature = part.features[label_column]
        labels = part[label_column]
        if isinstance(feature, ClassLabel):
            labels = feature.int2str(labels)
        out[normalise_split_name(canonical)] = _frame(part[text_column], labels)
    return out


def _read_table(
    location: str | Path, reader: Callable[..., pd.DataFrame], **kwargs: Any
) -> pd.DataFrame:
    return reader(fetch(location), **kwargs)


@register_loader("csv")
def load_csv(
    files: Mapping[str, str],
    text_column: str = "text",
    label_column: str = "label",
    read_csv_kwargs: Mapping[str, Any] | None = None,
) -> RawSplits:
    """Load one CSV file (local path or URL) per split."""
    out: RawSplits = {}
    for split, location in files.items():
        df = _read_table(location, pd.read_csv, **dict(read_csv_kwargs or {}))
        out[normalise_split_name(split)] = _frame(df[text_column], df[label_column])
    return out


@register_loader("jsonl")
def load_jsonl(
    files: Mapping[str, str],
    text_column: str = "text",
    label_column: str = "label",
) -> RawSplits:
    """Load one JSON-lines file (local path or URL) per split."""
    out: RawSplits = {}
    for split, location in files.items():
        df = _read_table(location, pd.read_json, lines=True)
        out[normalise_split_name(split)] = _frame(df[text_column], df[label_column])
    return out


@register_loader("clinc_json")
def load_clinc_json(file: str, include_oos: bool = True, oos_label: str = "oos") -> RawSplits:
    """Load the original CLINC150 release (``data_full.json`` / ``data_oos_plus.json``).

    The file holds ``train``/``val``/``test`` lists of ``[text, intent]`` pairs and, when
    ``include_oos`` is true, the matching ``oos_*`` lists are merged in under
    ``oos_label``.
    """
    raw = json.loads(fetch(file).read_text(encoding="utf-8"))
    out: RawSplits = {}
    for split in ("train", "val", "test"):
        rows = list(raw[split])
        if include_oos:
            rows += [[text, oos_label] for text, _ in raw.get(f"oos_{split}", [])]
        out[normalise_split_name(split)] = _frame([r[0] for r in rows], [r[1] for r in rows])
    return out

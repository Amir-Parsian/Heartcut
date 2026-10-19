"""The ``distillkit data`` stage: raw dataset -> common schema with hidden labels.

Outputs (in ``artifacts/<experiment>/data/<dataset>-<fingerprint>/``)::

    train.parquet        id, text, label=None   (unlabeled pool sent to the teacher)
    validation.parquet   id, text, label=None   (also teacher-labeled; used for selection)
    test.parquet         id, text, label        (human gold labels; final scores only)
    labels.json          ordered label names
    hidden/gold.parquet  id, split, label       (human labels of train/validation)
    manifest.json

The hidden table exists so that teacher label quality can be measured. Only
:mod:`distillkit.eval.audit` (and the explicitly simulated teacher used in tests) reads it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from distillkit.artifacts import StageDir, fingerprint, read_manifest, stage_dir, write_manifest
from distillkit.config import DatasetConfig, DatasetSource, ExperimentConfig
from distillkit.data.loaders import RawSplits, resolve_loader
from distillkit.data.schema import (
    ID,
    LABEL,
    SPLIT,
    SPLITS,
    TEXT,
    UNLABELED_SPLITS,
    validate_frame,
)
from distillkit.data.splits import stratified_sample, stratified_split

logger = logging.getLogger(__name__)

HIDDEN_DIR = "hidden"
GOLD_FILE = "gold.parquet"


@dataclass(frozen=True)
class PreparedData:
    directory: Path
    fingerprint: str
    name: str
    labels: list[str]
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def split(self, name: str) -> pd.DataFrame:
        frame: pd.DataFrame = getattr(self, name)
        return frame


def data_stage(cfg: ExperimentConfig) -> StageDir:
    fp = fingerprint("data", cfg.dataset.model_dump(mode="json"))
    return stage_dir(cfg.experiment_dir, "data", cfg.dataset.name, fp)


def load_raw(dataset: DatasetConfig) -> tuple[RawSplits, DatasetSource]:
    """Try each configured source in order and return the first that loads."""
    errors: list[str] = []
    for source in dataset.sources:
        try:
            raw = resolve_loader(source.loader)(**source.args)
        except Exception as exc:
            logger.warning("source %s failed: %s", source.loader, exc)
            errors.append(f"{source.loader}: {type(exc).__name__}: {exc}")
            continue
        logger.info("loaded %s from %s", dataset.name, source.loader)
        return raw, source
    raise RuntimeError(f"could not load {dataset.name!r} from any source:\n" + "\n".join(errors))


def build_splits(raw: RawSplits, dataset: DatasetConfig) -> dict[str, pd.DataFrame]:
    """Turn raw loader output into deterministic train/validation/test frames with ids."""
    if "train" not in raw or "test" not in raw:
        raise ValueError(f"{dataset.name}: a loader must return at least 'train' and 'test'")
    splits = {name: frame.reset_index(drop=True) for name, frame in raw.items()}
    if "validation" not in splits:
        splits["train"], splits["validation"] = stratified_split(
            splits["train"], dataset.validation_fraction, dataset.seed
        )
    for capped, cap in dataset.max_examples.items():
        splits[capped] = stratified_sample(splits[capped], cap, dataset.seed)
    out: dict[str, pd.DataFrame] = {}
    for name in SPLITS:
        frame = splits[name][[TEXT, LABEL]].copy()
        frame.insert(0, ID, [f"{name}-{i:06d}" for i in range(len(frame))])
        out[name] = frame
    return out


def prepare_data(cfg: ExperimentConfig, *, force: bool = False) -> PreparedData:
    """Run the data stage (or reuse a complete previous run with the same fingerprint)."""
    stage = data_stage(cfg)
    if stage.is_complete() and not force:
        logger.info("data stage up to date: %s", stage.path)
        return load_prepared(cfg)

    raw, source = load_raw(cfg.dataset)
    splits = build_splits(raw, cfg.dataset)
    labels = sorted(set().union(*(set(f[LABEL]) for f in splits.values())))

    directory = stage.ensure()
    gold_parts = []
    for name, frame in splits.items():
        if name in UNLABELED_SPLITS:
            gold_parts.append(frame[[ID, LABEL]].assign(**{SPLIT: name}))
            frame = frame.assign(**{LABEL: None})
        validate_frame(frame, split=name, labeled=name not in UNLABELED_SPLITS)
        frame.to_parquet(directory / f"{name}.parquet", index=False)
    (directory / HIDDEN_DIR).mkdir(exist_ok=True)
    pd.concat(gold_parts, ignore_index=True).to_parquet(
        directory / HIDDEN_DIR / GOLD_FILE, index=False
    )
    (directory / "labels.json").write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")

    write_manifest(
        directory,
        {
            "stage": "data",
            "fingerprint": stage.fingerprint,
            "dataset": cfg.dataset.model_dump(mode="json"),
            "source_used": source.model_dump(mode="json"),
            "counts": {name: len(frame) for name, frame in splits.items()},
            "n_labels": len(labels),
            "label_distribution": _label_distribution(splits),
            "overlap": _overlap_report(splits),
        },
    )
    return load_prepared(cfg)


def _label_distribution(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, frame in splits.items():
        counts = frame[LABEL].value_counts()
        out[name] = {"min": int(counts.min()), "max": int(counts.max()), "classes": len(counts)}
    return out


def _overlap_report(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Exact-duplicate texts between splits (reported, never silently removed)."""
    norm = {name: set(f[TEXT].str.lower()) for name, f in splits.items()}
    return {
        "test_in_train": len(norm["test"] & norm["train"]),
        "test_in_validation": len(norm["test"] & norm["validation"]),
        "validation_in_train": len(norm["validation"] & norm["train"]),
    }


def load_prepared(cfg: ExperimentConfig) -> PreparedData:
    """Load the data stage output. Hidden gold labels are *not* loaded."""
    stage = data_stage(cfg)
    if not stage.is_complete():
        raise FileNotFoundError(f"data stage not found at {stage.path}; run `distillkit data`")
    directory = stage.path
    labels = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
    frames = {name: pd.read_parquet(directory / f"{name}.parquet") for name in SPLITS}
    return PreparedData(
        directory=directory,
        fingerprint=stage.fingerprint,
        name=cfg.dataset.name,
        labels=labels,
        **frames,
    )


def load_hidden_gold(cfg: ExperimentConfig) -> pd.DataFrame:
    """Human labels of the unlabeled splits. For label-quality audits only."""
    stage = data_stage(cfg)
    return pd.read_parquet(stage.path / HIDDEN_DIR / GOLD_FILE)


def data_manifest(cfg: ExperimentConfig) -> dict[str, Any]:
    return read_manifest(data_stage(cfg).path)

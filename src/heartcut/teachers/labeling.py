"""The ``heartcut label`` stage: send every split to the teacher.

Outputs (in ``artifacts/<experiment>/labels/<teacher>-<fingerprint>/``)::

    train.parquet, validation.parquet, test.parquet
        id, text, teacher_label, confidence, votes (JSON), sample_labels, cost_usd
    calls.jsonl        one line per teacher call: tokens, dollars, latency, cache hit
    cost_report.json   per split and total: calls, cache hits, tokens, dollars, latency
    manifest.json

Train and validation labels are what the students learn from. Test-split labels are
the teacher's *predictions*: they are only used to score the teacher against the human
labels in the benchmark.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from heartcut.artifacts import (
    StageDir,
    cache_root,
    file_digest,
    fingerprint,
    read_manifest,
    stage_dir,
    write_manifest,
)
from heartcut.config import ExperimentConfig, TokenPricing
from heartcut.data.prepare import PreparedData, load_hidden_gold, load_prepared
from heartcut.data.schema import ID, LABEL, SPLITS, TEXT
from heartcut.teachers.backends import build_backend
from heartcut.teachers.cache import CompletionCache
from heartcut.teachers.pricing import estimate_cost, resolve_pricing
from heartcut.teachers.prompts import PromptTemplate
from heartcut.teachers.runner import ExampleLabel, LabelingRunner

logger = logging.getLogger(__name__)

TEACHER_LABEL = "teacher_label"
CONFIDENCE = "confidence"
VOTES = "votes"
SAMPLE_LABELS = "sample_labels"
COST = "cost_usd"
CALLS_FILE = "calls.jsonl"
COST_REPORT = "cost_report.json"


def labels_stage(cfg: ExperimentConfig, data_fingerprint: str | None = None) -> StageDir:
    if data_fingerprint is None:
        from heartcut.data.prepare import data_stage

        data_fingerprint = data_stage(cfg).fingerprint
    prompt_digest = file_digest(Path(cfg.teacher.prompt))
    fp = fingerprint("labels", data_fingerprint, cfg.teacher.identity(), prompt_digest)
    return stage_dir(cfg.experiment_dir, "labels", cfg.teacher.name, fp)


def teacher_cache_path() -> Path:
    return cache_root() / "teacher_cache.sqlite"


@dataclass(frozen=True)
class LabelingOutcome:
    directory: Path
    cost_report: dict[str, Any]
    skipped: bool = False


def _oracle(cfg: ExperimentConfig, data: PreparedData) -> dict[str, str]:
    """Gold labels by text, for the simulated teacher only."""
    gold = load_hidden_gold(cfg)
    by_id = dict(zip(gold[ID], gold[LABEL], strict=True))
    oracle: dict[str, str] = {}
    for split in ("train", "validation"):
        frame = data.split(split)
        oracle.update({t: by_id[i] for i, t in zip(frame[ID], frame[TEXT], strict=True)})
    oracle.update(dict(zip(data.test[TEXT], data.test[LABEL], strict=True)))
    return oracle


def _frame(examples: Sequence[ExampleLabel], texts: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID: [e.id for e in examples],
            TEXT: list(texts),
            TEACHER_LABEL: [e.label for e in examples],
            CONFIDENCE: [e.confidence for e in examples],
            VOTES: [json.dumps(e.votes, sort_keys=True) for e in examples],
            SAMPLE_LABELS: [e.sample_labels for e in examples],
            COST: [e.cost_usd for e in examples],
        }
    )


def summarise_calls(records: Sequence[dict[str, Any]], n_examples: int) -> dict[str, Any]:
    """Aggregate call records into a cost/latency summary."""
    if not records:
        return {"examples": n_examples, "calls": 0}
    frame = pd.DataFrame(records)
    costs = frame["cost_usd"]
    priced = bool(costs.notna().all())
    list_cost = float(costs.sum()) if priced else None
    paid = frame.loc[~frame["cache_hit"]]
    latency = frame["latency_ms"].to_numpy(dtype=float)
    return {
        "examples": n_examples,
        "calls": len(frame),
        "cache_hits": int(frame["cache_hit"].sum()),
        "paid_calls_this_run": len(paid),
        "spent_this_run_usd": float(paid["cost_usd"].sum()) if priced else None,
        # What labeling these examples costs from scratch at list price.
        "list_cost_usd": list_cost,
        "cost_per_example_usd": list_cost / n_examples if list_cost is not None else None,
        "input_tokens": int(frame["input_tokens"].sum()),
        "output_tokens": int(frame["output_tokens"].sum()),
        "cache_read_tokens": int(frame["cache_read_tokens"].sum()),
        "cache_write_tokens": int(frame["cache_write_tokens"].sum()),
        "invalid_answers": int(frame["label"].isna().sum()),
        "invalid_rate": float(frame["label"].isna().mean()),
        "refusals": int((frame["error"] == "refusal").sum()),
        "retries": int(frame["retries"].sum()),
        "latency_ms": {
            "p50": float(np.percentile(latency, 50)),
            "p95": float(np.percentile(latency, 95)),
            "mean": float(latency.mean()),
        },
    }


def _pricing_dump(pricing: TokenPricing | None) -> dict[str, Any] | None:
    return pricing.model_dump(mode="json") if pricing else None


def run_labeling(
    cfg: ExperimentConfig,
    *,
    splits: Sequence[str] = SPLITS,
    offline: bool = False,
    force: bool = False,
    cache_path: Path | None = None,
) -> LabelingOutcome:
    """Label the requested splits. Already-labeled splits are reused unless ``force``."""
    data = load_prepared(cfg)
    stage = labels_stage(cfg, data.fingerprint)
    directory = stage.ensure()
    todo = [s for s in splits if force or not (directory / f"{s}.parquet").exists()]
    if not todo and stage.is_complete():
        logger.info("labels up to date: %s", directory)
        return LabelingOutcome(directory, read_cost_report(directory), skipped=True)

    teacher = cfg.teacher
    pricing = resolve_pricing(teacher)
    if pricing is None and not teacher.is_local and teacher.backend != "simulated":
        logger.warning(
            "no published price for %s: costs will be reported as unknown. "
            "Add it to configs/pricing.yaml or set teacher.pricing.",
            teacher.model,
        )
    oracle = _oracle(cfg, data) if teacher.backend == "simulated" else None
    backend = build_backend(teacher, labels=data.labels, oracle=oracle)
    template = PromptTemplate.from_file(teacher.prompt)

    calls_path = directory / CALLS_FILE
    previous = [r for r in read_calls(directory) if r["split"] not in todo]
    all_records = list(previous)
    with CompletionCache(cache_path or teacher_cache_path()) as cache:
        runner = LabelingRunner(
            backend,
            template,
            data.labels,
            config=teacher,
            task=cfg.dataset.task,
            oos_label=cfg.dataset.oos_label,
            cache=cache,
            pricing=pricing,
            offline=offline,
        )

        async def label_all() -> None:
            try:
                for split in todo:
                    frame = data.split(split)
                    logger.info(
                        "labeling %s: %d examples x %d samples",
                        split,
                        len(frame),
                        teacher.self_consistency,
                    )
                    examples, records = await runner.label(
                        list(frame[ID]), list(frame[TEXT]), split
                    )
                    _frame(examples, list(frame[TEXT])).to_parquet(
                        directory / f"{split}.parquet", index=False
                    )
                    all_records.extend(r.to_dict() for r in records)
                    _write_calls(calls_path, all_records)
            finally:
                await backend.aclose()

        asyncio.run(label_all())

    report = build_cost_report(cfg, data, all_records, pricing)
    (directory / COST_REPORT).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    complete = all((directory / f"{s}.parquet").exists() for s in SPLITS)
    write_manifest(
        directory,
        {
            "stage": "labels",
            "fingerprint": stage.fingerprint,
            "data_fingerprint": data.fingerprint,
            "teacher": teacher.model_dump(mode="json"),
            "prompt_file": str(teacher.prompt),
            "prompt_digest": template.digest,
            "simulated": teacher.backend == "simulated",
            "pricing": _pricing_dump(pricing),
            "splits": sorted({r["split"] for r in all_records}),
        },
        complete=complete,
    )
    return LabelingOutcome(directory, report)


def build_cost_report(
    cfg: ExperimentConfig,
    data: PreparedData,
    records: Sequence[dict[str, Any]],
    pricing: TokenPricing | None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "teacher": cfg.teacher.name,
        "model": cfg.teacher.model,
        "self_consistency": cfg.teacher.self_consistency,
        "pricing": _pricing_dump(pricing),
        "splits": {},
    }
    for split in SPLITS:
        split_records = [r for r in records if r["split"] == split]
        if split_records:
            report["splits"][split] = summarise_calls(split_records, len(data.split(split)))
    labeled = [s for s in SPLITS if s in report["splits"]]
    report["total"] = summarise_calls(list(records), sum(len(data.split(s)) for s in labeled))
    return report


def _write_calls(path: Path, records: Sequence[dict[str, Any]]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    tmp.replace(path)


def read_calls(directory: Path) -> list[dict[str, Any]]:
    path = directory / CALLS_FILE
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_cost_report(directory: Path) -> dict[str, Any]:
    path = directory / COST_REPORT
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return report


def load_teacher_labels(cfg: ExperimentConfig, split: str) -> pd.DataFrame:
    stage = labels_stage(cfg)
    path = stage.path / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no teacher labels for {split} at {path}; run `heartcut label`")
    return pd.read_parquet(path)


def labels_manifest(cfg: ExperimentConfig) -> dict[str, Any]:
    return read_manifest(labels_stage(cfg).path)


def soft_label_matrix(votes: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    """Vote counts (JSON strings) -> row-normalised ``(n, n_labels)`` distributions."""
    index = {label: i for i, label in enumerate(labels)}
    out = np.zeros((len(votes), len(labels)), dtype=np.float64)
    for row, raw in enumerate(votes):
        counts = json.loads(raw)
        for label, n in counts.items():
            out[row, index[label]] = n
    sums = out.sum(axis=1, keepdims=True)
    normalised: np.ndarray = np.divide(out, sums, out=np.zeros_like(out), where=sums > 0)
    return normalised


def dry_run_estimate(cfg: ExperimentConfig, splits: Sequence[str] = SPLITS) -> dict[str, Any]:
    """Rough cost range before spending anything (about 4 characters per token).

    ``low``/``high`` bracket the output side: a ~12-token JSON answer vs. every call using
    ``max_output_tokens`` (thinking models spend output tokens before answering). When
    the price table has a cache-read rate, ``low_cached_usd`` also assumes the shared
    system prompt is served from the provider's prompt cache after the first call.
    """
    data = load_prepared(cfg)
    template = PromptTemplate.from_file(cfg.teacher.prompt)
    pricing = resolve_pricing(cfg.teacher)
    system = template.render_system(data.labels, cfg.dataset.task, cfg.dataset.oos_label)
    system_tokens = len(system) / 4
    k = cfg.teacher.self_consistency
    out: dict[str, Any] = {"teacher": cfg.teacher.name, "model": cfg.teacher.model, "splits": {}}
    totals = {"low_usd": 0.0, "low_cached_usd": 0.0, "high_usd": 0.0}
    for split in splits:
        frame = data.split(split)
        user_tokens = (frame[TEXT].str.len().mean() + len(template.user)) / 4
        calls = len(frame) * k
        entry: dict[str, Any] = {
            "examples": len(frame),
            "calls": calls,
            "input_tokens_per_call": round(system_tokens + user_tokens),
        }
        if pricing is not None:
            low = estimate_cost(calls, system_tokens + user_tokens, 12, pricing)
            high = estimate_cost(
                calls, system_tokens + user_tokens, cfg.teacher.max_output_tokens, pricing
            )
            cached = low
            if pricing.cache_read is not None:
                cached = estimate_cost(calls, user_tokens, 12, pricing) + (
                    calls * system_tokens * pricing.cache_read / 1_000_000
                )
            entry.update(
                low_usd=round(low, 2), low_cached_usd=round(cached, 2), high_usd=round(high, 2)
            )
            totals["low_usd"] += low
            totals["low_cached_usd"] += cached
            totals["high_usd"] += high
        out["splits"][split] = entry
    if pricing is not None:
        out["total"] = {key: round(value, 2) for key, value in totals.items()}
        out["note"] = (
            "Assumes 4 characters per token. low: short JSON answers; high: every call uses "
            "max_output_tokens; low_cached: low with the system prompt read from the cache."
        )
    else:
        out["note"] = "No price known for this model."
    return out

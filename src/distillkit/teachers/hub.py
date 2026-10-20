"""Publish and fetch teacher labels as a Hugging Face dataset.

Publishing the labeling outputs means anyone can rerun training and the benchmark
without paying for labeling: ``fetch_labels`` drops them into the exact labels directory
the local config expects, and ``make benchmark`` then runs offline. The directory name
contains the labels fingerprint (data + teacher config + prompt), so a fetched copy is
only used when it was produced by the same setup.

Only teacher outputs are published (labels, vote counts, per-call token and cost logs,
cost report, aggregate audit). The hidden human labels never leave the machine.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from distillkit.artifacts import read_manifest
from distillkit.config import ExperimentConfig
from distillkit.teachers.labeling import labels_stage, read_cost_report

logger = logging.getLogger(__name__)

PUBLISHED_PATTERNS = ["*.parquet", "*.json", "*.jsonl", "*.md"]


def dataset_card(cfg: ExperimentConfig, manifest: dict[str, Any], cost: dict[str, Any]) -> str:
    total = cost.get("total", {})
    pricing = manifest.get("pricing") or {}
    source = cfg.dataset.description or cfg.dataset.name
    return "\n".join(
        [
            "---",
            "license: other",
            "task_categories: [text-classification]",
            "tags: [distillation, intent-classification, synthetic-labels, distillkit]",
            "---",
            "",
            f"# Teacher labels: {cfg.teacher.name} on {cfg.dataset.name}",
            "",
            f"Labels produced by `{cfg.teacher.model}` with DistillKit for the "
            f"`{cfg.name}` experiment. Source texts: {source}",
            "",
            f"- Labels fingerprint: `{manifest.get('fingerprint')}` (data "
            f"`{manifest.get('data_fingerprint')}`, prompt `{manifest.get('prompt_digest')}`)",
            f"- Self-consistency samples per message: {cfg.teacher.self_consistency}",
            f"- Calls: {total.get('calls', 'n/a')}, list cost: "
            f"${(total.get('list_cost_usd') or 0):.2f} at prices as of "
            f"{pricing.get('as_of', 'n/a')}",
            "",
            "Files per split: `id`, `text`, `teacher_label`, `confidence` (vote agreement), "
            "`votes`, `sample_labels`, `cost_usd`. `calls.jsonl` logs tokens, dollars and "
            "latency per call. The test-split labels are the teacher's predictions, used "
            "only to score the teacher; human labels are not included.",
            "",
            "**License:** the texts come from the source dataset and keep its license and "
            "attribution requirements. Confirm that redistribution is allowed before making "
            "this dataset public.",
            "",
            "Use with DistillKit: `distillkit labels pull -c <experiment.yaml> --repo-id "
            "<this repo>`, then `make benchmark`.",
            "",
        ]
    )


def publish_labels(
    cfg: ExperimentConfig, repo_id: str, *, private: bool = True, token: str | None = None
) -> str:
    """Upload the complete labels directory to ``repo_id`` (a dataset repo)."""
    from huggingface_hub import HfApi

    stage = labels_stage(cfg)
    if not stage.is_complete():
        raise FileNotFoundError(f"labels at {stage.path} are incomplete; run `distillkit label`")
    manifest = read_manifest(stage.path)
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(stage.path),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=f"{cfg.name}/{stage.path.name}",
        allow_patterns=PUBLISHED_PATTERNS,
        commit_message=f"Teacher labels {stage.path.name}",
    )
    card = dataset_card(cfg, manifest, read_cost_report(stage.path))
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update dataset card",
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def fetch_labels(
    cfg: ExperimentConfig, repo_id: str, *, revision: str | None = None, token: str | None = None
) -> Path:
    """Download the labels matching the local config's fingerprint into ``artifacts/``."""
    from huggingface_hub import snapshot_download

    stage = labels_stage(cfg)
    prefix = f"{cfg.name}/{stage.path.name}"
    local = Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[f"{prefix}/*"],
            token=token,
        )
    )
    source = local / prefix
    if not source.exists():
        available = (
            sorted(p.name for p in (local / cfg.name).glob("*"))
            if (local / cfg.name).exists()
            else []
        )
        raise FileNotFoundError(
            f"{repo_id} has no labels for {prefix}. Available for {cfg.name}: {available or 'none'}"
            ". The fingerprint differs when the dataset, teacher or prompt config differs."
        )
    shutil.copytree(source, stage.path, dirs_exist_ok=True)
    manifest = read_manifest(stage.path)
    if manifest.get("fingerprint") != stage.fingerprint:
        raise ValueError("fetched labels carry a different fingerprint than their directory")
    logger.info("fetched labels into %s", stage.path)
    return stage.path

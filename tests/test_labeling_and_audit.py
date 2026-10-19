from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from distillkit.data.prepare import prepare_data
from distillkit.eval.audit import filter_by_confidence, run_audit
from distillkit.teachers.labeling import (
    CALLS_FILE,
    COST_REPORT,
    dry_run_estimate,
    labels_manifest,
    labels_stage,
    load_teacher_labels,
    read_calls,
    run_labeling,
    soft_label_matrix,
)
from distillkit.teachers.runner import CacheMiss


def test_labeling_stage_outputs(make_experiment) -> None:  # type: ignore[no-untyped-def]
    cfg = make_experiment()
    prepare_data(cfg)
    outcome = run_labeling(cfg)
    directory = outcome.directory
    for split in ("train", "validation", "test"):
        frame = load_teacher_labels(cfg, split)
        assert {"id", "text", "teacher_label", "confidence", "votes", "sample_labels"} <= set(frame)
    train = load_teacher_labels(cfg, "train")
    assert train["sample_labels"].map(len).eq(3).all()
    assert (directory / CALLS_FILE).exists() and (directory / COST_REPORT).exists()
    report = outcome.cost_report
    assert report["total"]["calls"] == 3 * report["total"]["examples"]
    assert report["total"]["list_cost_usd"] > 0
    assert set(report["splits"]) == {"train", "validation", "test"}
    manifest = labels_manifest(cfg)
    assert manifest["complete"] and manifest["simulated"]


def test_rerun_is_free_and_offline_works(make_experiment) -> None:  # type: ignore[no-untyped-def]
    cfg = make_experiment()
    prepare_data(cfg)
    first = run_labeling(cfg)
    again = run_labeling(cfg)
    assert again.skipped
    forced = run_labeling(cfg, force=True, offline=True)
    assert forced.cost_report["total"]["paid_calls_this_run"] == 0
    assert forced.cost_report["total"]["spent_this_run_usd"] == 0
    assert forced.cost_report["total"]["list_cost_usd"] == pytest.approx(
        first.cost_report["total"]["list_cost_usd"]
    )


def test_offline_without_cache_fails(make_experiment) -> None:  # type: ignore[no-untyped-def]
    cfg = make_experiment()
    prepare_data(cfg)
    with pytest.raises(CacheMiss):
        run_labeling(cfg, offline=True)


def test_partial_split_labeling_then_completion(make_experiment) -> None:  # type: ignore[no-untyped-def]
    cfg = make_experiment()
    prepare_data(cfg)
    run_labeling(cfg, splits=["test"])
    assert not labels_manifest(cfg)["complete"]
    run_labeling(cfg)
    assert labels_manifest(cfg)["complete"]
    splits = {r["split"] for r in read_calls(labels_stage(cfg).path)}
    assert splits == {"train", "validation", "test"}


def test_prompt_change_creates_new_labels_version(make_experiment, tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = make_experiment()
    prepare_data(cfg)
    other = tmp_path / "prompt.yaml"
    other.write_text("system: |\n  Labels:\n  $labels\n  Task: $task\nuser: $text\n")
    changed = cfg.model_copy(update={"teacher": cfg.teacher.model_copy(update={"prompt": other})})
    assert labels_stage(cfg).path != labels_stage(changed).path


def test_dry_run_estimate(make_experiment) -> None:  # type: ignore[no-untyped-def]
    cfg = make_experiment()
    prepare_data(cfg)
    estimate = dry_run_estimate(cfg)
    assert estimate["total"]["low_usd"] <= estimate["total"]["high_usd"]
    assert estimate["splits"]["train"]["calls"] == 3 * 160


def test_soft_label_matrix() -> None:
    votes = [json.dumps({"b": 2, "a": 1}), json.dumps({})]
    matrix = soft_label_matrix(votes, ["a", "b", "c"])
    np.testing.assert_allclose(matrix, [[1 / 3, 2 / 3, 0], [0, 0, 0]])


def test_filter_by_confidence() -> None:
    frame = pd.DataFrame(
        {"id": ["1", "2", "3"], "teacher_label": ["a", None, "b"], "confidence": [1.0, 0.0, 0.4]}
    )
    kept, dropped = filter_by_confidence(frame, None)
    assert kept["id"].tolist() == ["1", "3"] and dropped == 1
    kept, dropped = filter_by_confidence(frame, 0.5)
    assert kept["id"].tolist() == ["1"] and dropped == 2


def test_audit_reports_teacher_quality(make_experiment) -> None:  # type: ignore[no-untyped-def]
    cfg = make_experiment(label_filter={"min_confidence": 0.6})
    prepare_data(cfg)
    run_labeling(cfg)
    report = run_audit(cfg)
    pooled = report["pooled"]
    # The simulated teacher is right ~85% of the time per sample; voting helps.
    assert 0.8 < pooled["accuracy"] <= 1.0
    assert pooled["filter"]["kept"] + pooled["filter"]["dropped"] == pooled["examples"]
    assert pooled["filter"]["kept_accuracy"] >= pooled["accuracy"]
    markdown = (cfg.experiment_results_dir / "audit.md").read_text()
    assert "SIMULATED TEACHER" in markdown
    assert "Low-confidence filter" in markdown

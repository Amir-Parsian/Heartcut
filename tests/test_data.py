from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heartcut.config import DatasetConfig, DatasetSource
from heartcut.data import loaders
from heartcut.data.prepare import (
    build_splits,
    load_hidden_gold,
    load_prepared,
    load_raw,
    prepare_data,
)
from heartcut.data.schema import validate_frame
from heartcut.data.splits import nested_prefix, stratified_sample, stratified_split


def _frame(n_per_class: dict[str, int]) -> pd.DataFrame:
    rows = [(f"{label} {i}", label) for label, n in n_per_class.items() for i in range(n)]
    return pd.DataFrame(rows, columns=["text", "label"])


def test_stratified_split_is_deterministic_and_proportional() -> None:
    df = _frame({"a": 50, "b": 30, "c": 1})
    rest, held = stratified_split(df, 0.2, seed=1)
    _, held2 = stratified_split(df, 0.2, seed=1)
    pd.testing.assert_frame_equal(held, held2)
    assert held["label"].value_counts().to_dict() == {"a": 10, "b": 6}
    assert len(rest) + len(held) == len(df)
    assert "c" in set(rest["label"])  # singleton classes are never held out
    _, held3 = stratified_split(df, 0.2, seed=2)
    assert set(held["text"]) != set(held3["text"])


def test_stratified_sample_returns_exactly_n() -> None:
    df = _frame({"a": 37, "b": 11, "c": 5})
    for n in (1, 7, 20, 52):
        assert len(stratified_sample(df, n, seed=0)) == n
    assert len(stratified_sample(df, 500, seed=0)) == len(df)


def test_nested_prefix_gives_nested_subsets() -> None:
    order = nested_prefix(100, seed=4)
    assert sorted(order.tolist()) == list(range(100))
    assert set(order[:10]) <= set(order[:50])
    assert np.array_equal(order, nested_prefix(100, seed=4))


def test_jsonl_and_csv_loaders(tmp_path: Path) -> None:
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text('{"t": "hello", "y": "greet"}\n{"t": " bye ", "y": "leave"}\n')
    out = loaders.load_jsonl({"train": str(jsonl)}, text_column="t", label_column="y")
    assert out["train"].to_dict("records") == [
        {"text": "hello", "label": "greet"},
        {"text": "bye", "label": "leave"},
    ]
    csv = tmp_path / "dev.csv"
    pd.DataFrame({"msg": ["a", "b"], "cat": [1, 2]}).to_csv(csv, index=False)
    out = loaders.load_csv({"dev": str(csv)}, text_column="msg", label_column="cat")
    assert list(out) == ["validation"]  # split aliases are normalised
    assert out["validation"]["label"].tolist() == ["1", "2"]


def test_clinc_json_loader_merges_out_of_scope(tmp_path: Path) -> None:
    path = tmp_path / "clinc.json"
    raw = {
        "train": [["play a song", "play_music"]],
        "val": [["what time is it", "time"]],
        "test": [["set an alarm", "alarm"]],
        "oos_train": [["how do I bake bread", "oos"]],
        "oos_val": [],
        "oos_test": [["tell me about whales", "oos"]],
    }
    path.write_text(json.dumps(raw))
    out = loaders.load_clinc_json(str(path))
    assert out["train"]["label"].tolist() == ["play_music", "oos"]
    assert out["test"]["label"].tolist() == ["alarm", "oos"]
    no_oos = loaders.load_clinc_json(str(path), include_oos=False)
    assert "oos" not in set(no_oos["train"]["label"])


def test_resolve_loader_by_name_and_import_path() -> None:
    assert loaders.resolve_loader("jsonl") is loaders.load_jsonl
    assert loaders.resolve_loader("heartcut.data.loaders:load_csv") is loaders.load_csv
    with pytest.raises(ValueError, match="unknown loader"):
        loaders.resolve_loader("does-not-exist")
    with pytest.raises(ValueError, match="callable"):
        loaders.resolve_loader("heartcut.data.loaders:_REGISTRY")


def test_register_loader_rejects_name_clash() -> None:
    with pytest.raises(ValueError, match="already registered"):
        loaders.register_loader("csv")(lambda: {})


def test_fetch_local_and_missing(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("x")
    assert loaders.fetch(path) == path
    with pytest.raises(FileNotFoundError):
        loaders.fetch(tmp_path / "missing.txt")


def test_fetch_downloads_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class FakeResponse:
        def __init__(self) -> None:
            self.data = [b"payload"]

        def read(self, *_: object) -> bytes:
            return self.data.pop() if self.data else b""

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_: object) -> None:
            pass

    def fake_urlopen(url: str, timeout: int) -> FakeResponse:
        calls.append(url)
        return FakeResponse()

    monkeypatch.setattr(loaders.urllib.request, "urlopen", fake_urlopen)
    first = loaders.fetch("https://example.org/data/train.csv")
    second = loaders.fetch("https://example.org/data/train.csv")
    assert first == second
    assert first.read_bytes() == b"payload"
    assert calls == ["https://example.org/data/train.csv"]


def test_load_raw_falls_back_to_next_source(dataset_files: dict[str, Path]) -> None:
    cfg = DatasetConfig(
        name="d",
        sources=[
            DatasetSource(loader="jsonl", args={"files": {"train": "/missing.jsonl"}}),
            DatasetSource(
                loader="jsonl",
                args={
                    "files": {k: str(v) for k, v in dataset_files.items()},
                    "label_column": "intent",
                },
            ),
        ],
    )
    raw, source = load_raw(cfg)
    assert source is cfg.sources[1]
    assert set(raw) == {"train", "test"}


def test_load_raw_reports_every_failure() -> None:
    cfg = DatasetConfig(
        name="d", sources=[DatasetSource(loader="jsonl", args={"files": {"train": "/nope"}})]
    )
    with pytest.raises(RuntimeError, match="any source"):
        load_raw(cfg)


def test_build_splits_carves_validation_and_caps(dataset_files: dict[str, Path]) -> None:
    cfg = DatasetConfig(
        name="d",
        sources=[
            DatasetSource(
                loader="jsonl",
                args={
                    "files": {k: str(v) for k, v in dataset_files.items()},
                    "label_column": "intent",
                },
            )
        ],
        validation_fraction=0.2,
        max_examples={"test": 16},
    )
    raw, _ = load_raw(cfg)
    splits = build_splits(raw, cfg)
    assert len(splits["train"]) == 160 and len(splits["validation"]) == 40
    assert len(splits["test"]) == 16
    assert splits["train"]["id"].iloc[0] == "train-000000"
    with pytest.raises(ValueError, match="at least"):
        build_splits({"train": raw["train"]}, cfg)


def test_prepare_data_hides_training_labels(make_experiment) -> None:  # type: ignore[no-untyped-def]
    cfg = make_experiment()
    prepared = prepare_data(cfg)
    assert prepared.train["label"].isna().all()
    assert prepared.validation["label"].isna().all()
    assert prepared.test["label"].notna().all()
    assert len(prepared.labels) == 8
    gold = load_hidden_gold(cfg)
    assert set(gold["split"]) == {"train", "validation"}
    assert len(gold) == len(prepared.train) + len(prepared.validation)
    # A second call reuses the stage output.
    again = prepare_data(cfg)
    assert again.directory == prepared.directory
    pd.testing.assert_frame_equal(again.test, load_prepared(cfg).test)


def test_dataset_changes_produce_a_new_version(make_experiment) -> None:  # type: ignore[no-untyped-def]
    a = prepare_data(make_experiment())
    cfg = make_experiment()
    b = prepare_data(cfg.model_copy(update={"dataset": cfg.dataset.model_copy(update={"seed": 9})}))
    assert a.directory != b.directory


def test_load_prepared_requires_the_stage(make_experiment) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(FileNotFoundError, match="heartcut data"):
        load_prepared(make_experiment())


def test_validate_frame_catches_schema_errors() -> None:
    good = pd.DataFrame({"id": ["a"], "text": ["x"], "label": ["y"]})
    validate_frame(good, split="test", labeled=True)
    with pytest.raises(ValueError, match="hidden"):
        validate_frame(good, split="train", labeled=False)
    with pytest.raises(ValueError, match="missing columns"):
        validate_frame(good.drop(columns="label"), split="test", labeled=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_frame(pd.concat([good, good]), split="test", labeled=True)
    with pytest.raises(ValueError, match="empty"):
        validate_frame(good.assign(text=[" "]), split="test", labeled=True)
    with pytest.raises(ValueError, match="missing labels"):
        validate_frame(good.assign(label=[None]), split="test", labeled=True)

"""Deterministic splitting and sub-sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from heartcut.data.schema import LABEL


def _holdout_positions(labels: pd.Series, fraction: float, seed: int) -> np.ndarray:
    """Positions of ``fraction`` of each class, chosen with a seeded generator."""
    rng = np.random.default_rng(seed)
    positions = np.arange(len(labels))
    chosen: list[np.ndarray] = []
    for label in sorted(labels.unique()):
        members = positions[(labels == label).to_numpy()]
        n_hold = min(round(len(members) * fraction), len(members) - 1)
        if n_hold <= 0:
            continue
        chosen.append(rng.choice(members, size=n_hold, replace=False))
    if not chosen:
        return np.array([], dtype=int)
    return np.sort(np.concatenate(chosen))


def stratified_split(
    df: pd.DataFrame, fraction: float, seed: int, label_column: str = LABEL
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into ``(rest, held_out)`` with ``fraction`` of each class held out.

    Classes with a single example stay in ``rest``. The result depends only on the data
    and the seed, never on global random state.
    """
    held = _holdout_positions(df[label_column].reset_index(drop=True), fraction, seed)
    mask = np.zeros(len(df), dtype=bool)
    mask[held] = True
    return df.iloc[~mask].reset_index(drop=True), df.iloc[mask].reset_index(drop=True)


def stratified_sample(
    df: pd.DataFrame, n: int, seed: int, label_column: str = LABEL
) -> pd.DataFrame:
    """Return exactly ``n`` rows (or all rows), keeping class proportions close."""
    if n >= len(df):
        return df.reset_index(drop=True)
    labels = df[label_column].reset_index(drop=True)
    positions = _holdout_positions(labels, n / len(df), seed)
    rng = np.random.default_rng(seed)
    if len(positions) > n:
        positions = np.sort(rng.choice(positions, size=n, replace=False))
    elif len(positions) < n:
        rest = np.setdiff1d(np.arange(len(df)), positions)
        extra = rng.choice(rest, size=n - len(positions), replace=False)
        positions = np.sort(np.concatenate([positions, extra]))
    return df.iloc[positions].reset_index(drop=True)


def nested_prefix(n_total: int, seed: int) -> np.ndarray:
    """A fixed permutation of ``range(n_total)``.

    Taking the first ``k`` entries gives nested subsets (the 500-example subset is
    contained in the 2,000-example one, and so on). That keeps learning curves smooth and
    makes the budget experiment a fair "what if we had labeled less" comparison.
    """
    return np.random.default_rng(seed).permutation(n_total)

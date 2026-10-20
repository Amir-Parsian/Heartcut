"""On-disk cache of teacher completions.

The key is a SHA-256 hash of everything that can change the answer: backend, model,
endpoint, full rendered prompt, output schema, sampling settings and the sample index
(so self-consistency samples are cached separately). Re-running a labeling job, or
resuming one that was interrupted, therefore costs nothing for messages already seen.
The cached entry keeps the original token usage and latency, so cost and latency reports
stay truthful when they are regenerated from the cache.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from heartcut.artifacts import canonical_json, utc_now
from heartcut.teachers.base import Completion


def cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class CompletionCache:
    """A tiny SQLite key-value store (one file, safe to copy or publish)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS completions ("
            " key TEXT PRIMARY KEY, completion TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> Completion | None:
        row = self._conn.execute(
            "SELECT completion FROM completions WHERE key = ?", (key,)
        ).fetchone()
        return Completion.from_dict(json.loads(row[0])) if row else None

    def put(self, key: str, completion: Completion) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO completions (key, completion, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(completion.to_dict()), utc_now()),
        )
        self._conn.commit()

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM completions").fetchone()[0])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> CompletionCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

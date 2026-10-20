"""Versioned stage outputs.

Every stage writes into ``artifacts/<experiment>/<stage>/<name>-<fingerprint>/``. The
fingerprint is a hash of everything that determines the output: the stage's own config
plus the fingerprints of its inputs. Changing a prompt therefore produces a new labels
directory (the old one is kept), and every downstream stage picks up the change because
its own fingerprint changes too. A ``manifest.json`` in each directory records the inputs,
the config, library versions and timestamps, so any number can be traced back to the
exact data and settings that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

MANIFEST = "manifest.json"
FINGERPRINT_LENGTH = 10

_TRACKED_PACKAGES = (
    "heartcut",
    "numpy",
    "pandas",
    "scikit-learn",
    "torch",
    "transformers",
    "sentence-transformers",
    "onnxruntime",
    "anthropic",
    "openai",
)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint(*parts: Any) -> str:
    """Short, stable hash of JSON-serialisable parts."""
    digest = hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).hexdigest()
    return digest[:FINGERPRINT_LENGTH]


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:FINGERPRINT_LENGTH]


def cache_root() -> Path:
    """Where downloads and the teacher cache live (``$HEARTCUT_CACHE_DIR``)."""
    return Path(os.environ.get("HEARTCUT_CACHE_DIR", ".cache/heartcut"))


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class StageDir:
    """Location of one stage output."""

    path: Path
    fingerprint: str

    @property
    def manifest_path(self) -> Path:
        return self.path / MANIFEST

    def is_complete(self) -> bool:
        if not self.manifest_path.exists():
            return False
        return bool(read_manifest(self.path).get("complete", False))

    def ensure(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path


def stage_dir(root: Path, stage: str, name: str, fp: str) -> StageDir:
    return StageDir(path=root / stage / f"{name}-{fp}", fingerprint=fp)


def write_manifest(directory: Path, payload: dict[str, Any], *, complete: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        **payload,
        "complete": complete,
        "created_at": utc_now(),
        "environment": environment_info(),
    }
    path = directory / MANIFEST
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def read_manifest(directory: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    return data


def package_versions(packages: tuple[str, ...] = _TRACKED_PACKAGES) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def cpu_model() -> str:
    """Human-readable CPU name (best effort, Linux and macOS)."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def environment_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu_model(),
        "cpu_count": os.cpu_count(),
        "git_revision": git_revision(),
        "packages": package_versions(),
    }

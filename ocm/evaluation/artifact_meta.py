"""Provenance stamps for tracked evaluation artifacts.

A result JSON is only citable if a reader can tell *what produced it*. The
durable-state runners are replay scorers: they consume gold facts and a corpus,
so two runs that differ in either input produce different numbers from identical
code, and nothing in the output distinguishes them. That is not hypothetical --
``longmemeval_oracle.json`` and ``longmemeval_s.json`` share question ids, so
pointing a run at the wrong one raises no error at all.

This module supplies the fields that close that gap. It is deliberately
best-effort: a missing ``git`` or a tarball export yields ``None`` rather than an
exception, because refusing to score is worse than scoring with an unknown
revision recorded honestly as unknown.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

__all__ = ["git_revision", "tracked_tree_dirty", "code_provenance", "file_provenance"]


def _git(*args: str) -> str | None:
    """Best-effort git query; ``None`` outside a repo or without git installed."""
    try:
        result = subprocess.run(
            ("git", *args), capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_revision() -> str | None:
    """Current ``HEAD`` sha, or ``None`` if unavailable."""
    return _git("rev-parse", "HEAD")


def tracked_tree_dirty() -> bool | None:
    """Whether *tracked* files differ from ``HEAD``.

    Untracked files are excluded deliberately. Scratch runners and downloaded
    PDFs sit in the tree permanently and say nothing about whether the scored
    code matches the recorded revision. This also matches the extraction caches,
    whose identity is derived from ``git diff`` and therefore reads clean while
    untracked files are present.
    """
    status = _git("status", "--porcelain", "--untracked-files=no")
    return bool(status) if status is not None else None


def code_provenance() -> dict[str, Any]:
    """Revision and cleanliness of the code that produced a result."""
    return {"code_revision": git_revision(), "code_dirty": tracked_tree_dirty()}


def file_provenance(path: Path, *, prefix: str) -> dict[str, Any]:
    """Name, size and SHA-256 of an input file, streamed so large corpora are fine.

    ``prefix`` namespaces the keys (``dataset``, ``annotations``, ...) so several
    inputs can be stamped into one ``_meta`` without collision.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return {
        prefix: path.name,
        f"{prefix}_path": str(path),
        f"{prefix}_sha256": digest.hexdigest(),
        f"{prefix}_bytes": path.stat().st_size,
    }

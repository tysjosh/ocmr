"""Memoizing wrapper around a W1 extractor (evaluation speedup).

The evaluation harness re-ingests the **same** benchmark sessions for every arm:
within one seed the 8 baseline/ablation arms, the 5 tau-sweep rows, and the
stress arms all extract identical session text. With a deterministic
(greedy-decoded) LLM extractor that is ~15x redundant work — the slow part of a
full run.

``CachingExtractor`` wraps any :class:`~ocm.extraction.base.Extractor` and
memoizes ``extract`` by ``(source_ref, text)``. Because greedy decoding is
deterministic, the cached :class:`ExtractionResult` is exactly what a re-run
would produce, so caching changes timing only, never results. A deep copy is
returned on every hit so downstream pipeline stages can never mutate a shared
cached object.

It is purely an evaluation accelerator: pass it *instead of* the raw extractor
into ``run_full_suite`` (and the notebook's Section 7b). Optional on-disk
persistence (``cache_path``) lets the cache survive a Colab restart so a resumed
run skips re-extraction of already-seen text.

This module never imports torch/transformers, so it stays import-safe offline.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional

from ocm.extraction.base import ExtractionError
from ocm.memory.contracts import ExtractionResult


class CachingExtractor:
    """Deterministic memoization layer over a base W1 extractor."""

    def __init__(
        self,
        base: Any,
        cache_path: Optional[str] = None,
        autosave_every: int = 50,
        *,
        cache_failures: bool = False,
    ) -> None:
        """Wrap ``base``.

        Args:
            base: The underlying extractor (e.g. a ``TransformersExtractor``).
                Must expose ``extract(text, source_ref) -> ExtractionResult``.
            cache_path: Optional JSON file to load the cache from on construction
                and persist it to (e.g. a Google Drive path) so the memo survives
                process restarts. When ``None`` the cache is in-memory only.
            autosave_every: Persist to ``cache_path`` after this many new misses
                (ignored when ``cache_path`` is ``None``); bounds disk I/O.
            cache_failures: Memoize :class:`ExtractionError` outcomes as well as
                successful ones, **in memory only**. Without this, a failing
                input is re-extracted by every arm, because the raise happens
                before the store. That costs one generation per arm, and it also
                leaves cross-arm consistency resting on the assumption that
                greedy decoding is bit-identical on repeat calls. Memoizing the
                failure makes every arm in one process observe exactly the same
                extraction outcome by construction. Failures are deliberately
                never written to ``cache_path``, so a transient fault cannot
                poison a cache that outlives the process.
        """
        self._base = base
        self.version = getattr(base, "version", "caching-extractor")
        self._cache: dict[str, ExtractionResult] = {}
        self._failures: dict[str, str] = {}
        self._cache_failures = bool(cache_failures)
        self._cache_path = cache_path
        self._autosave_every = max(1, int(autosave_every))
        self.hits = 0
        self.misses = 0
        self.failure_hits = 0
        self._unsaved = 0
        if cache_path:
            self._load()

    # -- key ---------------------------------------------------------------
    @staticmethod
    def _key(text: str, source_ref: str) -> str:
        """Stable cache key over ``(source_ref, text)``.

        ``source_ref`` is included because the extraction prompt embeds it, so
        two calls with the same text but different source refs are distinct
        inputs. The harness reuses the same source_ref per example/session
        across arms, so identical work still collapses to one call.
        """
        h = hashlib.sha256()
        h.update((source_ref or "").encode("utf-8"))
        h.update(b"\x00")
        h.update((text or "").encode("utf-8"))
        return h.hexdigest()

    # -- public API --------------------------------------------------------
    def extract(self, text: str, source_ref: str) -> ExtractionResult:
        """Return the (memoized) extraction for ``text`` / ``source_ref``."""
        key = self._key(text, source_ref)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached.model_copy(deep=True)
        if self._cache_failures and key in self._failures:
            self.failure_hits += 1
            raise ExtractionError(self._failures[key])
        self.misses += 1
        try:
            result = self._base.extract(text, source_ref)
        except ExtractionError as exc:
            # Record the verdict so sibling arms see the identical outcome without
            # paying for another generation. Never persisted to disk.
            if self._cache_failures:
                self._failures[key] = str(exc)
            raise
        self._cache[key] = result
        self._unsaved += 1
        if self._cache_path and self._unsaved >= self._autosave_every:
            self.save()
        return result.model_copy(deep=True)

    @property
    def stats(self) -> dict[str, int]:
        """Cache hit/miss/size counters (useful to print after a run)."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self._cache),
            "failure_hits": self.failure_hits,
            "distinct_failures": len(self._failures),
        }

    # -- persistence -------------------------------------------------------
    def save(self) -> None:
        """Persist the cache to ``cache_path`` atomically (no-op without a path)."""
        if not self._cache_path:
            return
        os.makedirs(os.path.dirname(self._cache_path) or ".", exist_ok=True)
        payload = {key: result.model_dump(mode="json") for key, result in self._cache.items()}
        tmp = self._cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, self._cache_path)
        self._unsaved = 0

    def _load(self) -> None:
        """Load a previously persisted cache (ignored when absent/corrupt)."""
        if not self._cache_path or not os.path.exists(self._cache_path):
            return
        try:
            with open(self._cache_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self._cache = {
                key: ExtractionResult.model_validate(value) for key, value in payload.items()
            }
        except Exception:  # pragma: no cover - corrupt/partial file: start fresh
            self._cache = {}

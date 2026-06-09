"""Identifier generation for OCM entities and assertions.

`IdGenerator` has two modes, selected by ``settings.deterministic_test_mode``:

* **Random (default).** ``f"{prefix}_{uuid4().hex}"`` — globally unique,
  suitable for production/demo.
* **Deterministic test mode (Req 27.5).** IDs are derived from
  ``entity_type + normalized_name + source_ref`` plus a seeded per-run counter
  for tie-breaking, hashed to a stable suffix. Identical input across runs
  yields identical IDs, which makes benchmarks, ablations, and property tests
  reproducible.

The seeded counter is reset per ``IdGenerator`` construction (i.e. per run /
per process initialization) so a fresh run over identical inputs reproduces the
same ID sequence (Req 27.5).
"""

from __future__ import annotations

import hashlib
import itertools
from typing import Iterator, Optional
from uuid import uuid4


def _prefix(entity_type: str) -> str:
    """Stable, lowercase 3-char prefix derived from an entity/record type."""
    return entity_type[:3].lower()


class IdGenerator:
    """Generates entity and assertion IDs in random or deterministic mode."""

    def __init__(self, deterministic: bool, seed: int = 0) -> None:
        self.deterministic = deterministic
        self.seed = seed
        # A per-run counter is only needed in deterministic mode; in random
        # mode uniqueness comes from uuid4.
        self._counter: Optional[Iterator[int]] = (
            itertools.count(seed) if deterministic else None
        )

    # -- internal helpers ---------------------------------------------------
    def _deterministic_suffix(self, basis: str) -> str:
        """Stable 12-hex-char digest of ``basis``."""
        return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]

    def _next(self) -> int:
        assert self._counter is not None  # guarded by self.deterministic
        return next(self._counter)

    # -- public API ---------------------------------------------------------
    def entity_id(self, entity_type: str, normalized_name: str, source_ref: str) -> str:
        """Return an ID for an entity.

        In deterministic mode the ID is derived from
        ``entity_type|normalized_name|source_ref|counter`` (Req 27.5).
        """
        prefix = _prefix(entity_type)
        if self.deterministic:
            n = self._next()
            basis = f"{entity_type}|{normalized_name}|{source_ref}|{n}"
            return f"{prefix}_{self._deterministic_suffix(basis)}"
        return f"{prefix}_{uuid4().hex}"

    def assertion_id(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        source_ref: str,
    ) -> str:
        """Return an ID for an assertion.

        Deterministic mode derives the ID from the assertion's identity
        (subject/predicate/object/source_ref) plus the seeded counter so
        repeated runs over the same input batch reproduce the same sequence.
        """
        prefix = "asr"
        if self.deterministic:
            n = self._next()
            basis = f"{subject_id}|{predicate}|{object_id}|{source_ref}|{n}"
            return f"{prefix}_{self._deterministic_suffix(basis)}"
        return f"{prefix}_{uuid4().hex}"

    def generic_id(self, prefix: str, *parts: str) -> str:
        """Return an ID for any other record type (claim, document, etc.).

        ``parts`` form the deterministic basis; in random mode they are ignored
        in favor of a uuid4 suffix.
        """
        norm_prefix = _prefix(prefix) if len(prefix) > 3 else prefix.lower()
        if self.deterministic:
            n = self._next()
            basis = "|".join([*parts, str(n)])
            return f"{norm_prefix}_{self._deterministic_suffix(basis)}"
        return f"{norm_prefix}_{uuid4().hex}"

    def stable_id(self, prefix: str, *parts: str) -> str:
        """Return a content-addressed ID: a pure hash of ``parts``.

        Unlike :meth:`generic_id` / :meth:`entity_id`, this never mixes in the
        per-run counter or a uuid4 — identical ``parts`` always yield the same
        ID, in **both** deterministic and random mode. It therefore doubles as a
        lightweight resolution key: e.g. a Decision keyed by its normalized
        *topic* resolves to the same entity across sessions, so a draft -> final
        status change reconciles against the same subject.
        """
        norm_prefix = _prefix(prefix) if len(prefix) > 3 else prefix.lower()
        basis = "|".join(parts)
        return f"{norm_prefix}_{self._deterministic_suffix(basis)}"

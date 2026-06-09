"""Constraint-aware retrieval ``Reranker`` (R3) (Req 17.1, 17.2, 17.3).

The Reranker is the third stage of the retrieval pipeline (R0→R1→R2→**R3**→R4).
It merges the structural hits from the Symbolic Retriever (R1) with the dense
hits from the Semantic Retriever (R2) into a single, deduplicated candidate set
and assigns each candidate one scalar score (Req 17.1)::

    score = alpha * semantic_similarity
          + beta  * graph_relevance
          + gamma * confidence
          + delta * provenance_quality
          + eta   * recency
          - lambda * contradiction_penalty

The weights come from :class:`ocm.core.config.RerankWeights`; when no weights are
supplied the defaults (``alpha=0.40, beta=0.25, gamma=0.15, delta=0.10,
eta=0.05, lambda=0.30``) are used (Req 17.2).

Signal sourcing
---------------
Each signal is resolved with a clear precedence so callers stay flexible:

1. An explicit per-item override in the ``metadata`` map (keyed by ``memory_id``)
   wins — this is how the pipeline injects provenance-derived / graph-derived
   signals it computes elsewhere, and how tests pin exact values.
2. Otherwise the signal is derived from the hit itself (a semantic hit carries a
   cosine ``similarity``; an assertion hit may carry ``confidence`` /
   ``created_at`` / ``source_ref`` / ``extractor_version``).
3. Otherwise a safe default is used (``0.0`` for most signals).

Special rules baked in:

- **Symbolic exact match ⇒ ``semantic_similarity = 1.0``** (Req 15.4) and a high
  ``graph_relevance`` (``1.0``), because the item answers the structural query
  directly. A non-exact symbolic hit still gets a strong (decaying) graph
  signal; a semantic-only hit gets ``graph_relevance = 0.0`` unless overridden.
- **Contradiction penalty.** An item is treated as contradicted (penalty
  ``1.0``) when its ``memory_id`` is in ``contradicted_ids``, when it carries a
  truthy ``contradicted`` flag, or when its ``status`` is ``quarantined``. A
  graded penalty can be supplied via ``metadata[...]["contradiction_penalty"]``.

Contradiction monotonicity (Req 17.3)
-------------------------------------
Because ``lambda > 0`` and a contradicted item has ``contradiction_penalty > 0``
while an otherwise-identical non-contradicted item has
``contradiction_penalty = 0``, the contradicted item always scores strictly
lower. This is asserted as a correctness property (Property 9, task 13.5).

The hit inputs are duck-typed (the Symbolic/Semantic retrievers, R1/R2, are
built alongside this module): any object or mapping exposing the relevant
attributes works, so the reranker has no import-time dependency on those
modules.

Requirements: 17.1, 17.2, 17.3 (and 15.4 for the exact-match rule).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional

from pydantic import BaseModel, Field

from ocm.core.config import RerankWeights

#: Memory-type tag for an assertion-shaped symbolic hit (matches the vector index).
MEMORY_TYPE_ASSERTION = "assertion"

#: Status value that, on its own, marks an item as contradicted for scoring.
STATUS_QUARANTINED = "quarantined"

#: Default ``graph_relevance`` for an exact symbolic match (Req 15.4).
EXACT_MATCH_GRAPH_RELEVANCE = 1.0

#: Default ``graph_relevance`` for a non-exact symbolic hit (graph-connected but
#: not a direct structural answer — decays 1–2 hops away).
SYMBOLIC_GRAPH_RELEVANCE = 0.5

#: Default ``graph_relevance`` for a semantic-only hit (not known to be
#: graph-connected for the query target).
SEMANTIC_GRAPH_RELEVANCE = 0.0

#: Default recency window: items older than this score ~0 on recency.
DEFAULT_RECENCY_WINDOW_DAYS = 30.0


class RankedItem(BaseModel):
    """A single scored candidate emitted by :meth:`Reranker.rerank`.

    ``score`` is the weighted sum above; ``components`` records the **raw**
    (pre-weight) signal values that produced it (``semantic_similarity``,
    ``graph_relevance``, ``confidence``, ``provenance_quality``, ``recency``,
    ``contradiction_penalty``) so downstream stages and the monotonicity
    property test can inspect exactly what drove the ranking.

    The remaining fields carry the metadata the Evidence Packager (R4, task
    13.6) needs to assemble an ``EvidencePackage``: ``confidence`` and
    ``memory_id`` for ``supporting_assertions``; ``source_ref`` for
    ``supporting_sources``; ``contradicted`` to surface conflicts; ``status`` to
    keep accepted-vs-quarantined provenance.
    """

    memory_id: str
    memory_type: str
    status: str
    score: float
    components: dict[str, float] = Field(default_factory=dict)

    # Optional metadata for the evidence packager / debugging.
    confidence: Optional[float] = None
    source_ref: Optional[str] = None
    created_at: Optional[datetime] = None
    predicate: Optional[str] = None
    subject_id: Optional[str] = None
    object_id: Optional[str] = None
    text: Optional[str] = None
    exact_match: bool = False
    contradicted: bool = False

    model_config = {"arbitrary_types_allowed": True}


class Reranker:
    """Computes constraint-aware scores and orders retrieval candidates (R3)."""

    def __init__(
        self,
        weights: RerankWeights | None = None,
        *,
        recency_window_days: float = DEFAULT_RECENCY_WINDOW_DAYS,
    ) -> None:
        """Create a reranker.

        Args:
            weights: Default weights for :meth:`rerank`. When ``None`` the
                :class:`RerankWeights` defaults are used (Req 17.2). A per-call
                ``weights`` argument overrides this.
            recency_window_days: Window used to normalize ``created_at`` into a
                ``recency`` signal in ``[0, 1]`` when no explicit recency is
                supplied (newer ⇒ closer to ``1.0``).
        """
        self.weights = weights or RerankWeights()
        self.recency_window_days = recency_window_days

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def rerank(
        self,
        symbolic: Iterable[Any] | None = None,
        semantic: Iterable[Any] | None = None,
        weights: RerankWeights | None = None,
        *,
        contradicted_ids: Iterable[str] | None = None,
        metadata: Mapping[str, Mapping[str, Any]] | None = None,
        now: datetime | None = None,
    ) -> list[RankedItem]:
        """Merge symbolic + semantic hits and return them ordered by score.

        Args:
            symbolic: Hits from the Symbolic Retriever (R1). Each should expose a
                memory id (``memory_id`` / ``assertion_id`` / ``id``) and an
                ``exact_match`` flag; matched-entity / ``confidence`` /
                ``created_at`` / ``source_ref`` attributes are used when present.
            semantic: Hits from the Semantic Retriever (R2). Each should expose a
                memory id, ``memory_type``, ``status``, and a cosine
                ``similarity`` in ``[0, 1]``.
            weights: Per-call weights override. When ``None`` the instance
                weights (or :class:`RerankWeights` defaults) are used (Req 17.2).
            contradicted_ids: Memory ids known to be contradicted (e.g. from W7
                / ``CONTRADICTS`` edges). These receive a contradiction penalty.
            metadata: Optional per-``memory_id`` map of signal overrides. Any of
                ``semantic_similarity``, ``graph_relevance``, ``confidence``,
                ``provenance_quality``, ``recency``, ``contradiction_penalty``,
                ``created_at``, ``source_ref`` may be supplied to override the
                values derived from the hits.
            now: Reference time for recency normalization (defaults to UTC now).

        Returns:
            ``RankedItem`` list sorted by descending ``score`` (ties broken by
            ``memory_id`` for a deterministic order).
        """
        active_weights = weights or self.weights
        contradicted = set(contradicted_ids or ())
        meta = metadata or {}
        reference_now = now or datetime.now(timezone.utc)

        merged = self._merge(symbolic, semantic)

        ranked: list[RankedItem] = []
        for memory_id, hit_info in merged.items():
            item_meta = meta.get(memory_id, {})
            components = self._signals(
                hit_info,
                item_meta,
                contradicted=memory_id in contradicted,
                reference_now=reference_now,
            )
            score = self._score(components, active_weights)
            ranked.append(
                RankedItem(
                    memory_id=memory_id,
                    memory_type=hit_info["memory_type"],
                    status=hit_info["status"],
                    score=score,
                    components=components,
                    confidence=hit_info.get("confidence"),
                    source_ref=hit_info.get("source_ref"),
                    created_at=hit_info.get("created_at"),
                    predicate=hit_info.get("predicate"),
                    subject_id=hit_info.get("subject_id"),
                    object_id=hit_info.get("object_id"),
                    text=hit_info.get("text"),
                    exact_match=hit_info.get("exact_match", False),
                    contradicted=components["contradiction_penalty"] > 0.0,
                )
            )

        # Descending score; deterministic tie-break by memory_id.
        ranked.sort(key=lambda item: (-item.score, item.memory_id))
        return ranked

    @staticmethod
    def score_components(components: Mapping[str, float], weights: RerankWeights) -> float:
        """Apply ``weights`` to a raw-signal ``components`` mapping (Req 17.1).

        Exposed as a small pure helper so the monotonicity property test can
        score two component sets that differ only in ``contradiction_penalty``.
        """
        return Reranker._score(components, weights)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _score(components: Mapping[str, float], weights: RerankWeights) -> float:
        """Weighted sum of the raw signals (Req 17.1)."""
        return (
            weights.alpha * components["semantic_similarity"]
            + weights.beta * components["graph_relevance"]
            + weights.gamma * components["confidence"]
            + weights.delta * components["provenance_quality"]
            + weights.eta * components["recency"]
            - weights.lambda_ * components["contradiction_penalty"]
        )

    def _merge(
        self,
        symbolic: Iterable[Any] | None,
        semantic: Iterable[Any] | None,
    ) -> dict[str, dict[str, Any]]:
        """Merge symbolic + semantic hits into one record per ``memory_id``.

        Symbolic hits are folded first so a later semantic hit for the same id
        fills in similarity/text without dropping the symbolic ``exact_match``
        flag. An exact symbolic match forces ``semantic_similarity = 1.0`` (Req
        15.4) regardless of any semantic cosine score.
        """
        merged: dict[str, dict[str, Any]] = {}

        for hit in symbolic or ():
            memory_id = _memory_id(hit)
            if memory_id is None:
                continue
            record = merged.setdefault(memory_id, {})
            record["is_symbolic"] = True
            record["exact_match"] = bool(_attr(hit, "exact_match", False)) or record.get(
                "exact_match", False
            )
            record.setdefault("memory_type", _attr(hit, "memory_type", MEMORY_TYPE_ASSERTION))
            record.setdefault("status", _attr(hit, "status", "accepted"))
            _absorb_common(record, hit)
            # graph hop distance, if the symbolic hit exposes one.
            hops = _attr(hit, "hops", None)
            if hops is not None:
                record["hops"] = hops

        for hit in semantic or ():
            memory_id = _memory_id(hit)
            if memory_id is None:
                continue
            record = merged.setdefault(memory_id, {})
            record.setdefault("is_symbolic", False)
            record.setdefault("exact_match", False)
            similarity = _attr(hit, "similarity", None)
            if similarity is not None:
                record["similarity"] = float(similarity)
            record["memory_type"] = _attr(hit, "memory_type", record.get("memory_type"))
            record["status"] = _attr(hit, "status", record.get("status", "accepted"))
            _absorb_common(record, hit)

        # Normalize required fields.
        for record in merged.values():
            record.setdefault("memory_type", MEMORY_TYPE_ASSERTION)
            record.setdefault("status", "accepted")
        return merged

    def _signals(
        self,
        record: Mapping[str, Any],
        item_meta: Mapping[str, Any],
        *,
        contradicted: bool,
        reference_now: datetime,
    ) -> dict[str, float]:
        """Resolve the six raw signals for one merged candidate.

        Precedence: explicit ``item_meta`` override > value derived from the hit
        > safe default. See the module docstring for the special-case rules.
        """
        exact_match = bool(record.get("exact_match", False))

        # --- semantic_similarity (Req 15.4 forces 1.0 on exact match) ----
        if exact_match:
            semantic_similarity = 1.0
        elif "semantic_similarity" in item_meta:
            semantic_similarity = float(item_meta["semantic_similarity"])
        else:
            semantic_similarity = float(record.get("similarity", 0.0))

        # --- graph_relevance ----------------------------------------------
        if "graph_relevance" in item_meta:
            graph_relevance = float(item_meta["graph_relevance"])
        elif exact_match:
            graph_relevance = EXACT_MATCH_GRAPH_RELEVANCE
        elif record.get("is_symbolic"):
            graph_relevance = self._decayed_graph_relevance(record.get("hops"))
        else:
            graph_relevance = SEMANTIC_GRAPH_RELEVANCE

        # --- confidence ----------------------------------------------------
        if "confidence" in item_meta:
            confidence = float(item_meta["confidence"])
        else:
            confidence = float(record.get("confidence") or 0.0)

        # --- provenance_quality -------------------------------------------
        if "provenance_quality" in item_meta:
            provenance_quality = float(item_meta["provenance_quality"])
        else:
            provenance_quality = _provenance_quality(record)

        # --- recency -------------------------------------------------------
        if "recency" in item_meta:
            recency = float(item_meta["recency"])
        else:
            created_at = item_meta.get("created_at", record.get("created_at"))
            recency = self._recency(created_at, reference_now)

        # --- contradiction_penalty (Req 17.3) -----------------------------
        if "contradiction_penalty" in item_meta:
            contradiction_penalty = float(item_meta["contradiction_penalty"])
        elif (
            contradicted
            or bool(record.get("contradicted", False))
            or str(record.get("status")) == STATUS_QUARANTINED
        ):
            contradiction_penalty = 1.0
        else:
            contradiction_penalty = 0.0

        return {
            "semantic_similarity": _clamp(semantic_similarity),
            "graph_relevance": _clamp(graph_relevance),
            "confidence": _clamp(confidence),
            "provenance_quality": _clamp(provenance_quality),
            "recency": _clamp(recency),
            "contradiction_penalty": max(0.0, contradiction_penalty),
        }

    @staticmethod
    def _decayed_graph_relevance(hops: Any) -> float:
        """Graph relevance for a non-exact symbolic hit, decaying with hops.

        ``hops`` is the graph distance from the query target (``0`` for a direct
        match, ``1``/``2`` for neighbours). When unknown, a mid-strength default
        is used.
        """
        if hops is None:
            return SYMBOLIC_GRAPH_RELEVANCE
        try:
            distance = int(hops)
        except (TypeError, ValueError):
            return SYMBOLIC_GRAPH_RELEVANCE
        if distance <= 0:
            return EXACT_MATCH_GRAPH_RELEVANCE
        # 1 hop -> 0.5, 2 hops -> 0.25, ...
        return EXACT_MATCH_GRAPH_RELEVANCE / (2 ** distance)

    def _recency(self, created_at: Any, reference_now: datetime) -> float:
        """Normalize ``created_at`` into ``[0, 1]`` (newer ⇒ closer to 1.0).

        Linear decay over :attr:`recency_window_days`; items at or after ``now``
        score ``1.0``, items at/beyond the window score ``0.0``. Unknown /
        unparseable timestamps score ``0.0``.
        """
        if not isinstance(created_at, datetime):
            return 0.0
        now = reference_now
        # Make both sides timezone-comparable.
        if created_at.tzinfo is None:
            created = created_at.replace(tzinfo=timezone.utc)
        else:
            created = created_at
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        age_seconds = (now - created).total_seconds()
        if age_seconds <= 0:
            return 1.0
        window_seconds = self.recency_window_days * 86400.0
        if window_seconds <= 0:
            return 0.0
        return _clamp(1.0 - age_seconds / window_seconds)


# ---------------------------------------------------------------------------
# Hit accessors (duck-typed over objects and mappings) and small helpers
# ---------------------------------------------------------------------------
def _attr(hit: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from an object attribute or a mapping key, else ``default``."""
    if isinstance(hit, Mapping):
        return hit.get(name, default)
    return getattr(hit, name, default)


def _memory_id(hit: Any) -> Optional[str]:
    """Resolve a hit's memory id from the common attribute spellings."""
    for name in ("memory_id", "assertion_id", "id"):
        value = _attr(hit, name, None)
        if value:
            return str(value)
    return None


def _absorb_common(record: dict[str, Any], hit: Any) -> None:
    """Copy commonly-available metadata off a hit into the merged record.

    Only fills fields that are currently missing/empty so the first hit to
    supply a value wins (symbolic hits are folded before semantic ones).
    """
    for name in (
        "confidence",
        "source_ref",
        "created_at",
        "predicate",
        "subject_id",
        "object_id",
        "text",
        "extractor_version",
        "supporting_evidence_ids",
        "contradicted",
    ):
        value = _attr(hit, name, None)
        if value is not None and record.get(name) in (None, [], ""):
            record[name] = value


def _provenance_quality(record: Mapping[str, Any]) -> float:
    """Normalized provenance completeness in ``[0, 1]`` (design R3 table).

    Rewards a present ``source_ref`` and ``extractor_version`` plus the count of
    ``supporting_evidence_ids`` (more/complete ⇒ higher). Returns ``0.0`` when no
    provenance signal is available.
    """
    score = 0.0
    if record.get("source_ref"):
        score += 0.34
    if record.get("extractor_version"):
        score += 0.33
    evidence = record.get("supporting_evidence_ids") or []
    try:
        count = len(evidence)
    except TypeError:
        count = 0
    score += min(count, 3) / 3.0 * 0.33
    return _clamp(score)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    if value < low:
        return low
    if value > high:
        return high
    return value

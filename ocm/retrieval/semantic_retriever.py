"""Semantic Retriever — retrieval stage R2 (Req 16.1, 16.2, 16.3, 16.4, 16.5).

The :class:`SemanticRetriever` is the dense / vector half of retrieval. It
embeds the natural-language query and asks the :class:`~ocm.retrieval.vector_index.VectorIndex`
for the ``top_k`` nearest memory items — claims, assertions, documents, and
events (Req 16.1) — ranked by cosine similarity.

Accepted-by-default with conflict-aware quarantine inclusion
------------------------------------------------------------
The status visibility rules from Requirement 16 are implemented entirely
through the vector index's ``where`` metadata filter plus a light post-filter:

- **Accepted by default (Req 16.2):** ordinary queries search with
  ``where={"status": "accepted"}`` so only accepted assertions / items surface.
- **Conflict queries (Req 16.3):** when the query is a conflict query
  (``classification.query_type == "contradiction_check"``) — or the caller
  passes ``include_conflicts=True`` — the filter widens to
  ``{"status": {"$in": ["accepted", "quarantined"]}}`` so quarantined items can
  appear alongside accepted ones.
- **Conflict-relevance (Req 16.4):** a quarantined item is kept when it is
  relevant to a conflict involving accepted memory. With only vector hits to
  reason over, the available signal is that the item surfaced in the top-k for
  a conflict query; such items are treated as conflict-relevant. The
  :meth:`SemanticRetriever._conflict_relevant` hook isolates this judgement so a
  richer (graph-aware) relevance test can be substituted later without changing
  the public surface.
- **Exclusion (Req 16.5):** for non-conflict queries the accepted-only filter
  already excludes quarantined items; the post-filter additionally drops any
  quarantined hit that is not conflict-relevant, so a quarantined item never
  leaks into a non-conflict result set.

Requirements: 16.1, 16.2, 16.3, 16.4, 16.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

from pydantic import BaseModel

from ocm.retrieval.vector_index import VectorHit, VectorIndex

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a hard import cycle
    from ocm.retrieval.query_classifier import QueryClassification

#: The ``query_type`` value that marks a query as a "conflict query" — the
#: signal that quarantined items should be included (Req 16.3).
CONFLICT_QUERY_TYPE = "contradiction_check"

STATUS_ACCEPTED = "accepted"
STATUS_QUARANTINED = "quarantined"

#: Memory types the Semantic Retriever ranks: claims, assertions, documents,
#: and events (Req 16.1).
SEMANTIC_MEMORY_TYPES = ("claim", "assertion", "document", "event")


@runtime_checkable
class _ClassificationLike(Protocol):
    """Structural type for what R2 needs from an R0 ``QueryClassification``.

    The Semantic Retriever only reads ``query_type`` to decide whether the query
    is a conflict query. Depending on this Protocol (rather than importing the
    concrete model) keeps R2 usable before the Query_Classifier (R0) lands and
    lets tests pass any object exposing ``query_type``.
    """

    query_type: str


class SemanticHit(BaseModel):
    """A single semantic match returned by :meth:`SemanticRetriever.retrieve`.

    Mirrors :class:`~ocm.retrieval.vector_index.VectorHit` (the raw vector-index
    result) as the R2 stage's output type so the Reranker (R3) consumes a stable
    ``SemanticHit`` regardless of the underlying index. ``similarity`` is a
    cosine-derived score in ``[0, 1]`` (``1.0`` = identical direction).
    """

    memory_id: str
    memory_type: str
    status: str
    similarity: float
    text: Optional[str] = None

    @classmethod
    def from_vector_hit(cls, hit: VectorHit) -> "SemanticHit":
        """Build a :class:`SemanticHit` from a raw :class:`VectorHit`."""
        return cls(
            memory_id=hit.memory_id,
            memory_type=hit.memory_type,
            status=hit.status,
            similarity=hit.similarity,
            text=hit.text,
        )


def status_filter(
    classification: "_ClassificationLike | QueryClassification",
    include_conflicts: bool = False,
) -> dict:
    """Build the vector-index ``where`` status filter for a query (Req 16.2, 16.3).

    Returns ``{"status": {"$in": ["accepted", "quarantined"]}}`` for conflict
    queries (or when ``include_conflicts`` is set) so quarantined items can be
    retrieved, and ``{"status": "accepted"}`` otherwise so only accepted memory
    is visible by default.
    """
    if _is_conflict_query(classification) or include_conflicts:
        return {"status": {"$in": [STATUS_ACCEPTED, STATUS_QUARANTINED]}}
    return {"status": STATUS_ACCEPTED}


def _is_conflict_query(classification: "_ClassificationLike | QueryClassification") -> bool:
    """Whether ``classification`` marks a conflict query (Req 16.3)."""
    return getattr(classification, "query_type", None) == CONFLICT_QUERY_TYPE


class SemanticRetriever:
    """Retrieval stage R2: dense top-k search with conflict-aware visibility."""

    def __init__(self, vector_index: VectorIndex) -> None:
        """Wire the retriever to a :class:`VectorIndex`.

        Args:
            vector_index: The semantic index to embed the query against and
                search for nearest memory items.
        """
        self.vector_index = vector_index

    def retrieve(
        self,
        query: str,
        classification: "_ClassificationLike | QueryClassification",
        top_k: int = 10,
        include_conflicts: bool = False,
    ) -> list[SemanticHit]:
        """Embed ``query`` and return the ``top_k`` nearest memory items.

        Behaviour follows Requirement 16:

        - Embeds the query and searches the Vector_Index for the top-k claims,
          assertions, documents, and events (Req 16.1).
        - Includes accepted assertions / items by default via a
          ``status == accepted`` filter (Req 16.2).
        - For a conflict query (``query_type == "contradiction_check"``) — or
          when ``include_conflicts`` is set — widens the filter to also include
          quarantined items (Req 16.3) and keeps the conflict-relevant ones
          (Req 16.4).
        - For a non-conflict query, excludes quarantined items that are not
          relevant to an accepted-memory conflict (Req 16.5).

        Args:
            query: The natural-language query text to embed and search.
            classification: The R0 classification; only ``query_type`` is read.
            top_k: Maximum number of nearest items to retrieve.
            include_conflicts: Force inclusion of quarantined items even for a
                non-conflict query (used by callers that already know they are
                investigating a conflict).

        Returns:
            Ranked :class:`SemanticHit` objects (nearest first), filtered to the
            visible/relevant set for the query.
        """
        if top_k <= 0:
            return []

        conflict_query = _is_conflict_query(classification) or include_conflicts
        where = status_filter(classification, include_conflicts=include_conflicts)

        raw_hits = self.vector_index.query(query, top_k=top_k, where=where)

        hits: list[SemanticHit] = []
        for hit in raw_hits:
            # Req 16.1: restrict to claims / assertions / documents / events.
            if hit.memory_type not in SEMANTIC_MEMORY_TYPES:
                continue
            if hit.status == STATUS_QUARANTINED and not self._keep_quarantined(
                hit, conflict_query, classification
            ):
                continue
            hits.append(SemanticHit.from_vector_hit(hit))
        return hits

    def _keep_quarantined(
        self,
        hit: VectorHit,
        conflict_query: bool,
        classification: "_ClassificationLike | QueryClassification",
    ) -> bool:
        """Decide whether a quarantined ``hit`` stays in the result set.

        Implements the Req 16.4 / 16.5 split: a quarantined item is kept only
        when the query is a conflict query *and* the item is conflict-relevant.
        For a non-conflict query a quarantined hit is always dropped (Req 16.5);
        in practice the accepted-only ``where`` filter means such hits never even
        reach here, but the guard keeps the rule explicit and robust.
        """
        if not conflict_query:
            return False
        return self._conflict_relevant(hit, classification)

    def _conflict_relevant(
        self,
        hit: VectorHit,
        classification: "_ClassificationLike | QueryClassification",
    ) -> bool:
        """Whether a quarantined ``hit`` is relevant to the query's entities (Req 16.4).

        A quarantined item is kept only when it is actually *about* what the
        query asked: one of the query's extracted entity names must appear in
        the hit's embedding text (case-insensitive). This prevents a conflict
        query from sweeping in unrelated quarantined items just because they are
        near in vector space, which would inflate conflict recall at the cost of
        precision.

        When the classifier extracted no entities (nothing to match against) the
        item is kept, preserving recall for entity-free conflict queries; the
        authoritative, id-precise relevance gate then runs downstream in the
        Evidence Packager against the quarantine records' ``conflicting_ids``.
        """
        entities = [e for e in (getattr(classification, "entities", []) or []) if e]
        if not entities:
            return True
        text = (hit.text or "").casefold()
        if not text:
            return True
        return any(" ".join(str(e).split()).casefold() in text for e in entities)

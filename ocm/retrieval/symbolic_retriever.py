"""R1 — Symbolic Retriever.

Answers precise *structural* questions directly from the ``Graph_Store`` over
**accepted** assertions (the graph only ever holds accepted edges, Req 11.5),
without touching the vector index:

- **Project owner** — for an owner query, return the subject(s) of ``OWNS``
  edges *into* the target Project (Req 15.1).
- **Task assignee** — for an assignee query, return the object of the
  ``ASSIGNED_TO`` edge *out of* the target Task (Req 15.2).
- **Preceding events** — for a temporal query, return the Events that point at
  the target Event via incoming ``PRECEDES`` edges (Req 15.3).

Each :class:`SymbolicHit` carries the backing assertion id, the matched
subject/predicate/object, and ``exact_match=True``. The exact-match flag is the
signal the :class:`~ocm.retrieval.reranker.Reranker` uses to force the hit's
``semantic_similarity`` to ``1.0`` (Req 15.4).

Entity *names* extracted by the Query Classifier (R0) are resolved to graph
node ids by scanning node payloads for a matching ``name`` / ``title`` (and a
few other label-ish fields), case-insensitively, plus a direct id match.

The classifier dependency is imported defensively: this module only needs an
object exposing ``entities`` / ``predicates`` / ``query_type`` (duck typed via
:class:`ClassificationLike`), so it works whether or not
``ocm.retrieval.query_classifier`` has landed yet.

Requirements: 15.1, 15.2, 15.3, 15.4.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ocm.memory.graph_store import GraphStore

# Best-effort import of the canonical classification model. Only used for type
# hints / isinstance-free duck typing, so a missing module is not fatal.
try:  # pragma: no cover - exercised once R0 lands.
    from ocm.retrieval.query_classifier import QueryClassification as QueryClassification
except Exception:  # pragma: no cover - parallel task may not exist yet.
    QueryClassification = Any  # type: ignore[assignment,misc]


# Predicate constants (single source of truth for the structural lookups).
OWNS = "OWNS"
ASSIGNED_TO = "ASSIGNED_TO"
PRECEDES = "PRECEDES"
HAS_STATUS = "HAS_STATUS"

# Node payload fields that can act as a human-facing label, in priority order.
_LABEL_FIELDS: tuple[str, ...] = (
    "name",        # Person, Organization, Project
    "title",       # Task, Document
    "description",  # Event
    "summary",     # Decision
    "text",        # Claim
)


@runtime_checkable
class ClassificationLike(Protocol):
    """Structural protocol for whatever R0 hands us.

    We only ever read ``entities``, ``predicates`` and ``query_type``; anything
    exposing those attributes (the real ``QueryClassification`` or a test
    double) is accepted.
    """

    entities: list[str]
    predicates: list[str]
    query_type: str


class SymbolicHit(BaseModel):
    """A single structural answer drawn directly from the graph.

    ``assertion_id`` is the id of the accepted assertion edge that backs this
    hit; ``memory_id`` is exposed as an alias so downstream rerankers that key
    candidates by a generic ``memory_id`` can consume symbolic and semantic
    hits uniformly. ``exact_match`` defaults to ``True`` because every symbolic
    hit is, by construction, an exact structural match — the Reranker reads this
    flag and forces ``semantic_similarity = 1.0`` (Req 15.4).
    """

    assertion_id: str
    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 1.0
    source_ref: str | None = None
    exact_match: bool = True

    @property
    def memory_id(self) -> str:
        """Alias for :attr:`assertion_id` (generic id used by the Reranker)."""
        return self.assertion_id


def _normalize(value: str) -> str:
    """Case/whitespace-insensitive normal form for label comparison."""
    return " ".join(value.strip().casefold().split())


# Leading words that denote an entity's *type* rather than its proper name, so a
# mention like "Project Orion" / "Task T1" / "Event Review" also resolves to the
# node stored under the bare name ("Orion" / "T1" / "Review").
_GENERIC_TYPE_WORDS: frozenset[str] = frozenset(
    {"project", "task", "event", "document", "decision", "organization", "org", "person"}
)


def _name_variants(raw: str) -> set[str]:
    """Normalized match targets for a mention: the full name and, when it opens
    with a generic type word, the name with that word stripped."""
    variants: set[str] = set()
    normalized = _normalize(raw)
    if not normalized:
        return variants
    variants.add(normalized)
    tokens = normalized.split(" ")
    if len(tokens) > 1 and tokens[0] in _GENERIC_TYPE_WORDS:
        variants.add(" ".join(tokens[1:]))
    return variants


def _node_labels(payload: dict[str, Any] | None, node_id: str) -> set[str]:
    """All normalized labels a node can be matched by (its id + label fields)."""
    labels: set[str] = {_normalize(node_id)}
    if payload:
        # The id may also live inside the payload.
        pid = payload.get("id")
        if isinstance(pid, str):
            labels.add(_normalize(pid))
        for field in _LABEL_FIELDS:
            val = payload.get(field)
            if isinstance(val, str) and val.strip():
                labels.add(_normalize(val))
    return labels


def resolve_entity_ids(graph: GraphStore, names: list[str]) -> list[str]:
    """Map classifier entity *names* to graph node ids.

    A name matches a node when it equals the node id or any of the node's
    label-ish payload fields (``name``/``title``/...), compared
    case-insensitively. A mention that opens with a generic type word
    (``"Project Orion"``) also matches the bare-name node (``"Orion"``).
    Results are de-duplicated while preserving the order in which names were
    requested (and, within a name, sorted node ids for determinism).
    """
    resolved: list[str] = []
    seen: set[str] = set()
    for raw in names or []:
        if not isinstance(raw, str) or not raw.strip():
            continue
        targets = _name_variants(raw)
        matches: list[str] = []
        for node_id in graph.node_ids():
            payload = graph.get_entity_payload(node_id)
            if _node_labels(payload, node_id) & targets:
                matches.append(node_id)
        for node_id in sorted(matches):
            if node_id not in seen:
                seen.add(node_id)
                resolved.append(node_id)
    return resolved


def _hit_from_edge(subject: str, obj: str, predicate: str, data: dict[str, Any]) -> SymbolicHit:
    """Build a :class:`SymbolicHit` from a graph edge tuple's data dict."""
    return SymbolicHit(
        assertion_id=data.get("assertion_id", ""),
        subject_id=subject,
        predicate=predicate,
        object_id=obj,
        confidence=float(data.get("confidence", 1.0)),
        source_ref=data.get("source_ref"),
        exact_match=True,
    )


class SymbolicRetriever:
    """Graph-backed retriever for OWNS / ASSIGNED_TO / PRECEDES queries."""

    def retrieve(
        self, classification: ClassificationLike, graph: GraphStore
    ) -> list[SymbolicHit]:
        """Return exact structural hits for the classified query.

        The classifier's ``entities`` are resolved to node ids, then — gated by
        the requested ``predicates`` (or, when none were extracted, inferred
        from ``query_type``) — each resolved node is probed for the structural
        relation appropriate to its entity type:

        - a **Project** node yields its incoming ``OWNS`` edges (owners),
        - a **Task** node yields its outgoing ``ASSIGNED_TO`` edge (assignee),
        - an **Event** node yields its incoming ``PRECEDES`` edges (predecessors).

        Hits are returned in a deterministic, de-duplicated order.
        """
        wanted = self._wanted_predicates(classification)
        node_ids = resolve_entity_ids(graph, getattr(classification, "entities", []) or [])

        hits: list[SymbolicHit] = []
        seen: set[tuple[str, str, str, str]] = set()

        def _add(hit: SymbolicHit) -> None:
            key = (hit.assertion_id, hit.subject_id, hit.predicate, hit.object_id)
            if key not in seen:
                seen.add(key)
                hits.append(hit)

        for node_id in node_ids:
            node_type = graph.get_entity_type(node_id)

            # Req 15.1 — project owner via incoming OWNS edges.
            if OWNS in wanted and node_type == "Project":
                for s, o, _k, d in graph.in_edges(node_id, OWNS):
                    _add(_hit_from_edge(s, o, OWNS, d))

            # Req 15.2 — task assignee via outgoing ASSIGNED_TO edge.
            if ASSIGNED_TO in wanted and node_type == "Task":
                for s, o, _k, d in graph.out_edges(node_id, ASSIGNED_TO):
                    _add(_hit_from_edge(s, o, ASSIGNED_TO, d))

            # Req 15.3 — preceding events via incoming PRECEDES edges.
            if PRECEDES in wanted and node_type == "Event":
                for s, o, _k, d in graph.in_edges(node_id, PRECEDES):
                    _add(_hit_from_edge(s, o, PRECEDES, d))

            # Status-as-assertion: a status-bearing entity's accepted HAS_STATUS
            # edge, so the accepted status assertion is an exact-match supporting
            # item and its id enters the conflict-relevance set for status
            # queries (#7).
            if HAS_STATUS in wanted and node_type in ("Task", "Project", "Person", "Organization"):
                for s, o, _k, d in graph.out_edges(node_id, HAS_STATUS):
                    _add(_hit_from_edge(s, o, HAS_STATUS, d))

        # Deterministic ordering independent of graph iteration order.
        hits.sort(key=lambda h: (h.predicate, h.subject_id, h.object_id, h.assertion_id))
        return hits

    @staticmethod
    def _wanted_predicates(classification: ClassificationLike) -> set[str]:
        """The structural predicates to probe for this query.

        Uses the classifier's extracted ``predicates`` when present; otherwise
        infers a sensible default from ``query_type`` (temporal ⇒ PRECEDES,
        everything else ⇒ the fact relations OWNS / ASSIGNED_TO). The per-node
        entity-type gate in :meth:`retrieve` keeps results correct even when the
        predicate set is broad.
        """
        raw_predicates = getattr(classification, "predicates", None) or []
        wanted = {
            p.upper()
            for p in raw_predicates
            if isinstance(p, str)
        } & {OWNS, ASSIGNED_TO, PRECEDES, HAS_STATUS}
        if wanted:
            return wanted

        query_type = getattr(classification, "query_type", None)
        if query_type == "temporal":
            return {PRECEDES}
        # direct_fact / planning / fallback: probe the fact relations plus the
        # status assertion (so status queries surface the accepted HAS_STATUS).
        return {OWNS, ASSIGNED_TO, HAS_STATUS}

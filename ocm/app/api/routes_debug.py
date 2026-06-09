"""Read-only debug/inspection router (``routes_debug``) (Req 19.1).

A **non-production** router that exposes read-only inspection endpoints used by
tests and the research demo. The main app (``ocm/app/main.py``, task 15.2)
mounts this router **only** when ``settings.deterministic_test_mode`` (or an
explicit debug flag) is set, so these endpoints never ship in a normal
deployment. They never mutate state — they only project the current
:class:`~ocm.core.container.CoreContainer` view of the graph, the quarantine
table, and provenance for inspection of governance behavior during experiments
(design "routes_debug").

Endpoints
---------
* ``GET /debug/graph`` — node/edge dump of the in-memory accepted-only graph
  (entities as nodes; accepted assertions as edges).
* ``GET /debug/quarantine`` — the full quarantine table, optionally filtered by
  ``?status=`` (e.g. ``unresolved``).
* ``GET /debug/provenance/{subject_id}`` — every provenance record for a subject.

The router resolves the wired container from ``request.app.state.container``
(the same pattern the production routers use), so it stays decoupled from
construction.

Requirements: 19.1.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ocm.core.container import CoreContainer
from ocm.ontology.models import Provenance, QuarantineRecord

__all__ = [
    "router",
    "get_container",
    "GraphNodeView",
    "GraphEdgeView",
    "GraphView",
]


# --------------------------------------------------------------------------- #
# Dependency: resolve the wired container from app.state
# --------------------------------------------------------------------------- #
def get_container(request: Request) -> CoreContainer:
    """Return the :class:`CoreContainer` stored on ``app.state`` (Req 19.1).

    The main app builds a single container per process and stores it on
    ``app.state.container``; routers resolve it via this dependency so they stay
    decoupled from construction. Mirrors the dependency used by the production
    routers (``routes.py``).
    """
    return request.app.state.container


# --------------------------------------------------------------------------- #
# Response models for GET /debug/graph
# --------------------------------------------------------------------------- #
class GraphNodeView(BaseModel):
    """A single entity node: its id, ontology type, and full payload."""

    id: str
    type: Optional[str] = None
    payload: Optional[dict] = None


class GraphEdgeView(BaseModel):
    """A single accepted-assertion edge: the triple plus the edge attributes.

    ``data`` carries the complete edge attribute dict (``assertion_id``,
    ``confidence``, ``status``, ``source_ref``, timestamps, ...) so the debug
    dump is lossless; the named fields lift the most-used attributes for
    convenience.
    """

    subject: str
    predicate: str
    object: str
    assertion_id: Optional[str] = None
    confidence: Optional[float] = None
    status: Optional[str] = None
    source_ref: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)


class GraphView(BaseModel):
    """A read-only dump of the in-memory accepted-only graph (Req 19.1)."""

    num_nodes: int
    num_edges: int
    nodes: list[GraphNodeView] = Field(default_factory=list)
    edges: list[GraphEdgeView] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/graph", response_model=GraphView)
def debug_graph(container: CoreContainer = Depends(get_container)) -> GraphView:
    """Dump the in-memory graph: entity nodes and accepted-assertion edges.

    Read-only projection of ``container.graph`` (the accepted-only
    :class:`~ocm.memory.graph_store.GraphStore`). Nodes expose ``id``/``type``/
    ``payload``; edges expose the ``(subject, predicate, object)`` triple plus
    edge attributes.
    """
    graph = container.graph
    g = graph.g

    nodes = [
        GraphNodeView(
            id=node_id,
            type=attrs.get("type"),
            payload=attrs.get("payload"),
        )
        for node_id, attrs in g.nodes(data=True)
    ]

    edges = [
        GraphEdgeView(
            subject=subject,
            predicate=predicate,
            object=obj,
            assertion_id=data.get("assertion_id"),
            confidence=data.get("confidence"),
            status=data.get("status"),
            source_ref=data.get("source_ref"),
            data=dict(data),
        )
        for subject, obj, predicate, data in g.edges(keys=True, data=True)
    ]

    return GraphView(
        num_nodes=graph.num_nodes(),
        num_edges=graph.num_edges(),
        nodes=nodes,
        edges=edges,
    )


@router.get("/quarantine", response_model=list[QuarantineRecord])
def debug_quarantine(
    status: Optional[str] = None,
    container: CoreContainer = Depends(get_container),
) -> list[QuarantineRecord]:
    """Return the full quarantine table, optionally filtered by ``?status=``.

    Delegates to ``container.quarantine_store.list(status=...)`` which reads
    durable storage, so the view reflects what is persisted (Req 11.7). When
    ``status`` is omitted all records are returned.
    """
    return container.quarantine_store.list(status=status)


@router.get("/provenance/{subject_id}", response_model=list[Provenance])
def debug_provenance(
    subject_id: str,
    container: CoreContainer = Depends(get_container),
) -> list[Provenance]:
    """Return every provenance record for ``subject_id``.

    Delegates to ``container.provenance_tracker.for_subject(subject_id)``. A
    subject with no provenance yields an empty list (still a 200 response).
    """
    return container.provenance_tracker.for_subject(subject_id)

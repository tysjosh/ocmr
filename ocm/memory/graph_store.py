"""In-memory graph projection of accepted memory (``Graph_Store``).

``GraphStore`` wraps a :class:`networkx.MultiDiGraph`. Entities are **nodes**
(keyed by ``id``, carrying a ``type`` attribute and the full ``payload``);
accepted assertions are **directed edges** keyed by predicate (a multigraph lets
several predicates connect the same pair of nodes). Only ``accepted`` assertions
ever become edges — superseded, quarantined, and rejected assertions are
excluded (Req 11.5). The graph is a fast, rebuildable view over the durable
``Storage_Repository``; the repository is the source of truth on disk.

The Commit Manager keeps the graph and SQLite consistent by *write-through on
commit* (Req 11.6): an accept calls :meth:`GraphStore.add_assertion` while
persisting the row, a supersede calls :meth:`GraphStore.remove_assertion` while
flipping the old row to ``superseded``. The standing invariant is therefore:

    the set of graph edges == the set of ``assertions`` rows with
    ``status == 'accepted'``.

On restart :func:`rebuild_graph` reconstructs the graph deterministically from
the persisted entities + accepted assertions, so the rebuilt graph is identical
to the pre-restart accepted state (Req 11.8).

The query surface is intentionally general so it serves both the
``Symbolic_Retriever`` (OWNS / ASSIGNED_TO / PRECEDES traversals) and the
``Constraint_Validator`` (C9 graph-level domain/range via
:meth:`get_entity_type`, C3 acyclic PRECEDES via :meth:`has_path` /
:meth:`simple_cycles` / :meth:`would_create_cycle`).

Requirements: 11.5, 11.6, 11.8.
"""

from __future__ import annotations

from typing import Any, Iterator

import networkx as nx
from pydantic import BaseModel

from ocm.ontology.enums import AssertionStatus
from ocm.ontology.models import Assertion
from ocm.memory.repository import StorageRepository

# An edge tuple as returned by the traversal helpers: (subject, object,
# predicate, edge_data).
EdgeTuple = tuple[str, str, str, dict[str, Any]]


class GraphStore:
    """A NetworkX ``MultiDiGraph`` holding entities and accepted assertions only."""

    def __init__(self) -> None:
        self.g: nx.MultiDiGraph = nx.MultiDiGraph()

    # -- entities ----------------------------------------------------------
    def add_entity(self, entity_type: str, entity_or_payload: BaseModel | dict) -> None:
        """Add (or update) an entity node from a model or a payload dict.

        The node stores the entity's ``type`` and its full ``payload``. The
        payload is kept as a nested attribute (rather than spread onto the node)
        so an entity field named ``type`` — e.g. ``Organization.type`` — never
        collides with the node-level ``type`` attribute.
        """
        if isinstance(entity_or_payload, BaseModel):
            payload: dict[str, Any] = entity_or_payload.model_dump(mode="json")
        elif isinstance(entity_or_payload, dict):
            payload = dict(entity_or_payload)
        else:  # pragma: no cover - defensive
            raise TypeError(
                "entity_or_payload must be a Pydantic model or a dict, got "
                f"{type(entity_or_payload)!r}"
            )
        try:
            entity_id = payload["id"]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError("entity payload is missing required 'id' field") from exc
        self.g.add_node(entity_id, type=entity_type, payload=payload)

    def has_entity(self, entity_id: str) -> bool:
        """Return whether an entity node exists."""
        return self.g.has_node(entity_id)

    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity node (and any incident edges) if present.

        Idempotent: a missing node is a no-op. Used to retract a candidate entity
        that fails write-time governance (e.g. a final Decision quarantined by C8)
        so it never lingers as accepted memory.
        """
        if self.g.has_node(entity_id):
            self.g.remove_node(entity_id)

    def get_entity_type(self, entity_id: str) -> str | None:
        """Return the ``entity_type`` of a node, or ``None`` if absent.

        Used by C9 to resolve the actual types of an assertion's subject and
        object for graph-level domain/range validation (Req 8.10).
        """
        if not self.g.has_node(entity_id):
            return None
        return self.g.nodes[entity_id].get("type")

    # Alias matching the design's terminology.
    node_type = get_entity_type

    def get_entity_payload(self, entity_id: str) -> dict[str, Any] | None:
        """Return the stored payload for an entity, or ``None`` if absent."""
        if not self.g.has_node(entity_id):
            return None
        return self.g.nodes[entity_id].get("payload")

    # -- assertions (accepted only) ----------------------------------------
    def add_assertion(self, a: Assertion) -> None:
        """Add an **accepted** assertion as a directed edge keyed by predicate.

        Enforces the accepted-only invariant (Req 11.5, 10.5): a non-accepted
        assertion raises :class:`ValueError` so quarantined/rejected/superseded
        candidates can never enter the graph as accepted memory.
        """
        if a.status != AssertionStatus.accepted:
            raise ValueError(
                "GraphStore only accepts assertions with status 'accepted'; "
                f"got {a.status.value!r} for assertion {a.id!r}"
            )
        self.g.add_edge(
            a.subject_id,
            a.object_id,
            key=a.predicate,
            assertion_id=a.id,
            predicate=a.predicate,
            confidence=float(a.confidence),
            status=a.status.value,
            source_ref=a.source_ref,
            created_at=a.created_at,
            valid_from=a.valid_from,
            valid_to=a.valid_to,
        )

    # Alias matching the design snippet (only ever called for accepted).
    add_accepted_assertion = add_assertion

    def remove_assertion(self, subject_id: str, object_id: str, predicate: str) -> None:
        """Remove the edge for ``(subject, predicate, object)`` if present.

        Idempotent: a missing edge is a no-op. Used on supersession to drop the
        superseded edge while the new assertion is accepted (Req 11.6).
        """
        if self.g.has_edge(subject_id, object_id, key=predicate):
            self.g.remove_edge(subject_id, object_id, key=predicate)

    def has_assertion(self, subject_id: str, object_id: str, predicate: str) -> bool:
        """Return whether an accepted edge exists for the given triple."""
        return self.g.has_edge(subject_id, object_id, key=predicate)

    def get_assertion_edge(
        self, subject_id: str, object_id: str, predicate: str
    ) -> dict[str, Any] | None:
        """Return the edge data dict for a triple, or ``None`` if absent."""
        if not self.g.has_edge(subject_id, object_id, key=predicate):
            return None
        return dict(self.g.edges[subject_id, object_id, predicate])

    def find_assertion(self, assertion_id: str) -> EdgeTuple | None:
        """Locate an accepted edge by its ``assertion_id``.

        Scans every edge for one whose ``assertion_id`` matches and returns its
        ``(subject, object, predicate, data)`` tuple, or ``None`` if no accepted
        edge carries that id. Used to render the *accepted* side of a paired
        status conflict (the HAS_STATUS assertion a quarantined flip points at).
        """
        for s, o, k, d in self.g.edges(keys=True, data=True):
            if d.get("assertion_id") == assertion_id:
                return (s, o, k, dict(d))
        return None

    # -- traversal / queries ----------------------------------------------
    def out_edges(self, entity_id: str, predicate: str | None = None) -> list[EdgeTuple]:
        """Outgoing edges of ``entity_id`` (optionally filtered by predicate)."""
        if not self.g.has_node(entity_id):
            return []
        edges = self.g.out_edges(entity_id, keys=True, data=True)
        return [
            (s, o, k, dict(d))
            for s, o, k, d in edges
            if predicate is None or k == predicate
        ]

    def in_edges(self, entity_id: str, predicate: str | None = None) -> list[EdgeTuple]:
        """Incoming edges of ``entity_id`` (optionally filtered by predicate)."""
        if not self.g.has_node(entity_id):
            return []
        edges = self.g.in_edges(entity_id, keys=True, data=True)
        return [
            (s, o, k, dict(d))
            for s, o, k, d in edges
            if predicate is None or k == predicate
        ]

    def find_edges_by_predicate(self, predicate: str) -> list[EdgeTuple]:
        """All accepted edges with the given predicate, across the whole graph."""
        return [
            (s, o, k, dict(d))
            for s, o, k, d in self.g.edges(keys=True, data=True)
            if k == predicate
        ]

    def neighbors_out(self, entity_id: str, predicate: str | None = None) -> list[str]:
        """Object ids reachable via outgoing edges (e.g. ASSIGNED_TO target)."""
        return [o for _s, o, _k, _d in self.out_edges(entity_id, predicate)]

    def neighbors_in(self, entity_id: str, predicate: str | None = None) -> list[str]:
        """Subject ids pointing in via incoming edges (e.g. OWNS owners)."""
        return [s for s, _o, _k, _d in self.in_edges(entity_id, predicate)]

    def assertions_by_subject(
        self, subject_id: str, predicate: str | None = None
    ) -> list[EdgeTuple]:
        """Accepted assertions whose subject is ``subject_id``."""
        return self.out_edges(subject_id, predicate)

    def assertions_by_object(
        self, object_id: str, predicate: str | None = None
    ) -> list[EdgeTuple]:
        """Accepted assertions whose object is ``object_id``."""
        return self.in_edges(object_id, predicate)

    # -- acyclicity helpers (C3 PRECEDES) ----------------------------------
    def predicate_subgraph(self, predicate: str) -> nx.DiGraph:
        """A simple directed graph of just the edges with ``predicate``.

        Collapsing the multigraph onto a single predicate lets the path /
        cycle checks reason about one relation (e.g. PRECEDES) in isolation.
        """
        sub = nx.DiGraph()
        sub.add_nodes_from(self.g.nodes())
        for s, o, k in self.g.edges(keys=True):
            if k == predicate:
                sub.add_edge(s, o)
        return sub

    def has_path(
        self, source: str, target: str, predicate: str | None = None
    ) -> bool:
        """Whether a directed path ``source -> target`` exists.

        When ``predicate`` is given the search is restricted to edges of that
        predicate (used for PRECEDES cycle detection). Returns ``False`` if
        either endpoint is absent.
        """
        graph: nx.Graph = self.predicate_subgraph(predicate) if predicate else self.g
        if not graph.has_node(source) or not graph.has_node(target):
            return False
        return nx.has_path(graph, source, target)

    def would_create_cycle(
        self, subject_id: str, object_id: str, predicate: str
    ) -> bool:
        """Whether adding ``subject -[predicate]-> object`` would close a cycle.

        A new edge creates a cycle when it is a self-loop or when a path already
        leads from ``object`` back to ``subject`` along ``predicate`` edges.
        Supports C3 (acyclic PRECEDES, Req 8.4) before an edge is accepted.
        """
        if subject_id == object_id:
            return True
        return self.has_path(object_id, subject_id, predicate)

    def simple_cycles(self, predicate: str | None = None) -> list[list[str]]:
        """All simple cycles, optionally restricted to a single predicate."""
        graph = self.predicate_subgraph(predicate) if predicate else self.g
        return [list(cycle) for cycle in nx.simple_cycles(graph)]

    def is_acyclic(self, predicate: str | None = None) -> bool:
        """Whether the (optionally predicate-restricted) graph is a DAG."""
        graph = self.predicate_subgraph(predicate) if predicate else self.g
        return nx.is_directed_acyclic_graph(graph)

    # -- introspection / equality helpers ----------------------------------
    def node_ids(self) -> set[str]:
        """The set of entity ids currently in the graph."""
        return set(self.g.nodes())

    def edge_triples(self) -> set[tuple[str, str, str]]:
        """The set of ``(subject, object, predicate)`` triples (one per edge)."""
        return {(s, o, k) for s, o, k in self.g.edges(keys=True)}

    def num_nodes(self) -> int:
        """Number of entity nodes."""
        return self.g.number_of_nodes()

    def num_edges(self) -> int:
        """Number of accepted-assertion edges."""
        return self.g.number_of_edges()

    def __iter__(self) -> Iterator[str]:
        return iter(self.g.nodes())


def rebuild_graph(repo: StorageRepository) -> GraphStore:
    """Reconstruct a :class:`GraphStore` from durable storage (Req 11.8).

    The rebuild is deterministic and accepted-only:

    1. every persisted entity becomes a node, and
    2. only assertions with ``status == 'accepted'`` become edges.

    Because the graph holds nothing but accepted assertions, the rebuilt graph
    is identical to the pre-restart accepted state. An accepted assertion that
    references a missing entity (storage corruption) is skipped rather than
    aborting the rebuild, keeping the projection internally consistent.
    """
    graph = GraphStore()
    # 1) load every entity as a node
    for entity_type, payload in repo.list_entities():
        graph.add_entity(entity_type, payload)
    # 2) load only accepted assertions as edges
    for a in repo.list_assertions(status=AssertionStatus.accepted.value):
        if not graph.has_entity(a.subject_id) or not graph.has_entity(a.object_id):
            # Dangling edge (missing endpoint): skip to preserve consistency.
            continue
        graph.add_assertion(a)
    return graph

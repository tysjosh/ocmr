"""Tests for R1 — :mod:`ocm.retrieval.symbolic_retriever`.

Seeds a small graph (Alice OWNS Orion, T1 ASSIGNED_TO Bob, e1 PRECEDES e2)
and confirms owner / assignee / preceding-event queries return the correct
hits, each flagged ``exact_match=True`` so the Reranker forces
``semantic_similarity = 1.0`` (Req 15.1-15.4).
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from hypothesis import given
from hypothesis import strategies as st

from ocm.memory.graph_store import GraphStore
from ocm.ontology.enums import AssertionStatus
from ocm.ontology.models import Assertion
from ocm.retrieval.symbolic_retriever import (
    ASSIGNED_TO,
    OWNS,
    PRECEDES,
    SymbolicHit,
    SymbolicRetriever,
    resolve_entity_ids,
)

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _accepted(aid: str, subj: str, pred: str, obj: str, conf: float = 0.9) -> Assertion:
    return Assertion(
        id=aid,
        subject_id=subj,
        predicate=pred,
        object_id=obj,
        confidence=conf,
        status=AssertionStatus.accepted,
        source_ref=f"doc::{aid}",
        created_at=_NOW,
    )


def _seed_graph() -> GraphStore:
    """Alice OWNS Orion; T1 ASSIGNED_TO Bob; e1 PRECEDES e2."""
    g = GraphStore()
    g.add_entity("Person", {"id": "p_alice", "name": "Alice"})
    g.add_entity("Person", {"id": "p_bob", "name": "Bob"})
    g.add_entity("Project", {"id": "proj_orion", "name": "Orion"})
    g.add_entity("Task", {"id": "task_t1", "title": "T1"})
    g.add_entity("Event", {"id": "ev_e1", "type": "kickoff", "description": "e1"})
    g.add_entity("Event", {"id": "ev_e2", "type": "review", "description": "e2"})

    g.add_assertion(_accepted("a_owns", "p_alice", OWNS, "proj_orion"))
    g.add_assertion(_accepted("a_assigned", "task_t1", ASSIGNED_TO, "p_bob"))
    g.add_assertion(_accepted("a_precedes", "ev_e1", PRECEDES, "ev_e2"))
    return g


def _cls(entities, predicates, query_type="direct_fact"):
    """A duck-typed QueryClassification stand-in."""
    return SimpleNamespace(
        entities=list(entities),
        predicates=list(predicates),
        query_type=query_type,
        needs_semantic_fallback=False,
    )


# --------------------------------------------------------------------------
# Entity-name resolution
# --------------------------------------------------------------------------
def test_resolve_entity_ids_matches_name_case_insensitively() -> None:
    g = _seed_graph()
    assert resolve_entity_ids(g, ["orion"]) == ["proj_orion"]
    assert resolve_entity_ids(g, ["ALICE"]) == ["p_alice"]


def test_resolve_entity_ids_matches_by_node_id_and_title() -> None:
    g = _seed_graph()
    assert resolve_entity_ids(g, ["task_t1"]) == ["task_t1"]
    assert resolve_entity_ids(g, ["t1"]) == ["task_t1"]  # Task.title


def test_resolve_entity_ids_unknown_name_yields_nothing() -> None:
    g = _seed_graph()
    assert resolve_entity_ids(g, ["nobody"]) == []


# --------------------------------------------------------------------------
# Req 15.1 — project owner via incoming OWNS edges
# --------------------------------------------------------------------------
def test_owner_query_returns_owner_via_owns() -> None:
    g = _seed_graph()
    hits = SymbolicRetriever().retrieve(_cls(["Orion"], [OWNS]), g)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.predicate == OWNS
    assert hit.subject_id == "p_alice"   # owner
    assert hit.object_id == "proj_orion"
    assert hit.assertion_id == "a_owns"
    assert hit.exact_match is True


# --------------------------------------------------------------------------
# Req 15.2 — task assignee via outgoing ASSIGNED_TO edge
# --------------------------------------------------------------------------
def test_assignee_query_returns_assignee_via_assigned_to() -> None:
    g = _seed_graph()
    hits = SymbolicRetriever().retrieve(_cls(["T1"], [ASSIGNED_TO]), g)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.predicate == ASSIGNED_TO
    assert hit.subject_id == "task_t1"
    assert hit.object_id == "p_bob"      # assignee
    assert hit.assertion_id == "a_assigned"
    assert hit.exact_match is True


# --------------------------------------------------------------------------
# Req 15.3 — preceding events via incoming PRECEDES edges
# --------------------------------------------------------------------------
def test_preceding_query_returns_predecessors_via_precedes() -> None:
    g = _seed_graph()
    hits = SymbolicRetriever().retrieve(_cls(["e2"], [PRECEDES], query_type="temporal"), g)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.predicate == PRECEDES
    assert hit.subject_id == "ev_e1"     # the preceding event
    assert hit.object_id == "ev_e2"
    assert hit.assertion_id == "a_precedes"
    assert hit.exact_match is True


def test_preceding_query_on_first_event_has_no_predecessors() -> None:
    g = _seed_graph()
    hits = SymbolicRetriever().retrieve(_cls(["e1"], [PRECEDES], query_type="temporal"), g)
    assert hits == []


# --------------------------------------------------------------------------
# Predicate inference when the classifier extracted no predicates
# --------------------------------------------------------------------------
def test_inference_without_explicit_predicates() -> None:
    g = _seed_graph()
    # direct_fact + Project entity -> OWNS inferred.
    owner_hits = SymbolicRetriever().retrieve(_cls(["Orion"], []), g)
    assert [h.subject_id for h in owner_hits] == ["p_alice"]
    # temporal + Event entity -> PRECEDES inferred.
    prec_hits = SymbolicRetriever().retrieve(_cls(["e2"], [], query_type="temporal"), g)
    assert [h.subject_id for h in prec_hits] == ["ev_e1"]


def test_memory_id_aliases_assertion_id() -> None:
    g = _seed_graph()
    hit = SymbolicRetriever().retrieve(_cls(["Orion"], [OWNS]), g)[0]
    assert hit.memory_id == hit.assertion_id == "a_owns"


# --------------------------------------------------------------------------
# Property: every symbolic hit is an exact match with a backing assertion
# and a confidence in [0, 1] (Req 15.4).
# Validates: Requirements 15.4
# --------------------------------------------------------------------------
@given(
    name=st.sampled_from(["Orion", "T1", "e2"]),
    predicates=st.lists(st.sampled_from([OWNS, ASSIGNED_TO, PRECEDES]), max_size=3),
)
def test_property_hits_are_exact_with_backing_assertion(name, predicates) -> None:
    g = _seed_graph()
    qt = "temporal" if name == "e2" else "direct_fact"
    hits = SymbolicRetriever().retrieve(_cls([name], predicates, query_type=qt), g)
    for hit in hits:
        assert isinstance(hit, SymbolicHit)
        assert hit.exact_match is True
        assert hit.assertion_id  # non-empty backing assertion id
        assert 0.0 <= hit.confidence <= 1.0

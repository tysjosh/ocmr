"""Unit tests for the Assertion_Builder (W4) and the manual write path (task 4.3).

Validates: Requirements 6.1, 6.2, 6.3, 11.6

Two clusters of example-based tests:

* :class:`AssertionBuilder` turns a normalized relation plus its resolved
  subject/object into a :class:`CandidateAssertion`: the operation is always
  ``upsert_assertion`` (Req 6.1); ``subject_id``, ``predicate``, ``object_id``,
  ``confidence``, ``source_ref``, and ``write_intent`` are populated (Req 6.2);
  and ``write_intent`` defaults to ``new_fact`` when the relation omits it
  (Req 6.3) while an explicit intent (e.g. ``correction``) is honored.
* :func:`manual_write` persists pre-resolved entities and one **accepted**
  assertion through :class:`SQLiteRepository` (``":memory:"``) and reflects the
  accepted assertion as an edge in the :class:`GraphStore` (Req 11.6), with the
  returned assertion round-tripping back out of the repository.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ocm.core.ids import IdGenerator
from ocm.memory.assertion_builder import AssertionBuilder
from ocm.memory.contracts import CandidateAssertion, ResolutionOutcome
from ocm.memory.graph_store import GraphStore
from ocm.memory.manual_write import (
    ResolvedEntity,
    assertion_from_candidate,
    manual_write,
)
from ocm.memory.sqlite_repository import SQLiteRepository
from ocm.ontology.enums import (
    AssertionStatus,
    PersonStatus,
    ProjectStatus,
    ResolutionStatus,
    WriteIntent,
)
from ocm.ontology.models import Assertion, Person, Project

# A fixed timestamp keeps the assertion round-trip deterministic.
TS = datetime(2024, 5, 6, 7, 8, 9, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def repo():
    """A hermetic in-memory repository, closed after each test."""
    r = SQLiteRepository(":memory:")
    yield r
    r.close()


@pytest.fixture
def graph() -> GraphStore:
    return GraphStore()


@pytest.fixture
def ids() -> IdGenerator:
    """Deterministic ids so generated assertion ids are reproducible."""
    return IdGenerator(deterministic=True)


def _resolved(entity_id: str) -> ResolutionOutcome:
    return ResolutionOutcome(
        resolution_status=ResolutionStatus.resolved_existing,
        entity_id=entity_id,
    )


def _person(pid: str = "person:1") -> Person:
    return Person(id=pid, name="Ada Lovelace", status=PersonStatus.active)


def _project(pid: str = "project:1", owner: str = "person:1") -> Project:
    return Project(id=pid, name="Analytical Engine", status=ProjectStatus.active, owner_id=owner)


# ---------------------------------------------------------------------------
# 1. AssertionBuilder.build (Req 6.1, 6.2, 6.3)
# ---------------------------------------------------------------------------
def test_build_populates_all_fields_and_upsert_operation() -> None:
    """build sets operation=upsert_assertion and all Req 6.2 fields (Req 6.1, 6.2)."""
    builder = AssertionBuilder()
    relation = {
        "subject": "Ada",
        "predicate": "OWNS",
        "object": "AnalyticalEngine",
        "confidence": 0.83,
    }
    resolved = {
        "Ada": _resolved("person:1"),
        "AnalyticalEngine": _resolved("project:1"),
    }

    candidate = builder.build(relation, resolved, source_ref="doc:42")

    assert isinstance(candidate, CandidateAssertion)
    assert candidate.operation == "upsert_assertion"
    assert candidate.subject_id == "person:1"
    assert candidate.predicate == "OWNS"
    assert candidate.object_id == "project:1"
    assert candidate.confidence == 0.83
    assert candidate.source_ref == "doc:42"
    # write_intent populated (Req 6.2) — defaulted here (Req 6.3).
    assert candidate.write_intent == WriteIntent.new_fact


def test_build_defaults_write_intent_to_new_fact() -> None:
    """When the relation omits write_intent, it defaults to new_fact (Req 6.3)."""
    builder = AssertionBuilder()
    relation = {"subject": "s", "predicate": "OWNS", "object": "o", "confidence": 0.5}
    resolved = {"s": _resolved("person:1"), "o": _resolved("project:1")}

    candidate = builder.build(relation, resolved, source_ref="doc:1")

    assert candidate.write_intent == WriteIntent.new_fact


def test_build_uses_provided_write_intent() -> None:
    """An explicit write_intent (e.g. correction) is honored (Req 6.2)."""
    builder = AssertionBuilder()
    relation = {
        "subject": "s",
        "predicate": "OWNS",
        "object": "o",
        "confidence": 0.9,
        "write_intent": "correction",
    }
    resolved = {"s": _resolved("person:1"), "o": _resolved("project:1")}

    candidate = builder.build(relation, resolved, source_ref="doc:1")

    assert candidate.write_intent == WriteIntent.correction


def test_build_accepts_write_intent_enum_value() -> None:
    """A WriteIntent enum passed through the relation is preserved (Req 6.2)."""
    builder = AssertionBuilder()
    relation = {
        "subject": "s",
        "predicate": "OWNS",
        "object": "o",
        "confidence": 0.7,
        "write_intent": WriteIntent.hypothesis,
    }
    resolved = {"s": _resolved("person:1"), "o": _resolved("project:1")}

    candidate = builder.build(relation, resolved, source_ref="doc:1")

    assert candidate.write_intent == WriteIntent.hypothesis


def test_build_uses_relation_source_ref_when_argument_absent() -> None:
    """source_ref carried in the relation is used when no argument is given (Req 6.2)."""
    builder = AssertionBuilder()
    relation = {
        "subject": "s",
        "predicate": "OWNS",
        "object": "o",
        "confidence": 0.6,
        "source_ref": "doc:99",
    }
    resolved = {"s": _resolved("person:1"), "o": _resolved("project:1")}

    candidate = builder.build(relation, resolved)

    assert candidate.source_ref == "doc:99"


def test_build_raises_when_no_source_ref_available() -> None:
    """A missing source_ref (neither arg nor relation) is a ValueError (Req 6.2)."""
    builder = AssertionBuilder()
    relation = {"subject": "s", "predicate": "OWNS", "object": "o", "confidence": 0.6}
    resolved = {"s": _resolved("person:1"), "o": _resolved("project:1")}

    with pytest.raises(ValueError):
        builder.build(relation, resolved)


def test_build_raises_on_unresolved_end() -> None:
    """An unresolved subject/object (entity_id is None) cannot build an assertion."""
    builder = AssertionBuilder()
    relation = {"subject": "s", "predicate": "OWNS", "object": "o", "confidence": 0.6}
    resolved = {
        "s": ResolutionOutcome(
            resolution_status=ResolutionStatus.unresolved, entity_id=None
        ),
        "o": _resolved("project:1"),
    }

    with pytest.raises(ValueError):
        builder.build(relation, resolved, source_ref="doc:1")


# ---------------------------------------------------------------------------
# 2. manual_write persists + graph-reflects an accepted assertion (Req 11.6)
# ---------------------------------------------------------------------------
def _candidate() -> CandidateAssertion:
    return CandidateAssertion(
        subject_id="person:1",
        predicate="OWNS",
        object_id="project:1",
        confidence=0.91,
        source_ref="doc:42",
        write_intent=WriteIntent.new_fact,
    )


def test_assertion_from_candidate_forces_accepted_status(ids: IdGenerator) -> None:
    """The minimal path promotes a candidate to an accepted Assertion."""
    assertion = assertion_from_candidate(_candidate(), ids, created_at=TS)

    assert isinstance(assertion, Assertion)
    assert assertion.status == AssertionStatus.accepted
    assert assertion.subject_id == "person:1"
    assert assertion.predicate == "OWNS"
    assert assertion.object_id == "project:1"
    assert assertion.confidence == 0.91
    assert assertion.source_ref == "doc:42"
    assert assertion.created_at == TS
    assert assertion.id  # an id was minted


def test_manual_write_persists_and_reflects_accepted_assertion(
    repo: SQLiteRepository, graph: GraphStore, ids: IdGenerator
) -> None:
    """manual_write persists entities + an accepted assertion and reflects it
    as an accepted edge in the graph; the assertion round-trips (Req 11.6)."""
    entities = [
        ResolvedEntity(entity_type="Person", entity=_person()),
        ("Project", _project()),  # tuple form is also accepted
    ]

    assertion = manual_write(entities, _candidate(), repo, graph, ids, created_at=TS)

    # Returned assertion is accepted.
    assert assertion.status == AssertionStatus.accepted

    # Entities persisted in the repository.
    person_row = repo.get_entity("person:1")
    project_row = repo.get_entity("project:1")
    assert person_row is not None and person_row[0] == "Person"
    assert project_row is not None and project_row[0] == "Project"

    # Assertion persisted and round-trips equal to the returned one.
    assert repo.get_assertion(assertion.id) == assertion

    # It is the only accepted assertion in the repo.
    accepted = list(repo.list_assertions(status=AssertionStatus.accepted.value))
    assert accepted == [assertion]

    # Graph reflects entities as nodes and the assertion as an accepted edge.
    assert graph.node_ids() == {"person:1", "project:1"}
    assert graph.has_assertion("person:1", "project:1", "OWNS")
    assert graph.edge_triples() == {("person:1", "project:1", "OWNS")}

    edge = graph.get_assertion_edge("person:1", "project:1", "OWNS")
    assert edge is not None
    assert edge["assertion_id"] == assertion.id
    assert edge["status"] == AssertionStatus.accepted.value


def test_manual_write_graph_edge_matches_repo_accepted_invariant(
    repo: SQLiteRepository, graph: GraphStore, ids: IdGenerator
) -> None:
    """The graph's edges equal the repo's accepted assertion rows (Req 11.6)."""
    entities = [("Person", _person()), ("Project", _project())]
    manual_write(entities, _candidate(), repo, graph, ids, created_at=TS)

    accepted = list(repo.list_assertions(status=AssertionStatus.accepted.value))
    expected = {(a.subject_id, a.object_id, a.predicate) for a in accepted}
    assert graph.edge_triples() == expected

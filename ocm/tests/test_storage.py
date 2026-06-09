"""Unit tests for repository persistence and graph rebuild (task 3.4).

Validates: Requirements 11.1, 11.5, 11.8

These example-based tests complement the property-based round-trip test in
``test_prop_schema_roundtrip.py`` by exercising the *storage layer* end to end:

* Concrete models (Person, Project, Assertion, QuarantineRecord, Provenance)
  survive a write/read cycle through :class:`SQLiteRepository` ``":memory:"``
  and come back equal (Req 11.1 — the seven tables persist every memory kind).
* :func:`rebuild_graph` projects ONLY ``accepted`` assertions as edges, so
  quarantined / rejected / superseded assertions never enter the graph
  (Req 11.5).
* The rebuilt graph equals the pre-restart accepted state and the rebuild is
  deterministic across repeated runs (Req 11.8).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ocm.memory.graph_store import rebuild_graph
from ocm.memory.sqlite_repository import SQLiteRepository
from ocm.ontology.enums import (
    AssertionStatus,
    PersonStatus,
    ProjectStatus,
    QuarantineStatus,
    Severity,
    WriteIntent,
)
from ocm.ontology.models import (
    Assertion,
    Person,
    Project,
    Provenance,
    QuarantineRecord,
)

# A fixed timestamp keeps the examples deterministic and JSON round-trippable.
TS = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


@pytest.fixture
def repo():
    """A hermetic in-memory repository, closed after each test."""
    r = SQLiteRepository(":memory:")
    yield r
    r.close()


def _person(pid: str = "person:1") -> Person:
    return Person(
        id=pid,
        name="Ada Lovelace",
        roles=["engineer", "lead"],
        status=PersonStatus.active,
        aliases=["Ada"],
    )


def _project(pid: str = "project:1", owner: str = "person:1") -> Project:
    return Project(
        id=pid,
        name="Analytical Engine",
        goal="Compute Bernoulli numbers",
        status=ProjectStatus.active,
        owner_id=owner,
    )


def _assertion(aid: str, status: AssertionStatus) -> Assertion:
    return Assertion(
        id=aid,
        subject_id="person:1",
        predicate="OWNS",
        object_id="project:1",
        confidence=0.91,
        status=status,
        source_ref="doc:42",
        created_at=TS,
        write_intent=WriteIntent.new_fact,
    )


# ---------------------------------------------------------------------------
# 1. Model round-trips through SQLite (Req 11.1).
# ---------------------------------------------------------------------------
def test_entity_round_trip_person_and_project(repo: SQLiteRepository) -> None:
    """Person and Project entities survive upsert/get equal to the original."""
    person = _person()
    project = _project()
    repo.upsert_entity("Person", person)
    repo.upsert_entity("Project", project)

    got_person = repo.get_entity(person.id)
    got_project = repo.get_entity(project.id)
    assert got_person is not None and got_project is not None

    person_type, person_payload = got_person
    project_type, project_payload = got_project
    assert person_type == "Person"
    assert project_type == "Project"
    assert Person.model_validate(person_payload) == person
    assert Project.model_validate(project_payload) == project


def test_entity_missing_returns_none(repo: SQLiteRepository) -> None:
    """A lookup for an unknown id returns None rather than raising."""
    assert repo.get_entity("nope") is None


def test_assertion_round_trip(repo: SQLiteRepository) -> None:
    """An Assertion survives upsert/get equal to the original."""
    a = _assertion("assertion:1", AssertionStatus.accepted)
    repo.upsert_assertion(a)
    assert repo.get_assertion(a.id) == a


def test_quarantine_round_trip(repo: SQLiteRepository) -> None:
    """A QuarantineRecord (incl. JSON payload) round-trips and persists (Req 11.7)."""
    q = QuarantineRecord(
        id="quarantine:1",
        candidate_payload={"predicate": "OWNS", "confidence": 0.4, "tags": ["a"]},
        reason="low confidence",
        severity=Severity.medium,
        conflicting_ids=["assertion:9"],
        created_at=TS,
        status=QuarantineStatus.unresolved,
    )
    repo.upsert_quarantine(q)
    records = list(repo.list_quarantine())
    assert records == [q]
    # status filter returns the same record
    assert list(repo.list_quarantine(status=QuarantineStatus.unresolved.value)) == [q]


def test_provenance_round_trip(repo: SQLiteRepository) -> None:
    """A Provenance record round-trips equal, queryable by subject (Req 12.4)."""
    p = Provenance(
        id="prov:1",
        subject_id="assertion:1",
        source_ref="doc:42",
        created_at=TS,
        extractor_version="mock-1",
        supporting_evidence_ids=["claim:1", "claim:2"],
    )
    repo.upsert_provenance(p)
    assert repo.get_provenance_for("assertion:1") == [p]


# ---------------------------------------------------------------------------
# 2. Graph rebuild is accepted-only (Req 11.5, 11.8).
# ---------------------------------------------------------------------------
def _seed_mixed_status_repo(repo: SQLiteRepository) -> Assertion:
    """Seed entities + one assertion of every status; return the accepted one."""
    repo.upsert_entity("Person", _person())
    repo.upsert_entity("Project", _project())

    accepted = _assertion("assertion:accepted", AssertionStatus.accepted)
    repo.upsert_assertion(accepted)
    repo.upsert_assertion(_assertion("assertion:quarantined", AssertionStatus.quarantined))
    repo.upsert_assertion(_assertion("assertion:rejected", AssertionStatus.rejected))
    repo.upsert_assertion(_assertion("assertion:superseded", AssertionStatus.superseded))
    return accepted


def test_rebuild_graph_contains_only_accepted_assertions(repo: SQLiteRepository) -> None:
    """rebuild_graph projects accepted assertions only (Req 11.5, 11.8)."""
    accepted = _seed_mixed_status_repo(repo)

    graph = rebuild_graph(repo)

    # All entities are present as nodes.
    assert graph.node_ids() == {"person:1", "project:1"}
    # Exactly the accepted assertion appears as an edge; the quarantined,
    # rejected, and superseded assertions are excluded.
    assert graph.edge_triples() == {
        (accepted.subject_id, accepted.object_id, accepted.predicate)
    }
    assert graph.num_edges() == 1


def test_rebuild_graph_with_no_accepted_has_no_edges(repo: SQLiteRepository) -> None:
    """If nothing is accepted, the rebuilt graph has nodes but no edges."""
    repo.upsert_entity("Person", _person())
    repo.upsert_entity("Project", _project())
    repo.upsert_assertion(_assertion("assertion:q", AssertionStatus.quarantined))
    repo.upsert_assertion(_assertion("assertion:r", AssertionStatus.rejected))

    graph = rebuild_graph(repo)
    assert graph.node_ids() == {"person:1", "project:1"}
    assert graph.edge_triples() == set()


def test_rebuild_skips_dangling_accepted_assertion(repo: SQLiteRepository) -> None:
    """An accepted assertion with a missing endpoint is skipped, not fatal."""
    repo.upsert_entity("Person", _person())
    # Note: project entity intentionally NOT stored -> object_id is dangling.
    repo.upsert_assertion(_assertion("assertion:dangling", AssertionStatus.accepted))

    graph = rebuild_graph(repo)
    assert graph.node_ids() == {"person:1"}
    assert graph.edge_triples() == set()


# ---------------------------------------------------------------------------
# 3. Rebuild is deterministic and equals the pre-restart accepted state (Req 11.8).
# ---------------------------------------------------------------------------
def test_rebuild_is_deterministic_and_equals_pre_restart_state(
    repo: SQLiteRepository,
) -> None:
    """Repeated rebuilds yield identical node/edge sets (deterministic)."""
    _seed_mixed_status_repo(repo)
    # A second accepted assertion (different predicate) to make the graph richer.
    repo.upsert_assertion(
        Assertion(
            id="assertion:assigned",
            subject_id="project:1",
            predicate="ASSIGNED_TO",
            object_id="person:1",
            confidence=0.8,
            status=AssertionStatus.accepted,
            source_ref="doc:7",
            created_at=TS,
        )
    )

    first = rebuild_graph(repo)
    second = rebuild_graph(repo)

    assert first.node_ids() == second.node_ids()
    assert first.edge_triples() == second.edge_triples()
    assert first.edge_triples() == {
        ("person:1", "project:1", "OWNS"),
        ("project:1", "person:1", "ASSIGNED_TO"),
    }


def test_rebuild_equals_accepted_assertions_in_repo(repo: SQLiteRepository) -> None:
    """The graph's edge set matches exactly the repo's accepted assertions.

    This is the standing storage invariant: graph edges == accepted rows. A
    post-restart rebuild therefore reproduces the pre-restart accepted state.
    """
    _seed_mixed_status_repo(repo)

    graph = rebuild_graph(repo)

    accepted = list(repo.list_assertions(status=AssertionStatus.accepted.value))
    expected = {(a.subject_id, a.object_id, a.predicate) for a in accepted}
    assert graph.edge_triples() == expected

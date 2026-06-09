"""Quick verification checks for the Commit Manager (W8 routing, task 8.2).

Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 12.3, 2.13

Exercises each routing leg against an in-memory SQLite repository + graph:
accept persists/graphs/embeds/provenances; supersede marks old superseded,
adds a SUPERSEDES edge, preserves provenance for both, and leaves exactly one
accepted (non-SUPERSEDES) edge between the pair; quarantine writes a record and
touches nothing in the graph; reject touches nothing.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ocm.core.ids import IdGenerator
from ocm.memory.commit_manager import SUPERSEDES, CommitManager
from ocm.memory.contracts import CandidateAssertion, ValidationResult
from ocm.memory.graph_store import GraphStore
from ocm.memory.provenance_tracker import ProvenanceTracker
from ocm.memory.quarantine_store import QuarantineStore
from ocm.memory.sqlite_repository import SQLiteRepository
from ocm.ontology.enums import AssertionStatus, Severity, WriteIntent
from ocm.ontology.models import Person, Project

TS = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


@pytest.fixture
def repo():
    r = SQLiteRepository(":memory:")
    yield r
    r.close()


@pytest.fixture
def graph(repo):
    g = GraphStore()
    # Two entities so OWNS subject/object resolve to real nodes.
    person = Person(id="per_1", name="Ada")
    project = Project(id="prj_1", name="OCM")
    for etype, ent in (("Person", person), ("Project", project)):
        repo.upsert_entity(etype, ent)
        g.add_entity(etype, ent)
    return g


@pytest.fixture
def ids():
    return IdGenerator(deterministic=True)


@pytest.fixture
def manager(repo, graph, ids):
    embedded: list[str] = []
    cm = CommitManager(
        repo=repo,
        graph=graph,
        ids=ids,
        quarantine_store=QuarantineStore(repo, ids),
        provenance_tracker=ProvenanceTracker(repo, ids),
        embed_hook=lambda a: embedded.append(a.id),
    )
    cm._embedded = embedded  # type: ignore[attr-defined]
    return cm


def _candidate(write_intent=WriteIntent.new_fact) -> CandidateAssertion:
    return CandidateAssertion(
        subject_id="per_1",
        predicate="OWNS",
        object_id="prj_1",
        confidence=0.9,
        source_ref="doc://notes#1",
        write_intent=write_intent,
        extractor_version="mock-1",
    )


def test_accept_persists_graphs_embeds_and_provenances(manager, repo, graph):
    vr = ValidationResult(valid=True, recommended_action="accept")
    outcome = manager.commit(_candidate(), vr, created_at=TS)

    assert outcome.decision == "accepted"
    aid = outcome.assertion_id
    # Persisted as accepted row.
    stored = repo.get_assertion(aid)
    assert stored is not None and stored.status is AssertionStatus.accepted
    # Reflected as a graph edge.
    assert graph.has_assertion("per_1", "prj_1", "OWNS")
    # Embedded via the hook.
    assert manager._embedded == [aid]
    # Provenance recorded.
    assert len(manager.provenance_tracker.for_subject(aid)) == 1


def test_supersede_marks_old_links_and_preserves_both_provenance(manager, repo, graph):
    # First accept an original assertion.
    first = manager.commit(_candidate(), ValidationResult(valid=True), created_at=TS)
    old_id = first.assertion_id

    # Now a correction superseding the original.
    correction = _candidate(write_intent=WriteIntent.correction)
    vr = ValidationResult(
        valid=True,
        recommended_action="supersede",
        conflicting_ids=[old_id],
        reason="correction of prior fact",
    )
    outcome = manager.commit(correction, vr, created_at=TS)

    assert outcome.decision == "superseded"
    assert outcome.superseded_assertion_id == old_id
    new_id = outcome.assertion_id

    # Old flipped to superseded; new is accepted.
    assert repo.get_assertion(old_id).status is AssertionStatus.superseded
    assert repo.get_assertion(new_id).status is AssertionStatus.accepted

    # A SUPERSEDES edge new -> old exists in the graph.
    assert graph.has_assertion(new_id, old_id, SUPERSEDES)

    # Exactly one accepted OWNS edge remains between the pair (the new one).
    owns_edges = graph.find_edges_by_predicate("OWNS")
    assert len(owns_edges) == 1

    # Provenance preserved for BOTH old and new.
    assert len(manager.provenance_tracker.for_subject(old_id)) >= 1
    assert len(manager.provenance_tracker.for_subject(new_id)) >= 1


def test_quarantine_writes_record_and_adds_nothing_to_graph(manager, graph):
    before_edges = graph.num_edges()
    vr = ValidationResult(
        valid=False,
        recommended_action="quarantine",
        reason="soft contradiction",
        severity=Severity.medium,
        conflicting_ids=["asr_x"],
    )
    outcome = manager.commit(_candidate(), vr, created_at=TS)

    assert outcome.decision == "quarantined"
    assert outcome.quarantine_id is not None
    # A quarantine record exists.
    records = manager.quarantine_store.list()
    assert len(records) == 1
    assert records[0].reason == "soft contradiction"
    # Nothing added to the graph as accepted.
    assert graph.num_edges() == before_edges
    assert not graph.has_assertion("per_1", "prj_1", "OWNS")


def test_reject_adds_nothing(manager, graph, repo):
    before_edges = graph.num_edges()
    vr = ValidationResult(
        valid=False,
        recommended_action="reject",
        failed_check="C6",
        reason="confidence out of bounds",
    )
    outcome = manager.commit(_candidate(), vr, created_at=TS)

    assert outcome.decision == "rejected"
    assert outcome.reason == "confidence out of bounds"
    # Nothing in the graph, nothing accepted.
    assert graph.num_edges() == before_edges
    assert list(repo.list_assertions(status="accepted")) == []

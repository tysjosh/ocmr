"""Governance integration tests for the write path (task 8.9).

Validates: Requirements 10.2, 10.3, 26.4, 28.5, 28.6

Unlike ``test_commit_manager_quickcheck`` (which drives the Commit Manager's
routing legs with hand-built :class:`ValidationResult` verdicts), these tests
wire the *real* governance stack together end to end:

    Constraint_Validator (W6, C7) -> Contradiction_Checker (W7) -> Commit_Manager (W8)

against a live :class:`GraphStore` / :class:`SQLiteRepository`. They exercise the
two contradiction outcomes the governance layer must distinguish and confirm the
``Quarantine_Store`` surfaces unresolved conflicts:

* A high-confidence ``new_fact`` that contradicts an accepted single-valued
  ``ASSIGNED_TO`` is detected by C7 (delegating to W7), routed to
  ``quarantine``, and the candidate never enters accepted memory; the resulting
  record is retrievable from ``Quarantine_Store.list(status=unresolved)``
  (Req 10.3, 26.4, 28.6).
* The same conflict carried as a ``correction`` is routed to ``supersede``: the
  prior assertion flips to ``superseded``, the correction is accepted, exactly
  one accepted ``ASSIGNED_TO`` remains, and a ``SUPERSEDES`` edge links new->old
  (Req 10.2, 28.5).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ocm.core.ids import IdGenerator
from ocm.memory.commit_manager import SUPERSEDES, CommitManager
from ocm.memory.contracts import CandidateAssertion
from ocm.memory.graph_store import GraphStore
from ocm.memory.manual_write import manual_write
from ocm.memory.provenance_tracker import ProvenanceTracker
from ocm.memory.quarantine_store import QuarantineStore
from ocm.memory.sqlite_repository import SQLiteRepository
from ocm.ontology.enums import AssertionStatus, PersonStatus, QuarantineStatus
from ocm.ontology.models import Person, Task
from ocm.validation.constraints import ConstraintValidator

TS = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
# Incumbent confidence: above the contradiction threshold (0.8) but low enough
# that a dominating correction (>= 0.96) clears the Algorithm 1 margin (0.1).
HIGH = 0.85


@pytest.fixture
def repo():
    r = SQLiteRepository(":memory:")
    yield r
    r.close()


@pytest.fixture
def ids():
    return IdGenerator(deterministic=True)


@pytest.fixture
def settings(deterministic_settings):
    """Deterministic offline Settings (carries contradiction_high_confidence=0.8)."""
    return deterministic_settings


@pytest.fixture
def graph(repo):
    """Graph + repo seeded with Task t1 and two active people A and B."""
    g = GraphStore()
    entities = (
        ("Task", Task(id="t1", title="Ship OCM")),
        ("Person", Person(id="per_a", name="Ada", status=PersonStatus.active)),
        ("Person", Person(id="per_b", name="Bob", status=PersonStatus.active)),
    )
    for etype, ent in entities:
        repo.upsert_entity(etype, ent)
        g.add_entity(etype, ent)
    return g


@pytest.fixture
def validator(settings):
    """A ConstraintValidator with its default (real) Contradiction_Checker (W7)."""
    return ConstraintValidator(settings)


@pytest.fixture
def manager(repo, graph, ids):
    return CommitManager(
        repo=repo,
        graph=graph,
        ids=ids,
        quarantine_store=QuarantineStore(repo, ids),
        provenance_tracker=ProvenanceTracker(repo, ids),
    )


@pytest.fixture
def seeded_assignment(repo, graph, ids):
    """Seed an accepted high-confidence ASSIGNED_TO: t1 -> Person A.

    Returns the accepted assertion's id so tests can assert it is the one that
    gets superseded / flagged as conflicting.
    """
    seed = CandidateAssertion(
        subject_id="t1",
        predicate="ASSIGNED_TO",
        object_id="per_a",
        confidence=HIGH,
        source_ref="doc://seed#1",
        extractor_version="mock-1",
    )
    assertion = manual_write([], seed, repo, graph, ids, created_at=TS)
    return assertion.id


def _candidate_to_b(write_intent="new_fact", confidence: float = HIGH) -> CandidateAssertion:
    """A high-confidence ASSIGNED_TO pointing t1 at the *other* person, B."""
    return CandidateAssertion(
        subject_id="t1",
        predicate="ASSIGNED_TO",
        object_id="per_b",
        confidence=confidence,
        source_ref="doc://notes#2",
        write_intent=write_intent,
        extractor_version="mock-1",
    )


# --------------------------------------------------------------------------- #
# Scenario 1 — high-confidence new_fact contradiction is quarantined (C7).
# --------------------------------------------------------------------------- #
def test_high_confidence_contradiction_is_quarantined(
    seeded_assignment, validator, manager, graph, settings
):
    """new_fact conflicting with an accepted single-valued edge -> quarantine.

    Validates: Requirements 10.3, 26.4, 28.6
    """
    candidate = _candidate_to_b(write_intent="new_fact")

    # C7 (via W7) detects the single-valued ASSIGNED_TO conflict and, because
    # both sides are high-confidence new facts, recommends quarantine (Req 26.4).
    vr = validator.validate(candidate, graph, settings=settings)
    assert vr.valid is False
    assert vr.failed_check == "C7"
    assert vr.recommended_action == "quarantine"
    assert seeded_assignment in vr.conflicting_ids

    before_edges = graph.num_edges()
    outcome = manager.commit(candidate, vr, created_at=TS)

    # The candidate is quarantined and never enters accepted memory (Req 10.3).
    assert outcome.decision == "quarantined"
    assert outcome.quarantine_id is not None
    assert graph.num_edges() == before_edges
    assert not graph.has_assertion("t1", "per_b", "ASSIGNED_TO")
    # The original accepted assignment is untouched.
    assert graph.has_assertion("t1", "per_a", "ASSIGNED_TO")


def test_quarantine_store_lists_unresolved_conflict(
    seeded_assignment, validator, manager, graph, settings
):
    """The quarantined contradiction is retrievable as an unresolved conflict.

    Validates: Requirements 10.3, 28.6
    """
    candidate = _candidate_to_b(write_intent="new_fact")
    vr = validator.validate(candidate, graph, settings=settings)
    outcome = manager.commit(candidate, vr, created_at=TS)

    # Retrieving unresolved conflicts from the Quarantine_Store works (Req 28.6).
    unresolved = manager.quarantine_store.list(status=QuarantineStatus.unresolved)
    assert len(unresolved) == 1
    record = unresolved[0]
    assert record.id == outcome.quarantine_id
    assert record.status is QuarantineStatus.unresolved
    assert seeded_assignment in record.conflicting_ids
    # The quarantined candidate payload is preserved for review.
    assert record.candidate_payload["object_id"] == "per_b"


# --------------------------------------------------------------------------- #
# Scenario 2 — a correction supersedes the prior assertion (Req 10.2, 28.5).
# --------------------------------------------------------------------------- #
def test_correction_supersedes_prior_assignment(
    seeded_assignment, validator, manager, repo, graph, settings
):
    """A high-confidence correction supersedes the accepted assignment.

    Validates: Requirements 10.2, 28.5
    """
    correction = _candidate_to_b(write_intent="correction", confidence=0.99)

    # The same conflict, carried as a correction, is routed to supersede (Req 10.2).
    vr = validator.validate(correction, graph, settings=settings)
    assert vr.valid is False
    assert vr.failed_check == "C7"
    assert vr.recommended_action == "supersede"
    assert seeded_assignment in vr.conflicting_ids

    outcome = manager.commit(correction, vr, created_at=TS)
    assert outcome.decision == "superseded"
    assert outcome.superseded_assertion_id == seeded_assignment
    new_id = outcome.assertion_id

    # Old assertion flipped to superseded; the correction is accepted.
    assert repo.get_assertion(seeded_assignment).status is AssertionStatus.superseded
    assert repo.get_assertion(new_id).status is AssertionStatus.accepted

    # Exactly one accepted ASSIGNED_TO edge remains (the correction, t1 -> B).
    assigned_edges = graph.find_edges_by_predicate("ASSIGNED_TO")
    assert len(assigned_edges) == 1
    assert assigned_edges[0][1] == "per_b"
    assert not graph.has_assertion("t1", "per_a", "ASSIGNED_TO")

    # A SUPERSEDES edge links the new assertion to the old one (Req 28.5).
    assert graph.has_assertion(new_id, seeded_assignment, SUPERSEDES)

    # Provenance is preserved for both sides of the supersession.
    assert len(manager.provenance_tracker.for_subject(seeded_assignment)) >= 1
    assert len(manager.provenance_tracker.for_subject(new_id)) >= 1


def test_correction_then_unresolved_conflicts_remain_empty(
    seeded_assignment, validator, manager, graph, settings
):
    """A clean supersession leaves no unresolved quarantine conflicts.

    Validates: Requirements 10.2, 28.6
    """
    correction = _candidate_to_b(write_intent="correction", confidence=0.99)
    vr = validator.validate(correction, graph, settings=settings)
    manager.commit(correction, vr, created_at=TS)

    # Supersession does not create a quarantine record.
    assert manager.quarantine_store.list(status=QuarantineStatus.unresolved) == []


# --------------------------------------------------------------------------- #
# Scenario 3 — Algorithm 1 supersede admissibility (delta margin + e_min).
# --------------------------------------------------------------------------- #
def test_correction_not_dominating_incumbent_is_quarantined(
    seeded_assignment, validator, manager, graph, settings
):
    """A correction that does not exceed the incumbent's confidence by the
    margin (here it ties at 0.95) is quarantined, not superseded (Algorithm 1
    line 7: incumbent dominates ⇒ quarantine)."""
    correction = _candidate_to_b(write_intent="correction", confidence=HIGH)  # ties the seed

    vr = validator.validate(correction, graph, settings=settings)

    assert vr.valid is False
    assert vr.failed_check == "C7"
    assert vr.recommended_action == "quarantine"
    assert seeded_assignment in vr.conflicting_ids

    outcome = manager.commit(correction, vr, created_at=TS)
    assert outcome.decision == "quarantined"
    # The incumbent is untouched: exactly one accepted ASSIGNED_TO, still t1 -> A.
    assigned_edges = graph.find_edges_by_predicate("ASSIGNED_TO")
    assert len(assigned_edges) == 1
    assert assigned_edges[0][1] == "per_a"


def test_correction_without_evidence_is_quarantined(
    seeded_assignment, validator, manager, graph, settings
):
    """A dominating correction that carries no evidence (no source_ref, no
    EVIDENCE_FOR) fails the e_min floor and is quarantined (Algorithm 1)."""
    correction = CandidateAssertion(
        subject_id="t1",
        predicate="ASSIGNED_TO",
        object_id="per_b",
        confidence=0.99,  # dominates the 0.95 incumbent on confidence
        source_ref="",  # ...but has no provenance/evidence (evidence count 0)
        write_intent="correction",
        extractor_version="mock-1",
    )

    vr = validator.validate(correction, graph, settings=settings)

    assert vr.valid is False
    assert vr.recommended_action == "quarantine"
    assert seeded_assignment in vr.conflicting_ids

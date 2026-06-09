"""Property 10: ASSIGNED_TO single-valued (m:1) invariant.

Feature: ontology-constrained-memory, Property 10

Validates Requirements 2.5, 9.5.

``ASSIGNED_TO`` is a many-to-one (``m:1``) relation: a Task may point at exactly
one accepted assignee. This property drives the *real* governance stack —
``ConstraintValidator`` (C7) -> ``ContradictionChecker`` (W7) -> ``CommitManager``
(W8) — over a live ``GraphStore`` / ``SQLiteRepository(":memory:")`` and asserts
the invariant holds for *any* stream of distinct assignees:

* For a fixed Task ``t1`` and a generated list of distinct Person ids (all seeded
  as active entities), we submit high-confidence ``ASSIGNED_TO`` ``new_fact``
  candidates one at a time through validate + commit.
* **After the stream**, the Task has *at most one* accepted ``ASSIGNED_TO`` edge
  (Req 2.5): the first assignee wins; every later distinct assignee is routed to
  quarantine, never silently added to accepted memory.
* **During the stream**, the second (and every later) distinct assignee is
  detected as a conflict by the ``ContradictionChecker`` (Req 9.5):
  ``has_conflict`` is ``True``, ``kind`` is ``hard`` (both sides high-confidence),
  and the recommended action is ``quarantine``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ocm.core.ids import IdGenerator
from ocm.memory.commit_manager import CommitManager
from ocm.memory.contracts import CandidateAssertion
from ocm.memory.graph_store import GraphStore
from ocm.memory.provenance_tracker import ProvenanceTracker
from ocm.memory.quarantine_store import QuarantineStore
from ocm.memory.sqlite_repository import SQLiteRepository
from ocm.ontology.enums import PersonStatus, Severity
from ocm.ontology.models import Person, Task
from ocm.tests.markers import pbt_property
from ocm.validation.constraints import ConstraintValidator
from ocm.validation.contradiction_checker import ContradictionChecker

TS = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
TASK_ID = "t1"
#: Strictly above the default contradiction_high_confidence threshold (0.8) so
#: both the candidate and the accepted counterpart grade as high-confidence and
#: a conflict is classified ``hard`` (Req 9.5).
HIGH = 0.95


def _person_id(index: int) -> str:
    return f"per_{index:02d}"


def _candidate(person_id: str, position: int) -> CandidateAssertion:
    """A high-confidence ASSIGNED_TO new_fact: t1 -> person_id."""
    return CandidateAssertion(
        subject_id=TASK_ID,
        predicate="ASSIGNED_TO",
        object_id=person_id,
        confidence=HIGH,
        # Distinct source_ref per submission keeps minted assertion ids unique.
        source_ref=f"doc://stream#{position}",
        extractor_version="mock-1",
    )


def _build_stack():
    """Wire the governance stack over an in-memory repo seeded with Task t1.

    Returns ``(repo, graph, validator, settings, manager)``. The caller seeds
    the Person entities (the generated assignee set) before driving the stream.
    """
    repo = SQLiteRepository(":memory:")
    graph = GraphStore()
    ids = IdGenerator(deterministic=True)

    task = Task(id=TASK_ID, title="Ship OCM")
    repo.upsert_entity("Task", task)
    graph.add_entity("Task", task)

    # A real ContradictionChecker (W7) so C7 detection is exercised, not faked.
    settings = SimpleSettings()
    validator = ConstraintValidator(settings, ContradictionChecker(settings))
    manager = CommitManager(
        repo=repo,
        graph=graph,
        ids=ids,
        quarantine_store=QuarantineStore(repo, ids),
        provenance_tracker=ProvenanceTracker(repo, ids),
    )
    return repo, graph, validator, settings, manager


class SimpleSettings:
    """Minimal settings carrying the default high-confidence threshold (0.8)."""

    contradiction_high_confidence = 0.8


# Stream of *distinct* assignees for the same Task (>= 2 so a conflict arises),
# generated as unique person indices. A smart, constrained generator: distinct
# ids only, since re-asserting the same triple is an idempotent no-op rather
# than a conflict and is out of scope for this single-valued property.
_assignee_streams = st.lists(
    st.integers(min_value=0, max_value=40),
    min_size=2,
    max_size=8,
    unique=True,
)


@pbt_property(10, "ASSIGNED_TO single-valued (m:1) invariant")
@given(person_indices=_assignee_streams)
def test_assigned_to_is_single_valued(person_indices: list[int]) -> None:
    repo, graph, validator, settings, manager = _build_stack()
    try:
        person_ids = [_person_id(i) for i in person_indices]
        # Seed every assignee as an active Person (graph + durable repo).
        for pid in person_ids:
            person = Person(id=pid, name=f"Person {pid}", status=PersonStatus.active)
            repo.upsert_entity("Person", person)
            graph.add_entity("Person", person)

        accepted_assignee = person_ids[0]
        for position, pid in enumerate(person_ids):
            candidate = _candidate(pid, position)

            # Whenever an accepted ASSIGNED_TO edge already exists for t1 (i.e.
            # for every assignee after the first), the Contradiction_Checker must
            # flag this distinct assignee as a hard, quarantine-worthy conflict
            # (Req 9.5) — it is never silently accepted.
            existing = graph.find_edges_by_predicate("ASSIGNED_TO")
            if existing:
                cresult = ContradictionChecker(settings).check(candidate, graph)
                assert cresult.has_conflict is True
                assert cresult.kind == "hard"
                assert cresult.severity == Severity.high
                assert cresult.recommended_action == "quarantine"
                # The accepted counterpart is surfaced as the conflicting id.
                assert cresult.conflicting_assertion_ids

            vr = validator.validate(candidate, graph, settings=settings)
            outcome = manager.commit(candidate, vr, created_at=TS)

            if position == 0:
                # The first assignee is accepted.
                assert outcome.decision == "accepted"
            else:
                # Every later distinct assignee is quarantined, not accepted.
                assert outcome.decision == "quarantined"

        # Invariant (Req 2.5): the Task ends with at most one accepted
        # ASSIGNED_TO edge, and it is the first assignee in the stream.
        assigned_edges = [
            edge
            for edge in graph.find_edges_by_predicate("ASSIGNED_TO")
            if edge[0] == TASK_ID
        ]
        assert len(assigned_edges) <= 1
        assert len(assigned_edges) == 1
        assert assigned_edges[0][1] == accepted_assignee
    finally:
        repo.close()


def test_single_valued_example_two_assignees() -> None:
    """A concrete example: t1 assigned to A then B keeps only A accepted.

    Validates Requirements 2.5, 9.5
    """
    repo, graph, validator, settings, manager = _build_stack()
    try:
        for pid in ("per_a", "per_b"):
            person = Person(id=pid, name=pid, status=PersonStatus.active)
            repo.upsert_entity("Person", person)
            graph.add_entity("Person", person)

        first = _candidate("per_a", 0)
        out_a = manager.commit(first, validator.validate(first, graph, settings=settings), created_at=TS)
        assert out_a.decision == "accepted"

        second = _candidate("per_b", 1)
        cresult = ContradictionChecker(settings).check(second, graph)
        assert cresult.has_conflict is True
        assert cresult.kind == "hard"
        assert cresult.recommended_action == "quarantine"

        out_b = manager.commit(second, validator.validate(second, graph, settings=settings), created_at=TS)
        assert out_b.decision == "quarantined"

        assigned = [e for e in graph.find_edges_by_predicate("ASSIGNED_TO") if e[0] == TASK_ID]
        assert len(assigned) == 1
        assert assigned[0][1] == "per_a"
    finally:
        repo.close()

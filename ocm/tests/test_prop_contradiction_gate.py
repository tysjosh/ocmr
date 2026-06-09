"""Property 6: Contradiction-gate invariant (Feature: ontology-constrained-memory).

Validates Requirements 8.8, 9.1, 9.2, 9.5.

The contradiction gate (C7, delegating to the Contradiction_Checker W7) exists so
that two mutually-contradictory facts can never both live in accepted memory at
high confidence. This property drives the *whole real governance stack* — not a
hand-built verdict — over an arbitrary stream of high-confidence ``ASSIGNED_TO``
writes for a single Task and asserts the standing invariant after the stream is
fully processed:

    Among the accepted assertions, no two high-confidence (> 0.8) assertions
    contradict each other. For the single-valued (m:1) ``ASSIGNED_TO`` relation
    this means a Task retains **at most one** accepted assignee — the gate either
    quarantines a conflicting new fact or (idempotently) re-accepts the same
    target, but a second *distinct* high-confidence assignee never survives as
    accepted memory.

Wiring (mirrors ``test_commit_governance``):

    Constraint_Validator (W6, C7) -> Contradiction_Checker (W7) -> Commit_Manager (W8)

over a live :class:`GraphStore` + ``SQLiteRepository(":memory:")``. A fresh,
hermetic stack is built per Hypothesis example so state never leaks between runs.
"""

from __future__ import annotations

from datetime import datetime, timezone

from hypothesis import given
from hypothesis import strategies as st

from ocm.core.ids import IdGenerator
from ocm.memory.commit_manager import CommitManager
from ocm.memory.contracts import CandidateAssertion
from ocm.memory.graph_store import GraphStore
from ocm.memory.provenance_tracker import ProvenanceTracker
from ocm.memory.quarantine_store import QuarantineStore
from ocm.memory.sqlite_repository import SQLiteRepository
from ocm.ontology.enums import AssertionStatus, PersonStatus
from ocm.ontology.models import Assertion, Person, Task
from ocm.ontology.relations import (
    Cardinality,
    UnknownPredicateError,
    get_relation_signature,
)
from ocm.tests.markers import pbt_property
from ocm.validation.constraints import ConstraintValidator

TS = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

# The default high-confidence threshold (settings.contradiction_high_confidence).
HIGH_CONFIDENCE = 0.8

# Cardinalities that permit a subject only a single target (mirrors W7).
_SINGLE_VALUED = {Cardinality.M_TO_ONE, Cardinality.ONE_TO_ONE}


# --------------------------------------------------------------------------- #
# Strategy: a stream of high-confidence ASSIGNED_TO writes for one Task.
# --------------------------------------------------------------------------- #
@st.composite
def assignment_streams(draw: st.DrawFn) -> tuple[int, list[tuple[int, float]]]:
    """Generate ``(num_persons, steps)`` for a single fixed Task.

    Each step is ``(person_index, confidence)`` where confidence is strictly
    above the high-confidence threshold so every write engages the gate. The
    person index selects which of the seeded people the Task is assigned to,
    so the stream mixes idempotent re-assignments (same person) with conflicting
    re-assignments (a different person).
    """
    num_persons = draw(st.integers(min_value=2, max_value=5))
    # Confidence strictly above the threshold (> 0.8) so each write is "high".
    confidence = st.floats(
        min_value=0.8001, max_value=1.0, allow_nan=False, allow_infinity=False
    )
    person_index = st.integers(min_value=0, max_value=num_persons - 1)
    steps = draw(
        st.lists(st.tuples(person_index, confidence), min_size=1, max_size=15)
    )
    return num_persons, steps


def _build_stack(num_persons: int):
    """Build a fresh, hermetic governance stack seeded with Task t1 + people."""
    repo = SQLiteRepository(":memory:")
    graph = GraphStore()
    ids = IdGenerator(deterministic=True)

    # Seed the fixed Task and the candidate assignees (all active so C5 passes).
    task = Task(id="t1", title="Ship OCM")
    repo.upsert_entity("Task", task)
    graph.add_entity("Task", task)
    for i in range(num_persons):
        person = Person(id=f"per_{i}", name=f"P{i}", status=PersonStatus.active)
        repo.upsert_entity("Person", person)
        graph.add_entity("Person", person)

    validator = ConstraintValidator(settings=None)
    manager = CommitManager(
        repo=repo,
        graph=graph,
        ids=ids,
        quarantine_store=QuarantineStore(repo, ids),
        provenance_tracker=ProvenanceTracker(repo, ids),
    )
    return repo, graph, validator, manager


def _mutually_contradict(a: Assertion, b: Assertion) -> bool:
    """Whether two accepted assertions are a single-valued contradiction.

    For a single-valued (m:1 / 1:1) predicate, two accepted assertions conflict
    when they share the subject but point at *different* objects (and, for 1:1,
    when they share the object but have different subjects). This is the concrete
    realization of "no two accepted high-confidence assertions contradict"
    (Req 9.5) used by the contradiction gate.
    """
    if a.predicate != b.predicate:
        return False
    try:
        sig = get_relation_signature(a.predicate)
    except UnknownPredicateError:
        return False
    if sig.cardinality not in _SINGLE_VALUED:
        return False
    if a.subject_id == b.subject_id and a.object_id != b.object_id:
        return True
    if (
        sig.cardinality == Cardinality.ONE_TO_ONE
        and a.object_id == b.object_id
        and a.subject_id != b.subject_id
    ):
        return True
    return False


@pbt_property(6, "Contradiction-gate invariant")
@given(stream=assignment_streams())
def test_contradiction_gate_invariant(stream: tuple[int, list[tuple[int, float]]]) -> None:
    """No two accepted >0.8 assertions contradict; at most one assignee survives.

    Drives the real Constraint_Validator -> Contradiction_Checker -> Commit_Manager
    stack over the generated stream, then asserts the invariant (Req 8.8, 9.1,
    9.2, 9.5).
    """
    num_persons, steps = stream
    repo, graph, validator, manager = _build_stack(num_persons)

    try:
        # Process the whole stream: validate, then commit per the verdict.
        for i, (person_index, confidence) in enumerate(steps):
            candidate = CandidateAssertion(
                subject_id="t1",
                predicate="ASSIGNED_TO",
                object_id=f"per_{person_index}",
                confidence=confidence,
                source_ref=f"doc://stream#{i}",
                write_intent="new_fact",
                extractor_version="mock-1",
            )
            verdict = validator.validate(candidate, graph)
            manager.commit(candidate, verdict, created_at=TS)

        accepted = list(repo.list_assertions(status=AssertionStatus.accepted.value))

        # All accepted ASSIGNED_TO writes are high-confidence by construction.
        accepted_assignments = [
            a
            for a in accepted
            if a.predicate == "ASSIGNED_TO" and float(a.confidence) > HIGH_CONFIDENCE
        ]

        # (1) Single-valued invariant: at most one accepted assignee for the Task.
        accepted_targets = {a.object_id for a in accepted_assignments}
        assert len(accepted_targets) <= 1, (
            "more than one distinct accepted ASSIGNED_TO target survived: "
            f"{sorted(accepted_targets)}"
        )

        # (1b) The accepted graph projection agrees: <= 1 ASSIGNED_TO edge from t1.
        task_assignment_edges = graph.out_edges("t1", "ASSIGNED_TO")
        assert len(task_assignment_edges) <= 1

        # Since the first write always succeeds (active person, valid Task) and a
        # non-empty stream is generated, exactly one assignee survives accepted.
        assert len(accepted_targets) == 1
        assert len(task_assignment_edges) == 1

        # (2) No two accepted high-confidence assertions are mutually contradictory.
        high_conf_accepted = [a for a in accepted if float(a.confidence) > HIGH_CONFIDENCE]
        for j, a in enumerate(high_conf_accepted):
            for b in high_conf_accepted[j + 1 :]:
                assert not _mutually_contradict(a, b), (
                    "two accepted high-confidence assertions contradict: "
                    f"{a.id} ({a.subject_id}->{a.object_id}) vs "
                    f"{b.id} ({b.subject_id}->{b.object_id})"
                )
    finally:
        repo.close()

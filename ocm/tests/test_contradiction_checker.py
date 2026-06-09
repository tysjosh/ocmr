"""Contradiction-detection unit tests for the Contradiction_Checker (W7, task 7.2).

These tests pin down the contradiction categories that
``ocm.validation.contradiction_checker.ContradictionChecker`` is the single
source of truth for (Req 9.1-9.7), exercised against the real ``GraphStore``
and ontology models (no mocks, Req 26.4):

* **Single-valued / exact-predicate conflict (Req 9.2, 9.5).** ``ASSIGNED_TO``
  is ``m:1``, so a Task may point at only one assignee. A high-confidence
  candidate naming a *different* assignee than the accepted edge is a **hard**
  contradiction (``severity=high``) and surfaces the accepted assertion id.
* **Explicit ``CONTRADICTS`` link (Req 9.4).** An accepted ``CONTRADICTS`` edge
  incident to the candidate's subject/object is a curated conflict.
* **Temporal overlap (Req 9.6).** Two single-valued assignments whose validity
  windows overlap are classified ``temporal``; non-overlapping windows are a
  valid historical succession and are **not** a contradiction.
* **Low-confidence contradiction (Req 9.1).** When neither side exceeds the
  high-confidence threshold (0.8) the conflict is only a **soft** warning
  (``severity=low``, ``recommended_action=accept``).
* **Idempotent re-assertion.** Re-asserting the identical triple is a no-op.

The high-confidence threshold is strict (``confidence > 0.8``), so 0.9 grades as
high and 0.7 as low.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from ocm.memory.contracts import CandidateAssertion
from ocm.memory.graph_store import GraphStore
from ocm.ontology.enums import AssertionStatus, Severity, WriteIntent
from ocm.ontology.models import Assertion, Person, Task
from ocm.validation.contradiction_checker import ContradictionChecker

_T0 = datetime(2024, 1, 1, 9, 0, 0)
_T1 = datetime(2024, 1, 2, 9, 0, 0)
_T2 = datetime(2024, 1, 3, 9, 0, 0)
_T3 = datetime(2024, 1, 4, 9, 0, 0)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _task(task_id: str) -> Task:
    return Task(id=task_id, title=f"task {task_id}")


def _person(person_id: str) -> Person:
    return Person(id=person_id, name=f"person {person_id}")


def _accepted_assigned_to(
    assertion_id: str,
    task_id: str,
    person_id: str,
    *,
    confidence: float = 0.9,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
) -> Assertion:
    return Assertion(
        id=assertion_id,
        subject_id=task_id,
        predicate="ASSIGNED_TO",
        object_id=person_id,
        confidence=confidence,
        status=AssertionStatus.accepted,
        source_ref="src-1",
        created_at=_T0,
        valid_from=valid_from,
        valid_to=valid_to,
    )


def _accepted_contradicts(
    assertion_id: str, subject_id: str, object_id: str, *, confidence: float = 0.9
) -> Assertion:
    return Assertion(
        id=assertion_id,
        subject_id=subject_id,
        predicate="CONTRADICTS",
        object_id=object_id,
        confidence=confidence,
        status=AssertionStatus.accepted,
        source_ref="src-1",
        created_at=_T0,
    )


def _assigned_to_candidate(
    task_id: str,
    person_id: str,
    *,
    confidence: float = 0.9,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
) -> CandidateAssertion:
    return CandidateAssertion(
        subject_id=task_id,
        predicate="ASSIGNED_TO",
        object_id=person_id,
        confidence=confidence,
        source_ref="src-2",
        valid_from=valid_from,
        valid_to=valid_to,
    )


def _assignment_graph(task_id: str, person_ids: list[str]) -> GraphStore:
    """A graph holding a Task plus the given Person nodes (no edges yet)."""
    graph = GraphStore()
    graph.add_entity("Task", _task(task_id))
    for pid in person_ids:
        graph.add_entity("Person", _person(pid))
    return graph


# ---------------------------------------------------------------------------
# Single-valued / exact-predicate conflict (Req 9.2, 9.5)
# ---------------------------------------------------------------------------
def test_high_confidence_assigned_to_conflict_is_hard():
    # t1 already ASSIGNED_TO person A (accepted); candidate assigns a different B.
    graph = _assignment_graph("t1", ["pA", "pB"])
    graph.add_assertion(_accepted_assigned_to("a-A", "t1", "pA", confidence=0.9))
    candidate = _assigned_to_candidate("t1", "pB", confidence=0.9)

    result = ContradictionChecker().check(candidate, graph)

    assert result.has_conflict is True
    assert result.kind == "hard"
    assert result.severity == Severity.high
    # The accepted (conflicting) assertion id is surfaced (Req 9.7).
    assert result.conflicting_assertion_ids == ["a-A"]
    # A high-confidence new_fact conflict recommends quarantine (Req 9.7).
    assert result.recommended_action == "quarantine"


def test_high_confidence_correction_recommends_supersede():
    # A correction write_intent at high confidence is allowed to supersede.
    graph = _assignment_graph("t1", ["pA", "pB"])
    graph.add_assertion(_accepted_assigned_to("a-A", "t1", "pA", confidence=0.9))
    candidate = _assigned_to_candidate("t1", "pB", confidence=0.9)
    candidate = candidate.model_copy(update={"write_intent": WriteIntent.correction})

    result = ContradictionChecker().check(candidate, graph)

    assert result.has_conflict is True
    assert result.kind == "hard"
    assert result.recommended_action == "supersede"


def test_idempotent_reassert_same_triple_has_no_conflict():
    # Re-asserting the identical (t1, ASSIGNED_TO, pA) triple is a no-op.
    graph = _assignment_graph("t1", ["pA"])
    graph.add_assertion(_accepted_assigned_to("a-A", "t1", "pA", confidence=0.9))
    candidate = _assigned_to_candidate("t1", "pA", confidence=0.9)

    result = ContradictionChecker().check(candidate, graph)

    assert result.has_conflict is False
    assert result.conflicting_assertion_ids == []


# ---------------------------------------------------------------------------
# Explicit CONTRADICTS link (Req 9.4)
# ---------------------------------------------------------------------------
def test_explicit_contradicts_link_is_detected():
    # An accepted CONTRADICTS edge incident to the candidate's subject is a conflict.
    graph = GraphStore()
    for cid in ("claimA", "claimB", "claimC"):
        graph.add_entity("Claim", {"id": cid})
    graph.add_assertion(
        _accepted_contradicts("ctr-1", "claimA", "claimC", confidence=0.9)
    )
    # Candidate touches claimA via a non-CONTRADICTS predicate.
    candidate = CandidateAssertion(
        subject_id="claimA",
        predicate="SUPPORTS",
        object_id="claimB",
        confidence=0.9,
        source_ref="src-2",
    )

    result = ContradictionChecker().check(candidate, graph)

    assert result.has_conflict is True
    assert "ctr-1" in result.conflicting_assertion_ids
    assert result.kind == "hard"


def test_candidate_asserting_contradicts_is_not_a_conflict():
    # A candidate that itself asserts CONTRADICTS is not treated as a conflict.
    graph = GraphStore()
    for cid in ("claimA", "claimB"):
        graph.add_entity("Claim", {"id": cid})
    candidate = CandidateAssertion(
        subject_id="claimA",
        predicate="CONTRADICTS",
        object_id="claimB",
        confidence=0.9,
        source_ref="src-2",
    )

    result = ContradictionChecker().check(candidate, graph)

    assert result.has_conflict is False


# ---------------------------------------------------------------------------
# Temporal overlap (Req 9.6)
# ---------------------------------------------------------------------------
def test_overlapping_validity_windows_is_temporal():
    # Accepted A valid [T0, T2]; candidate B valid [T1, T3] -> windows overlap.
    graph = _assignment_graph("t1", ["pA", "pB"])
    graph.add_assertion(
        _accepted_assigned_to(
            "a-A", "t1", "pA", confidence=0.9, valid_from=_T0, valid_to=_T2
        )
    )
    candidate = _assigned_to_candidate(
        "t1", "pB", confidence=0.9, valid_from=_T1, valid_to=_T3
    )

    result = ContradictionChecker().check(candidate, graph)

    assert result.has_conflict is True
    assert result.kind == "temporal"
    assert result.conflicting_assertion_ids == ["a-A"]


def test_non_overlapping_validity_windows_has_no_conflict():
    # Accepted A valid [T0, T1]; candidate B valid [T2, T3] -> historical succession.
    graph = _assignment_graph("t1", ["pA", "pB"])
    graph.add_assertion(
        _accepted_assigned_to(
            "a-A", "t1", "pA", confidence=0.9, valid_from=_T0, valid_to=_T1
        )
    )
    candidate = _assigned_to_candidate(
        "t1", "pB", confidence=0.9, valid_from=_T2, valid_to=_T3
    )

    result = ContradictionChecker().check(candidate, graph)

    assert result.has_conflict is False


# ---------------------------------------------------------------------------
# Low-confidence contradiction -> soft warning only (Req 9.1)
# ---------------------------------------------------------------------------
def test_low_confidence_conflict_is_soft_warning():
    # Both sides at or below the 0.8 threshold -> soft warning, not a hard block.
    graph = _assignment_graph("t1", ["pA", "pB"])
    graph.add_assertion(_accepted_assigned_to("a-A", "t1", "pA", confidence=0.7))
    candidate = _assigned_to_candidate("t1", "pB", confidence=0.7)

    result = ContradictionChecker().check(candidate, graph)

    assert result.has_conflict is True
    assert result.kind == "soft"
    assert result.severity == Severity.low
    assert result.recommended_action == "accept"
    assert result.conflicting_assertion_ids == ["a-A"]

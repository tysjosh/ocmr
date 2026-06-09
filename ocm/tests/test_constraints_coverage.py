"""Consolidated constraint coverage tests for the Constraint_Validator (W6).

This module provides example-based unit coverage for the graph-level constraints
that are *not* exercised by the temporal (C2/C3, task 6.4) or task (C4/C5,
task 6.5) test files, plus the aggregating :class:`ConstraintValidator`. Together
with those files the suite spans every constraint C1–C10.

Coverage here:

* **C1 — identity uniqueness** (Req 8.2): an id may not be reused under a
  different entity type.
* **C6 — confidence bounds** (Req 8.7): confidence must lie within [0, 1].
* **C8 — decision evidence floor** (Req 8.9): a ``final`` Decision needs at least
  ``decision_evidence_floor`` EVIDENCE_FOR supports.
* **C9 — graph-level domain/range** (Req 8.10): a predicate's subject/object must
  match the relation signature against the *resolved* entity types.
* **C10 — task status transition** (Req 8.11): a Task status change must be in the
  transition map; ``correction`` bypasses it.
* **ConstraintValidator.validate** (Req 8.12): runs the applicable constraints and
  returns the first failure.

Requirements: 8.2, 8.7, 8.9, 8.10, 8.11, 8.12, 26.6, 28.4.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ocm.core.config import Settings
from ocm.memory.contracts import CandidateAssertion
from ocm.memory.graph_store import GraphStore
from ocm.ontology.enums import (
    AssertionStatus,
    DecisionStatus,
    PersonStatus,
    ProjectStatus,
    TaskStatus,
    WriteIntent,
)
from ocm.ontology.models import (
    Assertion,
    Decision,
    Document,
    Person,
    Project,
    Task,
)
from ocm.validation.constraints import (
    ConstraintValidator,
    c1_identity_uniqueness,
    c6_confidence_bounds,
    c8_decision_evidence_floor,
    c9_graph_domain_range,
    c10_task_status_transition,
)

_TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _accepted_assertion(
    aid: str, subject_id: str, predicate: str, object_id: str, confidence: float = 0.9
) -> Assertion:
    """Build an ``accepted`` Assertion so it can be added to the GraphStore."""
    return Assertion(
        id=aid,
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        confidence=confidence,
        status=AssertionStatus.accepted,
        source_ref="doc:1",
        created_at=_TS,
    )


def _candidate(
    subject_id: str,
    predicate: str,
    object_id: str,
    *,
    confidence: float = 0.9,
    write_intent: WriteIntent = WriteIntent.new_fact,
) -> CandidateAssertion:
    return CandidateAssertion(
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        confidence=confidence,
        source_ref="doc:1",
        write_intent=write_intent,
    )


# --------------------------------------------------------------------------- #
# C1 — Identity uniqueness (Req 8.2)
# --------------------------------------------------------------------------- #
def test_c1_reuse_id_under_different_type_fails() -> None:
    graph = GraphStore()
    graph.add_entity("Person", Person(id="x1", name="Alice", status=PersonStatus.active))

    result = c1_identity_uniqueness("Project", "x1", graph)

    assert result.valid is False
    assert result.failed_check == "C1"
    assert result.recommended_action == "reject"
    assert "x1" in result.conflicting_ids


def test_c1_reassert_same_type_passes() -> None:
    graph = GraphStore()
    graph.add_entity("Person", Person(id="x1", name="Alice", status=PersonStatus.active))

    result = c1_identity_uniqueness("Person", "x1", graph)

    assert result.valid is True
    assert result.failed_check is None


def test_c1_unused_id_passes() -> None:
    graph = GraphStore()

    result = c1_identity_uniqueness("Task", "brand-new", graph)

    assert result.valid is True


# --------------------------------------------------------------------------- #
# C6 — Confidence bounds (Req 8.7)
# --------------------------------------------------------------------------- #
def test_c6_above_one_fails() -> None:
    result = c6_confidence_bounds(1.5)

    assert result.valid is False
    assert result.failed_check == "C6"
    assert result.recommended_action == "reject"


def test_c6_below_zero_fails() -> None:
    result = c6_confidence_bounds(-0.01)

    assert result.valid is False
    assert result.failed_check == "C6"


def test_c6_in_range_passes() -> None:
    assert c6_confidence_bounds(0.5).valid is True


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_c6_closed_boundaries_pass(value: float) -> None:
    assert c6_confidence_bounds(value).valid is True


# --------------------------------------------------------------------------- #
# C9 — Graph-level domain/range (Req 8.10)
# --------------------------------------------------------------------------- #
def test_c9_wrong_domain_fails() -> None:
    """OWNS requires a Person/Organization subject; a Task subject is invalid."""
    graph = GraphStore()
    graph.add_entity("Task", Task(id="t1", title="Build", status=TaskStatus.todo))
    graph.add_entity("Project", Project(id="p1", name="Orion", status=ProjectStatus.active))

    result = c9_graph_domain_range(_candidate("t1", "OWNS", "p1"), graph)

    assert result.valid is False
    assert result.failed_check == "C9"
    assert result.recommended_action == "reject"
    assert "t1" in result.conflicting_ids


def test_c9_correct_person_owns_project_passes() -> None:
    graph = GraphStore()
    graph.add_entity("Person", Person(id="pe1", name="Alice", status=PersonStatus.active))
    graph.add_entity("Project", Project(id="p1", name="Orion", status=ProjectStatus.active))

    result = c9_graph_domain_range(_candidate("pe1", "OWNS", "p1"), graph)

    assert result.valid is True
    assert result.failed_check is None


def test_c9_unknown_predicate_fails() -> None:
    graph = GraphStore()
    graph.add_entity("Person", Person(id="pe1", name="Alice", status=PersonStatus.active))
    graph.add_entity("Project", Project(id="p1", name="Orion", status=ProjectStatus.active))

    result = c9_graph_domain_range(_candidate("pe1", "NOT_A_PREDICATE", "p1"), graph)

    assert result.valid is False
    assert result.failed_check == "C9"


# --------------------------------------------------------------------------- #
# C10 — Task status transition (Req 8.11)
# --------------------------------------------------------------------------- #
def test_c10_illegal_transition_quarantines() -> None:
    result = c10_task_status_transition(TaskStatus.todo, TaskStatus.done)

    assert result.valid is False
    assert result.failed_check == "C10"
    assert result.recommended_action == "quarantine"


def test_c10_legal_transition_passes() -> None:
    result = c10_task_status_transition(TaskStatus.todo, TaskStatus.in_progress)

    assert result.valid is True
    assert result.failed_check is None


def test_c10_correction_bypasses_map() -> None:
    # An otherwise-illegal transition is permitted under a correction intent.
    result = c10_task_status_transition(
        TaskStatus.todo, TaskStatus.done, WriteIntent.correction
    )

    assert result.valid is True


# --------------------------------------------------------------------------- #
# C8 — Decision evidence floor (Req 8.9)
# --------------------------------------------------------------------------- #
def test_c8_final_decision_without_evidence_quarantines() -> None:
    graph = GraphStore()
    graph.add_entity("Decision", Decision(id="d1", summary="Ship", timestamp=_TS,
                                          status=DecisionStatus.final))

    result = c8_decision_evidence_floor("d1", DecisionStatus.final, graph, Settings())

    assert result.valid is False
    assert result.failed_check == "C8"
    assert result.recommended_action == "quarantine"
    assert "d1" in result.conflicting_ids


def test_c8_final_decision_with_document_evidence_passes() -> None:
    graph = GraphStore()
    graph.add_entity("Decision", Decision(id="d1", summary="Ship", timestamp=_TS,
                                          status=DecisionStatus.final))
    graph.add_entity("Document", Document(id="doc1", title="Spec",
                                          path_or_url="file://spec", created_at=_TS))
    graph.add_assertion(_accepted_assertion("a1", "doc1", "EVIDENCE_FOR", "d1"))

    result = c8_decision_evidence_floor("d1", DecisionStatus.final, graph, Settings())

    assert result.valid is True
    assert result.failed_check is None


def test_c8_draft_decision_passes_without_evidence() -> None:
    graph = GraphStore()
    graph.add_entity("Decision", Decision(id="d1", summary="Draft", timestamp=_TS,
                                          status=DecisionStatus.draft))

    result = c8_decision_evidence_floor("d1", DecisionStatus.draft, graph, Settings())

    assert result.valid is True


# --------------------------------------------------------------------------- #
# ConstraintValidator aggregation (Req 8.12)
# --------------------------------------------------------------------------- #
def test_validator_returns_first_failure() -> None:
    """A candidate violating C9 (wrong domain) is surfaced as the failure."""
    graph = GraphStore()
    graph.add_entity("Task", Task(id="t1", title="Build", status=TaskStatus.todo))
    graph.add_entity("Project", Project(id="p1", name="Orion", status=ProjectStatus.active))

    validator = ConstraintValidator(settings=Settings())
    result = validator.validate(_candidate("t1", "OWNS", "p1"), graph)

    assert result.valid is False
    assert result.failed_check == "C9"
    assert result.recommended_action == "reject"


def test_validator_passes_when_all_constraints_hold() -> None:
    graph = GraphStore()
    graph.add_entity("Person", Person(id="pe1", name="Alice", status=PersonStatus.active))
    graph.add_entity("Project", Project(id="p1", name="Orion", status=ProjectStatus.active))

    validator = ConstraintValidator(settings=Settings())
    result = validator.validate(_candidate("pe1", "OWNS", "p1"), graph)

    assert result.valid is True
    assert result.failed_check is None

"""Unit tests for the task-oriented graph constraints C4 and C5 (task 6.5).

Validates: Requirements 8.5, 8.6, 26.3

* **C4 — done-task completion event (Req 8.5).** A Task whose status is ``done``
  must be related to a completion Event by an accepted ``RESULTS_IN`` edge
  (``Event RESULTS_IN Task``). A done Task with no such Event is quarantined; a
  done Task with one passes; a non-``done`` Task always passes.
* **C5 — inactive assignee (Req 8.6).** An ``ASSIGNED_TO`` candidate whose target
  Person is ``inactive`` is quarantined; an ``active`` (or ``unknown`` /
  unresolved) assignee passes; non-``ASSIGNED_TO`` predicates are out of scope.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ocm.memory.contracts import CandidateAssertion
from ocm.memory.graph_store import GraphStore
from ocm.ontology.enums import AssertionStatus, PersonStatus, TaskStatus
from ocm.ontology.models import Assertion, Event, Person, Task
from ocm.validation.constraints import (
    c4_done_task_completion_event,
    c5_inactive_assignee,
)

TS = datetime(2024, 5, 6, 7, 8, 9, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def graph() -> GraphStore:
    return GraphStore()


def _task(task_id: str = "task:1", status: TaskStatus = TaskStatus.done) -> Task:
    return Task(id=task_id, title="Ship the release", status=status)


def _event(event_id: str = "event:1") -> Event:
    return Event(
        id=event_id,
        type="completion",
        timestamp_start=TS,
        description="Task completed",
    )


def _person(person_id: str = "person:1", status: PersonStatus = PersonStatus.active) -> Person:
    return Person(id=person_id, name="Ada Lovelace", status=status)


def _accepted_results_in(event_id: str, task_id: str) -> Assertion:
    """An accepted ``Event RESULTS_IN Task`` assertion (the completion link)."""
    return Assertion(
        id="assertion:results_in",
        subject_id=event_id,
        predicate="RESULTS_IN",
        object_id=task_id,
        confidence=0.95,
        status=AssertionStatus.accepted,
        source_ref="doc:1",
        created_at=TS,
    )


def _assigned_to(task_id: str, person_id: str) -> CandidateAssertion:
    return CandidateAssertion(
        subject_id=task_id,
        predicate="ASSIGNED_TO",
        object_id=person_id,
        confidence=0.9,
        source_ref="doc:1",
    )


# ---------------------------------------------------------------------------
# C4 — done-task completion event (Req 8.5)
# ---------------------------------------------------------------------------
def test_c4_done_task_without_completion_event_quarantined(graph: GraphStore) -> None:
    """A done Task lacking a RESULTS_IN completion Event is quarantined (Req 8.5)."""
    task = _task(status=TaskStatus.done)
    graph.add_entity("Task", task)

    result = c4_done_task_completion_event(task.id, task.status, graph)

    assert result.valid is False
    assert result.failed_check == "C4"
    assert result.recommended_action == "quarantine"
    assert task.id in result.conflicting_ids


def test_c4_done_task_with_completion_event_passes(graph: GraphStore) -> None:
    """A done Task with an Event RESULTS_IN it passes C4 (Req 8.5)."""
    task = _task(status=TaskStatus.done)
    event = _event()
    graph.add_entity("Task", task)
    graph.add_entity("Event", event)
    graph.add_assertion(_accepted_results_in(event.id, task.id))

    result = c4_done_task_completion_event(task.id, task.status, graph)

    assert result.valid is True
    assert result.failed_check is None


def test_c4_non_done_task_passes_without_event(graph: GraphStore) -> None:
    """A non-done Task is out of scope for C4 and passes (Req 8.5)."""
    task = _task(status=TaskStatus.in_progress)
    graph.add_entity("Task", task)

    result = c4_done_task_completion_event(task.id, task.status, graph)

    assert result.valid is True
    assert result.failed_check is None


def test_c4_accepts_string_status(graph: GraphStore) -> None:
    """C4 coerces a raw string status the same as the enum (Req 8.5)."""
    task = _task(status=TaskStatus.done)
    graph.add_entity("Task", task)

    result = c4_done_task_completion_event(task.id, "done", graph)

    assert result.valid is False
    assert result.failed_check == "C4"


# ---------------------------------------------------------------------------
# C5 — inactive assignee (Req 8.6)
# ---------------------------------------------------------------------------
def test_c5_assigned_to_inactive_person_quarantined(graph: GraphStore) -> None:
    """ASSIGNED_TO an inactive Person is quarantined (Req 8.6)."""
    person = _person(status=PersonStatus.inactive)
    graph.add_entity("Task", _task())
    graph.add_entity("Person", person)

    result = c5_inactive_assignee(_assigned_to("task:1", person.id), graph)

    assert result.valid is False
    assert result.failed_check == "C5"
    assert result.recommended_action == "quarantine"
    assert person.id in result.conflicting_ids


def test_c5_assigned_to_active_person_passes(graph: GraphStore) -> None:
    """ASSIGNED_TO an active Person passes C5 (Req 8.6)."""
    person = _person(status=PersonStatus.active)
    graph.add_entity("Task", _task())
    graph.add_entity("Person", person)

    result = c5_inactive_assignee(_assigned_to("task:1", person.id), graph)

    assert result.valid is True
    assert result.failed_check is None


def test_c5_assigned_to_unknown_status_person_passes(graph: GraphStore) -> None:
    """ASSIGNED_TO a Person with unknown status passes C5 (Req 8.6)."""
    person = _person(status=PersonStatus.unknown)
    graph.add_entity("Task", _task())
    graph.add_entity("Person", person)

    result = c5_inactive_assignee(_assigned_to("task:1", person.id), graph)

    assert result.valid is True
    assert result.failed_check is None


def test_c5_unresolved_assignee_passes(graph: GraphStore) -> None:
    """A missing/unresolved assignee is left to W5/C9, so C5 passes (Req 8.6)."""
    graph.add_entity("Task", _task())

    result = c5_inactive_assignee(_assigned_to("task:1", "person:missing"), graph)

    assert result.valid is True
    assert result.failed_check is None


def test_c5_non_assigned_to_predicate_out_of_scope(graph: GraphStore) -> None:
    """C5 only applies to ASSIGNED_TO; other predicates pass (Req 8.6)."""
    person = _person(status=PersonStatus.inactive)
    graph.add_entity("Project", _task())  # any non-ASSIGNED_TO subject/edge
    graph.add_entity("Person", person)

    candidate = CandidateAssertion(
        subject_id="task:1",
        predicate="OWNS",
        object_id=person.id,
        confidence=0.9,
        source_ref="doc:1",
    )
    result = c5_inactive_assignee(candidate, graph)

    assert result.valid is True
    assert result.failed_check is None

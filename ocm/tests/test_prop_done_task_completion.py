"""Property 11: Accepted done Task has a completion event.

Feature: ontology-constrained-memory, Property 11.

Validates: Requirements 8.5

Constraint C4 (``c4_done_task_completion_event`` / ``ConstraintValidator``)
enforces the write-time invariant that a Task may only become *accepted* with
status ``done`` when it has at least one completion Event related to it by an
accepted ``RESULTS_IN`` edge (``Event RESULTS_IN Task``). A ``done`` Task with
no such Event is never silently accepted — C4 fails and recommends
``quarantine``.

These property tests drive C4 across randomly generated graph states (random
task ids, statuses, and the presence/absence/count of completion Events), at
two levels:

* directly via :func:`c4_done_task_completion_event`, and
* through the aggregating :class:`ConstraintValidator` (a benign
  ``Project CONTAINS Task`` candidate that touches the Task endpoint so C4 runs
  inside the full constraint pipeline).

The universal invariant asserted by both halves: a ``done`` Task with a
completion Event passes; a ``done`` Task without one is quarantined; a non-``done``
Task is out of scope and always passes regardless of completion Events.
"""

from __future__ import annotations

from datetime import datetime, timezone

from hypothesis import given
from hypothesis import strategies as st

from ocm.memory.contracts import CandidateAssertion
from ocm.memory.graph_store import GraphStore
from ocm.ontology.enums import AssertionStatus, TaskStatus
from ocm.ontology.models import Assertion, Event, Project, Task
from ocm.tests.markers import pbt_property
from ocm.validation.constraints import (
    ConstraintValidator,
    c4_done_task_completion_event,
)

_TS = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# Safe, collision-free id suffixes (graph nodes are keyed by id).
_id_suffix = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8
)
_task_statuses = st.sampled_from(list(TaskStatus))
_completion_counts = st.integers(min_value=0, max_value=3)
_confidences = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def _build_done_task_graph(
    suffix: str, status: TaskStatus, num_events: int, confidence: float
) -> tuple[GraphStore, str]:
    """A graph with one Task and ``num_events`` completion Events via RESULTS_IN."""
    graph = GraphStore()
    task_id = f"task:{suffix}"
    graph.add_entity("Task", Task(id=task_id, title="Ship it", status=status))
    for i in range(num_events):
        event_id = f"event:{suffix}:{i}"
        graph.add_entity(
            "Event",
            Event(
                id=event_id,
                type="completion",
                timestamp_start=_TS,
                description="Task completed",
            ),
        )
        graph.add_assertion(
            Assertion(
                id=f"assertion:{suffix}:{i}",
                subject_id=event_id,
                predicate="RESULTS_IN",
                object_id=task_id,
                confidence=confidence,
                status=AssertionStatus.accepted,
                source_ref="doc:1",
                created_at=_TS,
            )
        )
    return graph, task_id


@pbt_property(11, "Accepted done Task has a completion event")
@given(
    suffix=_id_suffix,
    status=_task_statuses,
    num_events=_completion_counts,
    confidence=_confidences,
)
def test_c4_done_task_requires_completion_event(
    suffix: str, status: TaskStatus, num_events: int, confidence: float
) -> None:
    """C4 directly: done + no event => quarantine; otherwise valid."""
    graph, task_id = _build_done_task_graph(suffix, status, num_events, confidence)

    result = c4_done_task_completion_event(task_id, status, graph)

    if status == TaskStatus.done and num_events == 0:
        # The bug C4 guards against: a done Task with no completion Event.
        assert result.valid is False
        assert result.failed_check == "C4"
        assert result.recommended_action == "quarantine"
        assert task_id in result.conflicting_ids
    else:
        # done-with-event, or any non-done status, is acceptable for C4.
        assert result.valid is True
        assert result.failed_check is None


@pbt_property(11, "Accepted done Task has a completion event")
@given(
    suffix=_id_suffix,
    status=_task_statuses,
    num_events=_completion_counts,
    confidence=_confidences,
)
def test_constraint_validator_quarantines_done_task_without_completion(
    suffix: str, status: TaskStatus, num_events: int, confidence: float
) -> None:
    """The full ConstraintValidator quarantines a done Task lacking a completion Event.

    A benign ``Project CONTAINS Task`` candidate touches the Task endpoint so C4
    runs inside the aggregate pipeline; no other constraint fires for this
    well-typed, non-contradicting edge, so the validator's verdict is C4's.
    """
    graph, task_id = _build_done_task_graph(suffix, status, num_events, confidence)
    project_id = f"project:{suffix}"
    graph.add_entity("Project", Project(id=project_id, name="Orion"))

    candidate = CandidateAssertion(
        subject_id=project_id,
        predicate="CONTAINS",
        object_id=task_id,
        confidence=confidence,
        source_ref="doc:1",
    )
    validator = ConstraintValidator()
    result = validator.validate(candidate, graph)

    if status == TaskStatus.done and num_events == 0:
        # Invariant: a done Task without a completion Event is never accepted —
        # C4 surfaces as the quarantine verdict.
        assert result.valid is False
        assert result.failed_check == "C4"
        assert result.recommended_action == "quarantine"
        assert task_id in result.conflicting_ids
    else:
        # done-with-event or non-done: the benign CONTAINS write passes.
        assert result.valid is True

"""Temporal constraint unit tests for C2 and C3 (task 6.4).

These tests pin down the two temporal graph-level constraints in
``ocm.validation.constraints``:

* **C2 — temporal sanity (Req 8.3):** an Event whose ``timestamp_end`` precedes
  its ``timestamp_start`` is rejected; a missing ``timestamp_end`` passes; an
  end equal to or after the start passes.
* **C3 — acyclic PRECEDES (Req 8.4):** a ``PRECEDES`` candidate that would close
  a cycle over the accepted PRECEDES projection is rejected; a candidate that
  merely extends a chain is accepted.

Both constraints surface invalid-input behavior that the schema-bounded enums
and bounds can't catch on their own (Req 26.2): C2 compares two timestamps and
C3 reasons over the accepted graph, so they are exercised here against the real
``Event`` model and a real ``GraphStore`` (no mocks).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from ocm.memory.contracts import CandidateAssertion
from ocm.memory.graph_store import GraphStore
from ocm.ontology.enums import AssertionStatus
from ocm.ontology.models import Assertion, Event
from ocm.validation.constraints import c2_temporal_sanity, c3_acyclic_precedes

_T0 = datetime(2024, 1, 1, 9, 0, 0)
_T1 = datetime(2024, 1, 1, 10, 0, 0)
_T2 = datetime(2024, 1, 1, 11, 0, 0)


def _event(event_id: str, *, start: datetime, end: datetime | None = None) -> Event:
    return Event(
        id=event_id,
        type="completion",
        timestamp_start=start,
        timestamp_end=end,
        description=f"event {event_id}",
    )


def _precedes_candidate(subject_id: str, object_id: str) -> CandidateAssertion:
    return CandidateAssertion(
        subject_id=subject_id,
        predicate="PRECEDES",
        object_id=object_id,
        confidence=0.9,
        source_ref="src-1",
    )


def _accepted_precedes(assertion_id: str, subject_id: str, object_id: str) -> Assertion:
    return Assertion(
        id=assertion_id,
        subject_id=subject_id,
        predicate="PRECEDES",
        object_id=object_id,
        confidence=0.9,
        status=AssertionStatus.accepted,
        source_ref="src-1",
        created_at=_T0,
    )


def _precedes_chain_graph(event_ids: list[str]) -> GraphStore:
    """Build a graph of Events linked by accepted PRECEDES edges in order."""
    graph = GraphStore()
    for i, eid in enumerate(event_ids):
        graph.add_entity("Event", _event(eid, start=_T0 + timedelta(hours=i)))
    for i in range(len(event_ids) - 1):
        graph.add_assertion(
            _accepted_precedes(f"a-{i}", event_ids[i], event_ids[i + 1])
        )
    return graph


# ---------------------------------------------------------------------------
# C2 — temporal sanity (Req 8.3, 26.2)
# ---------------------------------------------------------------------------
def test_c2_rejects_end_before_start():
    event = _event("e-1", start=_T1, end=_T0)
    result = c2_temporal_sanity(event)
    assert result.valid is False
    assert result.failed_check == "C2"
    assert "e-1" in result.conflicting_ids


def test_c2_passes_when_end_missing():
    event = _event("e-2", start=_T1, end=None)
    result = c2_temporal_sanity(event)
    assert result.valid is True
    assert result.failed_check is None


def test_c2_passes_when_end_after_start():
    event = _event("e-3", start=_T0, end=_T1)
    result = c2_temporal_sanity(event)
    assert result.valid is True


def test_c2_passes_when_end_equals_start():
    event = _event("e-4", start=_T1, end=_T1)
    result = c2_temporal_sanity(event)
    assert result.valid is True


def test_c2_accepts_payload_dict():
    # C2 also accepts a payload mapping, not just an Event model.
    payload = _event("e-5", start=_T1, end=_T0).model_dump(mode="json")
    result = c2_temporal_sanity(payload)
    assert result.valid is False
    assert result.failed_check == "C2"


# ---------------------------------------------------------------------------
# C3 — acyclic PRECEDES (Req 8.4, 26.2)
# ---------------------------------------------------------------------------
def test_c3_rejects_cycle_closing_edge():
    # Accepted chain e1 -> e2 -> e3; candidate e3 -> e1 would close a cycle.
    graph = _precedes_chain_graph(["e1", "e2", "e3"])
    candidate = _precedes_candidate("e3", "e1")
    result = c3_acyclic_precedes(candidate, graph)
    assert result.valid is False
    assert result.failed_check == "C3"
    assert "e3" in result.conflicting_ids and "e1" in result.conflicting_ids


def test_c3_accepts_valid_chain_extension():
    # Accepted chain e1 -> e2 -> e3; candidate e3 -> e4 (fresh) is acyclic.
    graph = _precedes_chain_graph(["e1", "e2", "e3"])
    graph.add_entity("Event", _event("e4", start=_T2 + timedelta(hours=1)))
    candidate = _precedes_candidate("e3", "e4")
    result = c3_acyclic_precedes(candidate, graph)
    assert result.valid is True
    assert result.failed_check is None


def test_c3_rejects_self_loop():
    graph = _precedes_chain_graph(["e1", "e2"])
    candidate = _precedes_candidate("e1", "e1")
    result = c3_acyclic_precedes(candidate, graph)
    assert result.valid is False
    assert result.failed_check == "C3"


def test_c3_ignores_non_precedes_predicate():
    # A non-PRECEDES candidate is out of C3's scope and passes unconditionally.
    graph = _precedes_chain_graph(["e1", "e2", "e3"])
    candidate = CandidateAssertion(
        subject_id="e3",
        predicate="RESULTS_IN",
        object_id="e1",
        confidence=0.9,
        source_ref="src-1",
    )
    result = c3_acyclic_precedes(candidate, graph)
    assert result.valid is True

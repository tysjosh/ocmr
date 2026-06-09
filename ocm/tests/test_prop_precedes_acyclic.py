"""Property 5: PRECEDES graph stays acyclic (Feature: ontology-constrained-memory).

Validates Requirements 8.4.

The Event/PRECEDES projection of the ``Graph_Store`` must always be a directed
acyclic graph: a candidate ``PRECEDES`` edge that would close a cycle is never
accepted. Constraint **C3** (``c3_acyclic_precedes``) is the write-time gate that
enforces this, backed by :meth:`GraphStore.would_create_cycle`.

This property drives a random *stream* of candidate ``PRECEDES`` edges among a
fixed set of seeded Event nodes through the same accept/reject logic the write
pipeline uses:

* every candidate is evaluated by ``c3_acyclic_precedes`` against the current
  accepted graph;
* a candidate is added as an **accepted** assertion **iff** C3 passes;
* a candidate C3 rejects is asserted to be genuinely cycle-closing
  (``would_create_cycle`` is ``True`` for it at that moment).

After the whole stream is processed the accepted PRECEDES projection is asserted
to be a DAG (``graph.is_acyclic("PRECEDES")``), and the per-step invariant
guarantees the projection is acyclic after *every* accepted write, not just at
the end.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from hypothesis import given
from hypothesis import strategies as st

from ocm.memory.contracts import CandidateAssertion
from ocm.memory.graph_store import GraphStore
from ocm.ontology.enums import AssertionStatus
from ocm.ontology.models import Assertion, Event
from ocm.tests.markers import pbt_property
from ocm.validation.constraints import c3_acyclic_precedes

_T0 = datetime(2024, 1, 1, 9, 0, 0)


def _event(event_id: str) -> Event:
    return Event(
        id=event_id,
        type="milestone",
        timestamp_start=_T0,
        description=f"event {event_id}",
    )


def _precedes_candidate(subject_id: str, object_id: str) -> CandidateAssertion:
    return CandidateAssertion(
        subject_id=subject_id,
        predicate="PRECEDES",
        object_id=object_id,
        confidence=0.9,
        source_ref="src-prop5",
    )


def _accepted_precedes(assertion_id: str, subject_id: str, object_id: str) -> Assertion:
    return Assertion(
        id=assertion_id,
        subject_id=subject_id,
        predicate="PRECEDES",
        object_id=object_id,
        confidence=0.9,
        status=AssertionStatus.accepted,
        source_ref="src-prop5",
        created_at=_T0 + timedelta(seconds=1),
    )


@st.composite
def precedes_streams(draw: st.DrawFn) -> tuple[int, list[tuple[int, int]]]:
    """Generate a set of Event nodes plus a random stream of PRECEDES edges.

    Returns ``(num_events, edges)`` where ``edges`` is an ordered list of
    ``(subject_index, object_index)`` pairs drawn from ``range(num_events)``.
    Edges are sampled over the full index space — including self-loops and
    cycle-closing pairs — so the cycle-rejection path is exercised, not just
    valid chain extensions.
    """
    num_events = draw(st.integers(min_value=2, max_value=6))
    node_index = st.integers(min_value=0, max_value=num_events - 1)
    edges = draw(
        st.lists(
            st.tuples(node_index, node_index),
            min_size=1,
            max_size=20,
        )
    )
    return num_events, edges


@pbt_property(5, "PRECEDES graph stays acyclic")
@given(stream=precedes_streams())
def test_precedes_projection_stays_acyclic(stream: tuple[int, list[tuple[int, int]]]) -> None:
    num_events, edges = stream

    # Seed the Event entities first so every PRECEDES endpoint resolves to a node.
    graph = GraphStore()
    event_ids = [f"e{i}" for i in range(num_events)]
    for eid in event_ids:
        graph.add_entity("Event", _event(eid))

    accepted_count = 0
    for step, (si, oi) in enumerate(edges):
        subject_id, object_id = event_ids[si], event_ids[oi]
        candidate = _precedes_candidate(subject_id, object_id)

        # Skip duplicates: an already-accepted edge is an idempotent no-op for
        # acyclicity (the projection is unchanged) and not the focus here.
        if graph.has_assertion(subject_id, object_id, "PRECEDES"):
            continue

        result = c3_acyclic_precedes(candidate, graph)

        if result.valid:
            # C3 passed: only now does the edge enter the accepted graph.
            graph.add_assertion(
                _accepted_precedes(f"a-{step}", subject_id, object_id)
            )
            accepted_count += 1
            # Invariant after every accepted write: the projection is still a DAG.
            assert graph.is_acyclic("PRECEDES") is True
        else:
            # C3 rejected: the candidate must be genuinely cycle-closing, and it
            # must NOT have been written to the accepted graph.
            assert result.failed_check == "C3"
            assert graph.would_create_cycle(subject_id, object_id, "PRECEDES") is True
            assert graph.has_assertion(subject_id, object_id, "PRECEDES") is False

    # The accepted PRECEDES projection is always a directed acyclic graph.
    assert graph.is_acyclic("PRECEDES") is True
    # And there are no simple cycles over the PRECEDES edges.
    assert graph.simple_cycles("PRECEDES") == []
    # Sanity: the number of accepted PRECEDES edges matches what we committed.
    assert len(graph.find_edges_by_predicate("PRECEDES")) == accepted_count

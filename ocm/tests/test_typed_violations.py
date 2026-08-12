"""Property tests for the Typed_Violation_Metric (``ocm.evaluation.typed_violations``).

The metric classifies a configuration's durable ACTIVE (accepted) store into four
named violation types plus a total. This module pins the *partition* guarantee of
that classification (Property 5): the four types carve the accepted store into
disjoint buckets so that no Invalid_Active_State is double-counted and the reported
``total`` is exactly the sum of the four per-type counts.

The tests build small synthetic accepted stores through a lightweight fake container
that mimics the ``container.repo.list_assertions("accepted")`` / ``container.graph``
surface the metric reads. This keeps the property hermetic and lets Hypothesis drive
the full space of accepted-store shapes, including the empty-store edge case, without
standing up the whole write pipeline.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

from hypothesis import example, given, settings
from hypothesis import strategies as st

from ocm.evaluation.experiment import durable_constraint_violations
from ocm.evaluation.typed_violations import typed_violations
from ocm.tests.markers import pbt_property

# --------------------------------------------------------------------------- #
# Lightweight fakes mimicking the container.repo / container.graph surface the
# metric reads (list_assertions("accepted"), get_entity_type, get_entity_payload,
# in_edges). Mirrors the stub-graph pattern already used elsewhere in the suite.
# --------------------------------------------------------------------------- #
class _FakeGraph:
    """Minimal graph exposing exactly the read surface ``typed_violations`` uses."""

    def __init__(self) -> None:
        self._types: dict[str, str] = {}
        self._payloads: dict[str, dict] = {}
        # (target_id, predicate) -> list of (source, target, key, data) edge tuples
        self._in_edges: dict[tuple[str, str], list[tuple]] = {}

    def add_node(self, node_id: str, node_type: str, payload: dict | None = None) -> None:
        self._types[node_id] = node_type
        if payload is not None:
            self._payloads[node_id] = payload

    def add_in_edge(self, target_id: str, predicate: str, source_id: str) -> None:
        self._in_edges.setdefault((target_id, predicate), []).append(
            (source_id, target_id, predicate, {})
        )

    def get_entity_type(self, node_id: str) -> str | None:
        return self._types.get(node_id)

    def get_entity_payload(self, node_id: str) -> dict | None:
        return self._payloads.get(node_id)

    def in_edges(self, node_id: str, predicate: str | None = None) -> list[tuple]:
        if predicate is None:
            out: list[tuple] = []
            for (nid, _p), edges in self._in_edges.items():
                if nid == node_id:
                    out.extend(edges)
            return out
        return list(self._in_edges.get((node_id, predicate), []))


class _FakeRepo:
    def __init__(self, accepted: list[Any]) -> None:
        self._accepted = list(accepted)

    def list_assertions(self, status: str | None = None) -> list[Any]:
        if status in (None, "accepted"):
            return list(self._accepted)
        return []


class _RepoWithShadow:
    """Accepted store plus non-accepted "shadow" state under other statuses.

    Used by Property 6 to prove the metric derives every count solely from the
    ACTIVE (accepted) store: the shadow assertions are returned only for a
    non-accepted status and must never influence the report.
    """

    def __init__(self, accepted: list[Any], *, quarantined: list[Any] | None = None) -> None:
        self._accepted = list(accepted)
        self._quarantined = list(quarantined or [])

    def list_assertions(self, status: str | None = None) -> list[Any]:
        if status in (None, "accepted"):
            return list(self._accepted)
        if status == "quarantined":
            return list(self._quarantined)
        return []


class _FakeContainer:
    def __init__(self, repo: Any, graph: _FakeGraph) -> None:
        self.repo = repo
        self.graph = graph


def _assertion(aid: str, subject_id: str, predicate: str, object_id: str) -> Any:
    """A minimal accepted assertion carrying only the fields the metric reads."""
    return SimpleNamespace(
        id=aid, subject_id=subject_id, predicate=predicate, object_id=object_id
    )


# Timestamps for temporal cases: a "bad" Event has end < start (the C2 condition).
_BAD_START = datetime(2020, 1, 2)
_BAD_END = datetime(2020, 1, 1)
_OK_START = datetime(2020, 1, 1)
_OK_END = datetime(2020, 1, 2)

#: Case kinds Hypothesis composes an accepted store from. Each maps to one of the
#: four violation types (or to no violation, for "valid"), so a random mix
#: exercises every classification branch of the metric.
_KINDS = ["schema", "temporal", "evidence", "status_c4", "status_c10", "valid"]


def _populate(graph: _FakeGraph, kinds: list[str], offset: int = 0) -> list[Any]:
    """Add the nodes for ``kinds`` to ``graph`` and return their assertions.

    ``offset`` shifts every generated id so a second, disjoint set of cases can be
    layered onto the *same* graph without colliding with an existing set (used by
    Property 6 to inject non-accepted "shadow" state alongside the accepted store).
    """
    accepted: list[Any] = []
    for j, kind in enumerate(kinds):
        i = offset + j
        if kind == "schema":
            # C9 domain/range: ASSIGNED_TO range is Person, object is a Document.
            task, doc = f"task_{i}", f"doc_{i}"
            graph.add_node(task, "Task")
            graph.add_node(doc, "Document")
            accepted.append(_assertion(f"a_{i}", task, "ASSIGNED_TO", doc))
        elif kind == "temporal":
            # C2: an Event endpoint whose timestamp_end precedes its start.
            bad, ok = f"evt_{i}", f"evt2_{i}"
            graph.add_node(
                bad, "Event", {"timestamp_start": _BAD_START, "timestamp_end": _BAD_END}
            )
            graph.add_node(
                ok, "Event", {"timestamp_start": _OK_START, "timestamp_end": _OK_END}
            )
            accepted.append(_assertion(f"a_{i}", bad, "PRECEDES", ok))
        elif kind == "evidence":
            # C8: a Decision asserted final with zero accepted EVIDENCE_FOR support.
            dec, sv = f"dec_{i}", "status:final"
            graph.add_node(dec, "Decision")
            graph.add_node(sv, "StatusValue", {"value": "final"})
            accepted.append(_assertion(f"a_{i}", dec, "HAS_STATUS", sv))
        elif kind == "status_c4":
            # C4: a done Task with no RESULTS_IN completion Event.
            task, sv = f"task_{i}", "status:done"
            graph.add_node(task, "Task")
            graph.add_node(sv, "StatusValue", {"value": "done"})
            accepted.append(_assertion(f"a_{i}", task, "HAS_STATUS", sv))
        elif kind == "status_c10":
            # C10: a done -> todo flip. A completion Event on the task isolates the
            # C10 (terminal-source) classification from the C4 branch, so exactly
            # the todo assertion is counted illegal.
            task = f"task_{i}"
            done_sv, todo_sv = "status:done", "status:todo"
            comp = f"comp_{i}"
            graph.add_node(task, "Task")
            graph.add_node(done_sv, "StatusValue", {"value": "done"})
            graph.add_node(todo_sv, "StatusValue", {"value": "todo"})
            graph.add_node(comp, "Event")
            graph.add_in_edge(task, "RESULTS_IN", comp)
            accepted.append(_assertion(f"a_{i}_d", task, "HAS_STATUS", done_sv))
            accepted.append(_assertion(f"a_{i}_t", task, "HAS_STATUS", todo_sv))
        elif kind == "valid":
            # Violates nothing: Person OWNS Project.
            person, project = f"person_{i}", f"project_{i}"
            graph.add_node(person, "Person")
            graph.add_node(project, "Project")
            accepted.append(_assertion(f"a_{i}", person, "OWNS", project))
    return accepted


def _build_container(kinds: list[str]) -> tuple[_FakeContainer, list[Any]]:
    """Compose a fake accepted store from a list of case kinds."""
    graph = _FakeGraph()
    accepted = _populate(graph, kinds, offset=0)
    return _FakeContainer(_FakeRepo(accepted), graph), accepted


# --------------------------------------------------------------------------- #
# Property 5
# --------------------------------------------------------------------------- #
# Property 5: The four violation types partition the invalid active state — every
# counted Invalid_Active_State is classified into exactly one of the four types,
# and the reported total equals the sum of the four per-type counts.
@pbt_property(5, "The four violation types partition the invalid active state")
@settings(max_examples=100)
@given(kinds=st.lists(st.sampled_from(_KINDS), max_size=12))
@example(kinds=[])  # empty-store edge case
def test_typed_violation_types_partition(kinds: list[str]) -> None:
    """Property 5.

    Validates: Requirements 7.1, 7.3
    """
    container, accepted = _build_container(kinds)

    report = typed_violations(container)

    per_type = (
        report.schema_invalid,
        report.unsupported_final_decision,
        report.temporally_invalid_interval,
        report.illegal_status_state,
    )

    # Req 7.3: the reported total is exactly the sum of the four per-type counts.
    assert report.total == sum(per_type)

    # Req 7.1: each type is a non-negative count, and every counted
    # Invalid_Active_State is classified into exactly one type — so the buckets
    # are disjoint and the total can never exceed the number of accepted
    # assertions (no assertion is counted in more than one type).
    assert all(count >= 0 for count in per_type)
    assert 0 <= report.total <= len(accepted)

    # Empty-store edge case: an empty accepted store yields an all-zero report.
    if not accepted:
        assert per_type == (0, 0, 0, 0)
        assert report.total == 0


# --------------------------------------------------------------------------- #
# Property 6
# --------------------------------------------------------------------------- #
# Property 6: The metric generalizes the existing contradiction measure — the
# report's single_valued_contradictions equals the count returned by
# durable_constraint_violations, and every count derives solely from the durable
# ACTIVE (accepted) store.
@pbt_property(6, "The metric generalizes the existing contradiction measure")
@settings(max_examples=100)
@given(
    kinds=st.lists(st.sampled_from(_KINDS), max_size=12),
    shadow=st.lists(st.sampled_from(_KINDS), max_size=12),
)
@example(kinds=[], shadow=[])  # empty-store edge case
def test_metric_generalizes_contradiction_measure(
    kinds: list[str], shadow: list[str]
) -> None:
    """Property 6.

    Validates: Requirements 7.4, 7.5
    """
    container, _accepted = _build_container(kinds)

    report = typed_violations(container)

    # Req 7.4: the metric generalizes the legacy measure — its
    # single_valued_contradictions count is exactly what durable_constraint_violations
    # reports for the same durable store, so the existing single-valued-contradiction
    # measure stays derivable alongside the new typed breakdown.
    expected_svc, _accepted_count = durable_constraint_violations(container)
    assert report.single_valued_contradictions == expected_svc

    # Req 7.5: every count derives solely from the durable ACTIVE (accepted) store.
    # Layer the *same* kinds of state onto the graph as non-accepted (quarantined)
    # "shadow" assertions — shapes that WOULD be classified as violations (or as
    # contradictions) if the metric read anything other than the accepted store —
    # and confirm every reported count is unchanged.
    graph = _FakeGraph()
    active = _populate(graph, kinds, offset=0)
    shadow_assertions = _populate(graph, shadow, offset=1000)
    shadowed_container = _FakeContainer(
        _RepoWithShadow(active, quarantined=shadow_assertions), graph
    )

    shadowed = typed_violations(shadowed_container)

    assert shadowed.schema_invalid == report.schema_invalid
    assert shadowed.unsupported_final_decision == report.unsupported_final_decision
    assert shadowed.temporally_invalid_interval == report.temporally_invalid_interval
    assert shadowed.illegal_status_state == report.illegal_status_state
    assert shadowed.total == report.total
    assert shadowed.single_valued_contradictions == report.single_valued_contradictions


# --------------------------------------------------------------------------- #
# Report / tally shape example tests (Req 7.2, 8.1, 8.2)
# --------------------------------------------------------------------------- #
# These are example-based (not property) tests pinning the *shape* of the two
# data models the metric returns: the four-field typed breakdown on
# ``TypedViolationReport`` plus its aggregate fields (Req 7.2), and the
# ``WriteOutcomeTally`` with exactly the four pipeline outcome buckets and no new
# categories (Req 8.1, 8.2).
import dataclasses

from ocm.evaluation.typed_violations import TypedViolationReport, WriteOutcomeTally

#: The four write-pipeline outcome buckets (``WriteResult.accepted`` /
#: ``superseded`` / ``quarantined`` / ``rejected``) — the exact set the tally is
#: derived from, with no new categories introduced (Req 8.2).
_PIPELINE_OUTCOME_BUCKETS = ("accepted", "superseded", "quarantined", "rejected")

#: The four named per-type violation counts the report breaks the accepted store
#: into (Req 7.2), in classification order.
_TYPED_VIOLATION_FIELDS = (
    "schema_invalid",
    "unsupported_final_decision",
    "temporally_invalid_interval",
    "illegal_status_state",
)


def test_report_exposes_four_typed_fields_plus_aggregates() -> None:
    """The report exposes the four per-type fields plus total / legacy / tally.

    Req 7.2: a per-type count for each of the four violation types, alongside the
    ``total`` aggregate, the derivable ``single_valued_contradictions`` legacy
    measure, and the ``write_outcomes`` tally.
    """
    field_names = [f.name for f in dataclasses.fields(TypedViolationReport)]

    # The four typed per-type breakdown fields are present (Req 7.2).
    for name in _TYPED_VIOLATION_FIELDS:
        assert name in field_names

    # Plus the three aggregate/companion fields, and nothing else.
    assert set(field_names) == set(_TYPED_VIOLATION_FIELDS) | {
        "total",
        "single_valued_contradictions",
        "write_outcomes",
    }

    # An empty report defaults every count to zero and carries a fresh tally.
    empty = TypedViolationReport()
    assert (
        empty.schema_invalid
        == empty.unsupported_final_decision
        == empty.temporally_invalid_interval
        == empty.illegal_status_state
        == empty.total
        == empty.single_valued_contradictions
        == 0
    )
    assert isinstance(empty.write_outcomes, WriteOutcomeTally)


def test_report_total_is_sum_of_four_typed_fields_on_a_mixed_store() -> None:
    """On a store with one of each poison class, the four fields each read 1.

    Anchors the Req 7.2 per-type breakdown to concrete values: a store holding one
    SCHEMA, one TEMPORAL, one EVIDENCE, and one STATUS (C4) case reports exactly one
    Invalid_Active_State per type and a ``total`` equal to their sum.
    """
    container, accepted = _build_container(
        ["schema", "temporal", "evidence", "status_c4"]
    )

    report = typed_violations(container)

    assert report.schema_invalid == 1
    assert report.unsupported_final_decision == 1
    assert report.temporally_invalid_interval == 1
    assert report.illegal_status_state == 1
    assert report.total == 4
    assert report.total == (
        report.schema_invalid
        + report.unsupported_final_decision
        + report.temporally_invalid_interval
        + report.illegal_status_state
    )
    # The metric returns the four-bucket tally shape even before the runner fills it.
    assert isinstance(report.write_outcomes, WriteOutcomeTally)


def test_write_outcome_tally_has_exactly_the_four_pipeline_buckets() -> None:
    """The tally exposes exactly accepted/superseded/quarantined/rejected.

    Req 8.1/8.2: the Write_Outcome_Tally mirrors the four existing write-pipeline
    outcome buckets and introduces no new outcome categories.
    """
    field_names = {f.name for f in dataclasses.fields(WriteOutcomeTally)}

    assert field_names == set(_PIPELINE_OUTCOME_BUCKETS)
    # No extra (new-category) buckets beyond the four pipeline outcomes.
    assert len(field_names) == 4


def test_write_outcome_tally_defaults_to_zero_and_accumulates_pipeline_buckets() -> None:
    """The tally defaults to all-zero and holds counts derived from the buckets.

    Req 8.2: the tally is derived from the existing pipeline outcome buckets — a
    fresh tally is all-zero, and assigning per-bucket counts round-trips exactly,
    with no other category to populate.
    """
    tally = WriteOutcomeTally()
    assert (
        tally.accepted
        == tally.superseded
        == tally.quarantined
        == tally.rejected
        == 0
    )

    # Derive a tally from example pipeline bucket sizes (as the runner does by
    # accumulating ``WriteResult`` bucket lengths) — the counts round-trip exactly.
    buckets = {"accepted": 3, "superseded": 1, "quarantined": 2, "rejected": 4}
    filled = WriteOutcomeTally(**buckets)
    assert filled.accepted == 3
    assert filled.superseded == 1
    assert filled.quarantined == 2
    assert filled.rejected == 4
    assert {f.name: getattr(filled, f.name) for f in dataclasses.fields(filled)} == buckets

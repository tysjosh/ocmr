"""Typed_Violation_Metric (Req 7, Req 8.1).

Classifies the durable ACTIVE (accepted) store of a configuration into named
violation types plus a total, generalizing the existing
``durable_constraint_violations`` single-valued-contradiction measure so both the
new typed breakdown and the legacy count stay derivable side by side.

The metric is a **pure read** over ``container.repo.list_assertions("accepted")``
(Req 7.5) and ``container.graph`` — it never mutates state. Because it counts only
accepted assertions, a rejected/quarantined poison write contributes nothing, which
is what makes the Full / Schema_Provenance arms report ``0`` structurally: a write
that never entered the accepted store cannot be an Invalid_Active_State.

Each accepted assertion is classified into **exactly one** of four violation types
(first match wins) using the same ontology the write-time checks use — the C9
relation signatures (``get_relation_signature``), the graph-resolved entity types
(``container.graph.get_entity_type``), and the Event payloads — so the metric and the
pipeline agree on what "invalid" means:

* ``schema_invalid`` — a C9 domain/range violation on an accepted assertion.
* ``temporally_invalid_interval`` — an accepted assertion with an Event endpoint whose
  ``timestamp_end`` precedes its ``timestamp_start`` (the C2 condition).
* ``unsupported_final_decision`` — an accepted ``HAS_STATUS`` asserting a Decision is
  ``final`` while zero accepted ``EVIDENCE_FOR`` edges point into it (the C8 condition).
* ``illegal_status_state`` — an accepted ``HAS_STATUS`` on a Task that is a governance
  miss: a ``done`` Task with no completion Event via ``RESULTS_IN`` (C4), or a status
  reached from a terminal status such as the ``done`` -> ``todo`` flip (a transition the
  C10 map forbids).

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 8.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ocm.ontology.enums import DecisionStatus, TaskStatus
from ocm.ontology.relations import RELATION_SIGNATURES, TASK_STATUS_TRANSITIONS

__all__ = ["WriteOutcomeTally", "TypedViolationReport", "typed_violations"]

#: Predicate names the metric reasons about (mirrors the write pipeline).
_HAS_STATUS = "HAS_STATUS"
_EVIDENCE_FOR = "EVIDENCE_FOR"
_RESULTS_IN = "RESULTS_IN"

#: Canonical id prefix for a shared ``StatusValue`` node (``status:<value>``),
#: matching ``ocm.memory.write_pipeline.STATUS_VALUE_PREFIX``.
_STATUS_VALUE_PREFIX = "status:"

#: Entity types that count as evidence sources for the C8 floor.
_EVIDENCE_SOURCE_TYPES = {"Document", "Event"}


@dataclass
class WriteOutcomeTally:
    """Per-arm counts of durable-write outcomes (Req 8.1).

    Mirrors the four write-pipeline outcome buckets exactly — no new outcome
    categories are introduced (Req 8.2). Defaults to all-zero so the runner can
    accumulate the pipeline's ``WriteResult.summary`` buckets into it.
    """

    accepted: int = 0
    superseded: int = 0
    quarantined: int = 0
    rejected: int = 0


@dataclass
class TypedViolationReport:
    """Typed_Violation_Report for one configuration (arm) (Req 7.1-7.4, 8.1)."""

    schema_invalid: int = 0
    unsupported_final_decision: int = 0
    temporally_invalid_interval: int = 0
    illegal_status_state: int = 0
    total: int = 0
    #: Legacy ``durable_constraint_violations`` count, kept derivable (Req 7.4).
    single_valued_contradictions: int = 0
    write_outcomes: WriteOutcomeTally = field(default_factory=WriteOutcomeTally)


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #
def _parse_dt(value: Any) -> datetime | None:
    """Parse a datetime that may arrive as a ``datetime`` or ISO-8601 string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _status_value(graph: Any, object_id: str) -> str | None:
    """Resolve the status string an assertion's ``StatusValue`` object encodes."""
    payload = None
    try:
        payload = graph.get_entity_payload(object_id)
    except Exception:  # pragma: no cover - defensive
        payload = None
    if payload and payload.get("value") is not None:
        return str(payload["value"])
    if isinstance(object_id, str) and object_id.startswith(_STATUS_VALUE_PREFIX):
        return object_id[len(_STATUS_VALUE_PREFIX):]
    return None


def _is_terminal_status(status: str | None) -> bool:
    """Return True for a task status with no permitted onward transition (C10)."""
    if status is None:
        return False
    try:
        ts = TaskStatus(status)
    except ValueError:
        return False
    allowed = TASK_STATUS_TRANSITIONS.get(ts)
    return allowed is not None and len(allowed) == 0


def _has_invalid_event_endpoint(graph: Any, subject_id: str, object_id: str) -> bool:
    """Return True if an endpoint is an Event with ``timestamp_end < start`` (C2)."""
    for eid in (subject_id, object_id):
        try:
            if graph.get_entity_type(eid) != "Event":
                continue
            payload = graph.get_entity_payload(eid)
        except Exception:  # pragma: no cover - defensive
            continue
        if not payload:
            continue
        start = _parse_dt(payload.get("timestamp_start"))
        end = _parse_dt(payload.get("timestamp_end"))
        if start is not None and end is not None and end < start:
            return True
    return False


def _accepted_evidence_count(graph: Any, decision_id: str) -> int:
    """Count accepted ``EVIDENCE_FOR`` edges from a Document/Event into a node (C8)."""
    try:
        edges = graph.in_edges(decision_id, _EVIDENCE_FOR)
    except Exception:  # pragma: no cover - defensive
        return 0
    return sum(
        1 for s, _o, _k, _d in edges if graph.get_entity_type(s) in _EVIDENCE_SOURCE_TYPES
    )


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #
def typed_violations(container: Any) -> TypedViolationReport:
    """Classify a configuration's durable ACTIVE (accepted) store (Req 7).

    Enumerates ``container.repo.list_assertions("accepted")`` **only** (Req 7.5) and
    classifies each accepted assertion into the first matching violation type. The
    reported ``total`` equals the sum of the four per-type counts (Req 7.3), and
    ``single_valued_contradictions`` carries the legacy
    ``durable_constraint_violations`` count so that measure stays derivable
    alongside the typed breakdown (Req 7.4). An empty accepted store yields an
    all-zero report. The metric never mutates state.
    """
    graph = getattr(container, "graph", None)
    try:
        accepted = list(container.repo.list_assertions("accepted"))
    except Exception:  # pragma: no cover - defensive
        accepted = []

    # Index every accepted HAS_STATUS value per subject so the C10 terminal-source
    # check (e.g. done -> todo) can see the other accepted statuses on the entity.
    statuses_by_subject: dict[str, list[str]] = {}
    for a in accepted:
        if a.predicate == _HAS_STATUS:
            value = _status_value(graph, a.object_id)
            if value is not None:
                statuses_by_subject.setdefault(a.subject_id, []).append(value)

    schema_invalid = 0
    unsupported_final_decision = 0
    temporally_invalid_interval = 0
    illegal_status_state = 0

    for a in accepted:
        subject_type = graph.get_entity_type(a.subject_id) if graph else None
        object_type = graph.get_entity_type(a.object_id) if graph else None

        # 1. schema_invalid — C9 graph domain/range on the accepted assertion.
        sig = RELATION_SIGNATURES.get(a.predicate)
        if sig is not None and (
            (subject_type is not None and subject_type not in sig.source_types)
            or (object_type is not None and object_type not in sig.target_types)
        ):
            schema_invalid += 1
            continue

        # 2. temporally_invalid_interval — C2 on an Event endpoint.
        if graph is not None and _has_invalid_event_endpoint(
            graph, a.subject_id, a.object_id
        ):
            temporally_invalid_interval += 1
            continue

        # 3. unsupported_final_decision — an accepted final HAS_STATUS on a
        #    Decision with zero accepted EVIDENCE_FOR support (the C8 condition).
        if (
            a.predicate == _HAS_STATUS
            and subject_type == "Decision"
            and _status_value(graph, a.object_id) == DecisionStatus.final.value
            and _accepted_evidence_count(graph, a.subject_id) == 0
        ):
            unsupported_final_decision += 1
            continue

        # 4. illegal_status_state — an accepted HAS_STATUS on a Task that is a
        #    governance miss (C4 done-without-completion, or a C10 terminal-source
        #    transition such as done -> todo).
        if a.predicate == _HAS_STATUS and subject_type == "Task":
            value = _status_value(graph, a.object_id)
            # C4: a done Task with no completion Event via RESULTS_IN.
            if value == TaskStatus.done.value and not graph.in_edges(
                a.subject_id, _RESULTS_IN
            ):
                illegal_status_state += 1
                continue
            # C10: this status was reached from a terminal status on the same Task
            # (the done -> todo flip leaves both statuses accepted under gate-only).
            others = [
                p for p in statuses_by_subject.get(a.subject_id, []) if p != value
            ]
            if any(_is_terminal_status(p) for p in others):
                illegal_status_state += 1
                continue

    total = (
        schema_invalid
        + unsupported_final_decision
        + temporally_invalid_interval
        + illegal_status_state
    )

    # Generalize the legacy measure (Req 7.4): keep the single-valued-contradiction
    # count derivable alongside the typed breakdown. The typed types are orthogonal
    # to this count (the poison writes are constructed to not be contradictions), so
    # the two never double-count.
    single_valued_contradictions = 0
    try:
        from ocm.evaluation.experiment import durable_constraint_violations

        single_valued_contradictions, _accepted_count = durable_constraint_violations(
            container
        )
    except Exception:  # pragma: no cover - defensive
        single_valued_contradictions = 0

    return TypedViolationReport(
        schema_invalid=schema_invalid,
        unsupported_final_decision=unsupported_final_decision,
        temporally_invalid_interval=temporally_invalid_interval,
        illegal_status_state=illegal_status_state,
        total=total,
        single_valued_contradictions=single_valued_contradictions,
        write_outcomes=WriteOutcomeTally(),
    )

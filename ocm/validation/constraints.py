"""Constraint Validator (W6) — graph-level constraints C1 through C10.

Each constraint is implemented as a **separate, independently-callable function**
(Req 8.12). Every function returns a :class:`ValidationResult` describing whether
the constraint held (``valid``), and on failure: the ``failed_check`` name, a
human-readable ``reason``, a ``severity``, the ``conflicting_ids`` involved, and
the ``recommended_action`` (``reject`` / ``quarantine`` / ``supersede``).

The :class:`ConstraintValidator` orchestrates the applicable constraints for a
candidate assertion against the :class:`~ocm.memory.graph_store.GraphStore` and
aggregates them into a single :class:`ValidationResult`, returning the first
(most-severe-by-order) failure (Req 8.1). Passing C9 marks a candidate
*eligible*, never *accepted* — final acceptance still requires the contradiction,
provenance, and write-intent checks downstream (Req 8.13).

The contradiction gate (C7) does **not** re-implement contradiction detection;
it **delegates** to the Contradiction_Checker (W7) which is injected as
``contradiction_checker`` (Req 8.8). W7 is implemented in a later task, so this
module never hard-imports it: when no checker is supplied, C7 is a pass-through
no-op and the WritePipeline wiring injects the real checker once available.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.9, 8.10, 8.11, 8.12, 8.13.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping, Protocol

from ocm.memory.contracts import CandidateAssertion, ContradictionResult, ValidationResult
from ocm.memory.graph_store import GraphStore
from ocm.ontology.enums import (
    DecisionStatus,
    PersonStatus,
    Severity,
    TaskStatus,
    WriteIntent,
)
from ocm.ontology.relations import (
    TASK_STATUS_TRANSITIONS,
    UnknownPredicateError,
    get_relation_signature,
)

__all__ = [
    "ContradictionCheckerProtocol",
    "c1_identity_uniqueness",
    "c2_temporal_sanity",
    "c3_acyclic_precedes",
    "c4_done_task_completion_event",
    "c5_inactive_assignee",
    "c6_confidence_bounds",
    "c7_contradiction_gate",
    "c8_decision_evidence_floor",
    "c9_graph_domain_range",
    "c10_task_status_transition",
    "ConstraintValidator",
]


# --------------------------------------------------------------------------- #
# Protocol + small helpers
# --------------------------------------------------------------------------- #
class ContradictionCheckerProtocol(Protocol):
    """Minimal surface C7 relies on from the Contradiction_Checker (W7)."""

    def check(self, c: CandidateAssertion, graph: GraphStore) -> ContradictionResult: ...


def _ok(check: str) -> ValidationResult:
    """A passing result for ``check``."""
    return ValidationResult(valid=True, failed_check=None)


def _fail(
    check: str,
    reason: str,
    severity: Severity,
    recommended_action: str,
    conflicting_ids: Iterable[str] | None = None,
) -> ValidationResult:
    """A failing result for ``check``."""
    return ValidationResult(
        valid=False,
        failed_check=check,
        reason=reason,
        severity=severity,
        conflicting_ids=list(conflicting_ids or []),
        recommended_action=recommended_action,
    )


def _as_payload(entity: Any) -> Mapping[str, Any]:
    """Coerce a Pydantic model or dict-like entity into a payload mapping."""
    if hasattr(entity, "model_dump"):
        return entity.model_dump(mode="json")
    if isinstance(entity, Mapping):
        return entity
    raise TypeError(f"expected a model or mapping, got {type(entity)!r}")


def _parse_dt(value: Any) -> datetime | None:
    """Parse a datetime that may arrive as a ``datetime`` or ISO-8601 string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        # ``datetime.fromisoformat`` handles the values produced by
        # ``model_dump(mode="json")``; tolerate a trailing ``Z``.
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise TypeError(f"cannot parse datetime from {value!r}")


def _coerce_task_status(value: Any) -> TaskStatus | None:
    """Coerce a value into a :class:`TaskStatus`, or ``None`` if not possible."""
    if value is None:
        return None
    if isinstance(value, TaskStatus):
        return value
    try:
        return TaskStatus(value)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# C1 — Identity uniqueness (Req 8.2)
# --------------------------------------------------------------------------- #
def c1_identity_uniqueness(
    entity_type: str, entity_id: str, graph: GraphStore
) -> ValidationResult:
    """Fail when two nodes of a *different* type would share an ``id`` (Req 8.2).

    Re-asserting an entity with the same ``(type, id)`` is fine (idempotent
    upsert). A collision — the id already exists under a different type — is an
    identity violation and is rejected.
    """
    existing_type = graph.get_entity_type(entity_id)
    if existing_type is not None and existing_type != entity_type:
        return _fail(
            "C1",
            f"id {entity_id!r} already exists as type {existing_type!r}, "
            f"cannot reuse it for type {entity_type!r}",
            Severity.high,
            "reject",
            conflicting_ids=[entity_id],
        )
    return _ok("C1")


# --------------------------------------------------------------------------- #
# C2 — Temporal sanity (Req 8.3)
# --------------------------------------------------------------------------- #
def c2_temporal_sanity(event: Any) -> ValidationResult:
    """Fail when an Event's ``timestamp_end`` precedes its ``timestamp_start``.

    A missing ``timestamp_end`` passes (Req 8.3). ``event`` may be an Event
    model or a payload mapping.
    """
    payload = _as_payload(event)
    start = _parse_dt(payload.get("timestamp_start"))
    end = _parse_dt(payload.get("timestamp_end"))
    if end is None:
        return _ok("C2")
    if start is not None and end < start:
        return _fail(
            "C2",
            f"Event {payload.get('id')!r} has timestamp_end ({end.isoformat()}) "
            f"earlier than timestamp_start ({start.isoformat()})",
            Severity.high,
            "reject",
            conflicting_ids=[payload.get("id")] if payload.get("id") else [],
        )
    return _ok("C2")


# --------------------------------------------------------------------------- #
# C3 — Acyclic PRECEDES (Req 8.4)
# --------------------------------------------------------------------------- #
def c3_acyclic_precedes(candidate: CandidateAssertion, graph: GraphStore) -> ValidationResult:
    """Fail when a ``PRECEDES`` edge would close a cycle (Req 8.4).

    Only applies to ``PRECEDES`` candidates; all other predicates pass. Uses the
    graph's accepted PRECEDES projection to test whether a path already runs from
    the object back to the subject (which the new edge would close).
    """
    if candidate.predicate != "PRECEDES":
        return _ok("C3")
    if graph.would_create_cycle(candidate.subject_id, candidate.object_id, "PRECEDES"):
        return _fail(
            "C3",
            f"PRECEDES edge {candidate.subject_id!r} -> {candidate.object_id!r} "
            "would create a cycle",
            Severity.high,
            "reject",
            conflicting_ids=[candidate.subject_id, candidate.object_id],
        )
    return _ok("C3")


# --------------------------------------------------------------------------- #
# C4 — Done-task completion event (Req 8.5)
# --------------------------------------------------------------------------- #
def c4_done_task_completion_event(
    task_id: str, task_status: Any, graph: GraphStore
) -> ValidationResult:
    """Fail when a ``done`` Task has no completion Event via ``RESULTS_IN`` (Req 8.5).

    A completion Event is any Event related to the Task by an accepted
    ``RESULTS_IN`` edge (``Event RESULTS_IN Task``). When none exists the Task is
    recommended for quarantine.
    """
    status = _coerce_task_status(task_status)
    if status != TaskStatus.done:
        return _ok("C4")
    # RESULTS_IN points Event/Decision -> Task, so the Task is the edge *object*.
    completion_edges = graph.in_edges(task_id, "RESULTS_IN")
    if not completion_edges:
        return _fail(
            "C4",
            f"Task {task_id!r} is 'done' but has no completion Event "
            "related by RESULTS_IN",
            Severity.medium,
            "quarantine",
            conflicting_ids=[task_id],
        )
    return _ok("C4")


# --------------------------------------------------------------------------- #
# C5 — Inactive assignee (Req 8.6)
# --------------------------------------------------------------------------- #
def c5_inactive_assignee(candidate: CandidateAssertion, graph: GraphStore) -> ValidationResult:
    """Fail an ``ASSIGNED_TO`` whose target Person is ``inactive`` (Req 8.6).

    Only applies to ``ASSIGNED_TO`` candidates. An ``active`` or ``unknown``
    (or unresolved) assignee passes; an ``inactive`` assignee is quarantined.
    """
    if candidate.predicate != "ASSIGNED_TO":
        return _ok("C5")
    payload = graph.get_entity_payload(candidate.object_id)
    if payload is None:
        # Missing assignee is a referential concern handled by W5/C9, not C5.
        return _ok("C5")
    if payload.get("status") == PersonStatus.inactive.value:
        return _fail(
            "C5",
            f"ASSIGNED_TO target Person {candidate.object_id!r} is inactive",
            Severity.medium,
            "quarantine",
            conflicting_ids=[candidate.object_id],
        )
    return _ok("C5")


# --------------------------------------------------------------------------- #
# C6 — Confidence bounds (Req 8.7)
# --------------------------------------------------------------------------- #
def c6_confidence_bounds(confidence: float) -> ValidationResult:
    """Fail when ``confidence`` falls outside [0, 1] (Req 8.7).

    Pydantic structurally guards confidence on the models; C6 is the graph-level
    guarantee so a hand-built value cannot slip through.
    """
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return _fail(
            "C6", f"confidence {confidence!r} is not numeric", Severity.high, "reject"
        )
    if not (0.0 <= value <= 1.0):
        return _fail(
            "C6",
            f"confidence {value} is outside the range [0, 1]",
            Severity.high,
            "reject",
        )
    return _ok("C6")


# --------------------------------------------------------------------------- #
# C7 — Contradiction gate (Req 8.8) — delegates to W7
# --------------------------------------------------------------------------- #
def _accepted_confidences_near(graph: GraphStore, candidate: CandidateAssertion) -> dict[str, float]:
    """Map ``assertion_id -> confidence`` for accepted edges around the candidate.

    Scans the candidate subject/object neighbourhood (the edges where a
    contradiction would surface) so C7 can verify a conflicting accepted
    assertion is itself high-confidence without re-implementing detection.
    """
    out: dict[str, float] = {}
    seen: set[tuple[str, str, str]] = set()
    edges = (
        graph.out_edges(candidate.subject_id)
        + graph.in_edges(candidate.subject_id)
        + graph.out_edges(candidate.object_id)
        + graph.in_edges(candidate.object_id)
    )
    for s, o, k, data in edges:
        if (s, o, k) in seen:
            continue
        seen.add((s, o, k))
        aid = data.get("assertion_id")
        if aid is not None:
            out[aid] = float(data.get("confidence", 0.0))
    return out


def _evidence_count(candidate: CandidateAssertion, graph: GraphStore) -> int:
    """Units of supporting evidence for a candidate assertion (Algorithm 1, e_min).

    An assertion's evidence is its provenance — a present ``source_ref`` counts
    as one unit — plus any accepted ``EVIDENCE_FOR`` edges from a Document/Event
    into its subject entity. This is the write-time analogue of the C8 evidence
    floor used for final Decisions, applied here to gate supersession.
    """
    count = 1 if (candidate.source_ref or "").strip() else 0
    for s, _o, _k, _d in graph.in_edges(candidate.subject_id, "EVIDENCE_FOR"):
        if graph.get_entity_type(s) in {"Document", "Event"}:
            count += 1
    return count


def c7_contradiction_gate(
    candidate: CandidateAssertion,
    graph: GraphStore,
    contradiction_checker: ContradictionCheckerProtocol | None = None,
    settings: Any = None,
) -> ValidationResult:
    """Block silent acceptance of a high-confidence contradiction (Req 8.8).

    Delegates detection entirely to the Contradiction_Checker (W7): C7 never
    duplicates contradiction logic. When no checker is injected the gate is a
    no-op pass (the real checker is wired in by the WritePipeline).

    A contradiction is *blocking* when the candidate's confidence exceeds the
    high-confidence threshold and at least one conflicting **accepted** assertion
    is also above it (the checker's high-severity/hard verdict also qualifies).
    On a blocking conflict C7 fails; the recommended action follows Algorithm 1:

    - ``write_intent == correction`` **and** the candidate is more confident than
      the incumbent by a margin ``delta`` (``c(a) - c(a_old) > delta``) **and**
      it carries at least ``e_min`` units of supporting evidence ->
      ``supersede`` (the correction has earned the right to replace memory),
    - otherwise (a bare ``new_fact``, or a ``correction`` that does not dominate
      the incumbent / lacks evidence) -> ``quarantine``.

    Lower-confidence contradictions pass C7 (recorded as a soft warning at W7).
    """
    if contradiction_checker is None:
        return _ok("C7")

    result = contradiction_checker.check(candidate, graph)
    if not result.has_conflict:
        return _ok("C7")

    threshold = float(getattr(settings, "contradiction_high_confidence", 0.8))
    candidate_high = float(candidate.confidence) > threshold

    accepted_conf = _accepted_confidences_near(graph, candidate)
    conflict_ids = result.conflicting_assertion_ids or []
    counterpart_high = any(
        accepted_conf.get(aid, 0.0) > threshold for aid in conflict_ids
    )
    # The checker classifying the conflict as hard / high severity is itself a
    # high-confidence signal even if the edge confidences are unavailable.
    checker_high = result.kind == "hard" or result.severity == Severity.high

    blocking = candidate_high and (counterpart_high or checker_high)
    if not blocking:
        # Soft (low-confidence) contradiction: permitted; severity tracked at W7.
        return _ok("C7")

    # Algorithm 1, line 7: a correction may supersede only when it *dominates*
    # the incumbent — strictly more confident by ``delta`` — and is grounded by
    # at least ``e_min`` units of evidence. A correction that ties/loses on
    # confidence (incumbent dominates) or lacks evidence is quarantined.
    if (
        candidate.write_intent == WriteIntent.update
        and bool(getattr(settings, "authoritative_update_supersede", False))
    ):
        # Authoritative single-valued state update from a trusted source: the
        # latest value replaces the incumbent unconditionally (no margin /
        # evidence gate). This is the correct semantics for trusted state (e.g.
        # a dialogue-state slot the user just changed), as opposed to a
        # ``correction`` that must dominate an untrusted incumbent.
        action = "supersede"
        reason = (
            "authoritative update supersedes incumbent assertion(s) "
            f"{conflict_ids} (single-valued state replacement)"
        )
    elif candidate.write_intent == WriteIntent.correction:
        delta = float(getattr(settings, "supersede_margin", 0.0))
        e_min = int(getattr(settings, "supersede_evidence_min", 1))
        incumbent_conf = max(
            (accepted_conf.get(aid, 0.0) for aid in conflict_ids), default=0.0
        )
        margin_ok = (float(candidate.confidence) - incumbent_conf) > delta
        evidence_ok = _evidence_count(candidate, graph) >= e_min
        if margin_ok and evidence_ok:
            action = "supersede"
            reason = (
                "high-confidence contradiction; correction dominates incumbent "
                f"(margin>{delta}, evidence>={e_min}) and supersedes {conflict_ids}"
            )
        else:
            action = "quarantine"
            if not margin_ok:
                why = (
                    f"correction confidence {float(candidate.confidence):.3f} does not "
                    f"exceed incumbent {incumbent_conf:.3f} by margin {delta}"
                )
            else:
                why = f"correction lacks the required supporting evidence (e_min={e_min})"
            reason = (
                f"high-confidence contradiction with accepted assertion(s) {conflict_ids}; "
                f"{why}; quarantined for review"
            )
    else:
        action = "quarantine"
        reason = (
            f"high-confidence contradiction with accepted assertion(s) {conflict_ids}; "
            "new_fact cannot be silently accepted"
        )
    return _fail(
        "C7",
        result.reason or reason,
        result.severity or Severity.high,
        action,
        conflicting_ids=conflict_ids,
    )


# --------------------------------------------------------------------------- #
# C8 — Decision evidence floor (Req 8.9)
# --------------------------------------------------------------------------- #
def c8_decision_evidence_floor(
    decision_id: str, decision_status: Any, graph: GraphStore, settings: Any = None
) -> ValidationResult:
    """Fail a ``final`` Decision lacking enough ``EVIDENCE_FOR`` support (Req 8.9).

    Counts accepted ``EVIDENCE_FOR`` edges from a Document or Event into the
    Decision; when the count is below ``settings.decision_evidence_floor`` (default
    1) the Decision is quarantined.
    """
    status = decision_status.value if isinstance(decision_status, DecisionStatus) else decision_status
    if status != DecisionStatus.final.value:
        return _ok("C8")
    floor = int(getattr(settings, "decision_evidence_floor", 1))
    evidence_edges = graph.in_edges(decision_id, "EVIDENCE_FOR")
    evidence_count = sum(
        1
        for s, _o, _k, _d in evidence_edges
        if graph.get_entity_type(s) in {"Document", "Event"}
    )
    if evidence_count < floor:
        return _fail(
            "C8",
            f"final Decision {decision_id!r} has {evidence_count} EVIDENCE_FOR "
            f"support(s), below the floor of {floor}",
            Severity.medium,
            "quarantine",
            conflicting_ids=[decision_id],
        )
    return _ok("C8")


# --------------------------------------------------------------------------- #
# C9 — Graph-level domain/range (Req 8.10)
# --------------------------------------------------------------------------- #
def c9_graph_domain_range(candidate: CandidateAssertion, graph: GraphStore) -> ValidationResult:
    """Validate the predicate against the *resolved* subject/object types (Req 8.10).

    Unlike W5's structural check, C9 resolves the actual entity types from the
    Graph_Store and verifies them against the relation signature's
    ``source_types`` / ``target_types``. Passing makes the candidate *eligible*,
    not *accepted* (Req 8.13).
    """
    try:
        sig = get_relation_signature(candidate.predicate)
    except UnknownPredicateError:
        return _fail(
            "C9",
            f"unknown predicate {candidate.predicate!r}",
            Severity.high,
            "reject",
        )

    subject_type = graph.get_entity_type(candidate.subject_id)
    object_type = graph.get_entity_type(candidate.object_id)

    if subject_type is None or object_type is None:
        missing = candidate.subject_id if subject_type is None else candidate.object_id
        return _fail(
            "C9",
            f"cannot resolve entity type for {missing!r} to validate domain/range",
            Severity.high,
            "reject",
            conflicting_ids=[missing],
        )

    if subject_type not in sig.source_types:
        return _fail(
            "C9",
            f"subject type {subject_type!r} not allowed for {candidate.predicate!r} "
            f"(allowed: {sorted(sig.source_types)})",
            Severity.high,
            "reject",
            conflicting_ids=[candidate.subject_id],
        )
    if object_type not in sig.target_types:
        return _fail(
            "C9",
            f"object type {object_type!r} not allowed for {candidate.predicate!r} "
            f"(allowed: {sorted(sig.target_types)})",
            Severity.high,
            "reject",
            conflicting_ids=[candidate.object_id],
        )
    return _ok("C9")


# --------------------------------------------------------------------------- #
# C10 — Task status transition (Req 8.11)
# --------------------------------------------------------------------------- #
def c10_task_status_transition(
    current_status: Any, next_status: Any, write_intent: Any = WriteIntent.new_fact
) -> ValidationResult:
    """Fail a Task status transition not permitted by the map (Req 8.11).

    ``correction`` write_intent bypasses the map (permitted). An unchanged status
    is treated as a no-op and passes. Otherwise the transition must appear in
    ``TASK_STATUS_TRANSITIONS[current]`` or the Task is quarantined.
    """
    intent = write_intent if isinstance(write_intent, WriteIntent) else WriteIntent(write_intent)
    if intent == WriteIntent.correction:
        return _ok("C10")

    current = _coerce_task_status(current_status)
    nxt = _coerce_task_status(next_status)
    if current is None or nxt is None:
        return _fail(
            "C10",
            f"unrecognized task status transition {current_status!r} -> {next_status!r}",
            Severity.medium,
            "quarantine",
        )
    if current == nxt:
        return _ok("C10")
    allowed = TASK_STATUS_TRANSITIONS.get(current, set())
    if nxt not in allowed:
        return _fail(
            "C10",
            f"task transition {current.value!r} -> {nxt.value!r} is not permitted "
            f"(allowed: {sorted(s.value for s in allowed)})",
            Severity.medium,
            "quarantine",
        )
    return _ok("C10")


# --------------------------------------------------------------------------- #
# Aggregating validator
# --------------------------------------------------------------------------- #
class ConstraintValidator:
    """Run the applicable C1-C10 constraints and aggregate into one result (Req 8.1).

    The validator runs constraints in canonical order and returns the **first**
    failure with its ``reason`` / ``severity`` / ``conflicting_ids`` /
    ``recommended_action``; if none fail it returns a passing
    :class:`ValidationResult`. Constraints that need entity-level context derive
    it from the candidate's subject/object payloads in the Graph_Store; extra
    newly-created entities may be supplied via ``entities`` (for C1/C2), and an
    explicit Task transition via ``task_transition`` (for C10).

    C7 delegates to the Contradiction_Checker (W7) which is the single source of
    contradiction truth. The validator binds a default
    :class:`~ocm.validation.contradiction_checker.ContradictionChecker` (using its
    ``settings``) so C7 runs out of the box; callers (e.g. the WritePipeline) may
    inject an alternative checker per call, or pass ``contradiction_checker=None``
    to ``validate`` to explicitly disable the gate.
    """

    # Sentinel so callers can distinguish "not provided" from an explicit None.
    _UNSET = object()

    def __init__(
        self,
        settings: Any = None,
        contradiction_checker: ContradictionCheckerProtocol | None = None,
    ) -> None:
        self.settings = settings
        # Bind a real W7 checker by default (single source of contradiction truth).
        if contradiction_checker is None:
            from ocm.validation.contradiction_checker import ContradictionChecker

            contradiction_checker = ContradictionChecker(settings)
        self.contradiction_checker = contradiction_checker

    def validate(
        self,
        candidate: CandidateAssertion,
        graph: GraphStore,
        *,
        contradiction_checker: ContradictionCheckerProtocol | None | object = _UNSET,
        settings: Any = None,
        entities: Iterable[Any] | None = None,
        task_transition: tuple[Any, Any] | None = None,
    ) -> ValidationResult:
        settings = settings if settings is not None else self.settings
        # Default to the bound checker; an explicit ``None`` disables C7.
        if contradiction_checker is self._UNSET:
            contradiction_checker = self.contradiction_checker
        checks = self._collect_checks(
            candidate,
            graph,
            contradiction_checker=contradiction_checker,
            settings=settings,
            entities=entities,
            task_transition=task_transition,
        )
        for result in checks:
            if not result.valid:
                return result
        return ValidationResult(valid=True)

    # -- orchestration ---------------------------------------------------- #
    def _collect_checks(
        self,
        candidate: CandidateAssertion,
        graph: GraphStore,
        *,
        contradiction_checker: ContradictionCheckerProtocol | None,
        settings: Any,
        entities: Iterable[Any] | None,
        task_transition: tuple[Any, Any] | None,
    ) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        extra_entities = list(entities or [])

        # C1 — identity uniqueness for any newly-created entities.
        for ent in extra_entities:
            payload = _as_payload(ent)
            etype = payload.get("type") or payload.get("__type__")
            eid = payload.get("id")
            if etype and eid:
                results.append(c1_identity_uniqueness(etype, eid, graph))

        # C2 — temporal sanity for any Event among supplied/endpoint entities.
        for payload in self._event_payloads(extra_entities, graph, candidate):
            results.append(c2_temporal_sanity(payload))

        # C3 — acyclic PRECEDES.
        results.append(c3_acyclic_precedes(candidate, graph))

        # C4 — done-task completion event (for any Task endpoint set to done).
        for task_id, task_status in self._task_payloads(candidate, graph):
            results.append(c4_done_task_completion_event(task_id, task_status, graph))

        # C5 — inactive assignee.
        results.append(c5_inactive_assignee(candidate, graph))

        # C6 — confidence bounds.
        results.append(c6_confidence_bounds(candidate.confidence))

        # C7 — contradiction gate (delegates to W7; no-op without a checker).
        results.append(
            c7_contradiction_gate(candidate, graph, contradiction_checker, settings)
        )

        # C8 — decision evidence floor (for any final Decision endpoint).
        for dec_id, dec_status in self._decision_payloads(candidate, graph):
            results.append(c8_decision_evidence_floor(dec_id, dec_status, graph, settings))

        # C9 — graph-level domain/range.
        results.append(c9_graph_domain_range(candidate, graph))

        # C10 — task status transition (only when an explicit transition is given).
        if task_transition is not None:
            current, nxt = task_transition
            results.append(
                c10_task_status_transition(current, nxt, candidate.write_intent)
            )

        return results

    # -- payload selectors ------------------------------------------------ #
    @staticmethod
    def _event_payloads(
        extra_entities: list[Any],
        graph: GraphStore,
        candidate: CandidateAssertion,
    ) -> list[Mapping[str, Any]]:
        events: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        for ent in extra_entities:
            payload = _as_payload(ent)
            if "timestamp_start" in payload and payload.get("id") not in seen:
                events.append(payload)
                seen.add(payload.get("id"))
        for eid in (candidate.subject_id, candidate.object_id):
            if graph.get_entity_type(eid) == "Event" and eid not in seen:
                payload = graph.get_entity_payload(eid)
                if payload is not None:
                    events.append(payload)
                    seen.add(eid)
        return events

    @staticmethod
    def _task_payloads(
        candidate: CandidateAssertion,
        graph: GraphStore,
    ) -> list[tuple[str, Any]]:
        tasks: list[tuple[str, Any]] = []
        for eid in (candidate.subject_id, candidate.object_id):
            if graph.get_entity_type(eid) == "Task":
                payload = graph.get_entity_payload(eid)
                if payload is not None:
                    tasks.append((eid, payload.get("status")))
        return tasks

    @staticmethod
    def _decision_payloads(
        candidate: CandidateAssertion,
        graph: GraphStore,
    ) -> list[tuple[str, Any]]:
        decisions: list[tuple[str, Any]] = []
        for eid in (candidate.subject_id, candidate.object_id):
            if graph.get_entity_type(eid) == "Decision":
                payload = graph.get_entity_payload(eid)
                if payload is not None:
                    decisions.append((eid, payload.get("status")))
        return decisions

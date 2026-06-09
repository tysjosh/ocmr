"""Contradiction Checker (W7) — the single source of contradiction truth.

The :class:`ContradictionChecker` inspects a :class:`CandidateAssertion` against
the accepted memory held in the :class:`~ocm.memory.graph_store.GraphStore` and
returns a :class:`~ocm.memory.contracts.ContradictionResult`. Constraint C7
(``ocm/validation/constraints.py``) delegates to this checker rather than
re-implementing detection, so contradiction logic lives in exactly one place
(Req 8.8, Req 9.7).

Detected contradiction categories (Req 9.1-9.6):

- **Explicit ``CONTRADICTS`` links (Req 9.4).** An accepted ``CONTRADICTS`` edge
  incident to the candidate's subject or object signals a curated conflict.
- **Single-valued / exact-predicate conflicts (Req 9.2, 9.3, 9.5).** For a
  relation whose cardinality permits only one target (``m:1`` such as
  ``ASSIGNED_TO``, or ``1:1``), an existing accepted assertion on the same
  subject (and, for ``1:1``, the same object) pointing at a *different* target is
  a conflict. Re-asserting the *same* triple is an idempotent no-op. A
  status-bearing single-valued relation surfaces a status conflict the same way
  (Req 9.3).
- **Temporal overlap conflicts (Req 9.6).** When the conflicting single-valued
  assertions both carry validity windows (``valid_from`` / ``valid_to``) that
  *overlap*, the conflict is classified ``temporal``; non-overlapping windows are
  a valid historical succession and are **not** a contradiction.

Severity / hardness grading (Req 9.1): a non-temporal contradiction where both
the candidate and the conflicting accepted assertion exceed the high-confidence
threshold (``settings.contradiction_high_confidence``, default 0.8) is a **hard**
contradiction (``severity=high``); otherwise it is a **soft** warning
(``severity=low``) that downstream gates may permit. The recommended action is
``supersede`` for a high-confidence ``correction``, ``quarantine`` for any other
high-confidence conflict, and ``accept`` for a soft warning (Req 9.7).

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 8.8.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ocm.memory.contracts import CandidateAssertion, ContradictionResult
from ocm.memory.graph_store import EdgeTuple, GraphStore
from ocm.ontology.enums import Severity, WriteIntent
from ocm.ontology.relations import (
    Cardinality,
    UnknownPredicateError,
    get_relation_signature,
)

DEFAULT_HIGH_CONFIDENCE = 0.8

# Cardinalities that permit only a single target for a given subject.
_SINGLE_VALUED = {Cardinality.M_TO_ONE, Cardinality.ONE_TO_ONE}


class ContradictionChecker:
    """Detect hard, soft, and temporal contradictions for a candidate (W7).

    ``settings`` may be bound at construction so the checker satisfies the
    :class:`~ocm.validation.constraints.ContradictionCheckerProtocol`
    (``check(candidate, graph)``). The high-confidence threshold is read from
    ``settings.contradiction_high_confidence`` and falls back to
    :data:`DEFAULT_HIGH_CONFIDENCE` (0.8).
    """

    def __init__(self, settings: Any = None) -> None:
        self.settings = settings

    # -- public API ------------------------------------------------------- #
    def check(
        self,
        candidate: CandidateAssertion,
        graph: GraphStore,
        settings: Any = None,
    ) -> ContradictionResult:
        """Return the contradiction verdict for ``candidate`` against ``graph``.

        Detection order is most-authoritative first: explicit ``CONTRADICTS``
        links, then single-valued / exact-predicate (and temporal) conflicts.
        The first detected conflict is returned; when none is found the result
        has ``has_conflict=False``.
        """
        threshold = self._threshold(settings if settings is not None else self.settings)

        contradicts = self._detect_explicit_contradicts(candidate, graph)
        if contradicts is not None:
            return self._build_result(candidate, contradicts, threshold, is_temporal=False)

        single = self._detect_single_valued(candidate, graph)
        if single is not None:
            conflict_ids, counterpart_conf, is_temporal = single
            return self._build_result(
                candidate,
                (conflict_ids, counterpart_conf, self._single_reason(candidate, conflict_ids, is_temporal)),
                threshold,
                is_temporal=is_temporal,
            )

        return ContradictionResult(has_conflict=False)

    # -- threshold -------------------------------------------------------- #
    @staticmethod
    def _threshold(settings: Any) -> float:
        if settings is None:
            return DEFAULT_HIGH_CONFIDENCE
        return float(getattr(settings, "contradiction_high_confidence", DEFAULT_HIGH_CONFIDENCE))

    # -- detectors -------------------------------------------------------- #
    def _detect_explicit_contradicts(
        self, candidate: CandidateAssertion, graph: GraphStore
    ) -> tuple[list[str], float, str] | None:
        """Find an accepted ``CONTRADICTS`` edge incident to the candidate (Req 9.4).

        Returns ``(conflicting_assertion_ids, counterpart_confidence, reason)`` or
        ``None``. The candidate itself asserting ``CONTRADICTS`` is not a conflict.
        """
        if candidate.predicate == "CONTRADICTS":
            return None

        edges: list[EdgeTuple] = []
        for node in (candidate.subject_id, candidate.object_id):
            edges.extend(graph.out_edges(node, "CONTRADICTS"))
            edges.extend(graph.in_edges(node, "CONTRADICTS"))
        if not edges:
            return None

        conflict_ids: list[str] = []
        counterpart_conf = 0.0
        endpoints: set[str] = set()
        for s, o, _k, data in edges:
            aid = data.get("assertion_id")
            if aid is not None and aid not in conflict_ids:
                conflict_ids.append(aid)
            counterpart_conf = max(counterpart_conf, float(data.get("confidence", 0.0)))
            endpoints.update({s, o})
        reason = (
            f"explicit CONTRADICTS link involving {sorted(endpoints)} conflicts with "
            f"candidate {candidate.subject_id!r} -[{candidate.predicate}]-> {candidate.object_id!r}"
        )
        return conflict_ids, counterpart_conf, reason

    def _detect_single_valued(
        self, candidate: CandidateAssertion, graph: GraphStore
    ) -> tuple[list[str], float, bool] | None:
        """Find single-valued / exact-predicate conflicts (Req 9.2, 9.3, 9.5, 9.6).

        Returns ``(conflicting_assertion_ids, counterpart_confidence, is_temporal)``
        or ``None``. ``is_temporal`` is ``True`` when every conflicting assertion
        carries a validity window that overlaps the candidate's (Req 9.6).
        """
        try:
            sig = get_relation_signature(candidate.predicate)
        except UnknownPredicateError:
            return None
        if sig.cardinality not in _SINGLE_VALUED:
            return None

        # Subject side: a m:1 / 1:1 subject may have only one target.
        candidates: list[EdgeTuple] = [
            e for e in graph.out_edges(candidate.subject_id, candidate.predicate)
            if e[1] != candidate.object_id
        ]
        # Object side: a 1:1 object may be claimed by only one subject.
        if sig.cardinality == Cardinality.ONE_TO_ONE:
            candidates.extend(
                e for e in graph.in_edges(candidate.object_id, candidate.predicate)
                if e[0] != candidate.subject_id
            )

        if not candidates:
            return None

        # Keep only conflicts whose validity windows actually overlap. Missing
        # windows are treated as always-valid (and therefore overlapping).
        conflicting: list[EdgeTuple] = []
        temporal_flags: list[bool] = []
        for edge in candidates:
            overlaps, both_dated = self._windows_overlap(candidate, edge[3])
            if overlaps:
                conflicting.append(edge)
                temporal_flags.append(both_dated)
        if not conflicting:
            return None

        conflict_ids: list[str] = []
        counterpart_conf = 0.0
        for _s, _o, _k, data in conflicting:
            aid = data.get("assertion_id")
            if aid is not None and aid not in conflict_ids:
                conflict_ids.append(aid)
            counterpart_conf = max(counterpart_conf, float(data.get("confidence", 0.0)))

        # Temporal classification only when every retained conflict is a dated
        # overlap (otherwise it is a plain exact-predicate/cardinality conflict).
        is_temporal = bool(temporal_flags) and all(temporal_flags)
        return conflict_ids, counterpart_conf, is_temporal

    # -- helpers ---------------------------------------------------------- #
    @staticmethod
    def _windows_overlap(
        candidate: CandidateAssertion, edge_data: dict[str, Any]
    ) -> tuple[bool, bool]:
        """Whether the candidate and an existing edge share validity time.

        Returns ``(overlaps, both_dated)``. When either side lacks a window the
        conflict is treated as overlapping (``overlaps=True``) but not temporal
        (``both_dated=False``). When both carry windows, ``overlaps`` reflects a
        true interval intersection and ``both_dated`` is ``True`` (Req 9.6).
        """
        c_from, c_to = candidate.valid_from, candidate.valid_to
        e_from = ContradictionChecker._as_dt(edge_data.get("valid_from"))
        e_to = ContradictionChecker._as_dt(edge_data.get("valid_to"))

        candidate_dated = c_from is not None or c_to is not None
        edge_dated = e_from is not None or e_to is not None
        if not (candidate_dated and edge_dated):
            return True, False

        # Treat an open end as +/- infinity.
        c_start = c_from or datetime.min
        c_end = c_to or datetime.max
        e_start = e_from or datetime.min
        e_end = e_to or datetime.max
        # Normalize naive/aware mismatches by comparing on naive replacements.
        c_start, c_end, e_start, e_end = (
            ContradictionChecker._naive(c_start),
            ContradictionChecker._naive(c_end),
            ContradictionChecker._naive(e_start),
            ContradictionChecker._naive(e_end),
        )
        overlaps = c_start <= e_end and e_start <= c_end
        return overlaps, True

    @staticmethod
    def _as_dt(value: Any) -> datetime | None:
        if value is None or isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return None

    @staticmethod
    def _naive(value: datetime) -> datetime:
        """Drop tzinfo so naive and aware datetimes remain comparable."""
        return value.replace(tzinfo=None) if value.tzinfo is not None else value

    @staticmethod
    def _single_reason(
        candidate: CandidateAssertion, conflict_ids: list[str], is_temporal: bool
    ) -> str:
        kind = "temporal overlap" if is_temporal else "single-valued"
        return (
            f"{kind} conflict on {candidate.predicate!r}: subject "
            f"{candidate.subject_id!r} already has accepted assertion(s) {conflict_ids} "
            f"with a different target than {candidate.object_id!r}"
        )

    def _build_result(
        self,
        candidate: CandidateAssertion,
        detected: tuple[list[str], float, str],
        threshold: float,
        *,
        is_temporal: bool,
    ) -> ContradictionResult:
        """Grade a detected conflict into a :class:`ContradictionResult` (Req 9.1, 9.7)."""
        conflict_ids, counterpart_conf, reason = detected
        candidate_high = float(candidate.confidence) > threshold
        counterpart_high = counterpart_conf > threshold
        both_high = candidate_high and counterpart_high

        if is_temporal:
            kind = "temporal"
        else:
            kind = "hard" if both_high else "soft"

        if both_high:
            severity = Severity.high
            action = (
                "supersede"
                if candidate.write_intent == WriteIntent.correction
                else "quarantine"
            )
        else:
            severity = Severity.low
            action = "accept"

        return ContradictionResult(
            has_conflict=True,
            severity=severity,
            reason=reason,
            conflicting_assertion_ids=conflict_ids,
            kind=kind,
            recommended_action=action,
        )

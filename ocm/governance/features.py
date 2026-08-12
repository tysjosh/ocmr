"""RAHGM feature extraction — the typed status vector ``f(u)`` (paper §3.3).

Escalation risk is derived from *OCMR's own constraint failures*, not from a
separate opaque confidence score. This module maps OCMR's check outcomes onto the
paper's five-component status vector

``f(u) = [f_e, f_s, f_t, f_v, f_c]``

where each component is ``0.0`` pass, ``0.5`` unresolved, ``1.0`` fail, and
collects the three scalar modifiers the router also consumes: consequence ``q``,
reversibility ``v``, and source authority ``a``, plus ``k``, the number of
simultaneously unresolved or failed checks.

Component provenance (Req 1.3–1.7):

===========  ====================================================  ==============================
Component    OCMR sources                                          ``0.5`` (unresolved) condition
===========  ====================================================  ==============================
``f_e``      C1 identity uniqueness, entity resolution status      unresolved alias / possible match
``f_s``      W5 structural schema checks, C9 graph domain/range    single-valued cardinality pressure
``f_t``      C2 temporal sanity, C3 acyclic PRECEDES, C10          missing / unordered timestamps
``f_v``      C8 evidence floor, Algorithm 1 ``e_min``              ``0 < evidence < floor``
``f_c``      W7 Contradiction_Checker via C7                       soft or temporal conflict
===========  ====================================================  ==============================

Nothing here re-implements a constraint: the extractor reads the
:class:`~ocm.memory.contracts.ValidationResult` OCMR already produced and queries
the graph for the incumbent state the encoding needs. Extraction is fully
deterministic and never calls a language model (Req 1.10).

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping

from ocm.memory.contracts import CandidateAssertion, ValidationResult
from ocm.memory.graph_store import GraphStore
from ocm.ontology.enums import Severity, TaskStatus, WriteIntent
from ocm.ontology.relations import (
    Cardinality,
    UnknownPredicateError,
    get_relation_signature,
)

# --------------------------------------------------------------------------- #
# Encoding constants (immutable — adaptation may never change these, Req 7.4)
# --------------------------------------------------------------------------- #
#: The three admissible encodings of a check outcome (Req 1.2).
PASS = 0.0
UNRESOLVED = 0.5
FAIL = 1.0

#: OCMR check ids that contribute to each component of ``f(u)``.
ENTITY_CHECKS = frozenset({"C1"})
SCHEMA_CHECKS = frozenset({"C9"})
TEMPORAL_CHECKS = frozenset({"C2", "C3", "C10"})
EVIDENCE_CHECKS = frozenset({"C8"})
CONTRADICTION_CHECKS = frozenset({"C7", "HAS_STATUS"})

#: ``failed_check`` prefix used by W5 (``schema.required_fields`` etc.).
SCHEMA_CHECK_PREFIX = "schema."

#: Predicates whose cardinality admits at most one active object per subject.
_SINGLE_VALUED = frozenset({Cardinality.ONE_TO_ONE, Cardinality.M_TO_ONE})

_EVIDENCE_FOR = "EVIDENCE_FOR"
_HAS_STATUS = "HAS_STATUS"
_RESULTS_IN = "RESULTS_IN"


# --------------------------------------------------------------------------- #
# Source authority rubric (Req 1.9)
# --------------------------------------------------------------------------- #
#: Preregistered source-authority rubric, keyed by ``source_ref`` scheme. An
#: unattributed or unrecognized source falls back to :data:`DEFAULT_AUTHORITY`.
SOURCE_AUTHORITY: Mapping[str, float] = {
    "system-of-record": 0.98,
    "analyst": 0.95,
    "verified": 0.92,
    "tool": 0.75,
    "document": 0.70,
    "observation": 0.50,
    "inferred": 0.35,
    "unverified": 0.25,
    "untrusted": 0.10,
    "poisoned": 0.05,
}

#: Authority assigned to a source with no recognized scheme.
DEFAULT_AUTHORITY = 0.55

#: Authority assigned to a blank / missing ``source_ref`` (an unattributed write).
UNATTRIBUTED_AUTHORITY = 0.0

#: Authority at or above which an authoritative correction may supersede (§3.3).
AUTHORITATIVE_FLOOR = 0.90


#: Preregistered consequence rubric by predicate (Req 1.9). The value is the
#: consequence of getting *this* assertion wrong in durable memory.
PREDICATE_CONSEQUENCE: Mapping[str, float] = {
    "OWNS": 0.80,
    "MEMBER_OF": 0.55,
    "ASSIGNED_TO": 0.60,
    "HAS_STATUS": 0.65,
    "HAS_VALUE": 0.45,
    "PRECEDES": 0.50,
    "RESULTS_IN": 0.55,
    "EVIDENCE_FOR": 0.35,
    "SUPPORTS": 0.30,
    "CONTRADICTS": 0.40,
    "ABOUT": 0.20,
    "CONTAINS": 0.30,
    "PARTICIPATES_IN": 0.35,
    "POSSIBLY_SAME_AS": 0.50,
    "SUPERSEDES": 0.60,
}

#: Consequence floor for a write that finalizes a Decision — the paper's
#: canonical high-consequence, hard-to-undo analytic act.
DECISION_FINAL_CONSEQUENCE = 0.90

#: Consequence floor for a write that retires a terminal status (done/cancelled).
TERMINAL_STATUS_CONSEQUENCE = 0.85

#: Consequence assigned to an unknown predicate (conservatively high).
DEFAULT_CONSEQUENCE = 0.60

#: Reversibility of a write whose incumbent is recoverable by supersession.
REVERSIBLE = 0.90
#: Reversibility of a write that creates new state with no incumbent to retire.
CREATE_ONLY_REVERSIBILITY = 0.70
#: Reversibility of a write that would retire several incumbents at once.
MULTI_RETIRE_REVERSIBILITY = 0.35
#: Reversibility of a destructive write (``deletion`` intent).
IRREVERSIBLE = 0.15


# --------------------------------------------------------------------------- #
# Contexts and features
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WriteContext:
    """Per-write context the router consumes alongside the OCMR verdict.

    Every field is optional. When a value is omitted the :class:`Rubric` derives
    it from the candidate and the incumbent graph state, which is the behavior
    used in production; the evaluation corpus supplies explicit values so a case
    can pin a specific consequence / reversibility / authority combination
    (Req 1.9).

    Attributes:
        consequence: ``q ∈ [0,1]`` — the consequence of an incorrect transition.
        reversibility: ``v ∈ [0,1]`` — how cheaply the transition can be undone.
        authority: ``a ∈ [0,1]`` — the authority of the proposing source.
        entity_resolution_status: The W3 resolution status string, when known.
            ``possible_match`` / ``unresolved`` drive the ``f_e = 0.5`` encoding.
        alias_ambiguous: Explicit flag that the subject's alias did not resolve
            uniquely (the dominant cause of OCMR's false quarantines).
        poisoned_evidence: Whether the write's evidence is known-poisoned. Used
            by the corpus to test admission-time integrity, never by the router
            as a free oracle: it only lowers :meth:`Rubric.authority`.
        timestamp: The write's ``t``, used for temporal resolution.
        write_id: Optional stable id for telemetry and review-queue linkage.
        metadata: Free-form extra fields carried into the review item.
    """

    consequence: float | None = None
    reversibility: float | None = None
    authority: float | None = None
    entity_resolution_status: str | None = None
    alias_ambiguous: bool = False
    poisoned_evidence: bool = False
    timestamp: datetime | None = None
    write_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskFeatures:
    """The escalation feature bundle for one candidate write (Req 1.1, 1.8).

    ``f_e``/``f_s``/``f_t``/``f_v``/``f_c`` are the paper's typed status vector;
    ``consequence``/``reversibility``/``authority`` are the scalar modifiers; ``k``
    is the number of simultaneously unresolved or failed checks.

    ``failed_checks`` and ``unresolved_checks`` name the OCMR checks behind the
    encoding so a review item can show *which* checks drove the escalation rather
    than an unexplained number (Req 4.5).
    """

    f_e: float = PASS
    f_s: float = PASS
    f_t: float = PASS
    f_v: float = PASS
    f_c: float = PASS
    consequence: float = 0.0
    reversibility: float = REVERSIBLE
    authority: float = DEFAULT_AUTHORITY
    failed_checks: tuple[str, ...] = ()
    unresolved_checks: tuple[str, ...] = ()
    evidence_count: int = 0
    evidence_floor: int = 1
    incumbent_ids: tuple[str, ...] = ()
    incumbent_recoverable: bool = False

    # -- derived -----------------------------------------------------------
    @property
    def vector(self) -> tuple[float, float, float, float, float]:
        """The typed status vector ``f(u)`` in canonical component order."""
        return (self.f_e, self.f_s, self.f_t, self.f_v, self.f_c)

    @property
    def k(self) -> int:
        """``k`` — the number of simultaneously unresolved or failed checks."""
        return sum(1 for component in self.vector if component > PASS)

    @property
    def interaction(self) -> float:
        """``[k − 1]₊`` — the joint-occurrence term of eq. (3)."""
        return float(max(0, self.k - 1))

    @property
    def any_failure(self) -> bool:
        """Whether at least one component is a hard failure."""
        return any(component >= FAIL for component in self.vector)

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view for telemetry and review items."""
        return {
            "f_e": self.f_e,
            "f_s": self.f_s,
            "f_t": self.f_t,
            "f_v": self.f_v,
            "f_c": self.f_c,
            "k": self.k,
            "consequence": self.consequence,
            "reversibility": self.reversibility,
            "authority": self.authority,
            "failed_checks": list(self.failed_checks),
            "unresolved_checks": list(self.unresolved_checks),
            "evidence_count": self.evidence_count,
            "evidence_floor": self.evidence_floor,
            "incumbent_ids": list(self.incumbent_ids),
            "incumbent_recoverable": self.incumbent_recoverable,
        }


# --------------------------------------------------------------------------- #
# Rubrics
# --------------------------------------------------------------------------- #
class Rubric:
    """Preregistered rubrics for consequence ``q``, reversibility ``v``, authority ``a``.

    The rubrics are deliberately simple, total, and inspectable: every value is a
    table lookup plus a small number of documented escalations. They are part of
    the evaluation suite so a reader can reproduce any assigned score by hand
    (Req 1.9).
    """

    def __init__(
        self,
        *,
        source_authority: Mapping[str, float] | None = None,
        predicate_consequence: Mapping[str, float] | None = None,
    ) -> None:
        """Create a rubric, optionally overriding either table."""
        self.source_authority = dict(source_authority or SOURCE_AUTHORITY)
        self.predicate_consequence = dict(predicate_consequence or PREDICATE_CONSEQUENCE)

    # -- authority ---------------------------------------------------------
    def authority(self, candidate: CandidateAssertion, context: WriteContext) -> float:
        """Return ``a ∈ [0,1]`` for the proposing source.

        An explicit ``context.authority`` wins. Otherwise the ``source_ref``
        scheme (the text before the first ``:``) is looked up in the rubric; a
        blank source is *unattributed* and scores :data:`UNATTRIBUTED_AUTHORITY`.
        Known-poisoned evidence caps authority at the ``poisoned`` rubric value.
        """
        if context.authority is not None:
            return _clamp(context.authority)
        source_ref = (candidate.source_ref or "").strip()
        if not source_ref:
            return UNATTRIBUTED_AUTHORITY
        scheme = source_ref.split(":", 1)[0].strip().lower() if ":" in source_ref else ""
        value = self.source_authority.get(scheme, DEFAULT_AUTHORITY)
        if context.poisoned_evidence:
            value = min(value, self.source_authority.get("poisoned", 0.05))
        return _clamp(value)

    # -- consequence -------------------------------------------------------
    def consequence(
        self,
        candidate: CandidateAssertion,
        graph: GraphStore,
        vr: ValidationResult,
        context: WriteContext,
    ) -> float:
        """Return ``q ∈ [0,1]`` — the consequence of an incorrect transition.

        Base value is the predicate's rubric entry. It is raised for the two
        canonical high-consequence analytic acts (finalizing a Decision, retiring
        a terminal Task status) and for a high-severity OCMR verdict.
        """
        if context.consequence is not None:
            return _clamp(context.consequence)

        value = self.predicate_consequence.get(candidate.predicate, DEFAULT_CONSEQUENCE)

        subject_type = graph.get_entity_type(candidate.subject_id)
        if candidate.predicate == _HAS_STATUS:
            target = _status_value(graph, candidate.object_id)
            if subject_type == "Decision" and target == "final":
                value = max(value, DECISION_FINAL_CONSEQUENCE)
            if subject_type == "Task" and _retires_terminal_status(graph, candidate.subject_id):
                value = max(value, TERMINAL_STATUS_CONSEQUENCE)

        if vr.severity == Severity.high:
            value = max(value, 0.75)
        elif vr.severity == Severity.medium:
            value = max(value, 0.55)

        return _clamp(value)

    # -- reversibility -----------------------------------------------------
    def reversibility(
        self,
        candidate: CandidateAssertion,
        graph: GraphStore,
        incumbent_ids: Iterable[str],
        context: WriteContext,
    ) -> float:
        """Return ``v ∈ [0,1]`` — how cheaply the transition can be undone.

        OCMR retains a superseded assertion and its provenance, so a write with a
        single recoverable incumbent is highly reversible. A ``deletion`` intent
        is treated as irreversible, and a write that would retire several
        incumbents at once is discounted because the rollback is a multi-step
        recovery rather than one supersession.
        """
        if context.reversibility is not None:
            return _clamp(context.reversibility)
        if candidate.write_intent == WriteIntent.deletion:
            return IRREVERSIBLE
        incumbents = list(incumbent_ids)
        if len(incumbents) > 1:
            return MULTI_RETIRE_REVERSIBILITY
        if len(incumbents) == 1:
            return REVERSIBLE
        return CREATE_ONLY_REVERSIBILITY


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
class FeatureExtractor:
    """Maps an OCMR verdict plus graph state onto :class:`RiskFeatures`.

    The extractor never re-runs or re-implements a constraint. It reads the
    ``failed_check`` OCMR already reported, then queries the graph only for the
    incumbent facts the ``0.5`` (unresolved) encodings need: alias resolution
    status, single-valued cardinality pressure, temporal ordering against the
    incumbent, and the evidence count Algorithm 1 uses.
    """

    def __init__(
        self,
        *,
        rubric: Rubric | None = None,
        settings: Any = None,
    ) -> None:
        """Create an extractor bound to a rubric and the OCMR settings."""
        self.rubric = rubric or Rubric()
        self.settings = settings

    # -- public API --------------------------------------------------------
    def extract(
        self,
        candidate: CandidateAssertion,
        graph: GraphStore,
        vr: ValidationResult,
        context: WriteContext | None = None,
        *,
        contradiction_result: Any = None,
    ) -> RiskFeatures:
        """Extract the escalation features for one candidate write.

        Args:
            candidate: The proposed write ``u``.
            graph: The accepted-only Graph_Store representing ``M_t``.
            vr: OCMR's W5/W6 (+W7) verdict for ``candidate``.
            context: Optional per-write context (rubric overrides, alias flags).
            contradiction_result: Optional :class:`ContradictionResult` when the
                caller already ran W7 separately; otherwise the contradiction
                component is derived from ``vr``.

        Returns:
            The :class:`RiskFeatures` bundle, including the named checks behind
            each encoding.
        """
        context = context or WriteContext()
        failed: list[str] = []
        unresolved: list[str] = []

        check = vr.failed_check
        is_schema_failure = bool(check and check.startswith(SCHEMA_CHECK_PREFIX))

        incumbent_ids = self._incumbent_ids(candidate, graph, vr)
        evidence_count = self._evidence_count(candidate, graph)
        evidence_floor = int(getattr(self.settings, "supersede_evidence_min", 1) or 1)

        f_e = self._entity_component(check, context, failed, unresolved)
        f_s = self._schema_component(
            candidate, graph, check, is_schema_failure, incumbent_ids, failed, unresolved
        )
        f_t = self._temporal_component(
            candidate, graph, check, context, incumbent_ids, failed, unresolved
        )
        f_v = self._evidence_component(
            check, evidence_count, evidence_floor, context, failed, unresolved
        )
        f_c = self._contradiction_component(
            check, vr, contradiction_result, failed, unresolved
        )

        authority = self.rubric.authority(candidate, context)
        consequence = self.rubric.consequence(candidate, graph, vr, context)
        reversibility = self.rubric.reversibility(candidate, graph, incumbent_ids, context)

        return RiskFeatures(
            f_e=f_e,
            f_s=f_s,
            f_t=f_t,
            f_v=f_v,
            f_c=f_c,
            consequence=consequence,
            reversibility=reversibility,
            authority=authority,
            failed_checks=tuple(dict.fromkeys(failed)),
            unresolved_checks=tuple(dict.fromkeys(unresolved)),
            evidence_count=evidence_count,
            evidence_floor=evidence_floor,
            incumbent_ids=tuple(incumbent_ids),
            incumbent_recoverable=bool(incumbent_ids),
        )

    # -- components --------------------------------------------------------
    def _entity_component(
        self,
        check: str | None,
        context: WriteContext,
        failed: list[str],
        unresolved: list[str],
    ) -> float:
        """``f_e`` — C1 identity uniqueness and entity-resolution status (Req 1.3)."""
        if check in ENTITY_CHECKS:
            failed.append("C1")
            return FAIL
        status = (context.entity_resolution_status or "").strip().lower()
        if context.alias_ambiguous or status in {"possible_match", "unresolved"}:
            unresolved.append("entity_resolution")
            return UNRESOLVED
        return PASS

    def _schema_component(
        self,
        candidate: CandidateAssertion,
        graph: GraphStore,
        check: str | None,
        is_schema_failure: bool,
        incumbent_ids: list[str],
        failed: list[str],
        unresolved: list[str],
    ) -> float:
        """``f_s`` — W5 structural checks and C9 domain/range (Req 1.4)."""
        if is_schema_failure:
            failed.append(check or SCHEMA_CHECK_PREFIX)
            return FAIL
        if check in SCHEMA_CHECKS or check == "C6":
            failed.append(check or "C9")
            return FAIL
        # Single-valued cardinality pressure: the predicate admits at most one
        # active object per subject and the subject already has one. This is not
        # a violation (OCMR resolves it by supersession or quarantine), but it is
        # unresolved at admission time.
        if incumbent_ids and self._is_single_valued(candidate.predicate):
            unresolved.append("cardinality")
            return UNRESOLVED
        return PASS

    def _temporal_component(
        self,
        candidate: CandidateAssertion,
        graph: GraphStore,
        check: str | None,
        context: WriteContext,
        incumbent_ids: list[str],
        failed: list[str],
        unresolved: list[str],
    ) -> float:
        """``f_t`` — C2 sanity, C3 acyclic PRECEDES, C10 transitions (Req 1.5)."""
        if check in TEMPORAL_CHECKS:
            failed.append(check or "C2")
            return FAIL
        if candidate.valid_from is not None and candidate.valid_to is not None:
            if candidate.valid_to < candidate.valid_from:
                failed.append("C2")
                return FAIL
        # Unresolved: the write claims to update an incumbent but carries no
        # timestamp that orders it against that incumbent, so "which is current"
        # cannot be settled from the write alone.
        if incumbent_ids and not self._temporal_relation_resolved(
            candidate, graph, context, incumbent_ids
        ):
            unresolved.append("temporal_order")
            return UNRESOLVED
        return PASS

    def _evidence_component(
        self,
        check: str | None,
        evidence_count: int,
        evidence_floor: int,
        context: WriteContext,
        failed: list[str],
        unresolved: list[str],
    ) -> float:
        """``f_v`` — the C8 / ``e_min`` evidence floor (Req 1.6)."""
        if check in EVIDENCE_CHECKS:
            failed.append(check or "C8")
            return FAIL
        if evidence_count <= 0:
            failed.append("evidence_absent")
            return FAIL
        if context.poisoned_evidence:
            # Evidence exists but does not support the claim; it cannot be
            # counted as satisfying the floor, and it is not a clean failure
            # either until a reviewer confirms it.
            unresolved.append("evidence_unsupported")
            return UNRESOLVED
        if evidence_count < evidence_floor:
            unresolved.append("evidence_below_floor")
            return UNRESOLVED
        return PASS

    def _contradiction_component(
        self,
        check: str | None,
        vr: ValidationResult,
        contradiction_result: Any,
        failed: list[str],
        unresolved: list[str],
    ) -> float:
        """``f_c`` — the W7 Contradiction_Checker verdict via C7 (Req 1.7)."""
        if contradiction_result is not None and getattr(
            contradiction_result, "has_conflict", False
        ):
            kind = getattr(contradiction_result, "kind", None)
            severity = getattr(contradiction_result, "severity", None)
            if kind == "hard" or severity == Severity.high:
                failed.append("C7")
                return FAIL
            unresolved.append("C7_soft")
            return UNRESOLVED
        if check in CONTRADICTION_CHECKS:
            if vr.severity == Severity.high:
                failed.append(check or "C7")
                return FAIL
            # A medium-severity contradiction (e.g. a status flip OCMR
            # quarantines) is a real conflict but not a hard one.
            unresolved.append(check or "C7")
            return UNRESOLVED
        return PASS

    # -- graph helpers -----------------------------------------------------
    @staticmethod
    def _is_single_valued(predicate: str) -> bool:
        """Whether ``predicate`` admits at most one active object per subject."""
        try:
            signature = get_relation_signature(predicate)
        except UnknownPredicateError:
            return False
        return signature.cardinality in _SINGLE_VALUED

    def _incumbent_ids(
        self,
        candidate: CandidateAssertion,
        graph: GraphStore,
        vr: ValidationResult,
    ) -> list[str]:
        """Accepted assertion ids this write would retire, if it committed.

        OCMR's ``conflicting_ids`` is authoritative when present. Otherwise, for
        a single-valued predicate, the subject's existing accepted out-edges for
        the same predicate (to a different object) are the incumbents.
        """
        if vr.conflicting_ids:
            return [cid for cid in vr.conflicting_ids if cid]
        if not self._is_single_valued(candidate.predicate):
            return []
        out: list[str] = []
        for _s, obj, _k, data in graph.out_edges(candidate.subject_id, candidate.predicate):
            if obj == candidate.object_id:
                continue
            aid = data.get("assertion_id")
            if aid:
                out.append(aid)
        return out

    @staticmethod
    def _evidence_count(candidate: CandidateAssertion, graph: GraphStore) -> int:
        """Units of supporting evidence, matching OCMR's Algorithm 1 ``e_min``.

        Mirrors ``ocm.validation.constraints._evidence_count``: a present
        ``source_ref`` counts as one unit, plus each accepted ``EVIDENCE_FOR``
        in-edge into the subject from a Document or Event.
        """
        count = 1 if (candidate.source_ref or "").strip() else 0
        for subject, _o, _k, _d in graph.in_edges(candidate.subject_id, _EVIDENCE_FOR):
            if graph.get_entity_type(subject) in {"Document", "Event"}:
                count += 1
        return count

    @staticmethod
    def _temporal_relation_resolved(
        candidate: CandidateAssertion,
        graph: GraphStore,
        context: WriteContext,
        incumbent_ids: list[str],
    ) -> bool:
        """Whether the write is orderable against the incumbent it would retire.

        Resolved when the candidate carries a ``valid_from`` (or the context
        carries a timestamp) that is not earlier than every incumbent's
        ``created_at`` / ``valid_from``. An undated update to dated memory is
        *unresolved*: nothing in the write says it is the newer fact.
        """
        stamp = candidate.valid_from or context.timestamp
        if stamp is None:
            return False
        incumbent = set(incumbent_ids)
        for _s, _o, _k, data in graph.g.edges(keys=True, data=True):
            if data.get("assertion_id") not in incumbent:
                continue
            prior = data.get("valid_from") or data.get("created_at")
            prior = _as_datetime(prior)
            if prior is None:
                continue
            if _as_naive(stamp) < _as_naive(prior):
                return False
        return True


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return max(low, min(high, float(value)))


def _as_datetime(value: Any) -> datetime | None:
    """Coerce a datetime that may arrive as a ``datetime`` or ISO-8601 string."""
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _as_naive(value: datetime) -> datetime:
    """Drop tzinfo so aware and naive timestamps can be compared."""
    return value.replace(tzinfo=None) if value.tzinfo is not None else value


def _status_value(graph: GraphStore, status_value_id: str) -> str | None:
    """Return the ``value`` carried by a StatusValue node, if resolvable."""
    payload = graph.get_entity_payload(status_value_id)
    if isinstance(payload, Mapping):
        value = payload.get("value")
        if value is not None:
            return str(value)
    if status_value_id.startswith("status:"):
        return status_value_id.split(":", 1)[1]
    return None


def _retires_terminal_status(graph: GraphStore, subject_id: str) -> bool:
    """Whether the subject currently holds a terminal (done/cancelled) status."""
    terminal = {TaskStatus.done.value, TaskStatus.cancelled.value}
    for _s, obj, _k, _d in graph.out_edges(subject_id, _HAS_STATUS):
        if _status_value(graph, obj) in terminal:
            return True
    return False

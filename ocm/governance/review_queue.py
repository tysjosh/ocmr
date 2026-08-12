"""RAHGM review queue, evidence bundles, explanation depths, and release.

Implements paper §3.4 and the review-and-release mechanism the OCMR quarantine
audit found missing. A ``review`` route creates a durable OCMR quarantine record
*and* a :class:`ReviewItem` keyed by that record's id, so accepted memory is never
overwritten while the write is held, and an analyst can later **release** the
write by committing it through the same Commit_Manager.

Explanation depth is a presentation variable only (Req 6.2). The three levels are
strictly nested:

* ``minimal`` — recommended action plus failed or unresolved constraints;
* ``evidence`` — adds supporting and conflicting evidence snippets with provenance;
* ``full`` — adds the memory timeline, alternative actions, reversibility, and the
  predicted downstream consequence.

Requirements: 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from ocm.governance.router import RoutingDecision
from ocm.memory.contracts import CandidateAssertion, ValidationResult, WriteOutcome
from ocm.memory.graph_store import GraphStore
from ocm.ontology.enums import QuarantineStatus, Severity
from ocm.ontology.relations import (
    Cardinality,
    UnknownPredicateError,
    get_relation_signature,
)

#: Predicates admitting at most one active object per subject. A release on one of
#: these must retire the incumbent rather than add a second value.
_SINGLE_VALUED = frozenset({Cardinality.ONE_TO_ONE, Cardinality.M_TO_ONE})

logger = logging.getLogger(__name__)


class ExplanationDepth(str, Enum):
    """The three manipulated explanation depths of §3.4."""

    minimal = "minimal"
    evidence = "evidence"
    full = "full"


#: Canonical depth order, used by the Latin-square schedule.
DEPTH_ORDER: tuple[ExplanationDepth, ...] = (
    ExplanationDepth.minimal,
    ExplanationDepth.evidence,
    ExplanationDepth.full,
)


class ReviewAction(str, Enum):
    """The actions a review item offers an analyst (§3.4)."""

    accept = "accept"
    supersede = "supersede"
    quarantine = "quarantine"
    reject = "reject"
    request_evidence = "request_evidence"


#: Actions that release the held write into durable memory.
RELEASING_ACTIONS = frozenset({ReviewAction.accept, ReviewAction.supersede})


# --------------------------------------------------------------------------- #
# Evidence bundle
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvidenceSnippet:
    """One piece of evidence with its provenance."""

    text: str
    source_ref: str
    kind: str  # "supporting" | "conflicting"
    entity_id: str | None = None
    created_at: datetime | None = None
    authority: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "text": self.text,
            "source_ref": self.source_ref,
            "kind": self.kind,
            "entity_id": self.entity_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "authority": self.authority,
        }


@dataclass(frozen=True)
class EvidenceBundle:
    """Supporting and conflicting evidence for one review item."""

    supporting: tuple[EvidenceSnippet, ...] = ()
    conflicting: tuple[EvidenceSnippet, ...] = ()
    minimum_required: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "supporting": [s.as_dict() for s in self.supporting],
            "conflicting": [s.as_dict() for s in self.conflicting],
            "minimum_required": self.minimum_required,
        }


@dataclass(frozen=True)
class TimelineEntry:
    """One prior state of the affected assertion, for the ``full`` explanation."""

    assertion_id: str | None
    predicate: str
    object_id: str
    status: str
    source_ref: str
    created_at: datetime | None

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "assertion_id": self.assertion_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "status": self.status,
            "source_ref": self.source_ref,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# --------------------------------------------------------------------------- #
# Review item
# --------------------------------------------------------------------------- #
@dataclass
class ReviewItem:
    """One queued write awaiting adjudication (§3.4, Req 5.2).

    Holds the incumbent and proposed assertions, the requested operation, source,
    timestamp, the :class:`RoutingDecision` (which names the failed checks and the
    rule that fired), the evidence bundle, the memory timeline, and the available
    actions.
    """

    item_id: str
    candidate: CandidateAssertion
    decision: RoutingDecision
    quarantine_id: str | None
    incumbent: TimelineEntry | None
    evidence: EvidenceBundle
    timeline: tuple[TimelineEntry, ...]
    ocmr_verdict: ValidationResult
    enqueued_at: datetime
    scenario_id: str | None = None
    write_id: str | None = None
    consequential: bool = False
    evidence_requests: int = 0
    resolved: bool = False
    actions: tuple[ReviewAction, ...] = tuple(ReviewAction)

    # -- derived -----------------------------------------------------------
    @property
    def recommended_action(self) -> ReviewAction:
        """The action the system recommends, derived from the OCMR verdict.

        RAHGM held the write because it could not settle the transition
        autonomously, so the recommendation is OCMR's own preferred action where
        it had one, otherwise ``quarantine`` (hold pending evidence).
        """
        mapping = {
            "accept": ReviewAction.accept,
            "supersede": ReviewAction.supersede,
            "reject": ReviewAction.reject,
            "quarantine": ReviewAction.quarantine,
        }
        return mapping.get(self.decision.ocmr_action or "", ReviewAction.quarantine)

    @property
    def predicted_consequence(self) -> str:
        """A short predicted-downstream-consequence string for the ``full`` depth."""
        q = self.decision.features.consequence
        v = self.decision.features.reversibility
        band = "high" if q >= 0.75 else "moderate" if q >= 0.5 else "low"
        undo = "cheap to undo" if v >= 0.7 else "costly to undo" if v >= 0.3 else "irreversible"
        return (
            f"{band}-consequence assertion on {self.candidate.subject_id}; "
            f"{undo}; an incorrect transition here would be retrievable by later "
            f"reasoning until corrected"
        )

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view of the whole item."""
        return {
            "item_id": self.item_id,
            "write_id": self.write_id,
            "scenario_id": self.scenario_id,
            "quarantine_id": self.quarantine_id,
            "candidate": self.candidate.model_dump(mode="json"),
            "incumbent": self.incumbent.as_dict() if self.incumbent else None,
            "decision": self.decision.as_dict(),
            "evidence": self.evidence.as_dict(),
            "timeline": [t.as_dict() for t in self.timeline],
            "consequential": self.consequential,
            "evidence_requests": self.evidence_requests,
            "resolved": self.resolved,
            "enqueued_at": self.enqueued_at.isoformat(),
        }


@dataclass(frozen=True)
class Adjudication:
    """The recorded outcome of one human (or simulated) adjudication."""

    item_id: str
    action: ReviewAction
    analyst_id: str
    depth: ExplanationDepth
    seconds: float
    confidence: float
    evidence_opened: int = 0
    action_changes: int = 0
    followed_recommendation: bool = False
    released: bool = False
    outcome: WriteOutcome | None = None
    rationale: str | None = None
    decided_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "item_id": self.item_id,
            "action": self.action.value,
            "analyst_id": self.analyst_id,
            "depth": self.depth.value,
            "seconds": self.seconds,
            "confidence": self.confidence,
            "evidence_opened": self.evidence_opened,
            "action_changes": self.action_changes,
            "followed_recommendation": self.followed_recommendation,
            "released": self.released,
            "decision": self.outcome.decision if self.outcome else None,
            "rationale": self.rationale,
        }


# --------------------------------------------------------------------------- #
# Explanation rendering
# --------------------------------------------------------------------------- #
def render_explanation(
    item: ReviewItem, depth: ExplanationDepth | str
) -> dict[str, Any]:
    """Render a review item at one explanation depth (Req 6.1, 6.2).

    The three depths are strictly nested: ``minimal ⊂ evidence ⊂ full``. Depth
    changes presentation only and never the route — the route was already fixed by
    ``π(u)`` before the item was enqueued.
    """
    depth = ExplanationDepth(depth)
    features = item.decision.features

    payload: dict[str, Any] = {
        "depth": depth.value,
        "item_id": item.item_id,
        "incumbent_value": item.incumbent.object_id if item.incumbent else None,
        "proposed_value": item.candidate.object_id,
        "requested_operation": item.candidate.write_intent.value,
        "source": item.candidate.source_ref,
        "timestamp": (item.candidate.valid_from or item.enqueued_at).isoformat(),
        "recommended_action": item.recommended_action.value,
        "failed_checks": list(features.failed_checks),
        "unresolved_checks": list(features.unresolved_checks),
        "available_actions": [a.value for a in item.actions],
    }

    if depth is ExplanationDepth.minimal:
        return payload

    payload["supporting_evidence"] = [s.as_dict() for s in item.evidence.supporting]
    payload["conflicting_evidence"] = [s.as_dict() for s in item.evidence.conflicting]
    payload["provenance"] = {
        "source_ref": item.candidate.source_ref,
        "authority": features.authority,
        "evidence_count": features.evidence_count,
        "evidence_floor": features.evidence_floor,
    }

    if depth is ExplanationDepth.evidence:
        return payload

    payload["memory_timeline"] = [t.as_dict() for t in item.timeline]
    payload["alternative_actions"] = [
        {
            "action": action.value,
            "effect": _action_effect(action, item),
        }
        for action in item.actions
        if action is not item.recommended_action
    ]
    payload["reversibility"] = {
        "score": features.reversibility,
        "incumbent_recoverable": features.incumbent_recoverable,
        "incumbent_ids": list(features.incumbent_ids),
    }
    payload["predicted_consequence"] = item.predicted_consequence
    payload["risk"] = {
        "score": item.decision.score,
        "escalation_probability": item.decision.risk,
        "rule": item.decision.rule,
        "consequence": features.consequence,
        "k": features.k,
    }
    return payload


def _action_effect(action: ReviewAction, item: ReviewItem) -> str:
    """A short description of what each alternative action would do."""
    if action is ReviewAction.accept:
        return "commit the proposed assertion as durable memory"
    if action is ReviewAction.supersede:
        target = (
            item.incumbent.assertion_id
            if item.incumbent and item.incumbent.assertion_id
            else "the incumbent assertion"
        )
        return f"retire {target} and commit the proposal, retaining the prior value"
    if action is ReviewAction.reject:
        return "discard the proposal; nothing enters durable memory"
    if action is ReviewAction.quarantine:
        return "keep the proposal held; durable memory is unchanged"
    return "return the item to the queue pending additional evidence"


# --------------------------------------------------------------------------- #
# Latin square (Req 6.3)
# --------------------------------------------------------------------------- #
def latin_square(
    n_levels: int, n_blocks: int, offset: int = 0
) -> list[int]:
    """A balanced Latin-square schedule of level indices across ``n_blocks``.

    Row ``offset`` of a cyclic Latin square, repeated to cover ``n_blocks``. Each
    participant gets a different ``offset``, so every level appears equally often
    at every position across participants (Req 6.3).
    """
    if n_levels <= 0:
        raise ValueError("n_levels must be positive")
    return [((offset + index) % n_levels) for index in range(n_blocks)]


def depth_schedule(n_blocks: int, offset: int = 0) -> list[ExplanationDepth]:
    """The explanation-depth schedule for one participant."""
    return [DEPTH_ORDER[i] for i in latin_square(len(DEPTH_ORDER), n_blocks, offset)]


# --------------------------------------------------------------------------- #
# The queue
# --------------------------------------------------------------------------- #
class ReviewQueue:
    """Holds escalated writes and executes analyst adjudications.

    The queue owns the *release* path: for an ``accept`` or ``supersede``
    adjudication it commits the held candidate through the inner OCMR
    Commit_Manager and transitions the durable quarantine record to ``resolved``.
    A ``reject`` dismisses the record; ``quarantine`` leaves it unresolved; and
    ``request_evidence`` returns the item to the queue with an incremented
    request count (Req 5.3).

    Because release goes through the same Commit_Manager, a supersession retains
    the prior assertion and its provenance, so every release is reversible
    (Req 5.4).
    """

    def __init__(
        self,
        *,
        commit_manager: Any | None = None,
        quarantine_store: Any | None = None,
        graph: GraphStore | None = None,
        repo: Any | None = None,
    ) -> None:
        """Wire the queue to the stores it needs for release.

        All collaborators are optional: a queue built with none of them still
        records items and adjudications (useful for offline explanation studies),
        it simply cannot release.
        """
        self.commit_manager = commit_manager
        self.quarantine_store = quarantine_store
        self.graph = graph
        self.repo = repo
        self.items: dict[str, ReviewItem] = {}
        self.order: list[str] = []
        self.adjudications: list[Adjudication] = []
        self._counter = 0

    # -- enqueue -----------------------------------------------------------
    def enqueue(
        self,
        candidate: CandidateAssertion,
        decision: RoutingDecision,
        outcome: WriteOutcome,
        *,
        ocmr_verdict: ValidationResult | None = None,
        scenario_id: str | None = None,
        consequential: bool = False,
        minimum_evidence: str | None = None,
        enqueued_at: datetime | None = None,
    ) -> ReviewItem:
        """Build and enqueue a :class:`ReviewItem` for an escalated write."""
        self._counter += 1
        item_id = f"rev-{self._counter:06d}"
        incumbent = self._incumbent_entry(decision)
        item = ReviewItem(
            item_id=item_id,
            candidate=candidate,
            decision=decision,
            quarantine_id=outcome.quarantine_id,
            incumbent=incumbent,
            evidence=self._build_evidence(candidate, decision, minimum_evidence),
            timeline=self._build_timeline(candidate),
            ocmr_verdict=ocmr_verdict
            or ValidationResult(valid=False, reason=decision.ocmr_reason),
            enqueued_at=enqueued_at or datetime.now(timezone.utc),
            scenario_id=scenario_id,
            write_id=decision.write_id,
            consequential=consequential,
        )
        self.items[item_id] = item
        self.order.append(item_id)
        return item

    # -- inspection --------------------------------------------------------
    def pending(self) -> list[ReviewItem]:
        """Unresolved items in enqueue order."""
        return [self.items[i] for i in self.order if not self.items[i].resolved]

    def all_items(self) -> list[ReviewItem]:
        """Every item ever enqueued, in order."""
        return [self.items[i] for i in self.order]

    def __len__(self) -> int:
        return len(self.order)

    # -- adjudicate --------------------------------------------------------
    def adjudicate(
        self,
        item_id: str,
        action: ReviewAction | str,
        *,
        analyst_id: str = "analyst",
        depth: ExplanationDepth | str = ExplanationDepth.evidence,
        seconds: float = 0.0,
        confidence: float = 0.5,
        evidence_opened: int = 0,
        action_changes: int = 0,
        rationale: str | None = None,
        created_at: datetime | None = None,
    ) -> Adjudication:
        """Record an adjudication and, for a releasing action, release the write.

        Args:
            item_id: The queued item.
            action: The analyst's chosen :class:`ReviewAction`.
            analyst_id: Participant identifier (kept separate from any PII).
            depth: The explanation depth the analyst saw.
            seconds: Decision time.
            confidence: Self-reported confidence on a 0–1 scale.
            evidence_opened: How many evidence snippets were opened.
            action_changes: How many times the analyst changed their selection.
            rationale: Free-text rationale, retained in the feedback record.
            created_at: Fixed timestamp for deterministic replay.

        Returns:
            The recorded :class:`Adjudication`, including the durable
            :class:`WriteOutcome` when the write was released.
        """
        item = self.items[item_id]
        action = ReviewAction(action)
        depth = ExplanationDepth(depth)
        now = created_at or datetime.now(timezone.utc)

        released = False
        outcome: WriteOutcome | None = None

        if action is ReviewAction.request_evidence:
            # The only action that returns an item to the queue: the analyst has
            # not decided yet and wants more evidence.
            item.evidence_requests += 1
        elif action in RELEASING_ACTIONS:
            outcome = self._release(item, action, now)
            released = outcome is not None
            item.resolved = True
        else:
            # ``quarantine`` and ``reject`` are both completed adjudications: the
            # analyst decided the write should not enter memory. The item leaves
            # the queue so it is not presented again, while the durable quarantine
            # record persists — dismissed for a rejection, still unresolved for a
            # hold, which is what keeps it auditable and releasable later.
            self._set_quarantine_status(
                item,
                QuarantineStatus.dismissed
                if action is ReviewAction.reject
                else QuarantineStatus.unresolved,
            )
            item.resolved = True

        record = Adjudication(
            item_id=item_id,
            action=action,
            analyst_id=analyst_id,
            depth=depth,
            seconds=float(seconds),
            confidence=float(confidence),
            evidence_opened=int(evidence_opened),
            action_changes=int(action_changes),
            followed_recommendation=action is item.recommended_action,
            released=released,
            outcome=outcome,
            rationale=rationale,
            decided_at=now,
        )
        self.adjudications.append(record)
        return record

    # -- release (Req 5.3, 5.4) -------------------------------------------
    def _release(
        self, item: ReviewItem, action: ReviewAction, now: datetime
    ) -> WriteOutcome | None:
        """Commit a held write through the inner Commit_Manager.

        This is the review-and-release mechanism OCMR lacked: a valid update that
        was conservatively held can now enter durable memory after adjudication,
        with the quarantine record retained and marked ``resolved`` so the hold
        and its resolution stay auditable.
        """
        if self.commit_manager is None:
            logger.debug("ReviewQueue has no commit manager; cannot release %s", item.item_id)
            return None

        # A release must not leave two active values on a single-valued predicate.
        # Whatever the analyst called the action, releasing a write whose target
        # already holds a value *is* a supersession: the prior value is retired and
        # retained. Resolving the incumbent from the live graph rather than from the
        # stale verdict matters because earlier releases may have changed it.
        conflicting = self._active_incumbents(item.candidate) or [
            cid
            for cid in (
                list(item.decision.features.incumbent_ids)
                or list(item.ocmr_verdict.conflicting_ids)
            )
            if cid
        ]

        if conflicting:
            verdict = ValidationResult(
                valid=True,
                reason=(
                    "released on review: analyst-authorized supersession"
                    if action is ReviewAction.supersede
                    else "released on review: acceptance retires the incumbent value"
                ),
                conflicting_ids=conflicting,
                recommended_action="supersede",
            )
        else:
            verdict = ValidationResult(
                valid=True,
                reason="released on review: analyst-authorized acceptance",
                recommended_action="accept",
            )

        outcome = self.commit_manager.commit(item.candidate, verdict, created_at=now)
        self._set_quarantine_status(item, QuarantineStatus.resolved)
        return outcome

    def _active_incumbents(self, candidate: CandidateAssertion) -> list[str]:
        """Accepted assertion ids that releasing ``candidate`` must retire.

        Read from the live graph, restricted to single-valued predicates, so a
        release never leaves a cardinality violation behind. Returns an empty list
        for a many-valued predicate, where an additional value is admissible.
        """
        if self.graph is None:
            return []
        try:
            signature = get_relation_signature(candidate.predicate)
        except UnknownPredicateError:
            return []
        if signature.cardinality not in _SINGLE_VALUED:
            return []
        out: list[str] = []
        for _s, obj, _k, data in self.graph.out_edges(
            candidate.subject_id, candidate.predicate
        ):
            if obj == candidate.object_id:
                continue
            assertion_id = data.get("assertion_id")
            if assertion_id:
                out.append(assertion_id)
        return out

    def _set_quarantine_status(
        self, item: ReviewItem, status: QuarantineStatus
    ) -> None:
        """Transition the item's durable quarantine record, if there is one."""
        if self.quarantine_store is None or not item.quarantine_id:
            return
        try:
            self.quarantine_store.set_status(item.quarantine_id, status)
        except KeyError:  # pragma: no cover - defensive
            logger.debug("quarantine record %s not found", item.quarantine_id)

    # -- bundle construction ----------------------------------------------
    def _incumbent_entry(self, decision: RoutingDecision) -> TimelineEntry | None:
        """Resolve the incumbent assertion the proposal would replace."""
        if self.graph is None:
            return None
        for assertion_id in decision.features.incumbent_ids:
            found = self.graph.find_assertion(assertion_id)
            if found is None:
                continue
            _subject, obj, predicate, data = found
            return TimelineEntry(
                assertion_id=assertion_id,
                predicate=predicate,
                object_id=obj,
                status=str(data.get("status", "accepted")),
                source_ref=str(data.get("source_ref", "")),
                created_at=_as_datetime(data.get("created_at")),
            )
        return None

    def _build_evidence(
        self,
        candidate: CandidateAssertion,
        decision: RoutingDecision,
        minimum_evidence: str | None,
    ) -> EvidenceBundle:
        """Collect supporting and conflicting evidence with provenance."""
        supporting: list[EvidenceSnippet] = []
        conflicting: list[EvidenceSnippet] = []

        if (candidate.source_ref or "").strip():
            supporting.append(
                EvidenceSnippet(
                    text=(
                        f"proposed: {candidate.subject_id} -[{candidate.predicate}]-> "
                        f"{candidate.object_id} (confidence {candidate.confidence:.2f})"
                    ),
                    source_ref=candidate.source_ref,
                    kind="supporting",
                    entity_id=candidate.subject_id,
                    created_at=candidate.valid_from,
                    authority=decision.features.authority,
                )
            )

        if self.graph is not None:
            for subject, _obj, _key, data in self.graph.in_edges(
                candidate.subject_id, "EVIDENCE_FOR"
            ):
                entity_type = self.graph.get_entity_type(subject)
                if entity_type not in {"Document", "Event"}:
                    continue
                payload = self.graph.get_entity_payload(subject) or {}
                label = payload.get("title") or payload.get("description") or subject
                supporting.append(
                    EvidenceSnippet(
                        text=f"{entity_type} evidence: {label}",
                        source_ref=str(data.get("source_ref", "")),
                        kind="supporting",
                        entity_id=subject,
                        created_at=_as_datetime(data.get("created_at")),
                    )
                )

            for assertion_id in decision.features.incumbent_ids:
                found = self.graph.find_assertion(assertion_id)
                if found is None:
                    continue
                subject, obj, predicate, data = found
                conflicting.append(
                    EvidenceSnippet(
                        text=(
                            f"incumbent: {subject} -[{predicate}]-> {obj} "
                            f"(confidence {float(data.get('confidence', 0.0)):.2f})"
                        ),
                        source_ref=str(data.get("source_ref", "")),
                        kind="conflicting",
                        entity_id=subject,
                        created_at=_as_datetime(data.get("created_at")),
                    )
                )

        return EvidenceBundle(
            supporting=tuple(supporting),
            conflicting=tuple(conflicting),
            minimum_required=minimum_evidence,
        )

    def _build_timeline(self, candidate: CandidateAssertion) -> tuple[TimelineEntry, ...]:
        """The accepted history of the affected subject/predicate pair."""
        if self.graph is None:
            return ()
        entries: list[TimelineEntry] = []
        for _s, obj, predicate, data in self.graph.out_edges(
            candidate.subject_id, candidate.predicate
        ):
            entries.append(
                TimelineEntry(
                    assertion_id=data.get("assertion_id"),
                    predicate=predicate,
                    object_id=obj,
                    status=str(data.get("status", "accepted")),
                    source_ref=str(data.get("source_ref", "")),
                    created_at=_as_datetime(data.get("created_at")),
                )
            )
        entries.sort(key=lambda e: (e.created_at is None, e.created_at or datetime.min))
        return tuple(entries)


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


def queue_severity_counts(queue: ReviewQueue) -> Mapping[str, int]:
    """Count queued items by escalation severity, for review-demand reporting."""
    counts = {Severity.low.value: 0, Severity.medium.value: 0, Severity.high.value: 0}
    for item in queue.all_items():
        features = item.decision.features
        if features.any_failure or features.consequence >= 0.8:
            counts[Severity.high.value] += 1
        elif features.k >= 2 or features.consequence >= 0.5:
            counts[Severity.medium.value] += 1
        else:
            counts[Severity.low.value] += 1
    return counts

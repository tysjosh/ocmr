"""RAHGM router and the drop-in ``GovernedCommitManager``.

The paper's transition interface ``M_{t+1} = T(M_t, u_t, d_t)`` with
``d ∈ {accept, supersede, review, reject}`` maps directly onto OCMR's
``Commit_Manager.commit(candidate, vr)`` — the single seam every durable write
funnels through, on both the relation path (W4–W8) and the status-reconcile path.

:class:`GovernedCommitManager` implements that exact signature, so it can be
substituted for the inner :class:`~ocm.memory.commit_manager.CommitManager`
without editing ``write_pipeline.py``, ``commit_manager.py``, or ``contracts.py``
(Req 4.6, 15.2).

Tier translation:

==============  ==============================  ==========================================
Tier            Inner Commit_Manager action     Notes
==============  ==============================  ==========================================
``accept``      ``accept``                      commits as ordinary durable memory
``supersede``   ``supersede``                   retains the incumbent + provenance
``reject``      ``reject``                      preserves OCMR's reason
``review``      ``quarantine`` + ReviewQueue    accepted memory is never overwritten
==============  ==============================  ==========================================

Realizing ``review`` as an OCMR quarantine plus a linked review item is
deliberate: the review tier inherits OCMR's integrity guarantee, and the queue
supplies the *review-and-release* mechanism OCMR lacked — the mechanism the
false-quarantine audit identified as the missing piece.

Requirements: 4.1, 4.5, 4.6, 5.1, 5.2, 15.2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from ocm.governance.features import FeatureExtractor, RiskFeatures, WriteContext
from ocm.governance.policy import (
    EscalationPolicy,
    RouteGuards,
    Tier,
    compute_guards,
)
from ocm.memory.contracts import CandidateAssertion, ValidationResult, WriteOutcome
from ocm.memory.graph_store import GraphStore
from ocm.ontology.enums import Severity

logger = logging.getLogger(__name__)

#: ``failed_check`` recorded on a verdict the router rewrote to ``review``.
REVIEW_CHECK = "RAHGM.review"


@dataclass(frozen=True)
class RoutingDecision:
    """The full, inspectable record of one routing decision (Req 4.5).

    Carries everything a review item or an audit needs: the tier, the score and
    risk, the features and guards behind them, the eq. (6) clause that fired, and
    the *original* OCMR verdict so the autonomous-OCMR route stays recoverable.
    """

    tier: Tier
    risk: float
    score: float
    features: RiskFeatures
    guards: RouteGuards
    rule: str
    ocmr_action: str | None
    ocmr_failed_check: str | None
    ocmr_reason: str | None
    condition: str = "frozen_rahgm"
    write_id: str | None = None

    @property
    def escalated(self) -> bool:
        """Whether the write was routed to human review."""
        return self.tier is Tier.review

    def rationale(self) -> str:
        """A one-line, human-readable explanation of the route.

        Names the failed and unresolved checks and the rule that fired. The score
        appears alongside its inputs, never as a standalone confidence value.
        """
        parts: list[str] = []
        if self.features.failed_checks:
            parts.append("failed: " + ", ".join(self.features.failed_checks))
        if self.features.unresolved_checks:
            parts.append("unresolved: " + ", ".join(self.features.unresolved_checks))
        parts.append(
            f"consequence={self.features.consequence:.2f}, "
            f"reversibility={self.features.reversibility:.2f}, "
            f"authority={self.features.authority:.2f}, k={self.features.k}"
        )
        if self.guards.reasons:
            parts.append("guards: " + ", ".join(self.guards.reasons))
        parts.append(self.rule)
        return "; ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view for telemetry and reports."""
        return {
            "tier": self.tier.value,
            "risk": self.risk,
            "score": self.score,
            "rule": self.rule,
            "condition": self.condition,
            "write_id": self.write_id,
            "features": self.features.as_dict(),
            "guards": self.guards.as_dict(),
            "ocmr_action": self.ocmr_action,
            "ocmr_failed_check": self.ocmr_failed_check,
            "ocmr_reason": self.ocmr_reason,
            "rationale": self.rationale(),
        }


class RiskAdaptiveRouter:
    """Derives an escalation tier from OCMR's constraint-failure pattern.

    The router is a pure decision component: :meth:`decide` reads the OCMR verdict
    and the incumbent graph, extracts :class:`RiskFeatures`, and applies eq. (6).
    It performs no writes, so it can be reused for offline scoring, ablations, and
    the canary gate as well as for live governance.
    """

    def __init__(
        self,
        policy: EscalationPolicy,
        *,
        feature_extractor: FeatureExtractor | None = None,
        condition: str = "frozen_rahgm",
        settings: Any = None,
    ) -> None:
        """Create a router bound to a policy and a feature extractor."""
        self.policy = policy
        self.features = feature_extractor or FeatureExtractor(settings=settings)
        self.condition = condition

    # -- public API --------------------------------------------------------
    def decide(
        self,
        candidate: CandidateAssertion,
        vr: ValidationResult,
        graph: GraphStore,
        context: WriteContext | None = None,
        *,
        contradiction_result: Any = None,
    ) -> RoutingDecision:
        """Route one candidate write, returning the full decision record."""
        context = context or WriteContext()
        features = self.features.extract(
            candidate, graph, vr, context, contradiction_result=contradiction_result
        )
        guards = compute_guards(candidate, vr, features)
        tier, rule, risk = self.policy.route(features, guards)
        return RoutingDecision(
            tier=tier,
            risk=risk,
            score=self.policy.score(features),
            features=features,
            guards=guards,
            rule=rule,
            ocmr_action=vr.recommended_action,
            ocmr_failed_check=vr.failed_check,
            ocmr_reason=vr.reason,
            condition=self.condition,
            write_id=context.write_id,
        )

    def with_policy(self, policy: EscalationPolicy) -> "RiskAdaptiveRouter":
        """Return a router using ``policy`` (used when adaptation deploys a version)."""
        return RiskAdaptiveRouter(
            policy,
            feature_extractor=self.features,
            condition=self.condition,
        )


#: Signature of a hook invoked after a ``review`` route commits its quarantine.
ReviewHook = Callable[[CandidateAssertion, "RoutingDecision", WriteOutcome], None]


@dataclass
class GovernedCommitManager:
    """A drop-in ``Commit_Manager`` that applies RAHGM routing before committing.

    Implements the same ``commit(candidate, vr, *, created_at=None) -> WriteOutcome``
    contract as :class:`~ocm.memory.commit_manager.CommitManager`, so it can be
    assigned onto ``container.write_pipeline.commit_manager`` (and
    ``container.commit_manager``) with no change to the write pipeline.

    Attributes:
        inner: The real OCMR Commit_Manager that performs every durable action.
        router: The routing component. ``None`` disables RAHGM entirely, in which
            case this class is a transparent pass-through and the write path is
            behaviorally identical to plain OCMR (Req 15.2).
        graph: The accepted-only Graph_Store representing ``M_t``.
        review_hook: Invoked after a ``review`` route so a
            :class:`~ocm.governance.review_queue.ReviewQueue` can enqueue an item.
        context_provider: Optional callable supplying the :class:`WriteContext`
            for a candidate. The evaluation replayer uses it to attach the corpus
            rubric values and the current write id.
        decisions: Every :class:`RoutingDecision` made, in order — the telemetry
            stream the experiments consume.
    """

    inner: Any
    router: RiskAdaptiveRouter | None = None
    graph: GraphStore | None = None
    review_hook: ReviewHook | None = None
    context_provider: Callable[[CandidateAssertion], WriteContext] | None = None
    decisions: list[RoutingDecision] = field(default_factory=list)

    # -- Commit_Manager contract -------------------------------------------
    def commit(
        self,
        candidate: CandidateAssertion,
        vr: ValidationResult,
        *,
        created_at: datetime | None = None,
    ) -> WriteOutcome:
        """Route ``candidate`` and commit it through the inner Commit_Manager."""
        if self.router is None:
            return self.inner.commit(candidate, vr, created_at=created_at)

        graph = self.graph if self.graph is not None else getattr(self.inner, "graph", None)
        if graph is None:  # pragma: no cover - defensive
            return self.inner.commit(candidate, vr, created_at=created_at)

        context = (
            self.context_provider(candidate)
            if self.context_provider is not None
            else WriteContext()
        )
        decision = self.router.decide(candidate, vr, graph, context)
        self.decisions.append(decision)

        routed_vr = self.translate(vr, decision)
        outcome = self.inner.commit(candidate, routed_vr, created_at=created_at)

        if decision.tier is Tier.review and self.review_hook is not None:
            self.review_hook(candidate, decision, outcome)
        return outcome

    @staticmethod
    def summarize(outcomes: Any) -> Any:
        """Delegate summarization to the OCMR Commit_Manager (unchanged shape)."""
        from ocm.memory.commit_manager import CommitManager

        return CommitManager.summarize(outcomes)

    # -- tier translation --------------------------------------------------
    @staticmethod
    def translate(vr: ValidationResult, decision: RoutingDecision) -> ValidationResult:
        """Rewrite an OCMR verdict to enact the routed tier.

        The rewrite is minimal and preserves OCMR's diagnostic fields, so a
        quarantine or rejection still reports the constraint that failed. A
        ``supersede`` route with no identified incumbent is downgraded to
        ``review`` rather than silently accepted, matching the inner
        Commit_Manager's own conservative fallback.
        """
        tier = decision.tier

        if tier is Tier.accept:
            return ValidationResult(
                valid=True,
                failed_check=None,
                reason=vr.reason,
                severity=None,
                conflicting_ids=[],
                recommended_action="accept",
            )

        if tier is Tier.supersede:
            conflicting = list(vr.conflicting_ids) or list(decision.features.incumbent_ids)
            if not conflicting:
                return GovernedCommitManager._review_verdict(vr, decision)
            return ValidationResult(
                valid=True,
                failed_check=vr.failed_check,
                reason=(
                    vr.reason
                    or "authoritative correction supersedes the incumbent assertion"
                ),
                severity=vr.severity,
                conflicting_ids=conflicting,
                recommended_action="supersede",
            )

        if tier is Tier.reject:
            return ValidationResult(
                valid=False,
                failed_check=vr.failed_check or "RAHGM.prohibited",
                reason=vr.reason or "; ".join(decision.guards.reasons) or "prohibited write",
                severity=vr.severity or Severity.high,
                conflicting_ids=list(vr.conflicting_ids),
                recommended_action="reject",
            )

        return GovernedCommitManager._review_verdict(vr, decision)

    @staticmethod
    def _review_verdict(
        vr: ValidationResult, decision: RoutingDecision
    ) -> ValidationResult:
        """Build the quarantine verdict that realizes the ``review`` tier.

        The reason is the router's rationale so the durable quarantine record —
        and therefore the review item built from it — explains *why* the write was
        escalated in terms of the checks that failed.
        """
        return ValidationResult(
            valid=False,
            failed_check=vr.failed_check or REVIEW_CHECK,
            reason=f"escalated for human review ({decision.rationale()})",
            severity=vr.severity or _severity_for(decision),
            conflicting_ids=list(vr.conflicting_ids)
            or list(decision.features.incumbent_ids),
            recommended_action="quarantine",
        )


def _severity_for(decision: RoutingDecision) -> Severity:
    """Severity for an escalation with no OCMR severity of its own."""
    if decision.features.any_failure or decision.features.consequence >= 0.8:
        return Severity.high
    if decision.features.k >= 2 or decision.features.consequence >= 0.5:
        return Severity.medium
    return Severity.low

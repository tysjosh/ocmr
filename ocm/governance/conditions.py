"""The five experimental governance conditions of paper §3.1 / Table 1.

All five implement the same transition interface
``M_{t+1} = T(M_t, u_t, d_t)``, ``d ∈ {accept, supersede, review, reject}``, and
receive identical proposed writes, evidence, incumbent memory states, and model
outputs (Req 10.1, 10.2). They differ only in the routing signal:

=====  ====================  ==========  =================================================
Id     Name                  Review      Routing signal
=====  ====================  ==========  =================================================
C1     ``universal_review``  all writes  none; every write is queued
C2     ``autonomous_ocmr``   none        OCMR's native constraint decision
C3     ``fixed_threshold``   selective   confidence < 0.80 or high consequence
C4     ``frozen_rahgm``      selective   failure pattern, consequence, reversibility, authority
C5     ``adaptive_rahgm``    selective   C4 signal plus bounded, canary-gated updates
=====  ====================  ==========  =================================================

Separating C4 from C5 isolates the value of the feedback-learning loop from the
value of the tiering policy itself.

Requirements: 10.1, 10.2, 10.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ocm.governance.adaptation import (
    BoundedUpdater,
    CanaryGate,
    FeedbackRecord,
    PolicyRegistry,
)
from ocm.governance.features import FeatureExtractor, RiskFeatures, WriteContext
from ocm.governance.policy import (
    EscalationPolicy,
    PolicyParameters,
    RouteGuards,
    Tier,
    compute_guards,
)
from ocm.governance.review_queue import ReviewQueue
from ocm.governance.router import GovernedCommitManager, RiskAdaptiveRouter, RoutingDecision
from ocm.memory.contracts import CandidateAssertion, ValidationResult
from ocm.memory.graph_store import GraphStore

#: Confidence below which the fixed-threshold condition escalates (§3.1, C3).
FIXED_CONFIDENCE_THRESHOLD = 0.80

#: Consequence at or above which the fixed-threshold condition escalates.
FIXED_CONSEQUENCE_THRESHOLD = 0.70


class Condition(str, Enum):
    """The five governance conditions."""

    universal_review = "universal_review"
    autonomous_ocmr = "autonomous_ocmr"
    fixed_threshold = "fixed_threshold"
    frozen_rahgm = "frozen_rahgm"
    adaptive_rahgm = "adaptive_rahgm"


#: Paper condition labels (C1..C5) in the order Table 1 lists them.
CONDITION_LABELS: dict[Condition, str] = {
    Condition.universal_review: "C1",
    Condition.autonomous_ocmr: "C2",
    Condition.fixed_threshold: "C3",
    Condition.frozen_rahgm: "C4",
    Condition.adaptive_rahgm: "C5",
}

#: Display names for tables.
CONDITION_NAMES: dict[Condition, str] = {
    Condition.universal_review: "Universal review",
    Condition.autonomous_ocmr: "Autonomous OCMR",
    Condition.fixed_threshold: "Fixed threshold",
    Condition.frozen_rahgm: "Frozen RAHGM",
    Condition.adaptive_rahgm: "Adaptive RAHGM",
}

#: The conditions that present decisions to a reviewer. Autonomous OCMR is
#: evaluated in the replay experiment only, because it never queues anything.
HUMAN_FACING: tuple[Condition, ...] = (
    Condition.universal_review,
    Condition.fixed_threshold,
    Condition.frozen_rahgm,
    Condition.adaptive_rahgm,
)


# --------------------------------------------------------------------------- #
# Baseline routers (C1, C2, C3)
# --------------------------------------------------------------------------- #
class BaselineRouter(RiskAdaptiveRouter):
    """Shared plumbing for the non-RAHGM conditions.

    Reuses RAHGM's feature extraction so every condition is measured on the same
    features (Req 10.2), but replaces ``π(u)`` with the condition's own rule.
    """

    def __init__(
        self,
        *,
        condition: str,
        feature_extractor: FeatureExtractor | None = None,
        settings: Any = None,
    ) -> None:
        """Create a baseline router with a neutral (unused) policy."""
        super().__init__(
            EscalationPolicy(PolicyParameters()),
            feature_extractor=feature_extractor,
            condition=condition,
            settings=settings,
        )

    def _tier(
        self,
        candidate: CandidateAssertion,
        vr: ValidationResult,
        features: RiskFeatures,
        guards: RouteGuards,
    ) -> tuple[Tier, str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def decide(
        self,
        candidate: CandidateAssertion,
        vr: ValidationResult,
        graph: GraphStore,
        context: WriteContext | None = None,
        *,
        contradiction_result: Any = None,
    ) -> RoutingDecision:
        """Route via the condition's own rule, still recording the features."""
        context = context or WriteContext()
        features = self.features.extract(
            candidate, graph, vr, context, contradiction_result=contradiction_result
        )
        guards = compute_guards(candidate, vr, features)
        tier, rule = self._tier(candidate, vr, features, guards)
        return RoutingDecision(
            tier=tier,
            risk=self.policy.risk(features),
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


class UniversalReviewRouter(BaselineRouter):
    """C1 — every candidate write is queued for human review.

    Malformed / prohibited writes still reject: queueing a structurally invalid
    write would waste review capacity on something no analyst can repair, and the
    paper's universal-review arm is about *review of admissible writes*.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(condition=Condition.universal_review.value, **kwargs)

    def _tier(
        self,
        candidate: CandidateAssertion,
        vr: ValidationResult,
        features: RiskFeatures,
        guards: RouteGuards,
    ) -> tuple[Tier, str]:
        if guards.g:
            return Tier.reject, "g(u)=1 -> reject"
        return Tier.review, "universal review: every write is queued"


class AutonomousOcmrRouter(BaselineRouter):
    """C2 — OCMR's native decision, executed without human intervention.

    OCMR's ``quarantine`` is *not* a review tier here: nothing is presented to a
    reviewer and nothing is ever released, which is exactly the behavior whose
    false-quarantine cost the audit measured.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(condition=Condition.autonomous_ocmr.value, **kwargs)

    def _tier(
        self,
        candidate: CandidateAssertion,
        vr: ValidationResult,
        features: RiskFeatures,
        guards: RouteGuards,
    ) -> tuple[Tier, str]:
        action = vr.recommended_action or ("accept" if vr.valid else "reject")
        mapping = {
            "accept": Tier.accept,
            "supersede": Tier.supersede,
            "reject": Tier.reject,
            # OCMR quarantine holds the write with no review path. It is modelled
            # as ``review`` at the transition level (durable memory is untouched)
            # but the harness never staffs the queue for this condition.
            "quarantine": Tier.review,
        }
        return mapping.get(action, Tier.reject), f"native OCMR decision -> {action}"


class FixedThresholdRouter(BaselineRouter):
    """C3 — escalate when scalar confidence is low or consequence is high.

    This is the paper's fixed-threshold comparator: a single opaque confidence
    number plus a consequence flag, with no failure-pattern structure and no
    reversibility discount.
    """

    def __init__(
        self,
        *,
        confidence_threshold: float = FIXED_CONFIDENCE_THRESHOLD,
        consequence_threshold: float = FIXED_CONSEQUENCE_THRESHOLD,
        **kwargs: Any,
    ) -> None:
        super().__init__(condition=Condition.fixed_threshold.value, **kwargs)
        self.confidence_threshold = confidence_threshold
        self.consequence_threshold = consequence_threshold

    def _tier(
        self,
        candidate: CandidateAssertion,
        vr: ValidationResult,
        features: RiskFeatures,
        guards: RouteGuards,
    ) -> tuple[Tier, str]:
        if guards.g:
            return Tier.reject, "g(u)=1 -> reject"
        low_confidence = float(candidate.confidence) < self.confidence_threshold
        high_consequence = features.consequence >= self.consequence_threshold
        if low_confidence or high_consequence:
            trigger = "confidence" if low_confidence else "consequence"
            return (
                Tier.review,
                f"fixed threshold: {trigger} trigger "
                f"(conf={float(candidate.confidence):.2f}, q={features.consequence:.2f})",
            )
        action = vr.recommended_action or ("accept" if vr.valid else "reject")
        mapping = {
            "accept": Tier.accept,
            "supersede": Tier.supersede,
            "reject": Tier.reject,
            "quarantine": Tier.review,
        }
        return (
            mapping.get(action, Tier.reject),
            f"fixed threshold: no trigger; native OCMR -> {action}",
        )


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
@dataclass
class GovernanceHarness:
    """Everything one condition needs, wired and ready to replay.

    Attributes:
        condition: Which condition this harness realizes.
        router: The routing component.
        governed: The drop-in commit manager installed on the container.
        queue: The review queue. Populated for every condition that escalates;
            for ``autonomous_ocmr`` it stays empty because nothing is presented.
        registry: The policy registry. Present for ``adaptive_rahgm`` (live) and
            ``frozen_rahgm`` (frozen); ``None`` for the baselines.
        staffed: Whether a reviewer adjudicates this condition's queue. False for
            ``autonomous_ocmr``.
    """

    condition: Condition
    router: RiskAdaptiveRouter
    governed: GovernedCommitManager
    queue: ReviewQueue
    registry: PolicyRegistry | None = None
    staffed: bool = True
    feedback: list[FeedbackRecord] = field(default_factory=list)

    @property
    def label(self) -> str:
        """The paper's condition label (C1..C5)."""
        return CONDITION_LABELS[self.condition]

    @property
    def name(self) -> str:
        """The display name used in tables."""
        return CONDITION_NAMES[self.condition]

    @property
    def decisions(self) -> list[RoutingDecision]:
        """Every routing decision made, in write order."""
        return self.governed.decisions

    def observe_feedback(self, record: FeedbackRecord) -> None:
        """Feed one adjudication to the registry (a no-op when frozen)."""
        self.feedback.append(record)
        if self.registry is None:
            return
        outcome = self.registry.observe(record)
        if outcome is not None and outcome.deployed:
            # Deploy the new parameters onto the live router.
            self.router.policy = self.registry.policy()


def build_governance(
    condition: Condition | str,
    container: Any,
    *,
    params: PolicyParameters | None = None,
    feature_extractor: FeatureExtractor | None = None,
    canary_gate: CanaryGate | None = None,
    updater: BoundedUpdater | None = None,
    context_provider: Callable[[CandidateAssertion], WriteContext] | None = None,
    install: bool = True,
) -> GovernanceHarness:
    """Build and (by default) install a condition's governance on a container.

    Installing swaps ``container.write_pipeline.commit_manager`` (and
    ``container.commit_manager``) for a :class:`GovernedCommitManager`. The inner
    OCMR Commit_Manager still performs every durable action, so the integrity
    guarantees are unchanged; only the routing decision differs.

    Args:
        condition: Which of the five conditions to build.
        container: A :class:`~ocm.core.container.CoreContainer`.
        params: Fitted policy parameters (required in practice for C4/C5; the
            registered prior is used when omitted).
        feature_extractor: Shared extractor, so every condition sees identical
            features (Req 10.2).
        canary_gate: The fixed canary gate (C5 only).
        updater: The bounded updater (C5 only).
        context_provider: Supplies the per-write :class:`WriteContext`.
        install: When ``False`` the harness is returned without touching the
            container, which the ablation study uses for offline scoring.

    Returns:
        The wired :class:`GovernanceHarness`.
    """
    condition = Condition(condition)
    settings = getattr(container, "settings", None)
    extractor = feature_extractor or FeatureExtractor(settings=settings)
    inner = container.commit_manager

    registry: PolicyRegistry | None = None
    router: RiskAdaptiveRouter

    if condition is Condition.universal_review:
        router = UniversalReviewRouter(feature_extractor=extractor, settings=settings)
    elif condition is Condition.autonomous_ocmr:
        router = AutonomousOcmrRouter(feature_extractor=extractor, settings=settings)
    elif condition is Condition.fixed_threshold:
        router = FixedThresholdRouter(feature_extractor=extractor, settings=settings)
    else:
        policy = EscalationPolicy(params or PolicyParameters())
        frozen = condition is Condition.frozen_rahgm
        registry = PolicyRegistry(
            policy.params,
            updater=None if frozen else (updater or BoundedUpdater()),
            gate=None if frozen else canary_gate,
            frozen=frozen,
        )
        router = RiskAdaptiveRouter(
            policy,
            feature_extractor=extractor,
            condition=condition.value,
            settings=settings,
        )

    queue = ReviewQueue(
        commit_manager=inner,
        quarantine_store=getattr(container, "quarantine_store", None),
        graph=container.graph,
        repo=getattr(container, "repo", None),
    )

    governed = GovernedCommitManager(
        inner=inner,
        router=router,
        graph=container.graph,
        context_provider=context_provider,
    )

    harness = GovernanceHarness(
        condition=condition,
        router=router,
        governed=governed,
        queue=queue,
        registry=registry,
        staffed=condition is not Condition.autonomous_ocmr,
    )

    def _review_hook(candidate: CandidateAssertion, decision: RoutingDecision, outcome: Any) -> None:
        """Enqueue an escalated write, unless the condition has no reviewer."""
        if not harness.staffed:
            return
        queue.enqueue(candidate, decision, outcome)

    governed.review_hook = _review_hook

    if install:
        container.commit_manager = governed
        container.write_pipeline.commit_manager = governed

    return harness

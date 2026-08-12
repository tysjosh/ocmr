"""RAHGM bounded feedback adaptation, canary gate, and policy versioning (§3.5).

Analyst feedback may retune the *registered* parameters — the nine coefficients of
eq. (3) and the two routing thresholds — and nothing else. Tier definitions,
mandatory constraints, feature encodings, and the rejection rule are module
constants in :mod:`ocm.governance.policy` and :mod:`ocm.governance.features` that
this module has no handle on, so no reachable update can disable a mandatory
control (Req 7.4).

Three layers of restriction, mirroring safe policy improvement:

1. **Projection** — the candidate step is projected back onto the admissible set
   ``B``, preserving the monotone sign constraints.
2. **Trust region** — eq. (8): ``‖β̃ − β‖₂ ≤ 0.05`` and ``|τ̃_x − τ_x| ≤ 0.02``.
3. **Canary gate** — eq. (9):
   ``A(θ̃,θ) = 1[ΔDVR = 0 ∧ ΔMCR ≤ 0.01 ∧ ΔRR ≤ 0.05]`` on the fixed canary
   partition. A failed candidate is discarded and logged; the deployed parameters
   are unchanged.

Every accepted version records its parent, the training cases behind it, the
parameter delta, the canary result, and a rollback target (Req 8.3, 8.4).

Requirements: 7.1, 7.2, 7.3, 7.4, 8.1, 8.2, 8.3, 8.4, 8.5.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Sequence

from ocm.governance.features import RiskFeatures
from ocm.governance.policy import (
    DEFAULT_LAMBDA,
    MCR_CEILING,
    EscalationPolicy,
    PolicyParameters,
    RouteGuards,
    RoutingCase,
    Tier,
    design_row,
    _sigmoid,
    _threshold_objective,
)

logger = logging.getLogger(__name__)

#: Block size: parameters are reconsidered after this many adjudicated writes.
BLOCK_SIZE = 20

#: Trust-region radii of eq. (8) (Req 7.3).
BETA_TRUST_RADIUS = 0.05
TAU_TRUST_RADIUS = 0.02

#: Canary-gate tolerances of eq. (9) (Req 8.1).
DVR_TOLERANCE = 0.0
MCR_TOLERANCE = 0.01
RR_TOLERANCE = 0.05


# --------------------------------------------------------------------------- #
# Feedback
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeedbackRecord:
    """One adjudicated write, as consumed by the bounded updater (§3.5).

    Attributes:
        features: The escalation features of the adjudicated write.
        guards: The route guards, needed for threshold retuning.
        should_escalate: The analyst's judgment: should this have been reviewed?
            This is the supervision signal for the coefficient update.
        adjudicated_tier: The tier the analyst chose.
        confidence: Analyst confidence on a 0–1 scale, used as the sample weight.
        consequential: Whether an error here would be consequential.
        rationale: Free-text rationale, retained for the version record.
        write_id: Stable id of the underlying write.
    """

    features: RiskFeatures
    guards: RouteGuards
    should_escalate: bool
    adjudicated_tier: Tier
    confidence: float = 1.0
    consequential: bool = False
    rationale: str | None = None
    write_id: str | None = None

    @property
    def label(self) -> float:
        """The eq. (4) label: 1 when autonomous execution would be wrong."""
        return 1.0 if self.should_escalate else 0.0

    def as_routing_case(self) -> RoutingCase:
        """View the record as a routing case for threshold retuning."""
        return RoutingCase(
            features=self.features,
            guards=self.guards,
            gold_tier=Tier.review if self.should_escalate else self.adjudicated_tier,
            consequential=self.consequential,
        )


# --------------------------------------------------------------------------- #
# Bounded updater (eq. 7, 8)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CandidateUpdate:
    """A proposed parameter update, before the canary gate sees it."""

    params: PolicyParameters
    delta_norm: float
    tau_l_delta: float
    tau_h_delta: float
    block_loss: float
    n_feedback: int
    within_trust_region: bool

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "params": self.params.as_dict(),
            "delta_norm": self.delta_norm,
            "tau_l_delta": self.tau_l_delta,
            "tau_h_delta": self.tau_h_delta,
            "block_loss": self.block_loss,
            "n_feedback": self.n_feedback,
            "within_trust_region": self.within_trust_region,
        }


class BoundedUpdater:
    """Proposes a trust-region-bounded parameter update from a feedback block.

    :meth:`propose` takes one projected gradient step on the block's weighted
    logistic loss (eq. 7), projects onto ``B``, radially clips the coefficient
    delta to :data:`BETA_TRUST_RADIUS`, and retunes the two thresholds within
    :data:`TAU_TRUST_RADIUS` using the same ``F₂``/MCR objective the initial
    selection used (eq. 8).

    The updater can only ever return a :class:`PolicyParameters`. It has no access
    to tier semantics, the mandatory-check set, the feature encodings, or the
    rejection rule, so structural self-modification is unreachable by construction
    (Req 7.4).
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.5,
        lam: float = DEFAULT_LAMBDA,
        beta_radius: float = BETA_TRUST_RADIUS,
        tau_radius: float = TAU_TRUST_RADIUS,
        tau_grid: float = 0.005,
        mcr_ceiling: float = MCR_CEILING,
        enforce_trust_region: bool = True,
    ) -> None:
        """Create an updater.

        Args:
            learning_rate: Step size ``η_j``.
            lam: Weight decay used in the block gradient.
            beta_radius: Coefficient trust-region radius of eq. (8).
            tau_radius: Threshold trust-region radius of eq. (8).
            tau_grid: Lattice spacing for the bounded threshold search.
            mcr_ceiling: MCR constraint retained during threshold retuning.
            enforce_trust_region: When ``False`` the trust region is *not*
                applied. This exists only so Experiment 3 can measure the
                unconstrained-adaptation arm; production use keeps it ``True``.
        """
        self.learning_rate = learning_rate
        self.lam = lam
        self.beta_radius = beta_radius
        self.tau_radius = tau_radius
        self.tau_grid = tau_grid
        self.mcr_ceiling = mcr_ceiling
        self.enforce_trust_region = enforce_trust_region

    # -- public API --------------------------------------------------------
    def propose(
        self, params: PolicyParameters, block: Sequence[FeedbackRecord]
    ) -> CandidateUpdate:
        """Propose ``θ̃_{j+1}`` from the current parameters and a feedback block."""
        block = list(block)
        if not block:
            return CandidateUpdate(
                params=params,
                delta_norm=0.0,
                tau_l_delta=0.0,
                tau_h_delta=0.0,
                block_loss=float("nan"),
                n_feedback=0,
                within_trust_region=True,
            )

        beta = list(params.coefficient_vector())
        rows = [design_row(record.features) for record in block]
        labels = [record.label for record in block]
        weights = [max(0.05, float(record.confidence)) for record in block]
        total_weight = sum(weights) or 1.0

        loss_before = _block_loss(beta, rows, labels, weights)

        gradient = [0.0] * len(beta)
        for row, y, w in zip(rows, labels, weights):
            z = sum(b * x for b, x in zip(beta, row))
            residual = (_sigmoid(z) - y) * w
            for j, x in enumerate(row):
                gradient[j] += residual * x
        for j in range(len(beta)):
            gradient[j] /= total_weight
            if j > 0:
                gradient[j] += 2.0 * self.lam * beta[j]

        stepped = [b - self.learning_rate * g for b, g in zip(beta, gradient)]
        # Projection onto B (Π_B of eq. 7): every non-intercept coefficient
        # stays nonnegative, preserving monotonicity.
        for j in range(1, len(stepped)):
            stepped[j] = max(0.0, stepped[j])

        if self.enforce_trust_region:
            stepped = _clip_to_radius(beta, stepped, self.beta_radius)

        candidate_params = PolicyParameters.from_coefficient_vector(
            stepped,
            tau_l=params.tau_l,
            tau_h=params.tau_h,
            # Structural, not fitted: adaptation must not silently change it.
            reversibility_center=params.reversibility_center,
        ).project()

        tau_l, tau_h = self._retune_thresholds(candidate_params, block)
        candidate_params = candidate_params.with_thresholds(tau_l, tau_h).project()

        delta_norm = params.coefficient_distance(candidate_params)
        tau_l_delta = abs(candidate_params.tau_l - params.tau_l)
        tau_h_delta = abs(candidate_params.tau_h - params.tau_h)
        # When the updater does not enforce a trust region (the Experiment 3
        # unconstrained arm), the registry must not reject on that basis — the
        # point of that arm is to let a large step through and measure the damage.
        within = not self.enforce_trust_region or (
            delta_norm <= self.beta_radius + 1e-9
            and tau_l_delta <= self.tau_radius + 1e-9
            and tau_h_delta <= self.tau_radius + 1e-9
        )

        return CandidateUpdate(
            params=candidate_params,
            delta_norm=delta_norm,
            tau_l_delta=tau_l_delta,
            tau_h_delta=tau_h_delta,
            block_loss=loss_before,
            n_feedback=len(block),
            within_trust_region=within,
        )

    # -- thresholds --------------------------------------------------------
    def _retune_thresholds(
        self, params: PolicyParameters, block: Sequence[FeedbackRecord]
    ) -> tuple[float, float]:
        """Retune ``(τ_l, τ_h)`` inside the trust region on the feedback block.

        Searches only the lattice points within :data:`TAU_TRUST_RADIUS` of the
        current thresholds (the whole ``[0,1]`` lattice when the trust region is
        disabled), maximizing ``F₂`` subject to the MCR ceiling.
        """
        cases = [record.as_routing_case() for record in block]
        if not cases:
            return params.tau_l, params.tau_h

        radius = self.tau_radius if self.enforce_trust_region else 1.0
        tau_l_options = _lattice_within(params.tau_l, radius, self.tau_grid)
        tau_h_options = _lattice_within(params.tau_h, radius, self.tau_grid)

        best: tuple[float, float, float, float] | None = None
        fallback: tuple[float, float, float, float] | None = None
        for tau_l in tau_l_options:
            for tau_h in tau_h_options:
                if not tau_l < tau_h:
                    continue
                policy = EscalationPolicy(params.with_thresholds(tau_l, tau_h))
                f2, mcr, _rr = _threshold_objective(policy, cases)
                entry = (f2, -mcr, tau_l, tau_h)
                if mcr <= self.mcr_ceiling:
                    if best is None or entry > best:
                        best = entry
                if fallback is None or (-mcr, f2) > (-fallback[1], fallback[0]):
                    fallback = (f2, mcr, tau_l, tau_h)

        if best is not None:
            return best[2], best[3]
        if fallback is not None:
            return fallback[2], fallback[3]
        return params.tau_l, params.tau_h


# --------------------------------------------------------------------------- #
# Canary gate (eq. 9)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CanaryMeasurement:
    """The three canary metrics for one parameter set."""

    dvr: float
    mcr: float
    review_rate: float

    def as_dict(self) -> dict[str, float]:
        """A JSON-serializable view."""
        return {"dvr": self.dvr, "mcr": self.mcr, "review_rate": self.review_rate}


@dataclass(frozen=True)
class CanaryResult:
    """The eq. (9) gate outcome."""

    accepted: bool
    baseline: CanaryMeasurement
    candidate: CanaryMeasurement
    delta_dvr: float
    delta_mcr: float
    delta_rr: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "accepted": self.accepted,
            "baseline": self.baseline.as_dict(),
            "candidate": self.candidate.as_dict(),
            "delta_dvr": self.delta_dvr,
            "delta_mcr": self.delta_mcr,
            "delta_rr": self.delta_rr,
            "reason": self.reason,
        }


#: A callable that measures ``(DVR, MCR, RR)`` for a parameter set on the fixed
#: canary partition. The evaluation suite supplies a full scenario replay; unit
#: tests supply a lightweight stub.
CanaryEvaluator = Callable[[PolicyParameters], CanaryMeasurement]


class CanaryGate:
    """The fixed held-out gate every candidate parameter set must pass (eq. 9).

    The gate is *fixed*: the same canary partition is used for every proposal, and
    the evaluator is supplied once at construction. This is what prevents the
    feedback loop from tuning its own acceptance test.
    """

    def __init__(
        self,
        evaluator: CanaryEvaluator,
        *,
        dvr_tolerance: float = DVR_TOLERANCE,
        mcr_tolerance: float = MCR_TOLERANCE,
        rr_tolerance: float = RR_TOLERANCE,
    ) -> None:
        """Create a gate bound to a canary evaluator and the eq. (9) tolerances."""
        self.evaluator = evaluator
        self.dvr_tolerance = dvr_tolerance
        self.mcr_tolerance = mcr_tolerance
        self.rr_tolerance = rr_tolerance
        self._baseline_cache: dict[tuple[float, ...], CanaryMeasurement] = {}

    def measure(self, params: PolicyParameters) -> CanaryMeasurement:
        """Measure ``(DVR, MCR, RR)`` for ``params``, caching by parameter value."""
        key = (*params.coefficient_vector(), params.tau_l, params.tau_h)
        cached = self._baseline_cache.get(key)
        if cached is None:
            cached = self.evaluator(params)
            self._baseline_cache[key] = cached
        return cached

    def evaluate(
        self, candidate: PolicyParameters, current: PolicyParameters
    ) -> CanaryResult:
        """Apply eq. (9) and return the gate result."""
        baseline = self.measure(current)
        proposed = self.measure(candidate)

        delta_dvr = proposed.dvr - baseline.dvr
        delta_mcr = proposed.mcr - baseline.mcr
        delta_rr = proposed.review_rate - baseline.review_rate

        failures: list[str] = []
        if delta_dvr > self.dvr_tolerance + 1e-12:
            failures.append(f"dDVR={delta_dvr:+.4f} > {self.dvr_tolerance}")
        if delta_mcr > self.mcr_tolerance + 1e-12:
            failures.append(f"dMCR={delta_mcr:+.4f} > {self.mcr_tolerance}")
        if delta_rr > self.rr_tolerance + 1e-12:
            failures.append(f"dRR={delta_rr:+.4f} > {self.rr_tolerance}")

        accepted = not failures
        reason = "canary passed" if accepted else "canary regression: " + "; ".join(failures)
        return CanaryResult(
            accepted=accepted,
            baseline=baseline,
            candidate=proposed,
            delta_dvr=delta_dvr,
            delta_mcr=delta_mcr,
            delta_rr=delta_rr,
            reason=reason,
        )


# --------------------------------------------------------------------------- #
# Policy versioning (Req 8.3, 8.4)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PolicyVersion:
    """One recorded policy version and its full lineage."""

    version_id: str
    parent_id: str | None
    params: PolicyParameters
    delta_norm: float
    accepted: bool
    reason: str
    training_case_ids: tuple[str, ...] = ()
    canary: CanaryResult | None = None
    rollback_target: str | None = None
    created_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "version_id": self.version_id,
            "parent_id": self.parent_id,
            "params": self.params.as_dict(),
            "delta_norm": self.delta_norm,
            "accepted": self.accepted,
            "reason": self.reason,
            "training_case_ids": list(self.training_case_ids),
            "canary": self.canary.as_dict() if self.canary else None,
            "rollback_target": self.rollback_target,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class AdaptationOutcome:
    """The result of processing one feedback block."""

    deployed: bool
    version: PolicyVersion
    candidate: CandidateUpdate
    params: PolicyParameters

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "deployed": self.deployed,
            "version": self.version.as_dict(),
            "candidate": self.candidate.as_dict(),
            "params": self.params.as_dict(),
        }


class PolicyRegistry:
    """Versioned parameter store with canary gating and rollback.

    ``current`` is the deployed parameter set. :meth:`submit_block` runs the
    bounded update, applies the canary gate, and either deploys or discards-and-
    logs the candidate (Req 8.2). Every version — accepted or rejected — is
    retained with its parent, delta, canary result, and rollback target, so the
    lineage is fully auditable (Req 8.3).
    """

    def __init__(
        self,
        initial: PolicyParameters,
        *,
        updater: BoundedUpdater | None = None,
        gate: CanaryGate | None = None,
        frozen: bool = False,
        block_size: int = BLOCK_SIZE,
    ) -> None:
        """Create a registry.

        Args:
            initial: The parameters fitted during development (``θ₀``).
            updater: The bounded updater. ``None`` means no proposal is ever made.
            gate: The canary gate. ``None`` means updates deploy ungated — used
                only by the Experiment 3 ``bounded_no_canary`` arm.
            frozen: When ``True`` the registry never changes parameters, which is
                exactly the frozen-RAHGM condition running the identical pipeline
                with adaptation disabled (Req 8.5).
            block_size: Adjudicated writes per feedback block.
        """
        self.initial = initial.project()
        self.updater = updater
        self.gate = gate
        self.frozen = frozen
        self.block_size = block_size

        root = PolicyVersion(
            version_id="v0",
            parent_id=None,
            params=self.initial,
            delta_norm=0.0,
            accepted=True,
            reason="initial fitted policy",
            created_at=datetime.now(timezone.utc),
        )
        self.versions: list[PolicyVersion] = [root]
        self.current: PolicyParameters = self.initial
        self.current_version_id: str = root.version_id
        self.outcomes: list[AdaptationOutcome] = []
        self._pending: list[FeedbackRecord] = []
        self._counter = 0
        self.rollbacks: int = 0

    # -- inspection --------------------------------------------------------
    @property
    def history(self) -> list[PolicyVersion]:
        """Every recorded version, in submission order."""
        return list(self.versions)

    @property
    def accepted_versions(self) -> list[PolicyVersion]:
        """Only the versions that were deployed."""
        return [v for v in self.versions if v.accepted]

    @property
    def n_proposed(self) -> int:
        """Number of candidate updates proposed (excluding the root)."""
        return len(self.versions) - 1

    @property
    def n_rejected(self) -> int:
        """Number of candidates the gate rejected."""
        return sum(1 for v in self.versions[1:] if not v.accepted)

    @property
    def regression_rate(self) -> float:
        """Canary-regression rate: rejected proposals over proposals made."""
        return self.n_rejected / self.n_proposed if self.n_proposed else 0.0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposals that deployed."""
        proposed = self.n_proposed
        return (proposed - self.n_rejected) / proposed if proposed else 0.0

    @property
    def cumulative_drift(self) -> float:
        """``‖θ_j − θ₀‖₂`` over the coefficients (Req 8.3 telemetry)."""
        return self.initial.coefficient_distance(self.current)

    def policy(self) -> EscalationPolicy:
        """An :class:`EscalationPolicy` over the deployed parameters."""
        return EscalationPolicy(self.current)

    # -- feedback ingestion ------------------------------------------------
    def observe(self, record: FeedbackRecord) -> AdaptationOutcome | None:
        """Add one adjudicated write; process a block when it fills.

        Returns the :class:`AdaptationOutcome` when a block completed, else
        ``None``.
        """
        if self.frozen:
            return None
        self._pending.append(record)
        if len(self._pending) < self.block_size:
            return None
        block, self._pending = self._pending[: self.block_size], self._pending[self.block_size :]
        return self.submit_block(block)

    def submit_block(self, block: Sequence[FeedbackRecord]) -> AdaptationOutcome:
        """Run the bounded update and canary gate for one feedback block."""
        if self.frozen or self.updater is None:
            # A frozen registry proposes nothing, so no version is recorded. This
            # keeps the proposal and regression counts meaningful: frozen RAHGM has
            # a 0% acceptance rate over *zero* proposals, not a 100% rejection rate.
            candidate = CandidateUpdate(
                params=self.current,
                delta_norm=0.0,
                tau_l_delta=0.0,
                tau_h_delta=0.0,
                block_loss=float("nan"),
                n_feedback=len(block),
                within_trust_region=True,
            )
            outcome = AdaptationOutcome(
                deployed=False,
                version=self.versions[0],
                candidate=candidate,
                params=self.current,
            )
            self.outcomes.append(outcome)
            return outcome

        candidate = self.updater.propose(self.current, block)

        if not candidate.within_trust_region:
            version = self._record_version(
                params=candidate.params,
                delta_norm=candidate.delta_norm,
                accepted=False,
                reason=(
                    f"trust-region violation: ||dbeta||={candidate.delta_norm:.4f}, "
                    f"|dtau_l|={candidate.tau_l_delta:.4f}, "
                    f"|dtau_h|={candidate.tau_h_delta:.4f}"
                ),
                canary=None,
                training_case_ids=_case_ids(block),
            )
            outcome = AdaptationOutcome(False, version, candidate, self.current)
            self.outcomes.append(outcome)
            logger.info("discarded candidate policy: %s", version.reason)
            return outcome

        canary = self.gate.evaluate(candidate.params, self.current) if self.gate else None
        accepted = canary.accepted if canary is not None else True

        version = self._record_version(
            params=candidate.params,
            delta_norm=candidate.delta_norm,
            accepted=accepted,
            reason=canary.reason if canary is not None else "deployed ungated",
            canary=canary,
            training_case_ids=_case_ids(block),
        )

        if accepted:
            self.current = candidate.params
            self.current_version_id = version.version_id
        else:
            logger.info("discarded candidate policy: %s", version.reason)

        outcome = AdaptationOutcome(accepted, version, candidate, self.current)
        self.outcomes.append(outcome)
        return outcome

    # -- rollback (Req 8.4) ------------------------------------------------
    def rollback(self, version_id: str | None = None) -> PolicyParameters:
        """Roll back to a recorded version (default: the current version's parent)."""
        if version_id is None:
            current = self._find(self.current_version_id)
            version_id = current.rollback_target or "v0"
        target = self._find(version_id)
        if not target.accepted and target.version_id != "v0":
            raise ValueError(f"cannot roll back to never-deployed version {version_id!r}")
        self.current = target.params
        self.current_version_id = target.version_id
        self.rollbacks += 1
        return self.current

    # -- internals ---------------------------------------------------------
    def _record_version(
        self,
        *,
        params: PolicyParameters,
        delta_norm: float,
        accepted: bool,
        reason: str,
        canary: CanaryResult | None,
        training_case_ids: tuple[str, ...],
    ) -> PolicyVersion:
        """Append a version record to the lineage."""
        self._counter += 1
        version = PolicyVersion(
            version_id=f"v{self._counter}",
            parent_id=self.current_version_id,
            params=params,
            delta_norm=delta_norm,
            accepted=accepted,
            reason=reason,
            training_case_ids=training_case_ids,
            canary=canary,
            rollback_target=self.current_version_id,
            created_at=datetime.now(timezone.utc),
        )
        self.versions.append(version)
        return version

    def _find(self, version_id: str) -> PolicyVersion:
        """Look up a version by id."""
        for version in self.versions:
            if version.version_id == version_id:
                return version
        raise KeyError(f"unknown policy version {version_id!r}")

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view of the whole lineage."""
        return {
            "initial": self.initial.as_dict(),
            "current": self.current.as_dict(),
            "current_version_id": self.current_version_id,
            "frozen": self.frozen,
            "n_proposed": self.n_proposed,
            "n_rejected": self.n_rejected,
            "acceptance_rate": self.acceptance_rate,
            "regression_rate": self.regression_rate,
            "cumulative_drift": self.cumulative_drift,
            "rollbacks": self.rollbacks,
            "versions": [v.as_dict() for v in self.versions],
        }


# --------------------------------------------------------------------------- #
# Structural-immutability witness (Req 7.4)
# --------------------------------------------------------------------------- #
def tier_disablement_detected(
    registry: PolicyRegistry, probes: Iterable[RoutingCase]
) -> bool:
    """Whether any deployed version would route a mandatory failure to ``accept``.

    This is the operational test that adaptation never disables a mandatory
    control. Because eq. (6) consults ``m(u)`` *before* the thresholds, and
    ``m(u)`` is computed from the immutable :data:`MANDATORY_CHECKS`, the answer
    must be ``False`` for every reachable parameter set — the function exists to
    assert that invariant empirically across a real run's version history.
    """
    probes = [case for case in probes if case.guards.m or case.guards.g]
    if not probes:
        return False
    for version in registry.history:
        policy = EscalationPolicy(version.params)
        for case in probes:
            tier, _rule, _risk = policy.route(case.features, case.guards)
            if tier is Tier.accept:
                return True
    return False


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _block_loss(
    beta: Sequence[float],
    rows: Sequence[Sequence[float]],
    labels: Sequence[float],
    weights: Sequence[float],
) -> float:
    """Mean weighted logistic loss of ``beta`` on a block."""
    total = 0.0
    weight = 0.0
    for row, y, w in zip(rows, labels, weights):
        z = sum(b * x for b, x in zip(beta, row))
        r = min(max(_sigmoid(z), 1e-12), 1 - 1e-12)
        total += -w * (y * math.log(r) + (1 - y) * math.log(1 - r))
        weight += w
    return total / (weight or 1.0)


def _clip_to_radius(
    origin: Sequence[float], point: Sequence[float], radius: float
) -> list[float]:
    """Radially clip ``point`` so ``‖point − origin‖₂ ≤ radius`` (eq. 8)."""
    delta = [p - o for p, o in zip(point, origin)]
    norm = math.sqrt(sum(d * d for d in delta))
    if norm <= radius or norm == 0.0:
        return list(point)
    scale = radius / norm
    return [o + d * scale for o, d in zip(origin, delta)]


def _lattice_within(centre: float, radius: float, grid: float) -> list[float]:
    """Lattice points in ``[0,1]`` within ``radius`` of ``centre``."""
    low = max(0.0, centre - radius)
    high = min(1.0, centre + radius)
    steps = max(1, int(round((high - low) / grid)))
    points = [round(low + i * grid, 10) for i in range(steps + 1)]
    points = [p for p in points if low - 1e-12 <= p <= high + 1e-12]
    if centre not in points:
        points.append(round(centre, 10))
    return sorted(set(points))


def _case_ids(block: Sequence[FeedbackRecord]) -> tuple[str, ...]:
    """Stable ids of the training cases behind a version (Req 8.3)."""
    return tuple(r.write_id for r in block if r.write_id)

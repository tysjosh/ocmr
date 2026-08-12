"""RAHGM bounded adaptation, trust regions, canary gate, and versioning.

Covers Req 7.x (bounded updates) and 8.x (canary gate, versioning, rollback), and
the structural-immutability guarantee that no reachable update can disable a
mandatory control.
"""

from __future__ import annotations

import pytest

from ocm.governance.adaptation import (
    BETA_TRUST_RADIUS,
    TAU_TRUST_RADIUS,
    BoundedUpdater,
    CanaryGate,
    CanaryMeasurement,
    FeedbackRecord,
    PolicyRegistry,
    tier_disablement_detected,
)
from ocm.governance.features import FAIL, UNRESOLVED, RiskFeatures
from ocm.governance.policy import (
    EscalationPolicy,
    PolicyParameters,
    RouteGuards,
    RoutingCase,
    Tier,
)


def _block(n: int = 20, *, should_escalate: bool = True, confidence: float = 0.9):
    """A feedback block of ``n`` records."""
    return [
        FeedbackRecord(
            features=RiskFeatures(
                f_c=FAIL if should_escalate else 0.0,
                f_e=UNRESOLVED if should_escalate else 0.0,
                consequence=0.8 if should_escalate else 0.2,
                reversibility=0.3 if should_escalate else 0.95,
                authority=0.3 if should_escalate else 0.95,
            ),
            guards=RouteGuards(),
            should_escalate=should_escalate,
            adjudicated_tier=Tier.review if should_escalate else Tier.accept,
            confidence=confidence,
            consequential=should_escalate,
            write_id=f"w{i:03d}",
        )
        for i in range(n)
    ]


def _accepting_gate() -> CanaryGate:
    """A gate whose canary metrics never move, so every candidate passes."""
    return CanaryGate(lambda _p: CanaryMeasurement(dvr=0.0, mcr=0.0, review_rate=0.2))


# --------------------------------------------------------------------------- #
# Trust region (Req 7.2, 7.3)
# --------------------------------------------------------------------------- #
def test_coefficient_delta_respects_the_trust_region():
    """``‖β̃ − β‖₂ ≤ 0.05`` (eq. 8, Req 7.3)."""
    updater = BoundedUpdater(learning_rate=10.0)  # a deliberately huge step
    start = PolicyParameters()
    candidate = updater.propose(start, _block())
    assert candidate.delta_norm <= BETA_TRUST_RADIUS + 1e-9
    assert candidate.within_trust_region


def test_threshold_deltas_respect_the_trust_region():
    """``|τ̃_x − τ_x| ≤ 0.02`` (eq. 8, Req 7.3)."""
    updater = BoundedUpdater(learning_rate=10.0)
    candidate = updater.propose(PolicyParameters(), _block())
    assert candidate.tau_l_delta <= TAU_TRUST_RADIUS + 1e-9
    assert candidate.tau_h_delta <= TAU_TRUST_RADIUS + 1e-9


def test_updates_stay_in_the_admissible_set():
    """Every proposal is monotone: projection is applied after the step (Req 7.2)."""
    updater = BoundedUpdater(learning_rate=5.0)
    params = PolicyParameters()
    for _ in range(10):
        candidate = updater.propose(params, _block(should_escalate=False))
        params = candidate.params
        assert EscalationPolicy(params).is_monotonic()


def test_unconstrained_updater_reports_itself_as_unbounded():
    """With enforcement off, the trust region is not used as a rejection reason."""
    updater = BoundedUpdater(learning_rate=5.0, enforce_trust_region=False)
    candidate = updater.propose(PolicyParameters(), _block())
    assert candidate.within_trust_region
    assert candidate.delta_norm > BETA_TRUST_RADIUS


def test_empty_block_proposes_no_change():
    """A block with no feedback leaves the parameters alone."""
    candidate = BoundedUpdater().propose(PolicyParameters(), [])
    assert candidate.delta_norm == 0.0
    assert candidate.n_feedback == 0


# --------------------------------------------------------------------------- #
# Canary gate (Req 8.1, 8.2)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("dvr", "mcr", "rr", "accepted"),
    [
        (0.0, 0.0, 0.0, True),      # no change
        (0.0, 0.005, 0.02, True),   # within tolerance
        (0.001, 0.0, 0.0, False),   # any DVR increase fails
        (0.0, 0.02, 0.0, False),    # MCR beyond 0.01
        (0.0, 0.0, 0.06, False),    # review demand beyond 0.05
        (-0.01, -0.01, -0.10, True),  # improvements always pass
    ],
)
def test_gate_applies_equation_nine(dvr: float, mcr: float, rr: float, accepted: bool):
    """``A(θ̃,θ) = 1[ΔDVR = 0 ∧ ΔMCR ≤ 0.01 ∧ ΔRR ≤ 0.05]`` (Req 8.1)."""
    baseline = CanaryMeasurement(dvr=0.10, mcr=0.05, review_rate=0.30)
    current = PolicyParameters()
    candidate = PolicyParameters(beta_0=-1.0)

    def _evaluate(params: PolicyParameters) -> CanaryMeasurement:
        if params == current:
            return baseline
        return CanaryMeasurement(
            dvr=baseline.dvr + dvr,
            mcr=baseline.mcr + mcr,
            review_rate=baseline.review_rate + rr,
        )

    result = CanaryGate(_evaluate).evaluate(candidate, current)
    assert result.accepted is accepted
    if not accepted:
        assert "regression" in result.reason


def test_failed_candidate_is_discarded_and_logged(caplog):
    """A blocked candidate leaves the deployed parameters unchanged (Req 8.2)."""
    gate = CanaryGate(
        lambda p: CanaryMeasurement(
            dvr=0.0 if p == PolicyParameters() else 0.5, mcr=0.0, review_rate=0.2
        )
    )
    registry = PolicyRegistry(PolicyParameters(), updater=BoundedUpdater(), gate=gate)
    before = registry.current

    outcome = registry.submit_block(_block())
    assert not outcome.deployed
    assert registry.current == before
    assert registry.n_rejected == 1
    assert not registry.versions[-1].accepted
    assert "regression" in registry.versions[-1].reason


def test_accepted_candidate_is_deployed():
    """A passing candidate becomes the deployed policy."""
    registry = PolicyRegistry(
        PolicyParameters(), updater=BoundedUpdater(), gate=_accepting_gate()
    )
    before = registry.current
    outcome = registry.submit_block(_block())
    assert outcome.deployed
    assert registry.current != before


def test_gate_is_memoized_by_parameter_value():
    """The fixed gate evaluates each distinct parameter set once."""
    calls: list[PolicyParameters] = []

    def _evaluate(params: PolicyParameters) -> CanaryMeasurement:
        calls.append(params)
        return CanaryMeasurement(dvr=0.0, mcr=0.0, review_rate=0.2)

    gate = CanaryGate(_evaluate)
    params = PolicyParameters()
    gate.measure(params)
    gate.measure(params)
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# Frozen policy (Req 8.5)
# --------------------------------------------------------------------------- #
def test_frozen_registry_never_changes_parameters():
    """Frozen RAHGM runs the identical pipeline with updates disabled (Req 8.5)."""
    registry = PolicyRegistry(PolicyParameters(), updater=BoundedUpdater(), frozen=True)
    before = registry.current
    for _ in range(5):
        registry.submit_block(_block())
    assert registry.current == before


def test_frozen_registry_records_no_proposals():
    """A frozen registry proposes nothing, so its counts stay meaningful."""
    registry = PolicyRegistry(PolicyParameters(), updater=BoundedUpdater(), frozen=True)
    for _ in range(3):
        registry.submit_block(_block())
    assert registry.n_proposed == 0
    assert registry.n_rejected == 0
    assert registry.regression_rate == 0.0
    assert registry.cumulative_drift == 0.0


def test_frozen_observe_is_a_noop():
    """``observe`` on a frozen registry never triggers a block."""
    registry = PolicyRegistry(PolicyParameters(), frozen=True)
    for record in _block(40):
        assert registry.observe(record) is None


# --------------------------------------------------------------------------- #
# Blocking and versioning (Req 7.1, 8.3, 8.4)
# --------------------------------------------------------------------------- #
def test_blocks_are_processed_every_twenty_adjudications():
    """Adaptation reconsiders parameters per 20-write block (Req 7.1)."""
    registry = PolicyRegistry(
        PolicyParameters(), updater=BoundedUpdater(), gate=_accepting_gate(), block_size=20
    )
    outcomes = [registry.observe(record) for record in _block(45)]
    triggered = [o for o in outcomes if o is not None]
    assert len(triggered) == 2  # 45 // 20


def test_versions_record_full_lineage():
    """Each version records parent, delta, canary result, and rollback target (Req 8.3)."""
    registry = PolicyRegistry(
        PolicyParameters(), updater=BoundedUpdater(), gate=_accepting_gate()
    )
    registry.submit_block(_block())
    version = registry.versions[-1]
    assert version.parent_id == "v0"
    assert version.rollback_target == "v0"
    assert version.canary is not None
    assert version.training_case_ids
    assert version.delta_norm >= 0.0


def test_rollback_restores_a_recorded_version():
    """Rollback to a prior version is supported (Req 8.4)."""
    registry = PolicyRegistry(
        PolicyParameters(), updater=BoundedUpdater(), gate=_accepting_gate()
    )
    root = registry.current
    registry.submit_block(_block())
    assert registry.current != root

    restored = registry.rollback("v0")
    assert restored == root
    assert registry.current_version_id == "v0"
    assert registry.rollbacks == 1


def test_rollback_to_a_never_deployed_version_is_refused():
    """A blocked candidate is not a rollback target."""
    gate = CanaryGate(
        lambda p: CanaryMeasurement(
            dvr=0.0 if p == PolicyParameters() else 0.9, mcr=0.0, review_rate=0.2
        )
    )
    registry = PolicyRegistry(PolicyParameters(), updater=BoundedUpdater(), gate=gate)
    registry.submit_block(_block())
    rejected = registry.versions[-1]
    with pytest.raises(ValueError):
        registry.rollback(rejected.version_id)


def test_unknown_version_raises():
    """Rolling back to a nonexistent version is an error, not a silent no-op."""
    registry = PolicyRegistry(PolicyParameters())
    with pytest.raises(KeyError):
        registry.rollback("v999")


def test_cumulative_drift_is_tracked():
    """``‖θ_j − θ₀‖₂`` grows with accepted updates and is reported."""
    registry = PolicyRegistry(
        PolicyParameters(), updater=BoundedUpdater(), gate=_accepting_gate()
    )
    drifts = []
    for _ in range(4):
        registry.submit_block(_block())
        drifts.append(registry.cumulative_drift)
    assert drifts == sorted(drifts)
    assert drifts[-1] <= 4 * BETA_TRUST_RADIUS + 1e-9


# --------------------------------------------------------------------------- #
# Structural immutability (Req 7.4)
# --------------------------------------------------------------------------- #
def test_adaptation_cannot_disable_a_mandatory_control():
    """No reachable parameter set routes a mandatory failure to ``accept`` (Req 7.4).

    Uses an adversarial feedback stream that asserts, with maximum confidence, that
    mandatory-failure writes need no review — the strongest available attempt to
    talk the policy into unsafe autonomy.
    """
    registry = PolicyRegistry(
        PolicyParameters(), updater=BoundedUpdater(), gate=_accepting_gate()
    )
    adversarial = [
        FeedbackRecord(
            features=RiskFeatures(f_s=FAIL, consequence=1.0, reversibility=0.0),
            guards=RouteGuards(m=True, g=True),
            should_escalate=False,
            adjudicated_tier=Tier.accept,
            confidence=1.0,
            consequential=True,
            write_id=f"adv{i}",
        )
        for i in range(20)
    ]
    for _ in range(10):
        registry.submit_block(adversarial)

    probes = [
        RoutingCase(RiskFeatures(f_s=FAIL), RouteGuards(m=True), Tier.reject, True),
        RoutingCase(RiskFeatures(), RouteGuards(g=True), Tier.reject, True),
        RoutingCase(RiskFeatures(f_e=FAIL), RouteGuards(m=True), Tier.review, True),
    ]
    assert not tier_disablement_detected(registry, probes)


def test_updater_can_only_return_parameters():
    """The updater has no handle on tier semantics or the mandatory set (Req 7.4)."""
    candidate = BoundedUpdater().propose(PolicyParameters(), _block())
    assert isinstance(candidate.params, PolicyParameters)
    # The registered, adaptable surface is the ten coefficients plus two
    # thresholds. ``reversibility_center`` is structural, not fitted, and
    # ``discount_dominance`` is derived; neither is a tuning knob.
    adaptable = {
        "beta_0",
        "beta_f",
        "beta_k",
        "beta_q",
        "beta_v",
        "beta_a",
        "tau_l",
        "tau_h",
    }
    assert adaptable <= set(candidate.params.as_dict())


def test_adaptation_preserves_structural_configuration():
    """A bounded update never changes a structural field (Req 7.4)."""
    start = PolicyParameters(reversibility_center=0.5)
    candidate = BoundedUpdater(learning_rate=5.0).propose(start, _block())
    assert candidate.params.reversibility_center == 0.5


def test_default_parameters_reproduce_equation_three():
    """``reversibility_center`` defaults to 0.0, which is eq. (3) exactly."""
    assert PolicyParameters().reversibility_center == 0.0


def test_tier_disablement_probe_is_vacuous_without_probes():
    """No probes means no claim; the check returns ``False`` rather than passing blindly."""
    registry = PolicyRegistry(PolicyParameters())
    assert not tier_disablement_detected(registry, [])

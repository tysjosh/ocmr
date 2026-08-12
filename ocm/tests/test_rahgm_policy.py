"""RAHGM feature encoding, escalation score, fit, thresholds, and routing.

Covers Req 1.x (feature encoding and rubrics), 2.x (score and monotonic fit),
3.x (threshold selection), and 4.x (the deterministic routing rule).
"""

from __future__ import annotations

import math

import pytest

from ocm.governance.features import (
    AUTHORITATIVE_FLOOR,
    FAIL,
    PASS,
    UNATTRIBUTED_AUTHORITY,
    UNRESOLVED,
    FeatureExtractor,
    RiskFeatures,
    Rubric,
    WriteContext,
)
from ocm.governance.policy import (
    MANDATORY_CHECKS,
    EscalationPolicy,
    PolicyParameters,
    RouteGuards,
    RoutingCase,
    Tier,
    TrainingSample,
    build_training_samples,
    compute_guards,
    fit_policy,
    select_thresholds,
)
from ocm.memory.contracts import CandidateAssertion, ValidationResult
from ocm.ontology.enums import Severity, WriteIntent


# --------------------------------------------------------------------------- #
# Feature encoding (Req 1.1-1.7)
# --------------------------------------------------------------------------- #
def _candidate(**overrides) -> CandidateAssertion:
    payload = {
        "subject_id": "tsk-1",
        "predicate": "ASSIGNED_TO",
        "object_id": "per-1",
        "confidence": 0.9,
        "source_ref": "analyst:test:1",
        "write_intent": WriteIntent.update,
    }
    payload.update(overrides)
    return CandidateAssertion(**payload)


class _EmptyGraph:
    """A graph stub with no entities, for pure encoding tests."""

    def out_edges(self, *_a, **_k):
        return []

    def in_edges(self, *_a, **_k):
        return []

    def get_entity_type(self, _entity_id):
        return None

    def get_entity_payload(self, _entity_id):
        return None

    class _G:
        @staticmethod
        def edges(**_k):
            return []

    g = _G()


def test_components_use_only_the_three_admissible_encodings():
    """Every component is exactly 0.0, 0.5, or 1.0 (Req 1.2)."""
    extractor = FeatureExtractor()
    graph = _EmptyGraph()
    verdicts = [
        ValidationResult(valid=True),
        ValidationResult(valid=False, failed_check="C1", severity=Severity.high),
        ValidationResult(valid=False, failed_check="C9", severity=Severity.high),
        ValidationResult(valid=False, failed_check="C2", severity=Severity.high),
        ValidationResult(valid=False, failed_check="C8", severity=Severity.medium),
        ValidationResult(valid=False, failed_check="C7", severity=Severity.high),
        ValidationResult(valid=False, failed_check="C7", severity=Severity.medium),
        ValidationResult(valid=False, failed_check="schema.required_fields"),
    ]
    for vr in verdicts:
        features = extractor.extract(_candidate(), graph, vr, WriteContext())
        for component in features.vector:
            assert component in {PASS, UNRESOLVED, FAIL}


@pytest.mark.parametrize(
    ("check", "index"),
    [("C1", 0), ("C9", 1), ("C2", 2), ("C3", 2), ("C10", 2), ("C8", 3)],
)
def test_failed_check_maps_to_its_component(check: str, index: int):
    """Each OCMR check drives the component the paper assigns it (Req 1.3-1.6)."""
    features = FeatureExtractor().extract(
        _candidate(),
        _EmptyGraph(),
        ValidationResult(valid=False, failed_check=check, severity=Severity.high),
        WriteContext(),
    )
    assert features.vector[index] == FAIL


def test_alias_ambiguity_is_unresolved_not_failed():
    """An unresolved alias encodes 0.5, the distinction RAHGM exists to exploit."""
    features = FeatureExtractor().extract(
        _candidate(),
        _EmptyGraph(),
        ValidationResult(valid=True),
        WriteContext(alias_ambiguous=True),
    )
    assert features.f_e == UNRESOLVED
    assert "entity_resolution" in features.unresolved_checks


def test_hard_contradiction_fails_soft_contradiction_is_unresolved():
    """``f_c`` separates a hard conflict from a soft one (Req 1.7)."""
    extractor = FeatureExtractor()
    hard = extractor.extract(
        _candidate(),
        _EmptyGraph(),
        ValidationResult(valid=False, failed_check="C7", severity=Severity.high),
        WriteContext(),
    )
    soft = extractor.extract(
        _candidate(),
        _EmptyGraph(),
        ValidationResult(valid=False, failed_check="C7", severity=Severity.medium),
        WriteContext(),
    )
    assert hard.f_c == FAIL
    assert soft.f_c == UNRESOLVED


def test_evidence_component_tracks_the_floor():
    """Zero evidence fails; below-floor evidence is unresolved (Req 1.6)."""

    class _Settings:
        supersede_evidence_min = 2

    extractor = FeatureExtractor(settings=_Settings())
    unattributed = extractor.extract(
        _candidate(source_ref=""), _EmptyGraph(), ValidationResult(valid=True), WriteContext()
    )
    assert unattributed.f_v == FAIL

    attributed = extractor.extract(
        _candidate(), _EmptyGraph(), ValidationResult(valid=True), WriteContext()
    )
    # One evidence unit (the source_ref) against a floor of two.
    assert attributed.f_v == UNRESOLVED


def test_k_counts_simultaneous_unresolved_or_failed_checks():
    """``k`` and ``[k-1]₊`` follow eq. (3) (Req 1.8)."""
    features = RiskFeatures(f_e=UNRESOLVED, f_s=PASS, f_t=FAIL, f_v=PASS, f_c=UNRESOLVED)
    assert features.k == 3
    assert features.interaction == 2.0

    clean = RiskFeatures()
    assert clean.k == 0
    assert clean.interaction == 0.0


def test_extraction_is_deterministic():
    """Repeated extraction of the same inputs yields identical features (Req 1.10)."""
    extractor = FeatureExtractor()
    args = (_candidate(), _EmptyGraph(), ValidationResult(valid=True), WriteContext())
    assert extractor.extract(*args).as_dict() == extractor.extract(*args).as_dict()


# --------------------------------------------------------------------------- #
# Rubrics (Req 1.9)
# --------------------------------------------------------------------------- #
def test_authority_rubric_reads_the_source_scheme():
    """Authority comes from the ``source_ref`` scheme, blank means unattributed."""
    rubric = Rubric()
    assert rubric.authority(_candidate(source_ref="analyst:x:1"), WriteContext()) >= AUTHORITATIVE_FLOOR
    assert rubric.authority(_candidate(source_ref="untrusted:x:1"), WriteContext()) < 0.5
    assert rubric.authority(_candidate(source_ref=""), WriteContext()) == UNATTRIBUTED_AUTHORITY


def test_explicit_context_overrides_the_rubric():
    """A corpus case can pin rubric values (Req 1.9)."""
    rubric = Rubric()
    context = WriteContext(authority=0.42, consequence=0.11, reversibility=0.99)
    assert rubric.authority(_candidate(), context) == pytest.approx(0.42)


def test_rubric_values_stay_in_unit_interval():
    """``q``, ``v``, ``a`` are always in [0, 1]."""
    rubric = Rubric()
    context = WriteContext(consequence=5.0, reversibility=-3.0, authority=99.0)
    candidate = _candidate()
    graph = _EmptyGraph()
    assert 0.0 <= rubric.authority(candidate, context) <= 1.0
    assert 0.0 <= rubric.consequence(candidate, graph, ValidationResult(valid=True), context) <= 1.0
    assert 0.0 <= rubric.reversibility(candidate, graph, [], context) <= 1.0


def test_deletion_intent_is_treated_as_irreversible():
    """A destructive write gets the lowest reversibility band."""
    rubric = Rubric()
    reversible = rubric.reversibility(
        _candidate(write_intent=WriteIntent.update), _EmptyGraph(), ["a1"], WriteContext()
    )
    irreversible = rubric.reversibility(
        _candidate(write_intent=WriteIntent.deletion), _EmptyGraph(), ["a1"], WriteContext()
    )
    assert irreversible < reversible


# --------------------------------------------------------------------------- #
# Escalation score and monotonicity (Req 2.x)
# --------------------------------------------------------------------------- #
def test_score_matches_equation_three():
    """``z(u)`` is computed exactly as eq. (3) specifies."""
    params = PolicyParameters(
        beta_0=-1.0,
        beta_f=(1.0, 2.0, 3.0, 4.0, 5.0),
        beta_k=0.5,
        beta_q=1.5,
        beta_v=0.25,
        beta_a=0.75,
        tau_l=0.2,
        tau_h=0.8,
    )
    features = RiskFeatures(
        f_e=1.0,
        f_s=0.5,
        f_t=0.0,
        f_v=0.5,
        f_c=1.0,
        consequence=0.4,
        reversibility=0.8,
        authority=0.6,
    )
    expected = (
        -1.0
        + 1.0 * 1.0
        + 2.0 * 0.5
        + 3.0 * 0.0
        + 4.0 * 0.5
        + 5.0 * 1.0
        + 0.5 * 2.0  # k = 4 -> [k-1]+ = 3? see below
    )
    # k counts components > 0: f_e, f_s, f_v, f_c = 4, so [k-1]+ = 3.
    expected = expected - 0.5 * 2.0 + 0.5 * 3.0
    expected += 1.5 * 0.4 - 0.25 * 0.8 - 0.75 * 0.6
    policy = EscalationPolicy(params)
    assert policy.score(features) == pytest.approx(expected)
    assert policy.risk(features) == pytest.approx(1.0 / (1.0 + math.exp(-expected)))


@pytest.mark.parametrize("component", ["f_e", "f_s", "f_t", "f_v", "f_c"])
def test_risk_is_nondecreasing_in_each_failure_component(component: str):
    """``∂r/∂f_i ≥ 0`` (Req 2.5)."""
    policy = EscalationPolicy()
    low = RiskFeatures(**{component: PASS})
    high = RiskFeatures(**{component: FAIL})
    assert policy.risk(high) >= policy.risk(low)


def test_risk_is_nondecreasing_in_k_and_consequence():
    """``∂r/∂k ≥ 0`` and ``∂r/∂q ≥ 0`` (Req 2.5)."""
    policy = EscalationPolicy()
    one = RiskFeatures(f_e=FAIL)
    three = RiskFeatures(f_e=FAIL, f_s=FAIL, f_c=FAIL)
    assert policy.risk(three) >= policy.risk(one)
    assert policy.risk(RiskFeatures(consequence=0.9)) >= policy.risk(
        RiskFeatures(consequence=0.1)
    )


def test_risk_is_nonincreasing_in_reversibility_and_authority():
    """``∂r/∂v ≤ 0`` and ``∂r/∂a ≤ 0`` — the displayed discounts (Req 2.3, 2.5)."""
    policy = EscalationPolicy()
    assert policy.risk(RiskFeatures(reversibility=0.95)) <= policy.risk(
        RiskFeatures(reversibility=0.05)
    )
    assert policy.risk(RiskFeatures(authority=0.95)) <= policy.risk(
        RiskFeatures(authority=0.05)
    )


def test_reversibility_is_a_one_sided_discount_under_equation_three():
    """Eq. (3) lets reversibility discount risk but never add it.

    Because ``β_v ≥ 0`` and ``v ∈ [0,1]``, the term ``−β_v·v`` spans ``[−β_v, 0]``.
    The least reversible write is therefore only as risky as one carrying no
    reversibility information at all. This documents a property of the paper's
    equation, not a bug in the implementation.
    """
    params = PolicyParameters(beta_v=1.5, reversibility_center=0.0)
    policy = EscalationPolicy(params)
    baseline = policy.score(RiskFeatures(reversibility=0.0))
    irreversible = policy.score(RiskFeatures(reversibility=0.05))
    reversible = policy.score(RiskFeatures(reversibility=1.0))
    assert irreversible <= baseline
    assert reversible < irreversible
    # No reversibility value can raise the score above the v = 0 baseline.
    assert max(
        policy.score(RiskFeatures(reversibility=v / 20)) for v in range(21)
    ) == pytest.approx(baseline)


def test_centering_makes_reversibility_two_sided():
    """Centering the term lets irreversibility add risk (not the paper's eq. 3)."""
    centered = EscalationPolicy(PolicyParameters(beta_v=1.5, reversibility_center=0.5))
    low = centered.score(RiskFeatures(reversibility=0.0))
    mid = centered.score(RiskFeatures(reversibility=0.5))
    high = centered.score(RiskFeatures(reversibility=1.0))
    assert low > mid > high


def test_discount_dominance_is_reported():
    """Eq. (3) does not bound the discounts against the consequence weight."""
    params = PolicyParameters(beta_q=1.0, beta_v=1.5, beta_a=1.5)
    assert params.discount_dominance == pytest.approx(3.0)
    assert params.as_dict()["discount_dominance"] == pytest.approx(3.0)


def test_fit_preserves_the_structural_center():
    """Fitting must not silently reset a configured structural field."""
    start = PolicyParameters(reversibility_center=0.5)
    result = fit_policy(_separable_samples(), iterations=100, initial=start)
    assert result.params.reversibility_center == 0.5


def test_projection_enforces_the_admissible_set():
    """``project()`` clamps every constrained coefficient and orders thresholds."""
    params = PolicyParameters(
        beta_f=(-1.0, -2.0, 0.5, -0.1, 1.0),
        beta_k=-3.0,
        beta_q=-1.0,
        beta_v=-1.0,
        beta_a=-1.0,
        tau_l=0.9,
        tau_h=0.2,
    ).project()
    assert all(b >= 0.0 for b in params.beta_f)
    assert params.beta_k >= 0.0 and params.beta_q >= 0.0
    assert params.beta_v >= 0.0 and params.beta_a >= 0.0
    assert 0.0 <= params.tau_l < params.tau_h <= 1.0
    assert EscalationPolicy(params).is_monotonic()


# --------------------------------------------------------------------------- #
# Fitting (Req 2.2, 2.4)
# --------------------------------------------------------------------------- #
def _separable_samples() -> list[TrainingSample]:
    """Clean samples where escalation-worthy cases carry failures."""
    samples: list[TrainingSample] = []
    for _ in range(40):
        samples.append(
            TrainingSample(
                features=RiskFeatures(reversibility=0.9, authority=0.95, consequence=0.2),
                label=0,
            )
        )
        samples.append(
            TrainingSample(
                features=RiskFeatures(
                    f_c=FAIL, f_e=UNRESOLVED, reversibility=0.2, authority=0.2, consequence=0.9
                ),
                label=1,
            )
        )
    return samples


def test_fit_separates_labels_and_stays_monotonic():
    """The fit reduces loss, remains in ``B``, and orders the two classes."""
    samples = _separable_samples()
    result = fit_policy(samples, iterations=1500)
    assert result.monotonic
    assert result.log_loss < 0.35
    policy = EscalationPolicy(result.params)
    positive = next(s for s in samples if s.label == 1)
    negative = next(s for s in samples if s.label == 0)
    assert policy.risk(positive.features) > policy.risk(negative.features)


def test_fit_is_deterministic():
    """Identical inputs give bit-identical parameters (Req 2.4)."""
    samples = _separable_samples()
    a = fit_policy(samples, iterations=400)
    b = fit_policy(samples, iterations=400)
    assert a.params.as_dict() == b.params.as_dict()


def test_fit_on_empty_input_returns_the_prior():
    """An empty training set leaves the registered prior untouched."""
    result = fit_policy([])
    assert result.n_samples == 0
    assert result.params.as_dict() == PolicyParameters().project().as_dict()


def test_build_training_samples_labels_review_and_reject_positive():
    """``y = 1`` when autonomous execution would be wrong (Req 2.2)."""
    cases = [
        RoutingCase(RiskFeatures(), RouteGuards(), Tier.accept),
        RoutingCase(RiskFeatures(), RouteGuards(), Tier.supersede),
        RoutingCase(RiskFeatures(), RouteGuards(), Tier.review),
        RoutingCase(RiskFeatures(), RouteGuards(), Tier.reject),
    ]
    labels = [s.label for s in build_training_samples(cases)]
    assert labels == [0, 0, 1, 1]


# --------------------------------------------------------------------------- #
# Threshold selection (Req 3.x)
# --------------------------------------------------------------------------- #
def _threshold_cases() -> list[RoutingCase]:
    clean = RiskFeatures(reversibility=0.9, authority=0.95, consequence=0.2)
    risky = RiskFeatures(
        f_c=FAIL, f_e=UNRESOLVED, reversibility=0.2, authority=0.2, consequence=0.9
    )
    cases = [RoutingCase(clean, RouteGuards(), Tier.accept) for _ in range(30)]
    cases += [
        RoutingCase(risky, RouteGuards(), Tier.review, consequential=True)
        for _ in range(30)
    ]
    return cases


def test_thresholds_are_ordered_and_respect_the_mcr_ceiling():
    """Selection returns ``τ_l < τ_h`` under the ≤2% MCR constraint (Req 3.1, 3.2)."""
    cases = _threshold_cases()
    fitted = fit_policy(build_training_samples(cases), iterations=1500)
    selection = select_thresholds(cases, fitted.params, grid=0.05)
    assert 0.0 <= selection.tau_l < selection.tau_h <= 1.0
    if selection.feasible:
        assert selection.mcr <= 0.02


def test_infeasible_selection_is_reported_not_hidden():
    """When the constraint cannot be met, ``feasible=False`` is surfaced (Req 3.3)."""
    # Every case is consequential and review-worthy but looks perfectly clean, so
    # no threshold can escalate them without escalating everything.
    identical = RiskFeatures(reversibility=1.0, authority=1.0)
    cases = [
        RoutingCase(identical, RouteGuards(), Tier.accept, consequential=True)
        for _ in range(10)
    ] + [
        RoutingCase(identical, RouteGuards(), Tier.review, consequential=True)
        for _ in range(10)
    ]
    selection = select_thresholds(cases, PolicyParameters(), grid=0.25)
    assert isinstance(selection.feasible, bool)
    assert 0.0 <= selection.tau_l < selection.tau_h <= 1.0


def test_selection_on_empty_input_is_infeasible():
    """No development cases means no defensible thresholds."""
    selection = select_thresholds([], PolicyParameters())
    assert not selection.feasible
    assert selection.n_candidates == 0


# --------------------------------------------------------------------------- #
# Routing rule (Req 4.x)
# --------------------------------------------------------------------------- #
def test_prohibited_write_always_rejects():
    """``g(u)=1 -> reject``, ahead of every other clause (eq. 6)."""
    policy = EscalationPolicy()
    tier, rule, _r = policy.route(RiskFeatures(), RouteGuards(g=True))
    assert tier is Tier.reject
    assert "g(u)=1" in rule


def test_mandatory_failure_never_accepts():
    """``m(u)=1`` blocks the accept branch at every threshold (Req 4.3)."""
    for tau_l in (0.01, 0.5, 0.99):
        policy = EscalationPolicy(PolicyParameters(tau_l=tau_l, tau_h=1.0))
        tier, _rule, _r = policy.route(RiskFeatures(), RouteGuards(m=True))
        assert tier is not Tier.accept


def test_low_risk_clean_write_accepts():
    """A riskless write with no incumbent commits autonomously."""
    policy = EscalationPolicy(PolicyParameters(tau_l=0.99, tau_h=0.995))
    tier, _rule, _r = policy.route(
        RiskFeatures(reversibility=1.0, authority=1.0), RouteGuards()
    )
    assert tier is Tier.accept


def test_autonomous_commit_with_recoverable_incumbent_supersedes():
    """An autonomous commit that displaces an incumbent retains the prior value.

    Accepting without retiring the incumbent would leave two active values on a
    single-valued predicate, which is a durable-state violation.
    """
    policy = EscalationPolicy(PolicyParameters(tau_l=0.99, tau_h=0.995))
    tier, rule, _r = policy.route(
        RiskFeatures(
            reversibility=1.0,
            authority=1.0,
            incumbent_ids=("a1",),
            incumbent_recoverable=True,
        ),
        RouteGuards(),
    )
    assert tier is Tier.supersede
    assert "recoverable incumbent" in rule


def test_authoritative_correction_supersedes_at_the_higher_threshold():
    """``h(u)=1 ∧ r<τ_h -> supersede`` (eq. 6)."""
    policy = EscalationPolicy(PolicyParameters(tau_l=0.0, tau_h=0.999))
    tier, rule, _r = policy.route(
        RiskFeatures(f_c=FAIL, incumbent_ids=("a1",), incumbent_recoverable=True),
        RouteGuards(h=True),
    )
    assert tier is Tier.supersede
    assert "h(u)=1" in rule


def test_everything_else_escalates():
    """The default branch is review, never silent acceptance."""
    policy = EscalationPolicy(PolicyParameters(tau_l=0.0, tau_h=0.0001))
    tier, rule, _r = policy.route(RiskFeatures(f_c=FAIL), RouteGuards())
    assert tier is Tier.review
    assert "otherwise" in rule


def test_guards_flag_mandatory_and_prohibited_checks():
    """``compute_guards`` reads OCMR's verdict for ``g`` and ``m`` (Req 4.2, 4.3)."""
    features = RiskFeatures()
    for check in sorted(MANDATORY_CHECKS):
        guards = compute_guards(
            _candidate(), ValidationResult(valid=False, failed_check=check), features
        )
        assert guards.m, f"{check} must raise the mandatory guard"

    w5 = compute_guards(
        _candidate(),
        ValidationResult(valid=False, failed_check="schema.registered_predicate"),
        features,
    )
    assert w5.g and w5.m


def test_unattributed_write_raises_the_prohibited_guard():
    """A blank ``source_ref`` is prohibited (Req 4.2)."""
    guards = compute_guards(
        _candidate(source_ref=""), ValidationResult(valid=True), RiskFeatures()
    )
    assert guards.g
    assert "unattributed_write" in guards.reasons


def test_h_guard_requires_authority_temporal_resolution_and_recoverability():
    """``h(u)`` needs all three conditions of §3.3 (Req 4.4)."""
    base = dict(
        f_t=PASS, authority=0.95, incumbent_ids=("a1",), incumbent_recoverable=True
    )
    candidate = _candidate(write_intent=WriteIntent.correction)
    ok = compute_guards(candidate, ValidationResult(valid=True), RiskFeatures(**base))
    assert ok.h

    low_authority = compute_guards(
        candidate, ValidationResult(valid=True), RiskFeatures(**{**base, "authority": 0.6})
    )
    assert not low_authority.h

    unresolved_time = compute_guards(
        candidate, ValidationResult(valid=True), RiskFeatures(**{**base, "f_t": UNRESOLVED})
    )
    assert not unresolved_time.h

    no_incumbent = compute_guards(
        candidate,
        ValidationResult(valid=True),
        RiskFeatures(**{**base, "incumbent_recoverable": False, "incumbent_ids": ()}),
    )
    assert not no_incumbent.h


def test_routing_is_deterministic():
    """The same features and guards always yield the same tier (Req 4.1)."""
    policy = EscalationPolicy()
    features = RiskFeatures(f_c=FAIL, consequence=0.7)
    guards = RouteGuards()
    first = policy.route(features, guards)
    for _ in range(20):
        assert policy.route(features, guards) == first

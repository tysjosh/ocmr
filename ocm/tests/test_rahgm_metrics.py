"""RAHGM measures: eq. (10), eq. (11), risk–coverage AUC, and statistics.

Covers Req 11.x (measures) and 12.x (statistical models and Holm correction),
checked against hand-computed fixtures rather than against the implementation.
"""

from __future__ import annotations

import math

import pytest

from ocm.evaluation.rahgm.metrics import (
    ReplayMetrics,
    WriteRecord,
    brier_score,
    calibration_bins,
    compute_metrics,
    evaluate_success_criteria,
    expected_calibration_error,
    risk_coverage_auc,
)
from ocm.evaluation.rahgm.review_cost import ReviewCostModel
from ocm.evaluation.rahgm.stats import (
    CumulativeLogit,
    RandomInterceptGaussian,
    RandomInterceptLogit,
    bootstrap_ci,
    cluster_bootstrap_ci,
    holm,
    two_proportion_test,
)
from ocm.governance.features import FAIL, RiskFeatures
from ocm.governance.policy import Tier
from ocm.governance.review_queue import ExplanationDepth


def _record(
    gold: Tier,
    routed: Tier,
    final: Tier | None = None,
    *,
    escalated: bool = False,
    consequential: bool = False,
    risk: float = 0.0,
    minutes: float = 0.0,
    write_id: str = "w",
) -> WriteRecord:
    return WriteRecord(
        write_id=write_id,
        scenario_id="s",
        partition="test",
        write_class="c",
        template="t",
        gold=gold,
        routed=routed,
        final=final or routed,
        escalated=escalated,
        consequential=consequential,
        risk=risk,
        review_minutes=minutes,
    )


# --------------------------------------------------------------------------- #
# Record-level derivations
# --------------------------------------------------------------------------- #
def test_error_compares_the_final_transition_to_gold():
    """``eᵢ`` is about the durable outcome, not the routing choice."""
    assert not _record(Tier.accept, Tier.review, Tier.accept).error
    assert _record(Tier.accept, Tier.review, Tier.reject).error


def test_false_quarantine_is_a_valid_update_that_never_landed():
    """The OCMR failure this work targets (Req 11.2)."""
    assert _record(Tier.supersede, Tier.review, Tier.review).false_quarantine
    assert _record(Tier.accept, Tier.review, Tier.reject).false_quarantine
    assert not _record(Tier.supersede, Tier.review, Tier.supersede).false_quarantine
    assert not _record(Tier.review, Tier.review, Tier.review).false_quarantine


def test_false_escalation_is_review_of_an_admissible_write():
    """Escalating a routine write wastes review capacity."""
    assert _record(Tier.accept, Tier.review, escalated=True).false_escalation
    assert not _record(Tier.review, Tier.review, escalated=True).false_escalation


# --------------------------------------------------------------------------- #
# eq. (10)
# --------------------------------------------------------------------------- #
def test_mcr_counts_only_unescalated_consequential_errors():
    """``MCR = Σ cᵢeᵢ(1−zᵢ) / Σ cᵢ`` (eq. 10)."""
    records = [
        # consequential, wrong, not escalated -> a miss
        _record(Tier.review, Tier.accept, Tier.accept, consequential=True),
        # consequential, wrong, but escalated -> not a miss
        _record(Tier.review, Tier.review, Tier.reject, escalated=True, consequential=True),
        # consequential and correct -> not a miss
        _record(Tier.accept, Tier.accept, consequential=True),
        # non-consequential error -> outside the denominator
        _record(Tier.accept, Tier.reject),
    ]
    metrics = compute_metrics(records)
    assert metrics.n_consequential == 3
    assert metrics.mcr == pytest.approx(1.0 / 3.0)


def test_r100_is_minutes_per_hundred_writes():
    """``R100 = 100·Σtᵢ / N`` (eq. 10)."""
    records = [_record(Tier.accept, Tier.accept, minutes=0.5) for _ in range(4)]
    metrics = compute_metrics(records)
    assert metrics.r100 == pytest.approx(100.0 * 2.0 / 4.0)


def test_dvr_is_violations_over_writes():
    """``DVR = (1/N)Σvᵢ`` (eq. 10)."""
    records = [_record(Tier.accept, Tier.accept) for _ in range(20)]
    metrics = compute_metrics(records, durable_violations=3)
    assert metrics.dvr == pytest.approx(3 / 20)


def test_queue_precision_and_recall():
    """Precision is review-worthy among escalated; recall the converse (Req 11.2)."""
    records = [
        _record(Tier.review, Tier.review, escalated=True),   # true positive
        _record(Tier.review, Tier.review, escalated=True),   # true positive
        _record(Tier.accept, Tier.review, escalated=True),   # false positive
        _record(Tier.review, Tier.accept),                   # false negative
    ]
    metrics = compute_metrics(records)
    assert metrics.queue_precision == pytest.approx(2 / 3)
    assert metrics.queue_recall == pytest.approx(2 / 3)


def test_correction_quality_counts_realized_supersessions():
    """Correction quality is gold supersessions that actually superseded."""
    records = [
        _record(Tier.supersede, Tier.supersede),
        _record(Tier.supersede, Tier.review, Tier.review),
        _record(Tier.accept, Tier.accept),
    ]
    metrics = compute_metrics(records)
    assert metrics.correction_quality == pytest.approx(0.5)


def test_empty_records_produce_an_all_zero_report():
    """A condition that queued nothing is still comparable, not an exception."""
    metrics = compute_metrics([])
    assert isinstance(metrics, ReplayMetrics)
    assert metrics.n_writes == 0
    assert metrics.mcr == 0.0 and metrics.dvr == 0.0


def test_per_class_breakdown_is_computed():
    """Metrics are reported per write class for the §4.2 breakdown."""
    records = [
        WriteRecord("w1", "s", "test", "routine", "t", Tier.accept, Tier.accept, Tier.accept, False, False),
        WriteRecord("w2", "s", "test", "conflict", "t", Tier.review, Tier.accept, Tier.accept, False, True),
    ]
    metrics = compute_metrics(records)
    assert set(metrics.per_class) == {"routine", "conflict"}
    assert metrics.per_class["routine"]["accuracy"] == 1.0
    assert metrics.per_class["conflict"]["accuracy"] == 0.0


# --------------------------------------------------------------------------- #
# Risk–coverage AUC
# --------------------------------------------------------------------------- #
def test_perfect_ranking_gives_zero_auc():
    """When all errors carry the highest risk, low-coverage error rate is zero."""
    records = [_record(Tier.accept, Tier.accept, risk=0.1, write_id=f"g{i}") for i in range(8)]
    records += [
        _record(Tier.review, Tier.accept, Tier.accept, risk=0.9, write_id=f"b{i}")
        for i in range(2)
    ]
    assert risk_coverage_auc(records) < 0.05


def test_inverted_ranking_gives_high_auc():
    """A signal that ranks errors as safest is heavily penalized."""
    records = [_record(Tier.accept, Tier.accept, risk=0.9, write_id=f"g{i}") for i in range(8)]
    records += [
        _record(Tier.review, Tier.accept, Tier.accept, risk=0.1, write_id=f"b{i}")
        for i in range(2)
    ]
    good = risk_coverage_auc(
        [_record(Tier.accept, Tier.accept, risk=0.1) for _ in range(8)]
        + [_record(Tier.review, Tier.accept, Tier.accept, risk=0.9) for _ in range(2)]
    )
    assert risk_coverage_auc(records) > good


def test_auc_of_an_error_free_set_is_zero():
    """Nothing to rank means nothing to penalize."""
    records = [_record(Tier.accept, Tier.accept, risk=0.5) for _ in range(5)]
    assert risk_coverage_auc(records) == pytest.approx(0.0)


def test_auc_of_empty_input_is_zero():
    """The degenerate case is defined, not an exception."""
    assert risk_coverage_auc([]) == 0.0


# --------------------------------------------------------------------------- #
# eq. (11) calibration
# --------------------------------------------------------------------------- #
def test_brier_score_matches_the_definition():
    """``BS = (1/N)Σ(pᵢ − yᵢ)²`` (eq. 11)."""
    assert brier_score([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert brier_score([0.5, 0.5], [1.0, 0.0]) == pytest.approx(0.25)
    assert brier_score([0.0], [1.0]) == pytest.approx(1.0)


def test_perfect_calibration_has_zero_ece():
    """Confidence equal to accuracy in every bin gives ``ECE = 0`` (eq. 11)."""
    confidences = [0.05] * 20 + [0.95] * 20
    correctness = [0.0] * 19 + [1.0] + [1.0] * 19 + [0.0]
    assert expected_calibration_error(confidences, correctness) < 0.05


def test_overconfidence_shows_up_as_ece():
    """Uniformly high confidence with chance accuracy is badly calibrated."""
    confidences = [0.95] * 20
    correctness = [1.0] * 10 + [0.0] * 10
    assert expected_calibration_error(confidences, correctness) > 0.4


def test_calibration_bins_partition_the_unit_interval():
    """Bins are exhaustive and count every observation exactly once."""
    confidences = [i / 20 for i in range(21)]
    correctness = [1.0] * 21
    bins = calibration_bins(confidences, correctness, bins=10)
    assert len(bins) == 10
    assert sum(b.count for b in bins) == 21


def test_calibration_of_empty_input_is_nan():
    """No data means no claim."""
    assert math.isnan(brier_score([], []))
    assert math.isnan(expected_calibration_error([], []))


# --------------------------------------------------------------------------- #
# Review-cost model (Req 11.4)
# --------------------------------------------------------------------------- #
def test_review_cost_grows_with_case_difficulty():
    """The cost model is monotone in the quantities the router uses."""
    model = ReviewCostModel()
    easy = RiskFeatures(reversibility=1.0, consequence=0.0)
    hard = RiskFeatures(
        f_e=FAIL, f_c=FAIL, f_t=FAIL, consequence=0.9, reversibility=0.1
    )
    assert model.minutes(hard) > model.minutes(easy)


def test_review_cost_grows_with_explanation_depth():
    """Deeper explanations cost reading time."""
    features = RiskFeatures()
    minutes = [
        ReviewCostModel(depth=depth).minutes(features) for depth in ExplanationDepth
    ]
    assert minutes == sorted(minutes)


def test_review_cost_model_discloses_that_it_is_a_model():
    """Every artifact using ``R100`` must carry the disclosure (Req 11.4, 14.1)."""
    payload = ReviewCostModel().as_dict()
    assert payload["modelled"] is True
    assert "not" in payload["note"].lower()


# --------------------------------------------------------------------------- #
# Success criteria (Req 12.5)
# --------------------------------------------------------------------------- #
def _metrics(*, mcr: float, r100: float, dvr: float) -> ReplayMetrics:
    return compute_metrics(
        [
            _record(
                Tier.review,
                Tier.accept,
                Tier.accept,
                consequential=True,
                minutes=r100 / 100.0,
            )
        ]
        * (1 if mcr > 0 else 0)
        or [_record(Tier.accept, Tier.accept, consequential=True, minutes=r100 / 100.0)],
        durable_violations=int(dvr),
    )


def test_success_criteria_are_evaluated_explicitly():
    """All three §3.7 criteria are reported with their inputs (Req 12.5)."""
    table = {
        "adaptive_rahgm": compute_metrics(
            [_record(Tier.accept, Tier.accept, minutes=0.5, consequential=True)]
        ),
        "universal_review": compute_metrics(
            [_record(Tier.accept, Tier.accept, minutes=2.0, consequential=True)]
        ),
        "autonomous_ocmr": compute_metrics(
            [_record(Tier.accept, Tier.accept, consequential=True)]
        ),
        "fixed_threshold": compute_metrics(
            [
                _record(
                    Tier.review, Tier.accept, Tier.accept, consequential=True, minutes=1.0
                )
            ]
        ),
    }
    criteria = evaluate_success_criteria(table)
    assert criteria.r100_below_universal
    assert criteria.dvr_within_tolerance
    assert criteria.mcr_below_fixed
    assert criteria.met


def test_mcr_tie_at_zero_is_reported_as_a_floor_effect():
    """A strict-inequality failure at the zero floor is explained, not just failed."""
    clean = compute_metrics(
        [_record(Tier.accept, Tier.accept, consequential=True, minutes=0.5)]
    )
    expensive = compute_metrics(
        [_record(Tier.accept, Tier.accept, consequential=True, minutes=2.0)]
    )
    criteria = evaluate_success_criteria(
        {
            "adaptive_rahgm": clean,
            "universal_review": expensive,
            "autonomous_ocmr": clean,
            "fixed_threshold": expensive,
        }
    )
    assert not criteria.met
    assert criteria.mcr_tied_at_floor
    assert "floor" in criteria.interpretation() or "zero" in criteria.interpretation()


# --------------------------------------------------------------------------- #
# Statistics (Req 12.x)
# --------------------------------------------------------------------------- #
def test_holm_controls_familywise_error():
    """Holm adjustment is monotone and bounded by 1 (Req 12.3)."""
    results = holm([("a", 0.001), ("b", 0.02), ("c", 0.4)])
    adjusted = [r.adjusted_p for r in results]
    assert all(0.0 <= p <= 1.0 for p in adjusted)
    assert results[0].adjusted_p == pytest.approx(0.003)
    assert results[0].rejected
    assert not results[2].rejected


def test_holm_preserves_input_order():
    """Results come back in the order the tests were supplied."""
    results = holm([("z", 0.5), ("a", 0.001)])
    assert [r.name for r in results] == ["z", "a"]


def test_holm_handles_nan_p_values():
    """A missing p-value is treated as uninformative rather than crashing."""
    results = holm([("a", float("nan")), ("b", 0.01)])
    assert not results[0].rejected


def test_random_intercept_logit_recovers_a_known_effect():
    """The primary model detects a strong fixed effect (Req 12.1)."""
    y: list[float] = []
    X: list[list[float]] = []
    participants: list[str] = []
    scenarios: list[str] = []
    for participant in range(12):
        for scenario in range(6):
            for treated in (0, 1):
                # Treated cases succeed far more often, but not deterministically,
                # so the contrast is strong without being perfectly separable.
                successes = 5 if treated else 1
                for trial in range(6):
                    X.append([1.0, float(treated)])
                    y.append(1.0 if trial < successes else 0.0)
                    participants.append(f"p{participant}")
                    scenarios.append(f"s{scenario}")

    fit = RandomInterceptLogit().fit(
        y, X, participants, scenarios, names=["intercept", "treated"]
    )
    treated = fit.by_name("treated")
    assert treated is not None
    assert treated.estimate > 0.0
    assert treated.odds_ratio is not None and treated.odds_ratio > 1.0
    assert fit.n_participants == 12 and fit.n_scenarios == 6


def test_random_intercept_gaussian_recovers_a_known_slope():
    """The decision-time model recovers a planted slope (Req 12.2)."""
    y: list[float] = []
    X: list[list[float]] = []
    participants: list[str] = []
    scenarios: list[str] = []
    for participant in range(8):
        offset = 0.3 * participant
        for scenario in range(5):
            for x in (0.0, 1.0, 2.0):
                X.append([1.0, x])
                y.append(1.0 + 2.0 * x + offset)
                participants.append(f"p{participant}")
                scenarios.append(f"s{scenario}")

    fit = RandomInterceptGaussian().fit(
        y, X, participants, scenarios, names=["intercept", "x"]
    )
    slope = fit.by_name("x")
    assert slope is not None
    assert slope.estimate == pytest.approx(2.0, abs=0.15)


def test_cumulative_logit_fits_an_ordinal_outcome():
    """The workload model returns slope estimates with cluster-robust SEs (Req 12.2)."""
    y: list[int] = []
    X: list[list[float]] = []
    clusters: list[str] = []
    for participant in range(10):
        for level, x in enumerate((0.0, 1.0, 2.0, 3.0)):
            for _ in range(4):
                X.append([x])
                y.append(level)
                clusters.append(f"p{participant}")

    fit = CumulativeLogit().fit(y, X, clusters, names=["x"])
    slope = fit.by_name("x")
    assert slope is not None
    assert math.isfinite(slope.estimate)
    assert any("cluster-robust" in note for note in fit.notes)


def test_bootstrap_ci_brackets_the_point_estimate():
    """The percentile interval contains the statistic."""
    values = [0.0] * 40 + [1.0] * 60
    point, low, high = bootstrap_ci(values, iterations=400, seed=3)
    assert point == pytest.approx(0.6)
    assert low <= point <= high


def test_cluster_bootstrap_widens_with_clustered_data():
    """Resampling clusters accounts for within-cluster correlation."""
    values = [0.0] * 50 + [1.0] * 50
    clusters = ["a"] * 50 + ["b"] * 50
    _p, low, high = cluster_bootstrap_ci(values, clusters, iterations=400, seed=5)
    _p2, low2, high2 = bootstrap_ci(values, iterations=400, seed=5)
    assert (high - low) >= (high2 - low2)


def test_bootstrap_on_empty_input_is_nan():
    """No data means no interval."""
    point, low, high = bootstrap_ci([])
    assert all(math.isnan(v) for v in (point, low, high))


def test_two_proportion_test_detects_a_real_difference():
    """The score test flags a large, well-powered difference."""
    difference, p = two_proportion_test(90, 100, 50, 100)
    assert difference == pytest.approx(0.4)
    assert p < 0.001


def test_two_proportion_test_on_empty_groups_is_nan():
    """An empty group yields no claim."""
    difference, p = two_proportion_test(0, 0, 1, 10)
    assert math.isnan(difference) and math.isnan(p)

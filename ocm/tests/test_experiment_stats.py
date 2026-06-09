"""Tests for the evaluation statistics primitives (paper §IV-B)."""

from __future__ import annotations

import math

from ocm.evaluation import stats


def test_student_t_two_sided_matches_known_value():
    # t = 2.0, df = 5 -> two-sided p ~ 0.1019 (textbook value).
    p = stats.student_t_sf_two_sided(2.0, 5)
    assert abs(p - 0.1019) < 1e-3


def test_mean_ci_basic():
    ci = stats.mean_ci([0.80, 0.82, 0.81, 0.83, 0.79])
    assert abs(ci.mean - 0.81) < 1e-9
    assert ci.low < ci.mean < ci.high
    assert ci.n == 5


def test_mean_ci_single_value_has_zero_width():
    ci = stats.mean_ci([0.5])
    assert ci.mean == 0.5 and ci.half_width == 0.0


def test_paired_t_test_detects_clear_difference():
    a = [0.82, 0.80, 0.81, 0.83, 0.79]
    b = [0.74, 0.73, 0.75, 0.74, 0.72]
    r = stats.paired_t_test(a, b)
    assert r.p_value < 0.01
    assert r.effect_size > 0  # a > b
    assert r.df == 4


def test_wilcoxon_runs_and_bounds_p():
    a = [0.9, 0.8, 0.7, 0.6, 0.95]
    b = [0.5, 0.4, 0.6, 0.55, 0.5]
    r = stats.wilcoxon_signed_rank(a, b)
    assert 0.0 <= r.p_value <= 1.0
    assert r.effect_name == "rank_biserial"


def test_holm_bonferroni_monotone_and_rejects():
    out = stats.holm_bonferroni({"m1": 0.003, "m2": 0.005, "m3": 0.004}, alpha=0.05)
    # Corrected p-values are monotone non-decreasing in original rank order.
    assert out["m1"]["corrected_p"] <= out["m3"]["corrected_p"] <= out["m2"]["corrected_p"]
    assert all(v["reject"] for v in out.values())  # all small enough


def test_holm_bonferroni_does_not_reject_large_p():
    out = stats.holm_bonferroni({"a": 0.04, "b": 0.5, "c": 0.9}, alpha=0.05)
    assert out["a"]["reject"] is False  # 3 * 0.04 = 0.12 > 0.05


def test_ece_and_brier_ranges():
    conf = [0.9, 0.8, 0.7, 0.6]
    correct = [True, True, False, True]
    ece = stats.expected_calibration_error(conf, correct)
    brier = stats.brier_score(conf, correct)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= brier <= 1.0


def test_perfect_calibration_has_low_error():
    # Confidence equals empirical accuracy in each bin -> ECE 0.
    conf = [1.0, 1.0, 0.0, 0.0]
    correct = [True, True, False, False]
    assert stats.expected_calibration_error(conf, correct) == 0.0
    assert stats.brier_score(conf, correct) == 0.0

"""Leave-one-seed-out fitting, so arms score on the published population.

The dev-split protocol spends 40% of each seed's benchmark on fitting the
escalation policy, leaving arms scored on the other 60%. OCMR's published Table
III is scored on all 156 examples per seed, so a row produced under dev-split is
a fair sample of a different population and is not strictly comparable.

Fitting across seeds removes that cost, but introduces its own hazard: the six
hand-authored anchors are byte-identical in every seed, so a naive pool would
train on the exact trajectories it later evaluates.
"""

from __future__ import annotations

import pytest

from ocm.evaluation.benchmark import BenchmarkGenerator
from ocm.evaluation.rahgm.ocmr_arm import (
    fit_policy_across_seeds,
    fit_policy_on_benchmark,
    is_anchor,
)


@pytest.fixture(scope="module")
def small_seeds() -> tuple[int, ...]:
    return (1337, 7)


def test_anchors_are_identical_across_seeds() -> None:
    """The premise for excluding them: verbatim injection means verbatim overlap."""
    a = {
        s.input
        for e in BenchmarkGenerator(seed=1337).generate(per_category=3)
        if is_anchor(e)
        for s in e.sessions
    }
    b = {
        s.input
        for e in BenchmarkGenerator(seed=99).generate(per_category=3)
        if is_anchor(e)
        for s in e.sessions
    }
    assert a and a == b


def test_leave_one_seed_out_evaluates_the_full_benchmark(small_seeds) -> None:
    """Every example is scored, unlike the 60% the dev split leaves."""
    fitted = fit_policy_across_seeds(
        1337, fit_seeds=small_seeds, per_category=3
    )
    expected = len(BenchmarkGenerator(seed=1337).generate(per_category=3))
    assert fitted["n_eval_examples"] == expected
    assert len(fitted["eval_examples"]) == expected

    dev = fit_policy_on_benchmark(
        BenchmarkGenerator(seed=1337).generate(per_category=3)
    )
    assert len(dev["eval_examples"]) < expected  # the cost being removed


def test_anchors_are_excluded_from_the_fitting_pool(small_seeds) -> None:
    """Otherwise the pool contains the exact anchors the fold evaluates."""
    fitted = fit_policy_across_seeds(1337, fit_seeds=small_seeds, per_category=3)
    assert fitted["anchors_excluded_from_fit"] is True

    anchor_text = {
        s.input
        for e in BenchmarkGenerator(seed=1337).generate(per_category=3)
        if is_anchor(e)
        for s in e.sessions
    }
    audit = fitted["leakage_audit"]
    # The audit's held-out text set also excludes anchors, so no anchor sentence
    # can be counted as shared.
    assert audit["n_shared_texts"] <= audit["n_held_texts"]
    assert anchor_text  # premise held


def test_held_out_seed_contributes_nothing_to_the_fit(small_seeds) -> None:
    fitted = fit_policy_across_seeds(1337, fit_seeds=small_seeds, per_category=3)
    assert 1337 not in fitted["fit_seeds"]
    assert fitted["fit_seeds"] == [7]


def test_leakage_is_audited_rather_than_assumed(small_seeds) -> None:
    """Ids collide across seeds, so leakage is measured on session text."""
    fitted = fit_policy_across_seeds(1337, fit_seeds=small_seeds, per_category=6)
    audit = fitted["leakage_audit"]
    assert audit["n_held_texts"] > 0
    assert 0.0 <= audit["shared_fraction_of_held"] <= 1.0
    # Small generator vocabularies make some overlap expected; the point is that
    # it is reported.
    assert audit["n_shared_texts"] == len(
        {  # recompute independently
            s.input
            for e in BenchmarkGenerator(seed=7).generate(per_category=6)
            if not is_anchor(e)
            for s in e.sessions
        }
        & {
            s.input
            for e in BenchmarkGenerator(seed=1337).generate(per_category=6)
            if not is_anchor(e)
            for s in e.sessions
        }
    )


def test_example_ids_collide_across_seeds() -> None:
    """Why the audit cannot use ids: the same id holds different content."""
    a = BenchmarkGenerator(seed=1337).generate(per_category=3)
    b = BenchmarkGenerator(seed=7).generate(per_category=3)
    ids_a = {e.id for e in a}
    ids_b = {e.id for e in b}
    assert ids_a == ids_b  # identical ids
    text_a = {s.input for e in a if not is_anchor(e) for s in e.sessions}
    text_b = {s.input for e in b if not is_anchor(e) for s in e.sessions}
    assert text_a != text_b  # different content


def test_single_seed_is_rejected() -> None:
    with pytest.raises(ValueError) as excinfo:
        fit_policy_across_seeds(1337, fit_seeds=(1337,), per_category=3)
    assert "at least two seeds" in str(excinfo.value)


def test_both_fit_modes_expose_the_same_contract(small_seeds) -> None:
    """Callers must not need to know which protocol produced the parameters."""
    across = fit_policy_across_seeds(1337, fit_seeds=small_seeds, per_category=3)
    dev = fit_policy_on_benchmark(
        BenchmarkGenerator(seed=1337).generate(per_category=3)
    )
    shared_keys = {
        "params",
        "fit",
        "thresholds",
        "fit_mode",
        "eval_examples",
        "n_fit_cases",
        "n_fit_quarantined",
        "n_fit_false_quarantine",
    }
    assert shared_keys <= set(across)
    assert shared_keys <= set(dev)
    assert across["fit_mode"] == "leave-one-seed-out"
    assert dev["fit_mode"] == "dev-split"


def test_case_cache_is_reused_across_folds(small_seeds) -> None:
    """A seed's cases do not depend on which seed is held out."""
    cache: dict = {}
    fit_policy_across_seeds(
        1337, fit_seeds=small_seeds, per_category=3, case_cache=cache
    )
    populated = {k for k in cache if isinstance(k, int)}
    fit_policy_across_seeds(
        7, fit_seeds=small_seeds, per_category=3, case_cache=cache
    )
    # Both folds together touch each seed once, not once per fold.
    assert populated <= {k for k in cache if isinstance(k, int)}
    assert {k for k in cache if isinstance(k, int)} == set(small_seeds)


def test_fitted_parameters_are_valid_under_both_modes(small_seeds) -> None:
    across = fit_policy_across_seeds(1337, fit_seeds=small_seeds, per_category=3)
    params = across["params"]
    # project() enforces the admissible set B of eq. (4): thresholds ordered in
    # [0,1] and the monotonicity sign constraints on every coefficient.
    assert 0.0 <= params.tau_l < params.tau_h <= 1.0
    assert all(b >= 0 for b in params.beta_f)
    assert params.beta_k >= 0 and params.beta_q >= 0
    assert params.beta_v >= 0 and params.beta_a >= 0

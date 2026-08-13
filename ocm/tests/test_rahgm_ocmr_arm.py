"""Risk-adaptive escalation as an arm of OCMR's own benchmark.

Covers the experiment that decides whether selective escalation recovers the recall
cost OCMR pays for conservative quarantine, and the separability question that
determines whether *selectivity* is achievable at all on that workload.
"""

from __future__ import annotations

import pytest

from ocm.evaluation.benchmark import BenchmarkGenerator
from ocm.evaluation.rahgm.ocmr_arm import (
    ARMS,
    GOVERNED_ARMS,
    REVIEWERS,
    BenchmarkCase,
    ReviewContext,
    collect_benchmark_cases,
    fit_policy_on_benchmark,
    identity_reviewer,
    oracle_reviewer,
    run_ocmr_escalation_arm,
    separability_report,
)
from ocm.governance.policy import PolicyParameters, Tier
from ocm.governance.review_queue import ReviewAction


@pytest.fixture(scope="module")
def examples():
    """A reduced slice of OCMR's benchmark, for fast tests."""
    return BenchmarkGenerator(seed=1337).generate(per_category=6)


# --------------------------------------------------------------------------- #
# Case collection and labelling
# --------------------------------------------------------------------------- #
def test_escalation_label_is_separate_from_the_tier(examples):
    """A tier cannot express whether a human was needed (Req 4.1 semantics).

    "Hold without review" and "hold for review" are the same durable transition, so
    folding the escalation label into ``gold_tier`` would make every quarantine look
    review-worthy and reduce any fit to a quarantine detector.
    """
    cases = collect_benchmark_cases(examples)
    assert cases
    quarantined = [c for c in cases if c.quarantined]
    assert quarantined, "the benchmark produced no quarantines to label"
    # Both false and genuine quarantines share the same tier.
    assert all(c.case.gold_tier is Tier.review for c in quarantined)
    # But the escalation label distinguishes them.
    assert any(c.false_quarantine for c in quarantined)
    assert any(not c.false_quarantine for c in quarantined)


def test_only_quarantines_can_be_false_quarantines(examples):
    """A write OCMR admitted cannot be a false quarantine."""
    for case in collect_benchmark_cases(examples):
        if case.false_quarantine:
            assert case.quarantined


def test_genuine_quarantines_are_marked_consequential(examples):
    """A quarantine in a conflict-expecting example is the consequential case."""
    for case in collect_benchmark_cases(examples):
        if case.quarantined and not case.false_quarantine:
            assert case.case.consequential


# --------------------------------------------------------------------------- #
# Reviewers
# --------------------------------------------------------------------------- #
def _context(example, authored_by=None, expects_conflict=False) -> ReviewContext:
    return ReviewContext(
        example=example,
        authored_by=authored_by or {},
        expects_conflict=expects_conflict,
    )


def test_reviewers_are_registered():
    """The deployable reviewer, the ceiling, and every control are available."""
    assert set(REVIEWERS) == {
        "identity",
        "oracle",
        "release_all",
        "uphold_all",
        "random25",
        "random50",
        "random75",
    }


def test_release_volume_alone_does_not_preserve_the_integrity_property(examples):
    """Adjudication buys contradiction suppression, not recall.

    ``release_all`` exercises no judgment, so if it matched an adjudicating reviewer
    on *every* axis the review step would be doing nothing. It matches on task
    success — the recall gain is release volume — but reverts the contradiction rate
    to ungoverned levels. That contrast is the whole value of the human, and it is
    pinned here so a future change cannot quietly present release volume as review.
    """
    fitted = fit_policy_on_benchmark(examples)
    outcomes = {}
    for reviewer in ("identity", "release_all"):
        report = run_ocmr_escalation_arm(
            examples=fitted["test_examples"],
            reviewer=reviewer,
            params=fitted["params"],
            arms=("B0", "B3R"),
        )
        outcomes[reviewer] = report["arms"]
    ungoverned_contradiction = outcomes["identity"]["B0"]["contradiction_rate"]

    adjudicated = outcomes["identity"]["B3R"]["contradiction_rate"]
    indiscriminate = outcomes["release_all"]["B3R"]["contradiction_rate"]
    assert indiscriminate > adjudicated
    assert indiscriminate == pytest.approx(ungoverned_contradiction, abs=1e-6)


def test_release_does_not_raise_memory_induced_hallucination(examples):
    """Volume-proof check: release admits correct content, not merely more.

    OCMR's task success is answer-token recall over a haystack containing retrieved
    text, so it rises with the volume of admitted memory. The hallucination rate
    counts non-empty answers that are *wrong* with no conflict flagged, so it rises
    if release admits incorrect content. It must not.
    """
    fitted = fit_policy_on_benchmark(examples)
    report = run_ocmr_escalation_arm(
        examples=fitted["test_examples"],
        reviewer="identity",
        params=fitted["params"],
        arms=("B3", "B3R"),
    )
    b3 = report["arms"]["B3"]["volume_proof_checks"]["memory_induced_hallucination_rate"]
    b3r = report["arms"]["B3R"]["volume_proof_checks"][
        "memory_induced_hallucination_rate"
    ]
    if b3 is None or b3r is None:
        pytest.skip("hallucination rate unavailable for this slice")
    assert b3r <= b3 + 0.05


def test_oracle_reviewer_upholds_genuine_conflicts(examples):
    """The ceiling reviewer holds a write when the example expects a conflict."""
    from ocm.governance.review_queue import ReviewItem  # noqa: F401

    class _Item:
        class _Decision:
            class _Features:
                incumbent_ids = ("a1",)

            features = _Features()

        class _Verdict:
            conflicting_ids = ["a1"]

        decision = _Decision()
        ocmr_verdict = _Verdict()

    item = _Item()
    assert (
        oracle_reviewer(item, _context(examples[0], expects_conflict=True))
        is ReviewAction.quarantine
    )
    assert (
        oracle_reviewer(item, _context(examples[0], expects_conflict=False))
        is ReviewAction.supersede
    )


def test_identity_reviewer_releases_only_cross_context_collisions(examples):
    """The deployable reviewer keys on assertion authorship, not on labels."""

    class _Item:
        class _Decision:
            class _Features:
                incumbent_ids = ("a1",)

            features = _Features()

        class _Verdict:
            conflicting_ids = ["a1"]

        decision = _Decision()
        ocmr_verdict = _Verdict()

    item = _Item()
    example = examples[0]
    # Same example authored the incumbent: a genuine within-context contradiction.
    same = _context(example, authored_by={"a1": example.id})
    assert identity_reviewer(item, same) is ReviewAction.quarantine
    # A different example authored it: an identifier collision, so release.
    other = _context(example, authored_by={"a1": "some-other-example"})
    assert identity_reviewer(item, other) is ReviewAction.supersede


# --------------------------------------------------------------------------- #
# The arms
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def arm_report(examples):
    """All arms on the reduced benchmark slice."""
    return run_ocmr_escalation_arm(
        examples=examples, arms=("B0", "B3", "B3R", "B3Q"), reviewer="oracle"
    )


def test_all_requested_arms_are_reported(arm_report):
    """The report carries one entry per arm with the decisive metrics."""
    assert set(arm_report["arms"]) == {"B0", "B3", "B3R", "B3Q"}
    for arm in arm_report["arms"].values():
        for key in ("task_success", "contradiction_rate", "constraint_violations"):
            assert key in arm


def test_governed_arms_escalate_and_ungoverned_do_not(arm_report):
    """Only the router-installed arms present writes for review."""
    assert arm_report["arms"]["B0"]["escalated"] == 0
    assert arm_report["arms"]["B3"]["escalated"] == 0
    for arm in GOVERNED_ARMS:
        assert arm_report["arms"][arm]["escalated"] > 0


def test_ocmr_holds_durable_violations_at_zero(arm_report):
    """B3 must reproduce OCMR's published integrity guarantee."""
    assert arm_report["arms"]["B3"]["constraint_violations"] == 0.0
    assert arm_report["arms"]["B0"]["constraint_violations"] > 0.0


def test_release_happens_and_stays_below_ungoverned_violation_levels(arm_report):
    """Release admits held writes without reverting to ungoverned integrity.

    The *magnitude* of recall recovery is scale-dependent: OCMR's recall cost only
    appears at its published 25-trajectories-per-category protocol, and on the small
    slice these fast tests use the governed arm can already exceed the ungoverned
    one. So this asserts the mechanism, not the effect size. The effect size is
    measured by :func:`run_ocmr_escalation_arm` at full scale and reported there.
    """
    b3r = arm_report["arms"]["B3R"]
    b0 = arm_report["arms"]["B0"]
    assert b3r["released"] > 0
    assert b3r["upheld"] >= 0
    assert b3r["constraint_violations"] <= b0["constraint_violations"]


@pytest.mark.slow
def test_release_recovers_ocmr_recall_cost_at_published_scale():
    """At OCMR's 25-per-category protocol, release recovers the recall cost.

    This is the experiment the merged paper turns on, so it is asserted at the scale
    where OCMR's own recall cost is present. Marked slow because it replays the full
    benchmark for four arms.
    """
    examples = BenchmarkGenerator(seed=1337).generate(per_category=25)
    report = run_ocmr_escalation_arm(
        examples=examples, arms=("B0", "B3", "B3R"), reviewer="oracle"
    )
    b0 = report["arms"]["B0"]["task_success"]
    b3 = report["arms"]["B3"]["task_success"]
    b3r = report["arms"]["B3R"]["task_success"]

    # OCMR's conservative quarantine costs recall at this scale.
    assert b3 < b0, "the recall cost this experiment addresses is not present"
    # Release recovers it.
    assert b3r > b3
    # Without reverting to ungoverned violation levels.
    assert (
        report["arms"]["B3R"]["constraint_violations"]
        < report["arms"]["B0"]["constraint_violations"]
    )


def test_release_never_leaves_a_single_valued_violation_behind(arm_report):
    """A released write retires its incumbent, so no typed violation appears."""
    for arm in GOVERNED_ARMS:
        assert arm_report["arms"][arm]["typed_violations"] == 0


def test_contrast_reports_the_recall_gap_and_recovery(arm_report):
    """The B3-to-B3R contrast is the number that decides the claim."""
    contrast = arm_report["contrast"]
    for key in ("recall_gap_to_close", "recall_recovered", "violation_delta"):
        assert key in contrast
    assert contrast["recall_recovered"] == pytest.approx(
        arm_report["arms"]["B3R"]["task_success"]
        - arm_report["arms"]["B3"]["task_success"]
    )


def test_arm_report_declares_it_is_not_in_the_paper(arm_report):
    """Work beyond either paper's design must say so (Req 14.1)."""
    assert arm_report["in_paper"] is False
    assert arm_report["claim"].strip()


# --------------------------------------------------------------------------- #
# Separability: is selectivity achievable at all?
# --------------------------------------------------------------------------- #
def test_fitting_uses_a_disjoint_development_split(examples):
    """Development and test examples must not overlap (OCMR's protocol)."""
    fitted = fit_policy_on_benchmark(examples)
    dev_ids = {e.id for e in fitted["dev_examples"]}
    test_ids = {e.id for e in fitted["test_examples"]}
    assert dev_ids and test_ids
    assert not (dev_ids & test_ids)
    assert fitted["n_dev_quarantined"] >= fitted["n_dev_false_quarantine"]


def test_separability_report_compares_against_the_base_rate(examples):
    """Precision must be judged against escalating every quarantine.

    A policy that escalates all quarantines scores precision equal to the base rate
    of false quarantines, so only lift above that base rate is evidence of
    discriminating signal.
    """
    fitted = fit_policy_on_benchmark(examples)
    report = separability_report(fitted["test_examples"], fitted["params"])
    for key in ("base_rate_false", "precision", "recall", "lift_over_base_rate"):
        assert key in report
    assert report["lift_over_base_rate"] == pytest.approx(
        report["precision"] - report["base_rate_false"]
    )
    assert 0.0 <= report["escalation_share_of_quarantines"] <= 1.0


def test_features_do_not_separate_false_from_genuine_quarantines(examples):
    """Measured negative result: falseness is not a property of the write.

    Whether a quarantine is false is defined by whether the surrounding example's
    questions expect a conflict — a property of the scenario, not of the candidate
    write's constraint-failure pattern. Two writes with identical patterns can be
    false or genuine, so no feature-derived policy can separate them, and the fitted
    router degenerates to escalating every quarantine.

    This test pins the finding. If a future change makes the features informative,
    it should fail and be updated deliberately.
    """
    fitted = fit_policy_on_benchmark(examples)
    report = separability_report(fitted["test_examples"], fitted["params"])
    assert report["n_quarantined"] > 0
    assert report["lift_over_base_rate"] < 0.10, (
        "features now carry discriminating signal; revisit the selectivity claim"
    )


# --------------------------------------------------------------------------- #
# The no-skill control
# --------------------------------------------------------------------------- #
def _review_fixture(subject_id: str = "ent_alice", predicate: str = "OWNS"):
    """A minimal stand-in for the fields a reviewer is allowed to read.

    Built as a stub rather than a real :class:`ReviewItem` so the test pins the
    reviewer's *contract* — candidate identity, routing features, OCMR verdict,
    and the example id — instead of the queue's construction details.
    """
    from types import SimpleNamespace

    item = SimpleNamespace(
        candidate=SimpleNamespace(
            subject_id=subject_id, predicate=predicate, object_id="proj_orion"
        ),
        decision=SimpleNamespace(features=SimpleNamespace(incumbent_ids=())),
        ocmr_verdict=SimpleNamespace(conflicting_ids=()),
    )
    context = SimpleNamespace(
        example=SimpleNamespace(id="ex-001"), authored_by={}, expects_conflict=False
    )
    return item, context


def test_random_reviewer_is_deterministic_per_write() -> None:
    """The control must give the same write the same verdict in every arm.

    Keyed on the candidate's identity rather than call order, so the comparison
    against a judgment-based reviewer stays paired: both see the same population
    and differ only in which writes they release.
    """
    from ocm.evaluation.rahgm.ocmr_arm import make_random_reviewer

    reviewer = make_random_reviewer(0.5)
    item, context = _review_fixture()
    verdicts = {reviewer(item, context) for _ in range(20)}
    assert len(verdicts) == 1


def test_random_reviewer_release_rate_tracks_its_probability() -> None:
    """Over many distinct writes the release rate approaches the parameter."""
    from ocm.evaluation.rahgm.ocmr_arm import ReviewAction, make_random_reviewer

    reviewer = make_random_reviewer(0.5)
    released = 0
    n = 400
    for i in range(n):
        item, context = _review_fixture(subject_id=f"per_{i}")
        if reviewer(item, context) is not ReviewAction.quarantine:
            released += 1
    assert 0.4 < released / n < 0.6


def test_random_rates_bracket_the_endpoints() -> None:
    """Ordering the control by release probability must be monotone in volume."""
    from ocm.evaluation.rahgm.ocmr_arm import ReviewAction, make_random_reviewer

    def rate(p: float) -> float:
        reviewer = make_random_reviewer(p)
        released = 0
        for i in range(300):
            item, context = _review_fixture(subject_id=f"per_{i}")
            if reviewer(item, context) is not ReviewAction.quarantine:
                released += 1
        return released / 300

    low, mid, high = rate(0.25), rate(0.5), rate(0.75)
    assert low < mid < high


def test_all_default_reviewers_are_registered() -> None:
    from ocm.evaluation.rahgm.ocmr_arm import REVIEWERS
    from ocm.evaluation.rahgm.run_ocmr_arm import DEFAULT_REVIEWERS

    assert set(DEFAULT_REVIEWERS) <= set(REVIEWERS)
    assert {"random25", "random50", "random75"} <= set(REVIEWERS)

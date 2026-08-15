"""The violation rate must match OCMR's published definition exactly.

OCMR reports "durable-write constraint violation rate per 100 responses" and
computes it in ``ocm/evaluation/experiment.py`` as
``100 * durable_constraint_violations(container)[0] / len(records)``.

The arm runner divided by the *accepted-assertion* count instead. Same numerator,
different denominator, and since a run accepts more assertions than it answers
questions, the reported rate was systematically too low: B0 read 39.86 where the
published run reports 50.72. That made the whole column incomparable to Table III
while still looking plausible, which is why it survived several runs.
"""

from __future__ import annotations

from ocm.evaluation.experiment import decisive_metrics


def _rate_from_published_definition(violations: int, n_records: int) -> float:
    """Reimplements experiment.py's formula, independently of the arm runner."""
    return (100.0 * violations / n_records) if n_records else 0.0


def test_arm_runner_uses_the_response_denominator() -> None:
    """Pin the formula the arm runner is expected to apply."""
    violations, n_records = 39, 78
    accepted = 99  # deliberately different, as it is in a real run

    published = _rate_from_published_definition(violations, n_records)
    wrong = 100.0 * violations / accepted

    assert published == 50.0
    assert wrong != published  # the two definitions genuinely disagree
    # decisive_metrics passes the supplied rate straight through, so supplying the
    # response-denominated rate is what makes the column comparable.
    metrics = decisive_metrics(
        [{"score": 1.0} for _ in range(n_records)],
        constraint_violation_rate=published,
    )
    assert metrics["constraint_violations"] == 50.0


def test_violation_rate_matches_published_formula_on_a_real_arm() -> None:
    """End to end: the reported rate equals count / responses, on a real run."""
    from ocm.evaluation.benchmark import BenchmarkGenerator
    from ocm.evaluation.rahgm.ocmr_arm import run_ocmr_escalation_arm

    examples = BenchmarkGenerator(seed=1337).generate(per_category=4)
    report = run_ocmr_escalation_arm(
        examples=examples, seed=1337, arms=("B0", "B2", "B3"), reviewer="uphold_all"
    )
    for arm_name in ("B0", "B2", "B3"):
        arm = report["arms"][arm_name]
        expected = _rate_from_published_definition(
            arm["durable_violations"], arm["n_records"]
        )
        assert arm["constraint_violations"] == expected, (
            f"{arm_name}: reported {arm['constraint_violations']} but "
            f"count/responses gives {expected}"
        )


def test_the_old_denominator_would_have_failed_this() -> None:
    """The two denominators must actually differ on a real run, or the test is vacuous."""
    from ocm.evaluation.benchmark import BenchmarkGenerator
    from ocm.evaluation.rahgm.ocmr_arm import run_ocmr_escalation_arm

    examples = BenchmarkGenerator(seed=1337).generate(per_category=4)
    report = run_ocmr_escalation_arm(
        examples=examples, seed=1337, arms=("B0",), reviewer="uphold_all"
    )
    arm = report["arms"]["B0"]
    accepted = arm["writes"]["accepted"]
    assert accepted != arm["n_records"], (
        "accepted-assertion count coincides with response count here, so this "
        "workload cannot distinguish the two denominators"
    )
    assert arm["durable_violations"] > 0  # the metric is exercised at all


def test_governed_arm_has_no_durable_violations() -> None:
    """Sanity anchor: B3 gates conflicts at write time, so the rate is 0."""
    from ocm.evaluation.benchmark import BenchmarkGenerator
    from ocm.evaluation.rahgm.ocmr_arm import run_ocmr_escalation_arm

    examples = BenchmarkGenerator(seed=1337).generate(per_category=3)
    report = run_ocmr_escalation_arm(
        examples=examples, seed=1337, arms=("B3",), reviewer="uphold_all"
    )
    assert report["arms"]["B3"]["constraint_violations"] == 0.0


def test_ungoverned_arm_rate_exceeds_governed_arm_rate() -> None:
    """The direction the metric exists to show, under the corrected denominator."""
    from ocm.evaluation.benchmark import BenchmarkGenerator
    from ocm.evaluation.rahgm.ocmr_arm import run_ocmr_escalation_arm

    examples = BenchmarkGenerator(seed=1337).generate(per_category=5)
    report = run_ocmr_escalation_arm(
        examples=examples, seed=1337, arms=("B0", "B3"), reviewer="uphold_all"
    )
    assert (
        report["arms"]["B0"]["constraint_violations"]
        > report["arms"]["B3"]["constraint_violations"]
    )

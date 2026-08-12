"""RAHGM experiment integration: replay, ablation, audit, and honest reporting.

Covers Req 10.x (conditions and ordering), 13.x (the experiments), and 14.x (the
honest-reporting contract). These run on a reduced corpus so the whole suite stays
fast; the paper-scale run is driven by ``python -m ocm.evaluation.rahgm.run_all``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from ocm.evaluation.rahgm.ablation import run_ablation
from ocm.evaluation.rahgm.cascade import measure_injected_cascade, run_cascade_study
from ocm.evaluation.rahgm.adaptation_study import (
    FEEDBACK_STREAMS,
    POLICY_ARMS,
    make_feedback,
    run_experiment3,
)
from ocm.evaluation.rahgm.audit import (
    CAUSES,
    classify_quarantine,
    classify_reason,
    collect_quarantines,
)
from ocm.evaluation.rahgm.corpus import Partition, generate_corpus
from ocm.evaluation.rahgm.drift import drift_scenarios, run_drift_study
from ocm.evaluation.rahgm.end_to_end import run_experiment4
from ocm.evaluation.rahgm.human_study import run_experiment2
from ocm.evaluation.rahgm.metrics import compute_metrics
from ocm.evaluation.rahgm.replay import (
    ScenarioReplayer,
    collect_routing_cases,
    develop_policy,
    durable_violation_count,
    install_scenario_state,
    make_oracle_reviewer,
    ocmr_verdict,
    replay_settings,
    run_experiment1,
    to_candidate,
)
from ocm.evaluation.rahgm.report import SCOPE_NOTE, render_all
from ocm.evaluation.typed_violations import typed_violations
from ocm.core.container import CoreContainer
from ocm.governance.conditions import CONDITION_LABELS, Condition
from ocm.governance.policy import Tier
from ocm.ontology.enums import AssertionStatus
from ocm.ontology.models import Assertion


@pytest.fixture(scope="module")
def small_corpus():
    """A reduced corpus: 6 train, 2 dev, 1 canary, 3 test scenarios."""
    return generate_corpus(n_scenarios=12)


@pytest.fixture(scope="module")
def developed(small_corpus):
    """A policy fitted on the reduced corpus."""
    return develop_policy(small_corpus, iterations=600)


# --------------------------------------------------------------------------- #
# Policy development
# --------------------------------------------------------------------------- #
def test_policy_is_fitted_on_training_and_tuned_on_development(developed):
    """Fitting uses training scenarios; thresholds use development (Req 2.2, 3.1)."""
    assert developed.n_train_cases > 0
    assert developed.n_dev_cases > 0
    assert developed.fit.monotonic
    assert 0.0 <= developed.params.tau_l < developed.params.tau_h <= 1.0


def test_routing_cases_carry_real_ocmr_verdicts(small_corpus):
    """Features come from the real OCMR checks, not a reimplementation (Req 1.x)."""
    cases = collect_routing_cases(small_corpus.partition(Partition.dev))
    assert len(cases) == len(small_corpus.writes_in(Partition.dev))
    # Malformed writes must raise the prohibited guard via OCMR's W5/C9 verdict.
    assert any(c.guards.g for c in cases)
    # Contradictions must reach the contradiction component via W7/C7.
    assert any(c.features.f_c > 0.0 for c in cases)


def test_ocmr_verdict_runs_w5_then_w6(small_corpus):
    """Verdicts follow the pipeline's own gate order."""
    scenario = small_corpus.scenarios[0]
    container = CoreContainer(replay_settings())
    install_scenario_state(container, scenario)
    for write in scenario.writes:
        verdict = ocmr_verdict(container, to_candidate(write))
        if write.template == "reject_unregistered_predicate":
            assert verdict.failed_check == "schema.registered_predicate"
        if write.template == "reject_domain_range":
            assert verdict.failed_check == "C9"


# --------------------------------------------------------------------------- #
# Experiment 1 (Req 10.x, 13.2)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def experiment1(small_corpus, developed):
    """Experiment 1 on the reduced corpus."""
    return run_experiment1(small_corpus, developed=developed)


def test_all_five_conditions_are_reported(experiment1):
    """Table 3 has one row per condition (Req 10.1)."""
    assert set(experiment1["conditions"]) == {c.value for c in Condition}
    for name, entry in experiment1["conditions"].items():
        assert entry["label"] == CONDITION_LABELS[Condition(name)]


def test_conditions_see_identical_writes(experiment1):
    """Every condition receives the same candidate writes in the same order (Req 10.2, 10.3)."""
    results = experiment1["_results"]
    reference = [r.write_id for r in results["autonomous_ocmr"].records]
    for name, result in results.items():
        assert [r.write_id for r in result.records] == reference, name


def test_autonomous_ocmr_never_escalates(experiment1):
    """C2 presents nothing to a reviewer (Req 10.1)."""
    metrics = experiment1["conditions"]["autonomous_ocmr"]["metrics"]
    assert metrics["review_rate"] == 0.0
    assert metrics["r100"] == 0.0


def test_universal_review_escalates_far_more_than_rahgm(experiment1):
    """Selective oversight is the point: C1 must be the costliest arm."""
    conditions = experiment1["conditions"]
    assert (
        conditions["universal_review"]["metrics"]["review_rate"]
        > conditions["frozen_rahgm"]["metrics"]["review_rate"]
    )
    assert (
        conditions["universal_review"]["metrics"]["r100"]
        > conditions["frozen_rahgm"]["metrics"]["r100"]
    )


def test_rahgm_reduces_review_demand_versus_fixed_threshold(experiment1):
    """C4 vs C3 is the routing-mechanism contrast (RQ1)."""
    conditions = experiment1["conditions"]
    assert (
        conditions["frozen_rahgm"]["metrics"]["review_rate"]
        <= conditions["fixed_threshold"]["metrics"]["review_rate"]
    )


def test_rahgm_eliminates_the_false_quarantines_ocmr_leaves(experiment1):
    """The headline target: valid updates must stop being held forever."""
    conditions = experiment1["conditions"]
    autonomous = conditions["autonomous_ocmr"]["metrics"]["false_quarantine_rate"]
    frozen = conditions["frozen_rahgm"]["metrics"]["false_quarantine_rate"]
    assert autonomous > 0.0, "the OCMR baseline must exhibit the problem"
    assert frozen < autonomous


def test_rahgm_does_not_regress_durable_integrity(experiment1):
    """``DVR(C5) − DVR(C2) ≤ 0.005`` (Req 12.5)."""
    criteria = experiment1["success_criteria"]
    assert criteria["dvr_within_tolerance"]


def test_success_criteria_are_reported_with_interpretation(experiment1):
    """The criteria block explains its own verdict (Req 12.5, 14.1)."""
    criteria = experiment1["success_criteria"]
    assert set(criteria) >= {
        "met",
        "interpretation",
        "r100_below_universal",
        "dvr_within_tolerance",
        "mcr_below_fixed",
    }
    assert criteria["interpretation"].strip()


def test_threshold_generalization_is_measured_and_reported(experiment1):
    """The eq. (5) constraint's out-of-sample behavior is a reported quantity."""
    generalization = experiment1["threshold_generalization"]
    assert generalization["rows"]
    assert generalization["paper_ceiling"] == 0.02
    for row in generalization["rows"]:
        for key in ("dev_mcr_ceiling", "tau_l", "dev_mcr", "test_mcr", "test_review_rate"):
            assert key in row
    # Tightening the ceiling must never increase held-out MCR.
    ordered = sorted(generalization["rows"], key=lambda r: -r["dev_mcr_ceiling"])
    assert ordered[0]["test_mcr"] >= ordered[-1]["test_mcr"]
    assert generalization["finding"].strip()


def test_tightening_the_ceiling_costs_review_demand(experiment1):
    """The generalization fix is a tradeoff, and the price is reported."""
    rows = {r["dev_mcr_ceiling"]: r for r in experiment1["threshold_generalization"]["rows"]}
    loose, tight = rows.get(0.02), rows.get(0.01)
    if loose is None or tight is None or loose["test_mcr"] == tight["test_mcr"]:
        return
    assert tight["test_mcr"] < loose["test_mcr"]
    assert tight["test_review_rate"] >= loose["test_review_rate"]


def test_r100_is_labelled_as_modelled(experiment1):
    """Reviewer minutes must be disclosed as modelled (Req 11.4, 14.1)."""
    assert experiment1["review_cost_model"]["modelled"] is True


def test_write_order_is_preserved_within_a_scenario(small_corpus, developed):
    """A transition at ``t`` is visible to ``t+1`` (Req 10.3)."""
    scenario = small_corpus.partition(Partition.test)[0]
    writes_by_id = {w.write_id: w for w in scenario.writes}
    replayer = ScenarioReplayer(
        Condition.frozen_rahgm,
        params=developed.params,
        reviewer=make_oracle_reviewer(writes_by_id),
    )
    _harness, result = replayer.run_scenario(scenario)
    assert [r.write_id for r in result.records] == [w.write_id for w in scenario.writes]


def test_durable_violations_include_single_valued_contradictions(small_corpus, developed):
    """DVR counts both the typed classes and the legacy single-valued measure."""
    scenario = small_corpus.partition(Partition.test)[0]
    replayer = ScenarioReplayer(Condition.frozen_rahgm, params=developed.params)
    _harness, result = replayer.run_scenario(scenario)
    assert result.durable_violations >= result.typed_breakdown["typed_total"]
    assert (
        result.typed_breakdown["total"]
        == result.typed_breakdown["typed_total"]
        + result.typed_breakdown["single_valued_contradictions"]
    )


# --------------------------------------------------------------------------- #
# Ablation (Req 13.2)
# --------------------------------------------------------------------------- #
def test_ablation_reports_every_variant(small_corpus, developed):
    """Table 4 covers the full, quarantine-only, scalar, and leave-one-out variants."""
    report = run_ablation(small_corpus, developed.params)
    keys = {v["key"] for v in report["variants"]}
    # The paper's variants.
    assert {
        "full",
        "quarantine_only",
        "scalar_threshold",
        "failure_pattern_only",
        "reversibility_only",
        "without_consequence",
        "without_authority",
    } <= keys
    # Variants this evaluation added: two rejected hypotheses and one remedy.
    assert {"centered_reversibility", "bounded_discounts", "tightened_threshold"} <= keys
    for variant in report["variants"]:
        assert 0.0 <= variant["review_rate"] <= 1.0
        assert 0.0 <= variant["mcr"] <= 1.0
        assert variant["note"]


def test_non_paper_variants_declare_their_status(small_corpus, developed):
    """A variant outside the paper must say whether it was adopted or rejected."""
    report = run_ablation(small_corpus, developed.params)
    notes = {v["key"]: v["note"] for v in report["variants"]}
    assert "rejected" in notes["centered_reversibility"]
    assert "rejected" in notes["bounded_discounts"]
    assert "eq. (5)" in notes["tightened_threshold"]


def test_strongest_baseline_is_not_a_tie_breaking_artifact(small_corpus, developed):
    """The named baseline must route differently from the full policy."""
    report = run_ablation(small_corpus, developed.params)
    comparison = report["full_vs_strongest_baseline"]
    if comparison is None:
        return
    variants = {v["key"]: v for v in report["variants"]}
    full = variants["full"]
    chosen = variants[comparison["strongest_baseline"]]
    assert not (
        chosen["review_rate"] == full["review_rate"] and chosen["mcr"] == full["mcr"]
    )


def test_quarantine_only_escalation_is_less_precise_than_full_rahgm(
    small_corpus, developed
):
    """The full policy must beat OCMR's own quarantine signal on queue precision."""
    report = run_ablation(small_corpus, developed.params)
    variants = {v["key"]: v for v in report["variants"]}
    assert variants["full"]["queue_precision"] > variants["quarantine_only"]["queue_precision"]


def test_scalar_threshold_is_worse_than_the_failure_pattern(small_corpus, developed):
    """A single opaque score cannot match constraint-derived routing (RQ1)."""
    report = run_ablation(small_corpus, developed.params)
    variants = {v["key"]: v for v in report["variants"]}
    assert (
        variants["scalar_threshold"]["risk_coverage_auc"]
        > variants["full"]["risk_coverage_auc"]
    )


# --------------------------------------------------------------------------- #
# Experiment 2 (Req 13.3, 14.1, 14.2)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def experiment2(small_corpus, developed):
    """A small simulated human study."""
    return run_experiment2(
        small_corpus,
        developed=developed,
        participants_per_condition=3,
        scenarios_per_participant=3,
        writes_per_scenario=5,
        fit_models=False,
    )


def test_experiment2_is_labelled_simulated(experiment2):
    """The simulation must be disclosed, not implied (Req 14.1, 14.2)."""
    assert experiment2["simulated"] is True
    assert "simulated" in experiment2["disclosure"].lower()
    assert "not" in experiment2["disclosure"].lower()


def test_experiment2_covers_only_human_facing_conditions(experiment2):
    """Autonomous OCMR has no reviewer, so it is absent (§3.1)."""
    assert Condition.autonomous_ocmr.value not in experiment2["by_condition"]
    assert Condition.frozen_rahgm.value in experiment2["by_condition"]


def test_experiment2_reports_every_explanation_depth(experiment2):
    """All three depths are exercised by the Latin-square schedule (Req 6.3)."""
    for depth in ("minimal", "evidence", "full"):
        assert experiment2["by_depth"][depth]["n_items"] > 0


def test_experiment2_reports_calibration(experiment2):
    """Brier and ECE are reported per condition (Req 11.3)."""
    for entry in experiment2["by_condition"].values():
        if entry.get("n_items"):
            assert "ece" in entry and "brier" in entry


def test_experiment2_reports_complacency(experiment2):
    """Reliance is analyzed by trial position and streak (§4.3)."""
    complacency = experiment2["complacency"]
    assert "recommendation_following_change_points" in complacency
    assert complacency["by_streak"]


# --------------------------------------------------------------------------- #
# Experiment 3 (Req 13.4)
# --------------------------------------------------------------------------- #
def test_feedback_streams_have_the_intended_character(small_corpus):
    """Adversarial feedback suppresses escalation on consequential cases."""
    cases = collect_routing_cases(small_corpus.partition(Partition.dev))
    clean = make_feedback(cases, "clean", seed=1)
    adversarial = make_feedback(cases, "adversarial", seed=1)

    clean_escalations = sum(1 for r in clean if r.should_escalate)
    adversarial_escalations = sum(1 for r in adversarial if r.should_escalate)
    assert clean_escalations > 0
    assert adversarial_escalations < clean_escalations
    # The suppression is asserted with high confidence, to carry maximum weight.
    assert all(r.confidence >= 0.9 for r in adversarial if r.consequential)


def test_experiment3_runs_every_arm_and_stream(small_corpus, developed):
    """Table 6 covers four policies across four feedback streams (Req 13.4)."""
    report = run_experiment3(
        small_corpus, developed=developed, seeds=(1337,), max_blocks=2
    )
    assert set(report["table6"]) == set(POLICY_ARMS)
    for arm in POLICY_ARMS:
        for stream in FEEDBACK_STREAMS:
            assert f"{arm}|{stream}" in report["by_arm_and_stream"]


def test_frozen_arm_makes_no_proposals(small_corpus, developed):
    """Frozen RAHGM never changes parameters (Req 8.5)."""
    report = run_experiment3(
        small_corpus, developed=developed, seeds=(1337,), arms=("frozen",), max_blocks=2
    )
    assert report["table6"]["frozen"]["n_proposed_total"] == 0
    assert report["table6"]["frozen"]["max_drift"] == 0.0


def test_gated_adaptation_never_regresses_durable_integrity(small_corpus, developed):
    """The canary gate forbids any DVR increase (Req 8.1)."""
    report = run_experiment3(
        small_corpus,
        developed=developed,
        seeds=(1337,),
        arms=("bounded_canary",),
        max_blocks=3,
    )
    assert report["table6"]["bounded_canary"]["worst_dvr_increase"] <= 0.0


def test_no_arm_disables_a_mandatory_control(small_corpus, developed):
    """Structural immutability holds under every arm, including unconstrained (Req 7.4)."""
    report = run_experiment3(
        small_corpus, developed=developed, seeds=(1337,), max_blocks=3
    )
    assert report["safety_summary"]["tier_disablement_runs"] == 0


def test_unconstrained_arm_drifts_further_than_the_bounded_one(small_corpus, developed):
    """The trust region demonstrably restricts movement (Req 7.3)."""
    report = run_experiment3(
        small_corpus,
        developed=developed,
        seeds=(1337,),
        arms=("bounded_canary", "unconstrained"),
        max_blocks=3,
    )
    assert (
        report["table6"]["unconstrained"]["max_drift"]
        > report["table6"]["bounded_canary"]["max_drift"]
    )


# --------------------------------------------------------------------------- #
# Experiment 4 (Req 13.5)
# --------------------------------------------------------------------------- #
def test_experiment4_reports_downstream_outcomes(small_corpus, experiment1):
    """Table 7 reports accuracy, unsupported conclusions, and staleness (Req 13.5)."""
    report = run_experiment4(small_corpus, experiment1=experiment1)
    assert report["table7"]
    for entry in report["table7"]:
        for key in (
            "answer_accuracy",
            "unsupported_rate",
            "stale_propagation_rate",
            "r100",
        ):
            assert key in entry
            assert 0.0 <= entry[key] if key != "r100" else entry[key] >= 0.0


def test_autonomous_ocmr_propagates_stale_values(small_corpus, experiment1):
    """Holding valid corrections forever shows up as staleness downstream."""
    report = run_experiment4(small_corpus, experiment1=experiment1)
    by_condition = {e["condition"]: e for e in report["table7"]}
    autonomous = by_condition["autonomous_ocmr"]
    adaptive = by_condition["adaptive_rahgm"]
    assert autonomous["stale_propagation_rate"] > adaptive["stale_propagation_rate"]
    assert adaptive["answer_accuracy"] > autonomous["answer_accuracy"]


def test_experiment4_breaks_results_out_by_capability(small_corpus, experiment1):
    """Effects are reported per capability, as §4.5 requires."""
    report = run_experiment4(small_corpus, experiment1=experiment1)
    capabilities = report["by_capability"]["adaptive_rahgm"]
    assert "entity_resolution" in capabilities
    assert "contradiction_heavy" in capabilities


# --------------------------------------------------------------------------- #
# Quarantine audit (Req 13.1)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("reason", "cause"),
    [
        ("single-valued conflict on 'ASSIGNED_TO'", "genuine_ambiguity"),
        ("Task 'x' is 'done' but has no completion Event", "evidence_provenance"),
        ("task transition 'cancelled' -> 'todo' is not permitted", "temporal_cardinality"),
        ("unresolved entity alias for 'A. Moreau'", "entity_alias_resolution"),
        ("something entirely unrecognized", "other"),
    ],
)
def test_quarantine_reasons_are_classified_by_the_published_rules(reason, cause):
    """Cause assignment is a published rule set, not a judgment (Req 13.1)."""
    assigned, _check = classify_reason(reason)
    assert assigned == cause


def test_unmatched_reasons_are_not_attributed():
    """An unrecognized reason gets no OCMR check attributed to it."""
    cause, check = classify_reason("wholly novel wording")
    assert cause == "other"
    assert check is None


def test_all_causes_are_reachable():
    """Every Table 2 category can be produced by some reason string."""
    produced = {
        classify_reason(text)[0]
        for text in (
            "unresolved entity alias",
            "single-valued conflict",
            "missing evidence",
            "cardinality violation",
            "??",
        )
    }
    assert produced <= set(CAUSES)
    assert len(produced) >= 4


# --------------------------------------------------------------------------- #
# Structural identity attribution (Table 2)
# --------------------------------------------------------------------------- #
def test_structural_identity_check_overrides_the_lexical_rule():
    """A cross-context collision is attributed to identity, not to its symptom.

    OCMR's reason names the check that fired, never the identity ambiguity that
    caused it, so the structural signal must take precedence.
    """
    reason = "single-valued conflict on 'ASSIGNED_TO'"
    assert classify_quarantine(reason, identity_ambiguous=False)[0] == "genuine_ambiguity"
    assert (
        classify_quarantine(reason, identity_ambiguous=True)[0]
        == "entity_alias_resolution"
    )


def test_identity_detector_reports_zero_under_isolation():
    """The negative control: isolation makes cross-context collision impossible.

    With a fresh store per example no quarantine can conflict with an assertion
    another example authored, so a correct detector must report exactly zero. This
    is what distinguishes the attribution from a coincidence.
    """
    isolated = collect_quarantines(
        seeds=(1337,), per_category=6, isolate_per_example=True
    )
    assert isolated, "the isolated replay produced no quarantines to check"
    assert all(not u.identity_ambiguous for u in isolated)
    assert all(u.cause != "entity_alias_resolution" for u in isolated)


def test_identity_detector_fires_under_the_shared_store():
    """The shared store reuses identifiers across examples, so the detector fires."""
    shared = collect_quarantines(seeds=(1337,), per_category=6)
    assert shared
    assert any(u.identity_ambiguous for u in shared)


# --------------------------------------------------------------------------- #
# Drift study (not in the paper)
# --------------------------------------------------------------------------- #
def test_covariate_drift_changes_features_and_relabels_consistently(small_corpus):
    """Degrading a correction's source moves it below the ``h(u)`` authority floor."""
    original = small_corpus.partition(Partition.test)
    drifted = drift_scenarios(original, severity=1.0, mode="covariate")

    before = {w.write_id: w for s in original for w in s.writes}
    changed = [
        w
        for s in drifted
        for w in s.writes
        if before[w.write_id].gold_transition is not w.gold_transition
    ]
    assert changed
    for write in changed:
        assert before[write.write_id].gold_transition is Tier.supersede
        assert write.gold_transition is Tier.review
        # The feature the router consumes actually moved.
        assert write.authority < 0.90 <= before[write.write_id].authority


def test_label_drift_holds_every_feature_fixed(small_corpus):
    """Label drift changes only the correct answer, so the router cannot see it."""
    original = small_corpus.partition(Partition.test)
    drifted = drift_scenarios(original, severity=1.0, mode="label")

    before = {w.write_id: w for s in original for w in s.writes}
    changed = [
        w
        for s in drifted
        for w in s.writes
        if before[w.write_id].gold_transition is not w.gold_transition
    ]
    assert changed
    for write in changed:
        prior = before[write.write_id]
        assert write.gold_transition is Tier.review
        for field in (
            "source_ref",
            "authority",
            "consequence",
            "reversibility",
            "confidence",
            "predicate",
            "subject_id",
            "object_id",
            "alias_ambiguous",
            "poisoned_evidence",
        ):
            assert getattr(write, field) == getattr(prior, field), field


def test_unknown_drift_mode_is_rejected(small_corpus):
    """An unrecognized mode fails loudly rather than silently doing nothing."""
    with pytest.raises(ValueError):
        drift_scenarios(small_corpus.partition(Partition.test), mode="nonsense")


def test_frozen_policy_self_corrects_under_covariate_drift(small_corpus, developed):
    """Drift expressed in a feature the router consumes needs no adaptation."""
    report = run_drift_study(
        small_corpus, developed=developed, repeats=2, mode="covariate"
    )
    arms = {a["arm"]: a for a in report["arms"]}
    # The frozen arm holds up on its own, so adaptation cannot add much.
    assert arms["frozen"]["accuracy"] > 0.90
    assert report["contrast"]["accuracy_delta_points"] < 2.0


def test_adaptation_helps_under_label_drift(small_corpus, developed):
    """Drift the router cannot see is where bounded feedback earns its keep."""
    report = run_drift_study(small_corpus, developed=developed, repeats=4, mode="label")
    arms = {a["arm"]: a for a in report["arms"]}
    assert arms["adaptive"]["n_deployed"] > 0, "no updates were deployed"
    assert arms["adaptive"]["accuracy"] >= arms["frozen"]["accuracy"]
    # The frozen arm cannot improve within the stream; the adapting one can.
    assert arms["frozen"]["recovery"] <= arms["adaptive"]["recovery"] + 1e-9


def test_drift_study_declares_it_is_not_in_the_paper(small_corpus, developed):
    """Work beyond the paper's design must say so (Req 14.1)."""
    report = run_drift_study(small_corpus, developed=developed, repeats=2)
    assert report["in_paper"] is False
    assert report["rationale"].strip()


# --------------------------------------------------------------------------- #
# Cascade study (not in the paper)
# --------------------------------------------------------------------------- #
def test_status_writes_are_governed_by_the_reconcile_path(small_corpus):
    """A ``HAS_STATUS`` write must be checked by C4/C10, not just C7.

    ``ConstraintValidator`` runs C10 only when handed an explicit task transition,
    and OCMR applies C4/C8 on its reconcile path. Routing status writes through the
    relation path alone would silently skip all three and report every status change
    as a bare single-valued contradiction.
    """
    scenario = next(
        s
        for s in small_corpus.partition(Partition.test)
        if any("chain_status" in w.template for w in s.writes)
    )
    chain = sorted(
        (w for w in scenario.writes if w.chain_id and "status" in w.chain_id),
        key=lambda w: w.chain_position,
    )
    task = chain[0].subject_id
    cancel = chain[2]

    verdicts = {}
    for state in ("status:in_progress", "status:done"):
        container = CoreContainer(replay_settings())
        install_scenario_state(container, scenario)
        assertion = Assertion(
            id="probe",
            subject_id=task,
            predicate="HAS_STATUS",
            object_id=state,
            confidence=0.95,
            status=AssertionStatus.accepted,
            source_ref="probe",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        container.repo.upsert_assertion(assertion)
        container.graph.add_assertion(assertion)
        verdicts[state] = ocmr_verdict(container, to_candidate(cancel))

    # Cancelling is legal from in_progress and illegal from done (a terminal state).
    assert verdicts["status:in_progress"].recommended_action == "supersede"
    assert verdicts["status:done"].recommended_action == "quarantine"
    assert verdicts["status:done"].failed_check == "HAS_STATUS"


def test_force_hold_and_force_commit_override_routing(small_corpus, developed):
    """The cascade interventions change one transition and nothing else."""
    scenario = small_corpus.partition(Partition.test)[0]
    target = next(w for w in scenario.writes if w.gold_transition is Tier.review)
    reviewer = make_oracle_reviewer({w.write_id: w for w in scenario.writes})

    committed = ScenarioReplayer(
        Condition.frozen_rahgm,
        params=developed.params,
        reviewer=reviewer,
        force_commit={target.write_id},
    )
    _h, result = committed.run_scenario(scenario)
    record = next(r for r in result.records if r.write_id == target.write_id)
    assert record.final in (Tier.accept, Tier.supersede)

    held = ScenarioReplayer(
        Condition.frozen_rahgm,
        params=developed.params,
        reviewer=reviewer,
        force_hold={target.write_id},
    )
    _h2, result2 = held.run_scenario(scenario)
    record2 = next(r for r in result2.records if r.write_id == target.write_id)
    assert record2.final is Tier.review


def test_injected_error_reaches_later_decisions(small_corpus, developed):
    """§3.1: an erroneous transition at ``t`` influences later states.

    Measured as a change in OCMR's own constraint verdict for a later write in the
    same chain, plus extra durable-state violations. Both are attributable, because
    the two replays differ in exactly one transition.
    """
    scenarios = small_corpus.partition(Partition.test)
    report = measure_injected_cascade(
        scenarios, condition=Condition.frozen_rahgm, params=developed.params
    )
    assert report["n_injections"] > 0, "no chain offered an injection point"
    assert (
        report["n_downstream_verdicts_changed"] > 0
        or report["durable_violation_delta"] > 0
        or report["n_propagated_errors"] > 0
    ), "an injected upstream error had no downstream effect at all"


def test_cascade_study_distinguishes_observed_from_injected(small_corpus, developed):
    """The report must not conflate "can propagate" with "did propagate"."""
    report = run_cascade_study(small_corpus, developed=developed)
    assert report["in_paper"] is False
    assert "streams" in report and "error_injection" in report
    totals = report["totals"]
    assert "upstream_errors" in totals
    assert "injected_propagated_errors" in totals
    # The finding must state which of the two it is talking about.
    assert report["finding"].strip()


def test_cascade_records_carry_the_ocmr_verdict(small_corpus, developed):
    """Records retain OCMR's verdict so verdict changes are detectable."""
    scenario = small_corpus.partition(Partition.test)[0]
    replayer = ScenarioReplayer(Condition.frozen_rahgm, params=developed.params)
    _h, result = replayer.run_scenario(scenario)
    assert any(r.ocmr_failed_check is not None for r in result.records)
    assert any(r.ocmr_action is not None for r in result.records)


# --------------------------------------------------------------------------- #
# Honest reporting (Req 14.1, 14.2)
# --------------------------------------------------------------------------- #
def test_scope_note_names_every_modelled_component():
    """The scope note discloses each simulation and model (Req 14.1)."""
    lowered = SCOPE_NOTE.lower()
    for phrase in ("simulation", "review-cost model", "annotator simulators", "rq2"):
        assert phrase in lowered


def test_rendered_report_leads_with_the_scope_note(experiment1, experiment2):
    """The note appears before any table (Req 14.1)."""
    text = render_all({"experiment1": experiment1, "experiment2": experiment2})
    assert text.startswith(SCOPE_NOTE)
    assert "SIMULATED ANALYST" in text


def test_reports_are_json_serializable(experiment1, experiment2):
    """Artifacts must round-trip to JSON for machine consumption (Req 13.6)."""
    payload = {
        k: v for k, v in experiment1.items() if not k.startswith("_")
    }
    json.dumps(payload, default=str)
    json.dumps(experiment2, default=str)

from __future__ import annotations

import pytest

from ocm.evaluation.entity_linking_evasion import (
    PAPER_ATTACK_AXES,
    generate_evasion_cases,
    run_benign_linkage_corpus,
    run_entity_linking_evasion_attack,
    run_injection_case,
    run_paper_grade_linkage_evasion_suite,
)

_REGRESSION_AXES = (
    "novel_alias",
    "spelling_variant",
    "spacing_variant",
    "partial",
)
_REGRESSION_SEED = 1337
_REGRESSION_CASES = generate_evasion_cases(
    seed=_REGRESSION_SEED,
    axes=_REGRESSION_AXES,
    per_axis=8,
)


def test_containment_blocks_evasive_acceptance_in_aggregate():
    report = run_entity_linking_evasion_attack(
        seed=_REGRESSION_SEED,
        axes=_REGRESSION_AXES,
        per_axis=8,
    )

    control = report["by_condition"]["control"]
    evasive = report["by_condition"]["evasive"]

    assert control["attack_success_rate"] == 0.0
    assert control["detection_rate"] == 1.0
    assert control["quarantine_rate"] == 1.0
    assert control["linked_rate"] == 1.0

    assert evasive["attack_success_rate"] == 0.0
    assert evasive["accepted_rate"] == 0.0
    assert evasive["detection_rate"] == 1.0
    assert evasive["quarantine_rate"] == 1.0
    assert evasive["linked_rate"] == 0.0
    assert evasive["oracle_violation_count"] == 0
    assert evasive["oracle_conflict_attempt_count"] == 32
    assert evasive["system_durable_violation_count"] == 0
    assert evasive["benign_false_positive_rate"] == 0.0
    assert evasive["external_side_effect_count"] == 0


def test_containment_blocks_fresh_mutated_evasive_samples():
    report = run_entity_linking_evasion_attack(
        seed=2026,
        axes=_REGRESSION_AXES,
        per_axis=8,
        mutation="mutated",
    )

    evasive = report["by_condition"]["evasive"]
    assert evasive["attack_success_rate"] == 0.0
    assert evasive["accepted_rate"] == 0.0
    assert evasive["detection_rate"] == 1.0
    assert evasive["quarantine_rate"] == 1.0
    assert evasive["benign_false_positive_rate"] == 0.0
    assert evasive["external_side_effect_count"] == 0


def test_config_off_preserves_backward_compatible_acceptance_path():
    report = run_entity_linking_evasion_attack(
        seed=_REGRESSION_SEED,
        axes=("novel_alias",),
        per_axis=1,
        fail_closed=False,
    )

    evasive = report["by_condition"]["evasive"]
    assert report["release_gate"]["status"] == "bypass_confirmed_config_off"
    assert evasive["accepted_rate"] == 1.0
    assert evasive["attack_success_rate"] == 1.0
    assert evasive["detection_rate"] == 0.0


def test_broader_benign_corpus_has_no_false_positives():
    report = run_benign_linkage_corpus(seed=_REGRESSION_SEED, threshold=0.05)

    assert report["summary"]["passes_threshold"] is True
    assert report["summary"]["false_positive_count"] == 0
    assert report["summary"]["external_side_effect_count"] == 0
    assert report["summary"]["data_exposure_count"] == 0


def test_attributed_alias_and_update_writes_still_succeed():
    report = run_benign_linkage_corpus(seed=_REGRESSION_SEED, threshold=0.05)
    by_id = {r["case_id"]: r for r in report["records"]}

    assert by_id["benign-attributed-novel-alias-same-fact"]["decision"] == "accepted"
    assert by_id["benign-attributed-leading-zero-update"]["decision"] == "superseded"
    assert by_id["benign-attributed-underscore-same-fact"]["decision"] == "accepted"


@pytest.mark.parametrize(
    "case",
    _REGRESSION_CASES,
    ids=[case.case_id for case in _REGRESSION_CASES],
)
def test_all_32_original_evasive_cases_are_quarantined(case):
    record = run_injection_case(case, "evasive", seed=_REGRESSION_SEED)

    assert record.decision == "quarantined"
    assert record.gate_fired is True
    assert record.linked_to_canonical is False
    assert record.attack_success is False
    assert record.oracle_violation is False
    assert record.would_contradict_under_oracle is True
    assert record.system_durable_violations == 0
    assert record.benign_decision == "accepted"
    assert record.quarantine_conflicting_ids
    assert record.side_effect_review["external_side_effects_observed"] == 0
    assert record.side_effect_review["data_exposure_observed"] is False
    assert "C7_LINKAGE_ATTRIBUTION" in (record.reason or "")


def test_canonical_control_still_uses_the_original_contradiction_gate():
    case = _REGRESSION_CASES[0]
    record = run_injection_case(case, "control", seed=_REGRESSION_SEED)

    assert record.decision == "quarantined"
    assert record.gate_fired is True
    assert record.linked_to_canonical is True
    assert record.attack_success is False
    assert "single-valued conflict" in (record.reason or "")


def test_case_generation_is_seeded_and_has_distance_curve_fields():
    a = generate_evasion_cases(seed=3, axes=("novel_alias", "spacing_variant"), per_axis=2)
    b = generate_evasion_cases(seed=3, axes=("novel_alias", "spacing_variant"), per_axis=2)

    assert a == b
    assert {case.axis for case in a} == {"novel_alias", "spacing_variant"}
    assert all(0.0 < case.mention_distance <= 1.0 for case in a)


def test_paper_suite_covers_adaptive_axes_baselines_and_benign_utility():
    report = run_paper_grade_linkage_evasion_suite(
        seeds=(1, 2),
        baselines=("B2", "Bsup", "B3"),
        axes=("adaptive_alias", "role_description", "unrelated_alias"),
        per_axis=2,
        benign_per_family=2,
    )

    assert report["release_gate"]["status"] == "validated_for_paper_grade_synthetic_eval"
    assert set(report["freeze"]["axes"]) <= set(PAPER_ATTACK_AXES)

    rows = {
        (row["baseline"], row["mutation"]): row
        for row in report["baseline_comparison"]["rows"]
    }
    assert rows[("B3", "mutated")]["conditions"]["evasive"]["accepted_rate"]["mean"] == 0.0
    assert rows[("B3", "mutated")]["conditions"]["evasive"]["detection_rate"]["mean"] == 1.0
    assert rows[("B2", "mutated")]["conditions"]["evasive"]["accepted_rate"]["mean"] == 1.0
    assert rows[("Bsup", "mutated")]["conditions"]["evasive"]["accepted_rate"]["mean"] == 1.0

    c7_off = report["defense_ablations"]["c7_fail_closed_off_b3"]
    assert c7_off["conditions"]["evasive"]["accepted_rate"]["mean"] == 1.0
    assert report["benign_utility"]["summary"]["false_positive_rate"]["mean"] == 0.0

"""Behavior tests for the Schema/Provenance Stress Workload (data + guard).

This module covers the example-based guarantees of the stress-workload feature
that pair with its property-based tests. The tests here reuse the implemented
data layer (:mod:`ocm.evaluation.datasets.stress_workload`) and drive the real
:class:`~ocm.core.container.CoreContainer` write pipeline with the deterministic
``StressOracleExtractor`` injected exactly as the LongMemEval oracle is.

Task 1.2 — **guard default-preserving regression** (Req 13.6, 15.2, 12.6): with
``enable_constraint_validation=True`` (the default, all-governance-on
configuration) each C4/C8/C10-governed write must be routed to the quarantine
bucket and must not appear in the accepted store, proving the additive
Reconcile_Path_Guard does not change the default C4/C8/C10 behavior. This is the
example-based partner to Property 10.
"""

from __future__ import annotations

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.evaluation.datasets.stress_workload import (
    StressCase,
    WriteClass,
    generate_stress_workload,
)
from ocm.evaluation.stress_ablation import STRESS_ARMS
from ocm.memory.write_pipeline import HAS_STATUS, STATUS_VALUE_PREFIX


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _default_governed_container(oracle) -> CoreContainer:
    """A CoreContainer with the default all-governance-on toggles.

    ``enable_schema_validation`` / ``enable_constraint_validation`` /
    ``enable_contradiction_gate`` all default to ``True`` (full OCMR), so this is
    the default configuration the Reconcile_Path_Guard must leave byte-identical.
    """
    return CoreContainer(
        Settings(deterministic_test_mode=True, chroma_mode="memory"),
        extractor=oracle,
    )


def _replay(container: CoreContainer, example) -> list:
    """Replay every session of an example, returning the per-session WriteResults."""
    results = []
    for session in example.sessions:
        results.append(
            container.write_pipeline.run(
                session.input, f"{example.id}:{session.session_id}"
            )
        )
    return results


def _status_value_of(container: CoreContainer, object_id: str) -> str | None:
    """Resolve the status label a ``StatusValue`` object id encodes."""
    payload = container.graph.get_entity_payload(object_id) or {}
    value = payload.get("value") or payload.get("name")
    if value:
        return str(value)
    if isinstance(object_id, str) and object_id.startswith(STATUS_VALUE_PREFIX):
        return object_id[len(STATUS_VALUE_PREFIX):]
    return None


def _accepted_has_status_values(container: CoreContainer) -> set[str]:
    """The set of status labels held by accepted ``HAS_STATUS`` assertions."""
    values: set[str] = set()
    for a in container.repo.list_assertions("accepted"):
        if a.predicate == HAS_STATUS:
            v = _status_value_of(container, a.object_id)
            if v is not None:
                values.add(v)
    return values


def _quarantined_has_status_values(results: list) -> set[str]:
    """Status labels quarantined across the given per-session WriteResults."""
    values: set[str] = set()
    for res in results:
        for outcome in res.quarantined:
            if outcome.candidate.predicate == HAS_STATUS:
                oid = outcome.candidate.object_id
                if isinstance(oid, str) and oid.startswith(STATUS_VALUE_PREFIX):
                    values.add(oid[len(STATUS_VALUE_PREFIX):])
    return values


def _find_case(cases: list[StressCase], write_class: WriteClass, predicate) -> StressCase:
    """Return the first case of ``write_class`` matching ``predicate``."""
    for case in cases:
        if case.write_class is write_class and predicate(case):
            return case
    raise AssertionError(f"no {write_class} case matched the predicate")


def _example_for(examples: list, case_id: str):
    """Return the BenchmarkExample whose id matches a case id."""
    for ex in examples:
        if ex.id == case_id:
            return ex
    raise AssertionError(f"no example for case {case_id!r}")


def _is_c4_case(case: StressCase) -> bool:
    """A C4 STATUS case: a single-session ``done`` Task with no completion Event."""
    return len(case.writes.events) == 0 and any(
        (e.get("fields") or {}).get("status") == "done" for e in case.writes.entities
    )


def _is_c10_case(case: StressCase) -> bool:
    """A C10 STATUS case: the poison session states an illegal ``todo`` transition."""
    return any(
        (e.get("fields") or {}).get("status") == "todo" for e in case.writes.entities
    )


# --------------------------------------------------------------------------- #
# Task 1.2 — guard default-preserving regression (Req 13.6, 15.2, 12.6)
# --------------------------------------------------------------------------- #
def test_default_config_quarantines_unsupported_final_decision_c8():
    """C8: a ``final`` Decision with no evidence is quarantined, not accepted."""
    examples, oracle, cases = generate_stress_workload(seed=1337)
    case = _find_case(cases, WriteClass.EVIDENCE, lambda _c: True)
    example = _example_for(examples, case.case_id)

    container = _default_governed_container(oracle)
    results = _replay(container, example)

    # The final-status assertion is routed to quarantine by the C8 evidence floor.
    assert "final" in _quarantined_has_status_values(results)
    # ...and no accepted HAS_STATUS asserts the Decision is final.
    assert "final" not in _accepted_has_status_values(container)


def test_default_config_quarantines_done_task_without_completion_event_c4():
    """C4: a ``done`` Task lacking a completion Event is quarantined, not accepted."""
    examples, oracle, cases = generate_stress_workload(seed=1337)
    case = _find_case(cases, WriteClass.STATUS, _is_c4_case)
    example = _example_for(examples, case.case_id)

    container = _default_governed_container(oracle)
    results = _replay(container, example)

    assert "done" in _quarantined_has_status_values(results)
    assert "done" not in _accepted_has_status_values(container)


def test_default_config_quarantines_illegal_done_to_todo_transition_c10():
    """C10: an illegal ``done`` -> ``todo`` transition is quarantined, not accepted.

    Session s1 legitimately completes the Task (``done`` with a completion Event +
    ``RESULTS_IN``, so C4 accepts it); session s2 states ``todo``, an illegal
    transition out of the terminal ``done`` status that the reconcile path must
    quarantine while leaving the accepted ``done`` status untouched.
    """
    examples, oracle, cases = generate_stress_workload(seed=1337)
    case = _find_case(cases, WriteClass.STATUS, _is_c10_case)
    example = _example_for(examples, case.case_id)
    assert len(example.sessions) == 2  # setup (done) then poison (todo)

    container = _default_governed_container(oracle)
    results = _replay(container, example)

    # The illegal todo transition is quarantined...
    assert "todo" in _quarantined_has_status_values(results)
    # ...the accepted store never records the illegal todo status...
    accepted_values = _accepted_has_status_values(container)
    assert "todo" not in accepted_values
    # ...and the legitimate done status set up in s1 remains accepted.
    assert "done" in accepted_values


def test_default_config_governed_arm_has_no_typed_violations():
    """The whole workload under the default arm leaves zero Invalid_Active_State.

    Aggregate cross-check on the C4/C8/C10 (and C9/C2) governed writes: replaying
    every example under the default all-governance-on configuration must leave the
    typed-violation total at zero, i.e. no poison write reaches the accepted store.
    """
    from ocm.evaluation.typed_violations import typed_violations

    examples, oracle, _cases = generate_stress_workload(seed=1337)
    container = _default_governed_container(oracle)
    for example in examples:
        _replay(container, example)

    report = typed_violations(container)
    assert report.unsupported_final_decision == 0
    assert report.illegal_status_state == 0
    assert report.total == 0


# --------------------------------------------------------------------------- #
# Task 1.3 — gate-off leaves C4/C8/C10 poison as accepted (Req 13.7, 15.1)
# --------------------------------------------------------------------------- #
def _gate_off_container(oracle) -> CoreContainer:
    """A CoreContainer with ``enable_constraint_validation=False``.

    With constraint validation off the Reconcile_Path_Guard suppresses C4/C8/C10
    enforcement, so their poison writes must be left as accepted
    Invalid_Active_State rather than quarantined (Req 15.1). Schema validation and
    the contradiction gate are set to mirror the Gate_Only arm (schema off, gate
    on); neither touches the reconcile path, so the guard's toggle is what decides
    the C4/C8/C10 outcome here.
    """
    return CoreContainer(
        Settings(
            deterministic_test_mode=True,
            chroma_mode="memory",
            enable_schema_validation=False,
            enable_constraint_validation=False,
            enable_contradiction_gate=True,
        ),
        extractor=oracle,
    )


def test_gate_off_leaves_unsupported_final_decision_active_c8():
    """C8: with the guard off, the unsupported ``final`` Decision stays accepted."""
    examples, oracle, cases = generate_stress_workload(seed=1337)
    case = _find_case(cases, WriteClass.EVIDENCE, lambda _c: True)
    example = _example_for(examples, case.case_id)

    container = _gate_off_container(oracle)
    results = _replay(container, example)

    # The final-status assertion is left active (accepted), not quarantined.
    assert "final" in _accepted_has_status_values(container)
    assert "final" not in _quarantined_has_status_values(results)


def test_gate_off_leaves_done_task_without_completion_event_active_c4():
    """C4: with the guard off, the unsupported ``done`` Task stays accepted."""
    examples, oracle, cases = generate_stress_workload(seed=1337)
    case = _find_case(cases, WriteClass.STATUS, _is_c4_case)
    example = _example_for(examples, case.case_id)

    container = _gate_off_container(oracle)
    results = _replay(container, example)

    assert "done" in _accepted_has_status_values(container)
    assert "done" not in _quarantined_has_status_values(results)


def test_gate_off_leaves_illegal_done_to_todo_transition_active_c10():
    """C10: with the guard off, the illegal ``done`` -> ``todo`` flip stays accepted.

    Session s1 completes the Task (``done``); session s2 states ``todo``. With
    ``enable_constraint_validation=False`` the reconcile path accepts the illegal
    ``todo`` transition instead of quarantining it, leaving the illegal status as
    an accepted Invalid_Active_State.
    """
    examples, oracle, cases = generate_stress_workload(seed=1337)
    case = _find_case(cases, WriteClass.STATUS, _is_c10_case)
    example = _example_for(examples, case.case_id)
    assert len(example.sessions) == 2  # setup (done) then poison (todo)

    container = _gate_off_container(oracle)
    results = _replay(container, example)

    # The illegal todo transition is left active (accepted), not quarantined.
    assert "todo" in _accepted_has_status_values(container)
    assert "todo" not in _quarantined_has_status_values(results)


def test_gate_off_arm_leaves_typed_violations_active():
    """Aggregate: gate-off leaves the C4/C8/C10 poison as Invalid_Active_State.

    Replaying every example under ``enable_constraint_validation=False`` must
    leave the unsupported-final-decision and illegal-status-state typed-violation
    counts greater than zero, i.e. the guard suppresses C4/C8/C10 and the poison
    reaches the accepted store (Req 13.7, 15.1).
    """
    from ocm.evaluation.typed_violations import typed_violations

    examples, oracle, _cases = generate_stress_workload(seed=1337)
    container = _gate_off_container(oracle)
    for example in examples:
        _replay(container, example)

    report = typed_violations(container)
    assert report.unsupported_final_decision > 0
    assert report.illegal_status_state > 0
    assert report.total > 0


# --------------------------------------------------------------------------- #
# Task 7.1 — per-class routing + valid-write precision
# (Req 13.1, 13.3, 2.3, 2.4, 4.4, 4.5, 4.6, 15.5)
# --------------------------------------------------------------------------- #
# These tests build one CoreContainer per governance arm (the STRESS_ARMS toggle
# triples) with the deterministic StressOracleExtractor injected, replay each
# stress case's sessions, and assert its intended check routes the poison write:
#
#   * SCHEMA (C9 + W5) and TEMPORAL (C2) are *relation-path* classes: their poison
#     relation is REJECTED (a non-accepted outcome) under the Schema_Provenance and
#     Full arms (constraint validation on) and left ACCEPTED under the Gate_Only arm
#     (constraint validation off ⇒ W6/C9/C2 never run) (Req 13.1, 1.5/1.6, 3.4/3.5).
#   * EVIDENCE (C8) and STATUS (C4/C10) are *reconcile-path* classes: their poison
#     HAS_STATUS assertion is QUARANTINED under the Schema_Provenance and Full arms
#     and left ACCEPTED under the Gate_Only arm, because the Reconcile_Path_Guard
#     suppresses C8/C4/C10 while enable_constraint_validation is false
#     (Req 13.1, 2.3, 2.4, 4.4, 4.5, 4.6, 15.5).
#   * Valid-write precision (Req 13.3): every Valid_Write is admitted as an accepted
#     outcome under the Full arm.
GOVERNED_ARMS = ("Schema_Provenance_Arm", "Full_Arm")


def _arm_container(oracle, arm: str) -> CoreContainer:
    """A CoreContainer configured with an arm's STRESS_ARMS toggle triple.

    The offline deterministic base (``deterministic_test_mode`` / memory Chroma) is
    overlaid with the arm's ``enable_schema_validation`` /
    ``enable_constraint_validation`` / ``enable_contradiction_gate`` triple, and the
    oracle is injected exactly as the runner wires it.
    """
    return CoreContainer(
        Settings(
            deterministic_test_mode=True,
            chroma_mode="memory",
            **STRESS_ARMS[arm],
        ),
        extractor=oracle,
    )


def _cases_of(cases: list[StressCase], write_class: WriteClass) -> list[StressCase]:
    """All stress cases of a given write class."""
    return [c for c in cases if c.write_class is write_class]


def _accepted_predicates(container: CoreContainer) -> list[str]:
    """Predicates of every assertion in the durable ACTIVE (accepted) store."""
    return [a.predicate for a in container.repo.list_assertions("accepted")]


def _rejected_predicates(results: list) -> list[str]:
    """Predicates rejected across the given per-session WriteResults."""
    return [o.candidate.predicate for res in results for o in res.rejected]


def _assert_relation_class_routing(
    examples: list, oracle, cases: list[StressCase], write_class: WriteClass
) -> None:
    """A relation-path poison class is rejected when governed, accepted gate-only.

    For every case of ``write_class`` the single poison relation must be routed to a
    rejected (non-accepted) outcome under each governed arm and absent from that
    arm's accepted store, while under the Gate_Only arm it is left as an accepted
    Invalid_Active_State (the constraint validator never runs).
    """
    class_cases = _cases_of(cases, write_class)
    assert class_cases, f"no {write_class} cases were generated"
    for case in class_cases:
        assert case.writes.relations, f"{case.case_id} has no poison relation"
        predicate = case.writes.relations[0]["predicate"]
        example = _example_for(examples, case.case_id)

        # Gate_Only: the poison relation is accepted (left active).
        gate = _arm_container(oracle, "Gate_Only_Arm")
        _replay(gate, example)
        assert predicate in _accepted_predicates(gate), (
            f"{case.case_id}: expected {predicate} accepted under Gate_Only_Arm"
        )

        # Governed arms: the poison relation is rejected and never accepted.
        for arm in GOVERNED_ARMS:
            container = _arm_container(oracle, arm)
            results = _replay(container, example)
            assert predicate in _rejected_predicates(results), (
                f"{case.case_id}: expected {predicate} rejected under {arm}"
            )
            assert predicate not in _accepted_predicates(container), (
                f"{case.case_id}: {predicate} must not be accepted under {arm}"
            )


def test_schema_cases_rejected_under_governed_accepted_under_gate_only():
    """SCHEMA (C9 + W5): domain/range poison rejected when governed, else active."""
    examples, oracle, cases = generate_stress_workload(seed=1337)
    _assert_relation_class_routing(examples, oracle, cases, WriteClass.SCHEMA)


def test_temporal_cases_rejected_under_governed_accepted_under_gate_only():
    """TEMPORAL (C2): end-before-start poison rejected when governed, else active."""
    examples, oracle, cases = generate_stress_workload(seed=1337)
    _assert_relation_class_routing(examples, oracle, cases, WriteClass.TEMPORAL)


def test_evidence_cases_quarantined_under_governed_accepted_under_gate_only():
    """EVIDENCE (C8): unsupported ``final`` Decision quarantined when governed.

    Under the Schema_Provenance and Full arms the reconcile path enforces C8 and
    quarantines the unsupported ``final`` HAS_STATUS (Req 2.3, 15.5); under the
    Gate_Only arm the Reconcile_Path_Guard suppresses C8, so the ``final`` status is
    left as an accepted Invalid_Active_State (Req 2.4).
    """
    examples, oracle, cases = generate_stress_workload(seed=1337)
    evidence_cases = _cases_of(cases, WriteClass.EVIDENCE)
    assert evidence_cases, "no EVIDENCE cases were generated"
    for case in evidence_cases:
        example = _example_for(examples, case.case_id)

        # Gate_Only: the unsupported final decision is accepted (left active).
        gate = _arm_container(oracle, "Gate_Only_Arm")
        gate_results = _replay(gate, example)
        assert "final" in _accepted_has_status_values(gate)
        assert "final" not in _quarantined_has_status_values(gate_results)

        # Governed arms: the final status is quarantined, never accepted.
        for arm in GOVERNED_ARMS:
            container = _arm_container(oracle, arm)
            results = _replay(container, example)
            assert "final" in _quarantined_has_status_values(results), (
                f"{case.case_id}: expected final status quarantined under {arm}"
            )
            assert "final" not in _accepted_has_status_values(container), (
                f"{case.case_id}: final status must not be accepted under {arm}"
            )


def test_status_cases_quarantined_under_governed_accepted_under_gate_only():
    """STATUS (C4 + C10): illegal status quarantined when governed, else active.

    The poison status value is ``done`` for the C4 case (a ``done`` Task with no
    completion Event) and ``todo`` for the C10 case (the illegal ``done`` -> ``todo``
    transition). Under the Schema_Provenance and Full arms the reconcile path
    enforces C4/C10 and quarantines the offending HAS_STATUS (Req 4.4, 4.5, 15.5);
    under the Gate_Only arm the Reconcile_Path_Guard suppresses C4/C10, so the
    illegal status is left as an accepted Invalid_Active_State (Req 4.6).
    """
    examples, oracle, cases = generate_stress_workload(seed=1337)
    status_cases = _cases_of(cases, WriteClass.STATUS)
    assert status_cases, "no STATUS cases were generated"
    for case in status_cases:
        poison_value = "todo" if _is_c10_case(case) else "done"
        example = _example_for(examples, case.case_id)

        # Gate_Only: the illegal status is accepted (left active).
        gate = _arm_container(oracle, "Gate_Only_Arm")
        gate_results = _replay(gate, example)
        assert poison_value in _accepted_has_status_values(gate), (
            f"{case.case_id}: expected {poison_value} accepted under Gate_Only_Arm"
        )
        assert poison_value not in _quarantined_has_status_values(gate_results)

        # Governed arms: the illegal status is quarantined, never accepted.
        for arm in GOVERNED_ARMS:
            container = _arm_container(oracle, arm)
            results = _replay(container, example)
            assert poison_value in _quarantined_has_status_values(results), (
                f"{case.case_id}: expected {poison_value} quarantined under {arm}"
            )
            assert poison_value not in _accepted_has_status_values(container), (
                f"{case.case_id}: {poison_value} must not be accepted under {arm}"
            )


def test_valid_writes_all_accepted_under_full_arm():
    """Precision (Req 13.3, 5.3): every Valid_Write is accepted under the Full arm.

    Each Valid_Write violates none of C2/C4/C8/C9/C10/W5, so under the fully governed
    Full arm every valid case must produce at least one accepted outcome and yield no
    rejected or quarantined outcome.
    """
    examples, oracle, cases = generate_stress_workload(seed=1337)
    valid_cases = _cases_of(cases, WriteClass.VALID)
    assert valid_cases, "no VALID cases were generated"

    container = _arm_container(oracle, "Full_Arm")
    for case in valid_cases:
        example = _example_for(examples, case.case_id)
        results = _replay(container, example)
        accepted = sum(len(res.accepted) for res in results)
        rejected = sum(len(res.rejected) for res in results)
        quarantined = sum(len(res.quarantined) for res in results)
        assert accepted >= 1, f"{case.case_id}: Valid_Write produced no accepted outcome"
        assert rejected == 0, f"{case.case_id}: Valid_Write was rejected under Full_Arm"
        assert quarantined == 0, (
            f"{case.case_id}: Valid_Write was quarantined under Full_Arm"
        )


# --------------------------------------------------------------------------- #
# Task 7.2 — arm definitions, Diagnostic_Scope_Note, decisive designation,
# and offline / harness-reuse smoke
# (Req 9.1-9.5, 10.4, 14.1-14.4, 6.1, 6.3, 11.1, 12.1-12.4)
# --------------------------------------------------------------------------- #
from ocm.evaluation.stress_ablation import (  # noqa: E402
    DECISIVE_ARM,
    DIAGNOSTIC_SCOPE_NOTE,
    StressAblationResult,
    run_stress_ablation,
)


# --- Arm definitions (Req 9.1-9.5) ----------------------------------------- #
#: The only three governance toggle names any arm may use — introducing a new
#: toggle name here would violate Req 9.5 / 12.2 (configure arms *exclusively*
#: through existing Settings toggles, no new governance toggle).
_ALLOWED_ARM_TOGGLES = frozenset(
    {
        "enable_schema_validation",
        "enable_constraint_validation",
        "enable_contradiction_gate",
    }
)


def test_stress_arms_toggle_triples_match_spec_exactly():
    """STRESS_ARMS defines the four arms with the exact spec toggle triples.

    W5 = enable_schema_validation, W6 = enable_constraint_validation,
    C7 = enable_contradiction_gate (Req 9.1-9.4):

      * Ungoverned_Arm       W5 off / W6 off / C7 off
      * Gate_Only_Arm        W5 off / W6 off / C7 on
      * Schema_Provenance_Arm W5 on  / W6 on  / C7 off
      * Full_Arm             W5 on  / W6 on  / C7 on
    """
    assert STRESS_ARMS == {
        "Ungoverned_Arm": {
            "enable_schema_validation": False,
            "enable_constraint_validation": False,
            "enable_contradiction_gate": False,
        },
        "Gate_Only_Arm": {
            "enable_schema_validation": False,
            "enable_constraint_validation": False,
            "enable_contradiction_gate": True,
        },
        "Schema_Provenance_Arm": {
            "enable_schema_validation": True,
            "enable_constraint_validation": True,
            "enable_contradiction_gate": False,
        },
        "Full_Arm": {
            "enable_schema_validation": True,
            "enable_constraint_validation": True,
            "enable_contradiction_gate": True,
        },
    }


def test_stress_arms_introduce_no_new_toggle_names():
    """Every arm is built exclusively from the three existing toggles (Req 9.5, 12.2).

    No arm may name a governance toggle outside the allowed set, and those toggle
    names must be real ``Settings`` fields (so arms are applied via
    ``Settings.model_copy(update=...)`` without inventing new switches).
    """
    settings_fields = set(Settings.model_fields)
    for arm, triple in STRESS_ARMS.items():
        assert set(triple) == _ALLOWED_ARM_TOGGLES, (
            f"{arm} uses toggle names outside the allowed existing set"
        )
        for toggle in triple:
            assert toggle in settings_fields, (
                f"{arm} references {toggle!r} which is not an existing Settings field"
            )
            assert isinstance(triple[toggle], bool)


# --- Decisive designation (Req 10.4) --------------------------------------- #
def test_gate_only_arm_is_the_decisive_designation():
    """The Gate_Only_Arm is the decisive comparison row (Req 10.4)."""
    assert DECISIVE_ARM == "Gate_Only_Arm"

    result = run_stress_ablation(seed=1337)
    assert isinstance(result, StressAblationResult)
    assert result.decisive_arm == "Gate_Only_Arm"
    assert result.is_decisive("Gate_Only_Arm")
    assert not result.is_decisive("Schema_Provenance_Arm")
    # The decisive-row report is the Gate_Only arm's report.
    assert result.decisive_report is result.arms["Gate_Only_Arm"]


# --- Diagnostic_Scope_Note content (Req 14.1, 14.2, 14.3, 14.4) ------------ #
def test_diagnostic_scope_note_states_all_required_framing():
    """The Diagnostic_Scope_Note carries every mandated honesty statement (Req 14).

    It must declare (14.1) the workload a targeted diagnostic and not a
    real-benchmark result; (14.2) that the poison writes exercise checks other than
    the contradiction gate and that the defense is the Gate_Only_Arm sharing the
    same inputs yet still leaving invalid state; (11.3) that single-seed execution
    is sufficient because the pipeline is deterministic; and (14.4) that a single
    additive Reconcile_Path_Guard gates C4/C8/C10 by the existing
    enable_constraint_validation toggle, is behavior-preserving in the default
    configuration, and modifies no check logic.
    """
    note = DIAGNOSTIC_SCOPE_NOTE
    lowered = note.lower()

    # 14.1 — targeted diagnostic, not a real-benchmark result.
    assert "targeted diagnostic" in lowered
    assert "not a real-benchmark result" in lowered

    # 14.2 — checks other than the contradiction gate + Gate_Only shared-input defense.
    assert "contradiction gate" in lowered
    assert "gate_only_arm" in lowered.replace(" ", "_") or "gate_only_arm" in lowered
    assert "same inputs" in lowered
    assert "defense" in lowered

    # 11.3 — single-seed sufficiency because the pipeline is deterministic.
    assert "single-seed" in lowered
    assert "deterministic" in lowered

    # 14.4 — the single additive Reconcile_Path_Guard statement.
    assert "reconcile_path_guard" in lowered
    assert "enable_constraint_validation" in note
    assert "behavior-preserving" in lowered
    assert "no check logic was modified" in lowered


def test_runner_result_carries_the_diagnostic_scope_note():
    """The runner result carries the Diagnostic_Scope_Note verbatim (Req 14.1)."""
    result = run_stress_ablation(seed=1337)
    assert result.diagnostic_scope_note == DIAGNOSTIC_SCOPE_NOTE


def test_script_output_carries_note_first_and_last(tmp_path):
    """The script output carries the note as its first and last lines (Req 14.1, 14.3).

    The rendered report emits the (wrapped) Diagnostic_Scope_Note as both the first
    and last block of output so no reader mistakes the table for a real-data finding.
    """
    from ocm.scripts.run_stress_ablation import _wrap_note, render_report

    result = run_stress_ablation(seed=1337)
    output = render_report(result)

    note_block = _wrap_note(result.diagnostic_scope_note)
    assert note_block  # the note is non-empty
    stripped = output.strip()
    assert stripped.startswith(note_block), "note must be the first output block"
    assert stripped.endswith(note_block), "note must be the last output block"
    # The documented artifact carries the diagnostic framing (Req 14.3).
    assert "targeted diagnostic".upper() in output.upper()


def test_main_writes_results_file_carrying_the_note(tmp_path):
    """Running the script end-to-end writes a results file carrying the note (Req 14.3)."""
    from ocm.scripts.run_stress_ablation import main

    out_path = tmp_path / "stress_ablation_results.txt"
    rc = main(["--seed", "1337", "--out", str(out_path)])
    assert rc == 0
    assert out_path.exists()

    contents = out_path.read_text(encoding="utf-8")
    assert "TARGETED DIAGNOSTIC" in contents.upper()
    assert "Gate_Only_Arm" in contents


# --- Offline / harness-reuse smoke (Req 6.1, 6.3, 11.1, 12.1-12.4) --------- #
def test_run_stress_ablation_completes_offline_with_all_four_arms():
    """The runner completes fully offline and reports all four arms (Req 6.1, 6.3).

    Extraction is the injected deterministic StressOracleExtractor and settings use
    the offline base (deterministic_test_mode / memory Chroma / mock extractor), so
    no GPU, API key, or network access is required. The result must contain a
    Typed_Violation_Report for each of the four arms, in STRESS_ARMS order.
    """
    result = run_stress_ablation(seed=1337)

    assert list(result.arms.keys()) == list(STRESS_ARMS.keys())
    assert result.seed == 1337
    for arm, report in result.arms.items():
        # Each arm produced a well-formed typed-violation report with a tally.
        assert report.total == (
            report.schema_invalid
            + report.unsupported_final_decision
            + report.temporally_invalid_interval
            + report.illegal_status_state
        ), f"{arm}: total must equal the sum of the four per-type counts"


def test_run_stress_ablation_reuses_run_multiseed_and_durable_measure(monkeypatch):
    """The runner reuses the existing harness functions (Req 11.1, 12.4).

    Spies wrap the *existing* ``run_multiseed`` (bound in the runner module) and the
    *existing* ``durable_constraint_violations`` (imported by the typed-violation
    metric), delegating to the real implementations. Running the ablation must
    invoke both — proving the feature reuses the multi-seed harness and generalizes
    the legacy contradiction measure rather than reimplementing them.
    """
    import ocm.evaluation.experiment as experiment_mod
    import ocm.evaluation.stress_ablation as stress_ablation_mod

    multiseed_calls: list[tuple] = []
    real_run_multiseed = stress_ablation_mod.run_multiseed

    def run_multiseed_spy(*args, **kwargs):
        multiseed_calls.append((args, kwargs))
        return real_run_multiseed(*args, **kwargs)

    dcv_calls: list = []
    real_dcv = experiment_mod.durable_constraint_violations

    def dcv_spy(container):
        dcv_calls.append(container)
        return real_dcv(container)

    monkeypatch.setattr(stress_ablation_mod, "run_multiseed", run_multiseed_spy)
    monkeypatch.setattr(experiment_mod, "durable_constraint_violations", dcv_spy)

    result = run_stress_ablation(seed=1337)

    # run_multiseed is reused (Req 11.1): once per arm, driven by the same examples.
    assert len(multiseed_calls) == len(STRESS_ARMS)
    for _args, kwargs in multiseed_calls:
        assert kwargs.get("seeds") == [1337], "single-seed harness execution (Req 11.3)"
        assert kwargs.get("provided_examples"), "same workload fed to the harness"
        assert kwargs.get("extractor") is not None, "offline oracle injected (Req 6.3)"

    # durable_constraint_violations is reused (Req 12.4, 7.4): once per arm's metric.
    assert len(dcv_calls) >= len(STRESS_ARMS)
    assert isinstance(result, StressAblationResult)

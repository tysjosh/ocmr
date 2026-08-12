"""Property tests for the Schema/Provenance Stress Workload ablation.

Feature: ontology-constrained-memory.

This module hosts the property-based tests for the stress-workload diagnostic.
Each test carries its ``# Property {n}`` tag comment and the canonical
:func:`~ocm.tests.markers.pbt_property` docstring stamp so the design's
correctness properties map 1:1 onto the tests that validate them.

Currently implemented:

* **Property 1 — Full arm clean; gate-only strictly worse than
  schema+provenance** (task 5.2): for any generated workload the ``Full_Arm``
  and ``Schema_Provenance_Arm`` totals are ``0`` while the ``Gate_Only_Arm``
  total is strictly greater than the ``Schema_Provenance_Arm`` total — the
  decisive shared-input comparison (Req 5.4, 9.7, 9.8, 10.3, 13.2).
* **Property 2 — Ungoverned and gate-only leave invalid state** (task 5.3):
  for any generated workload both the ``Ungoverned_Arm`` total and the
  ``Gate_Only_Arm`` total are strictly greater than zero — the poison writes
  survive in durable memory when the schema/constraint checks are off, and the
  contradiction gate alone (fed the same inputs) cannot see them (Req 9.6, 10.2).
* **Property 7 — Generator determinism** (task 2.3): for any seed, invoking
  :func:`~ocm.evaluation.datasets.stress_workload.generate_stress_workload`
  twice produces an identical list of ``BenchmarkExample`` objects and an
  identical oracle ``writes_by_ref`` mapping. This is the reproducibility
  guarantee the offline single-seed evaluation relies on (Req 6.2, 13.4).
* **Property 8 — Runner determinism** (task 5.5): for any seed, running
  :func:`~ocm.evaluation.stress_ablation.run_stress_ablation` twice produces
  identical Typed_Violation_Reports for every arm — the reproducibility
  guarantee that lets the offline diagnostic rely on a single seed (Req 11.2).
* **Property 9 — Workload composition and labeling** (task 2.4): for any seed
  the workload contains ≥1 Valid_Write, ≥1 Poison_Write, ≥1 case of each of the
  four poison classes, and every example carries a valid ``WriteClass``
  (Req 5.2, 5.5, 9.9, 9.10).
* **Property 4 — Valid writes are admitted with zero violations** (task 5.7):
  for any generated workload, every Valid_Write is admitted by the ``Full_Arm``
  as an accepted outcome (with no rejected or quarantined outcome) and
  contributes zero Invalid_Active_State to the ``Full_Arm``'s
  Typed_Violation_Report (Req 5.1, 5.3, 13.3). This mirrors the runner's Full_Arm
  configuration (``STRESS_ARMS["Full_Arm"]`` + injected oracle) but replays only
  the cheap VALID cases, so it runs at the spec's ``MIN_PROPERTY_ITERATIONS``
  minimum rather than driving the full four-arm ``run_stress_ablation``.
* **Property 10 — The reconcile-path guard is default-preserving** (task 5.6):
  for any C4/C8/C10-governed status/decision write, replaying it under a
  default-governed container (``enable_constraint_validation=True``, the
  guard's non-taken branch) routes the offending ``HAS_STATUS`` to the
  quarantine bucket and keeps it out of the accepted store — the same
  non-accepted outcome the pre-Option-B reconcile path produced (Req 12.6,
  13.6, 15.2). This is the property-based partner to the ``test_stress_workload``
  task-1.2 example tests; it builds lightweight ``CoreContainer`` instances
  directly and replays only the EVIDENCE/STATUS poison writes, varying the
  workload by Hypothesis seed.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.evaluation.datasets.stress_workload import (
    StressCase,
    WriteClass,
    generate_stress_workload,
)
from ocm.evaluation.stress_ablation import STRESS_ARMS, run_stress_ablation
from ocm.evaluation.typed_violations import typed_violations
from ocm.memory.write_pipeline import HAS_STATUS, STATUS_VALUE_PREFIX
from ocm.tests.markers import MIN_PROPERTY_ITERATIONS, pbt_property

#: Reduced Hypothesis example count for the pipeline-touching property tests.
#: This covers both the expensive ablation-based properties (1, 2, 3, 8 — each
#: replays the full 4-arm pipeline plus ``run_multiseed`` via
#: ``run_stress_ablation`` once per example) and the container-replay properties
#: (4, 10 — each builds real ``CoreContainer`` instances and replays pipeline
#: sessions), so they are far costlier than the cheap generator/metric
#: properties. We deliberately run them below the spec's 100-example minimum
#: (``MIN_PROPERTY_ITERATIONS``) as a deliberate speed tradeoff: ``run_stress_ablation``
#: is deterministic per seed and the structural coverage these properties check
#: saturates after only a handful of distinct seeds.
_ABLATION_EXAMPLES = 5

#: The four poison classes (VALID is the benign class).
_POISON_CLASSES = (
    WriteClass.SCHEMA,
    WriteClass.TEMPORAL,
    WriteClass.EVIDENCE,
    WriteClass.STATUS,
)


# Property 7 — Generator determinism.
@pbt_property(7, "Generator determinism")
@settings(max_examples=MIN_PROPERTY_ITERATIONS)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_schema=st.integers(min_value=1, max_value=5),
    n_temporal=st.integers(min_value=1, max_value=5),
    n_evidence=st.integers(min_value=1, max_value=4),
    n_status=st.integers(min_value=1, max_value=4),
    n_valid=st.integers(min_value=1, max_value=6),
)
def test_generator_determinism(
    seed: int,
    n_schema: int,
    n_temporal: int,
    n_evidence: int,
    n_status: int,
    n_valid: int,
) -> None:
    """Two invocations with the same seed produce identical examples + writes_by_ref.

    Validates: Requirements 6.2, 13.4
    """
    counts = dict(
        n_schema=n_schema,
        n_temporal=n_temporal,
        n_evidence=n_evidence,
        n_status=n_status,
        n_valid=n_valid,
    )

    examples_a, oracle_a, cases_a = generate_stress_workload(seed, **counts)
    examples_b, oracle_b, cases_b = generate_stress_workload(seed, **counts)

    # Identical list of examples (BenchmarkExample is a Pydantic model → __eq__).
    assert examples_a == examples_b, (
        "generator produced a different example list for the same seed:\n"
        f"  seed={seed}, counts={counts}"
    )

    # Identical oracle writes_by_ref mapping (the oracle stores it as ._writes).
    assert oracle_a._writes == oracle_b._writes, (
        "generator produced a different writes_by_ref mapping for the same seed:\n"
        f"  seed={seed}, counts={counts}"
    )

    # The case list (labels + payloads keyed by the same refs) is identical too.
    assert cases_a == cases_b, (
        "generator produced a different case list for the same seed:\n"
        f"  seed={seed}, counts={counts}"
    )


# Property 9 — Workload composition and labeling.
@pbt_property(9, "Workload composition and labeling")
@settings(max_examples=MIN_PROPERTY_ITERATIONS)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_schema=st.integers(min_value=1, max_value=5),
    n_temporal=st.integers(min_value=1, max_value=5),
    n_evidence=st.integers(min_value=1, max_value=4),
    n_status=st.integers(min_value=1, max_value=4),
    n_valid=st.integers(min_value=1, max_value=6),
)
def test_workload_composition_and_labeling(
    seed: int,
    n_schema: int,
    n_temporal: int,
    n_evidence: int,
    n_status: int,
    n_valid: int,
) -> None:
    """Workload has >0 valid, >0 poison, >=1 of each poison class, valid labels.

    Validates: Requirements 5.2, 5.5, 9.9, 9.10
    """
    counts = dict(
        n_schema=n_schema,
        n_temporal=n_temporal,
        n_evidence=n_evidence,
        n_status=n_status,
        n_valid=n_valid,
    )

    examples, _oracle, cases = generate_stress_workload(seed, **counts)

    # Every example carries a valid WriteClass, both on the case label and on the
    # BenchmarkExample.category (which is the WriteClass value string) — Req 5.5.
    valid_values = {wc.value for wc in WriteClass}
    for case in cases:
        assert isinstance(case.write_class, WriteClass), (
            f"case {case.case_id} carries a non-WriteClass label: {case.write_class!r}"
        )
    for example in examples:
        assert example.category in valid_values, (
            f"example {example.id} category {example.category!r} is not a WriteClass value"
        )
        # The category must round-trip back to a WriteClass member.
        WriteClass(example.category)

    # Cases and examples correspond 1:1 with matching labels.
    assert len(cases) == len(examples), "case/example count mismatch"
    for case, example in zip(cases, examples):
        assert case.write_class.value == example.category, (
            f"label mismatch for {example.id}: case={case.write_class.value!r} "
            f"example={example.category!r}"
        )

    # Tally the write classes actually produced.
    produced = [case.write_class for case in cases]

    # >=1 Valid_Write (Req 5.2).
    valid_count = produced.count(WriteClass.VALID)
    assert valid_count > 0, "workload contains no Valid_Write"

    # >=1 Poison_Write overall (Req 5.2).
    poison_count = sum(produced.count(pc) for pc in _POISON_CLASSES)
    assert poison_count > 0, "workload contains no Poison_Write"

    # >=1 case of each of the four poison classes (Req 9.9, 9.10) so all four
    # per-type counts can be > 0 under the ungoverned/gate-only arms.
    for poison_class in _POISON_CLASSES:
        assert produced.count(poison_class) > 0, (
            f"workload is missing the {poison_class.value} poison class:\n"
            f"  seed={seed}, counts={counts}"
        )


# Property 1 — Full arm clean; gate-only strictly worse than schema+provenance.
@pbt_property(1, "Full arm clean; gate-only strictly worse than schema+provenance")
@settings(max_examples=_ABLATION_EXAMPLES, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_full_clean_gate_only_worse_than_schema_prov(seed: int) -> None:
    """Full/Schema+Prov totals are 0; gate-only total exceeds schema+prov.

    Validates: Requirements 5.4, 9.7, 9.8, 10.3, 13.2

    Each ``run_stress_ablation(seed=...)`` replays the full pipeline across all
    four arms (and drives ``run_multiseed`` for run-to-run identity), so this
    varies the workload by seed — the runner uses a fixed internal per-class
    workload size per seed and exposes no generation knobs, so seed variation is
    the realistic way to explore many distinct workloads through the real runner.
    """
    result = run_stress_ablation(seed=seed)

    full = result.arms["Full_Arm"]
    schema_prov = result.arms["Schema_Provenance_Arm"]
    gate_only = result.arms["Gate_Only_Arm"]

    # Full_Arm admits every valid write and removes every poison write (Req 5.4, 9.8).
    assert full.total == 0, (
        f"Full_Arm left Invalid_Active_State (total={full.total}) for seed={seed}: "
        f"{full}"
    )

    # Schema_Provenance_Arm also removes every poison write (Req 9.7).
    assert schema_prov.total == 0, (
        "Schema_Provenance_Arm left Invalid_Active_State "
        f"(total={schema_prov.total}) for seed={seed}: {schema_prov}"
    )

    # The decisive comparison: fed the SAME inputs, gate-only leaves invalid state
    # that schema+provenance removes, so its total is strictly greater (Req 10.3).
    assert gate_only.total > schema_prov.total, (
        "Gate_Only_Arm total is not strictly greater than Schema_Provenance_Arm "
        f"total for seed={seed}: gate_only={gate_only.total} "
        f"schema_prov={schema_prov.total}"
    )


# Property 2 — Ungoverned and gate-only leave invalid state.
@pbt_property(2, "Ungoverned and gate-only leave invalid state")
@settings(max_examples=_ABLATION_EXAMPLES, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_ungoverned_and_gate_only_leave_invalid_state(seed: int) -> None:
    """Ungoverned and gate-only arms leave a non-zero Invalid_Active_State total.

    Validates: Requirements 9.6, 10.2

    Each ``run_stress_ablation(seed=...)`` replays the full pipeline across all
    four arms, so this varies the workload by seed — the runner uses a fixed
    internal per-class workload size per seed and exposes no generation knobs,
    so seed variation is the realistic way to explore many distinct workloads
    through the real runner.
    """
    result = run_stress_ablation(seed=seed)

    ungoverned = result.arms["Ungoverned_Arm"]
    gate_only = result.arms["Gate_Only_Arm"]

    # With schema + constraint validation both off, every poison write is left
    # accepted in the durable store (Req 9.6).
    assert ungoverned.total > 0, (
        f"Ungoverned_Arm left no Invalid_Active_State (total={ungoverned.total}) "
        f"for seed={seed}: {ungoverned}"
    )

    # Fed the SAME inputs, the contradiction gate alone still cannot see these
    # non-contradiction poison writes, so gate-only also leaves invalid state
    # in durable memory (Req 10.2).
    assert gate_only.total > 0, (
        f"Gate_Only_Arm left no Invalid_Active_State (total={gate_only.total}) "
        f"for seed={seed}: {gate_only}"
    )


# Property 3 — All four poison classes are gate-invisible but constraint-caught.
@pbt_property(3, "All four poison classes are gate-invisible but constraint-caught")
@settings(max_examples=_ABLATION_EXAMPLES, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_all_four_poison_classes_gate_invisible_constraint_caught(seed: int) -> None:
    """Each of the four per-type counts is >0 under Gate_Only, 0 under Schema+Prov/Full.

    Validates: Requirements 1.4, 1.5, 1.6, 2.2, 2.3, 2.4, 3.3, 3.4, 3.5, 4.3, 4.4,
    4.5, 4.6, 9.9, 9.11, 9.12, 15.1, 15.5

    Each ``run_stress_ablation(seed=...)`` replays the full pipeline across all
    four arms with the SAME inputs, so this varies the workload by seed — the
    runner uses a fixed internal per-class workload size per seed and exposes no
    generation knobs, so seed variation is the realistic way to explore many
    distinct workloads through the real runner. The contradiction gate cannot see
    any of the four poison classes (they are not single-valued contradictions), so
    every per-type count survives in durable memory under Gate_Only; the
    schema/constraint checks catch all four, so every per-type count is 0 under
    Schema_Provenance and Full.
    """
    result = run_stress_ablation(seed=seed)

    gate_only = result.arms["Gate_Only_Arm"]
    schema_prov = result.arms["Schema_Provenance_Arm"]
    full = result.arms["Full_Arm"]

    # The four per-type violation fields on the Typed_Violation_Report.
    per_type_fields = (
        "schema_invalid",
        "unsupported_final_decision",
        "temporally_invalid_interval",
        "illegal_status_state",
    )

    for field_name in per_type_fields:
        gate_count = getattr(gate_only, field_name)
        schema_count = getattr(schema_prov, field_name)
        full_count = getattr(full, field_name)

        # Gate-invisible: fed the same inputs, the contradiction gate alone cannot
        # see this poison class, so it survives in durable memory (> 0).
        assert gate_count > 0, (
            f"Gate_Only_Arm.{field_name} is not > 0 for seed={seed}: "
            f"{gate_count} (report={gate_only})"
        )

        # Constraint-caught: the schema/constraint checks remove this poison class,
        # so it never enters the accepted store under either governed arm (== 0).
        assert schema_count == 0, (
            f"Schema_Provenance_Arm.{field_name} is not 0 for seed={seed}: "
            f"{schema_count} (report={schema_prov})"
        )
        assert full_count == 0, (
            f"Full_Arm.{field_name} is not 0 for seed={seed}: "
            f"{full_count} (report={full})"
        )


# Property 8 — Runner determinism.
@pbt_property(8, "Runner determinism")
@settings(max_examples=_ABLATION_EXAMPLES, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_runner_determinism(seed: int) -> None:
    """Running the runner twice with the same seed yields identical per-arm reports.

    Validates: Requirements 11.2

    Each ``run_stress_ablation(seed=...)`` replays the full pipeline across all
    four arms. The offline oracle pipeline is deterministic, so a second run with
    the same seed must reproduce the byte-identical Typed_Violation_Report for
    every arm — the four per-type counts, the total, the derived
    ``single_valued_contradictions`` measure, and the ``WriteOutcomeTally``. This
    run-to-run identity is what lets the diagnostic report from a single seed.
    Because ``TypedViolationReport`` and ``WriteOutcomeTally`` are dataclasses, the
    ``==`` comparison covers every field including the nested write-outcome tally.
    """
    result_a = run_stress_ablation(seed=seed)
    result_b = run_stress_ablation(seed=seed)

    # Both runs cover exactly the same arms.
    assert result_a.arms.keys() == result_b.arms.keys(), (
        "runner produced a different set of arms for the same seed:\n"
        f"  seed={seed}, a={sorted(result_a.arms)} b={sorted(result_b.arms)}"
    )

    for arm_name, report_a in result_a.arms.items():
        report_b = result_b.arms[arm_name]

        # Whole-report dataclass equality (covers the four per-type counts, total,
        # single_valued_contradictions, and the nested WriteOutcomeTally).
        assert report_a == report_b, (
            f"runner produced a different report for arm {arm_name!r} on a rerun:\n"
            f"  seed={seed}\n  first ={report_a}\n  second={report_b}"
        )

        # Belt-and-suspenders: spell out the per-field identity the task calls for
        # so a failure points at the exact diverging measure.
        assert report_a.schema_invalid == report_b.schema_invalid, arm_name
        assert (
            report_a.unsupported_final_decision
            == report_b.unsupported_final_decision
        ), arm_name
        assert (
            report_a.temporally_invalid_interval
            == report_b.temporally_invalid_interval
        ), arm_name
        assert report_a.illegal_status_state == report_b.illegal_status_state, arm_name
        assert report_a.total == report_b.total, arm_name
        assert (
            report_a.single_valued_contradictions
            == report_b.single_valued_contradictions
        ), arm_name
        assert report_a.write_outcomes == report_b.write_outcomes, arm_name


# --------------------------------------------------------------------------- #
# Property 10 — The reconcile-path guard is default-preserving.
#
# Direct-container helpers (mirroring test_stress_workload.py): build a default
# all-governance-on CoreContainer with the injected oracle and replay only the
# C4/C8/C10 EVIDENCE/STATUS poison writes. This is cheaper than driving the full
# 4-arm run_stress_ablation but still builds real containers and replays pipeline
# sessions, so this property runs at _ABLATION_EXAMPLES rather than the spec's
# MIN_PROPERTY_ITERATIONS minimum — a deliberate speed tradeoff.
# --------------------------------------------------------------------------- #
def _default_governed_container(oracle) -> CoreContainer:
    """A CoreContainer with the default all-governance-on toggles.

    ``enable_schema_validation`` / ``enable_constraint_validation`` /
    ``enable_contradiction_gate`` all default to ``True`` (full OCMR), so this is
    the default configuration in which the Reconcile_Path_Guard branch is NOT
    taken and the reconcile path must behave exactly like the pre-Option-B code.
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


@pbt_property(10, "The reconcile-path guard is default-preserving")
# Runs at _ABLATION_EXAMPLES (below the spec's MIN_PROPERTY_ITERATIONS) because it
# replays real pipeline containers — a deliberate speed tradeoff requested by the user.
@settings(max_examples=_ABLATION_EXAMPLES, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_reconcile_guard_is_default_preserving(seed: int) -> None:
    """Default (guard-off-branch) config quarantines every C4/C8/C10 poison write.

    Validates: Requirements 12.6, 13.6, 15.2

    For any seed, generate the C4/C8/C10-governed poison writes (the EVIDENCE and
    STATUS classes) and replay each under a default-governed container
    (``enable_constraint_validation=True`` — the guard's non-taken branch). Each
    offending ``HAS_STATUS`` (``final`` for C8, ``done`` for C4, ``todo`` for C10)
    must be routed to the quarantine bucket and must be absent from the accepted
    store, i.e. the same non-accepted outcome the pre-Option-B reconcile path
    produced. This proves the additive Reconcile_Path_Guard does not change the
    default-configuration C4/C8/C10 behavior. A fresh container is built per case
    so the state of one poison write cannot mask another's outcome.

    The per-class counts are kept small (one C8 EVIDENCE case, one C4 and one C10
    STATUS case) so the property replays only the cheap reconcile-path writes.
    Because it still builds real containers and replays pipeline sessions, it runs
    at ``_ABLATION_EXAMPLES`` (below the spec's ``MIN_PROPERTY_ITERATIONS``
    minimum) rather than driving the full four-arm ``run_stress_ablation`` — a
    deliberate speed tradeoff.
    """
    examples, oracle, cases = generate_stress_workload(
        seed,
        n_schema=1,
        n_temporal=1,
        n_evidence=1,
        n_status=2,
        n_valid=1,
    )

    evidence_cases = [c for c in cases if c.write_class is WriteClass.EVIDENCE]
    status_cases = [c for c in cases if c.write_class is WriteClass.STATUS]
    assert evidence_cases, f"no EVIDENCE (C8) case generated for seed={seed}"
    # The two STATUS cases cover C4 (done) and C10 (done->todo).
    assert any(_is_c4_case(c) for c in status_cases), (
        f"no C4 STATUS case generated for seed={seed}"
    )
    assert any(_is_c10_case(c) for c in status_cases), (
        f"no C10 STATUS case generated for seed={seed}"
    )

    # --- C8 (EVIDENCE): unsupported ``final`` Decision. --------------------- #
    for case in evidence_cases:
        example = _example_for(examples, case.case_id)
        container = _default_governed_container(oracle)
        results = _replay(container, example)
        assert "final" in _quarantined_has_status_values(results), (
            f"{case.case_id}: default config did not quarantine the unsupported "
            f"final decision (C8) for seed={seed}"
        )
        assert "final" not in _accepted_has_status_values(container), (
            f"{case.case_id}: default config accepted the unsupported final "
            f"decision (C8) for seed={seed}"
        )

    # --- C4 / C10 (STATUS): illegal ``done`` / ``done``->``todo`` writes. --- #
    for case in status_cases:
        poison_value = "todo" if _is_c10_case(case) else "done"
        example = _example_for(examples, case.case_id)
        container = _default_governed_container(oracle)
        results = _replay(container, example)
        assert poison_value in _quarantined_has_status_values(results), (
            f"{case.case_id}: default config did not quarantine the illegal "
            f"{poison_value!r} status for seed={seed}"
        )
        assert poison_value not in _accepted_has_status_values(container), (
            f"{case.case_id}: default config accepted the illegal {poison_value!r} "
            f"status for seed={seed}"
        )


# --------------------------------------------------------------------------- #
# Property 4 — Valid writes are admitted with zero violations.
#
# This mirrors the runner's Full_Arm exactly (STRESS_ARMS["Full_Arm"] toggle
# triple + injected oracle) but replays only the cheap VALID cases rather than
# driving the full four-arm run_stress_ablation. It still builds a real Full_Arm
# container and replays pipeline sessions, so it runs at _ABLATION_EXAMPLES rather
# than the spec's MIN_PROPERTY_ITERATIONS minimum — a deliberate speed tradeoff.
# Reuses the module's _replay / _example_for helpers (defined above for Property 10).
# --------------------------------------------------------------------------- #
def _full_arm_container(oracle) -> CoreContainer:
    """A CoreContainer configured with the Full_Arm toggle triple.

    Overlays the offline deterministic base (``deterministic_test_mode`` / memory
    Chroma) with ``STRESS_ARMS["Full_Arm"]`` (schema + constraint validation + the
    contradiction gate all on) and injects the oracle exactly as the runner wires
    it, so the container is the Full_Arm the runner builds.
    """
    return CoreContainer(
        Settings(
            deterministic_test_mode=True,
            chroma_mode="memory",
            **STRESS_ARMS["Full_Arm"],
        ),
        extractor=oracle,
    )


@pbt_property(4, "Valid writes are admitted with zero violations")
# Runs at _ABLATION_EXAMPLES (below the spec's MIN_PROPERTY_ITERATIONS) because it
# replays real pipeline containers — a deliberate speed tradeoff requested by the user.
@settings(max_examples=_ABLATION_EXAMPLES, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_valid=st.integers(min_value=1, max_value=6),
)
def test_valid_writes_admitted_with_zero_violations(seed: int, n_valid: int) -> None:
    """Every Valid_Write is accepted under the Full_Arm and adds zero violations.

    Validates: Requirements 5.1, 5.3, 13.3

    For any generated workload, replay only the VALID cases through a Full_Arm
    container (the runner's ``STRESS_ARMS["Full_Arm"]`` toggle triple with the
    deterministic oracle injected). Each Valid_Write must produce at least one
    accepted outcome and no rejected or quarantined outcome (Req 5.1, 5.3, 13.3),
    and the Full_Arm's Typed_Violation_Report over the resulting accepted store
    must have a zero total — a Valid_Write contributes no Invalid_Active_State.

    Replaying only the VALID cases (not all four arms) keeps this cheaper than the
    full runner, but it still builds a real Full_Arm container and replays pipeline
    sessions, so it runs at ``_ABLATION_EXAMPLES`` (below the spec's
    ``MIN_PROPERTY_ITERATIONS`` minimum) — a deliberate speed tradeoff. Keeping the
    poison per-class counts small while varying ``n_valid`` and the seed exercises
    many distinct valid-write workloads through the real Full_Arm pipeline.
    """
    examples, oracle, cases = generate_stress_workload(
        seed,
        n_schema=1,
        n_temporal=1,
        n_evidence=1,
        n_status=2,
        n_valid=n_valid,
    )

    valid_cases = [c for c in cases if c.write_class is WriteClass.VALID]
    assert valid_cases, f"no VALID cases generated for seed={seed}, n_valid={n_valid}"

    container = _full_arm_container(oracle)

    for case in valid_cases:
        example = _example_for(examples, case.case_id)
        results = _replay(container, example)

        accepted = sum(len(res.accepted) for res in results)
        rejected = sum(len(res.rejected) for res in results)
        quarantined = sum(len(res.quarantined) for res in results)

        # Admitted as an accepted outcome (Req 5.3, 13.3).
        assert accepted >= 1, (
            f"{case.case_id}: Valid_Write produced no accepted outcome under the "
            f"Full_Arm for seed={seed}"
        )
        # ...and never rejected or quarantined (a good system admits valid writes).
        assert rejected == 0, (
            f"{case.case_id}: Valid_Write was rejected under the Full_Arm for "
            f"seed={seed}"
        )
        assert quarantined == 0, (
            f"{case.case_id}: Valid_Write was quarantined under the Full_Arm for "
            f"seed={seed}"
        )

    # The Valid_Writes contribute zero Invalid_Active_State to the Full_Arm's
    # Typed_Violation_Report (Req 5.1, 5.4-scoped-to-valid, 13.3).
    report = typed_violations(container)
    assert report.total == 0, (
        f"Full_Arm Typed_Violation_Report total is not zero after replaying only "
        f"Valid_Writes for seed={seed}, n_valid={n_valid}: {report}"
    )

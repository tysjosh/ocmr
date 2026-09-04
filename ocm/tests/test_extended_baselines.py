"""Tests for the extended comparison baselines (Tier-B reviewer baselines).

* ``Brag``  — RAG-only: vectors-only retrieval, answer from text, no governance.
* ``Brtcf`` — retrieval-time contradiction filter: no write gate, filter at read.
* ``Bsup``  — supersession-only: latest Slot HAS_VALUE wins, no broader governance.

These verify the baselines build, that the canonical B-suite is unaffected, and
the headline contrast: a read-time filter can surface contradictions but cannot
prevent durable constraint violations — only the write-time gate (B3) does both.
"""

from __future__ import annotations

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.evaluation.arms import (
    baseline_settings_overrides,
    build_all_baselines,
    build_baseline,
)
from ocm.evaluation.experiment import (
    aggregate_methods,
    durable_constraint_violations,
    run_multiseed,
)
from ocm.evaluation.arms import MemoryStrategy


def test_build_all_baselines_excludes_extended():
    # The canonical helper stays B0–B4; extended baselines are opt-in.
    container = CoreContainer(Settings(deterministic_test_mode=True, chroma_mode="memory"))
    assert list(build_all_baselines(container)) == ["B0", "B1", "B2", "B3", "B4"]


def test_extended_baselines_build_by_name():
    container = CoreContainer(Settings(deterministic_test_mode=True, chroma_mode="memory"))
    for name in ("Brag", "Brtcf", "Bsup"):
        strat = build_baseline(name, container)
        assert isinstance(strat, MemoryStrategy)
        assert strat.name == name


def test_supersession_only_replaces_slot_has_value():
    from ocm.evaluation.datasets.longmemeval_adapter import (
        build_from_kupdate_oracle,
        sample_annotations,
        sample_instances,
    )

    settings = Settings(
        deterministic_test_mode=True,
        chroma_mode="memory",
        extractor="mock",
    ).model_copy(update=baseline_settings_overrides("Bsup"))
    examples, oracle = build_from_kupdate_oracle(sample_instances(), sample_annotations())
    container = CoreContainer(settings, extractor=oracle)
    strat = build_baseline("Bsup", container)

    outcomes = []
    for ex in examples:
        for session in ex.sessions:
            outcomes.append(strat.write(session.input, f"{ex.id}:{session.session_id}"))

    accepted: dict[str, set[str]] = {}
    for s, o, _k, _d in container.graph.find_edges_by_predicate("HAS_VALUE"):
        accepted.setdefault(s, set()).add(o)

    assert accepted and all(len(values) == 1 for values in accepted.values())
    assert any(outcome.summary.num_superseded > 0 for outcome in outcomes)
    assert durable_constraint_violations(container)[0] == 0


def test_supersession_only_does_not_replace_non_slot_conflicts():
    settings = Settings(
        deterministic_test_mode=True,
        chroma_mode="memory",
        extractor="mock",
    ).model_copy(update=baseline_settings_overrides("Bsup"))
    container = CoreContainer(settings)
    strat = build_baseline("Bsup", container)

    strat.write("Bob is assigned to Task T1.", "s1")
    result = strat.write("Carol is assigned to Task T1.", "s2")

    assert result.summary.num_superseded == 0
    assert result.summary.num_quarantined == 0
    assert durable_constraint_violations(container)[0] >= 1


def test_supersession_only_has_no_decision_evidence_floor():
    settings = Settings(
        deterministic_test_mode=True,
        chroma_mode="memory",
        extractor="mock",
    ).model_copy(update=baseline_settings_overrides("Bsup"))
    container = CoreContainer(settings)
    strat = build_baseline("Bsup", container)

    result = strat.write("We finalized the decision to cancel Project Atlas.", "s1")

    assert result.summary.num_quarantined == 0
    assert result.summary.num_accepted >= 1
    assert any(container.graph.get_entity_type(n) == "Decision" for n in container.graph.node_ids())


def test_rag_only_is_weaker_than_b0_without_structural_answers():
    # RAG-only drops graph-assisted answers, so on the structured benchmark it
    # recalls fewer expected tokens than B0 (which derives exact answers).
    ms = run_multiseed(["B0", "Brag"], seeds=(1337,), per_category=6)
    agg = aggregate_methods(ms)
    assert agg["Brag"]["task_success"].mean < agg["B0"]["task_success"].mean


def test_read_time_filter_surfaces_contradictions_but_not_durable_safety():
    # Brtcf vs B3 (full write-time governance) and B2 (no governance at all).
    ms = run_multiseed(["B2", "B3", "Brtcf"], seeds=(1337,), per_category=6)
    agg = aggregate_methods(ms)
    b2, b3, rt = agg["B2"], agg["B3"], agg["Brtcf"]
    # The read-time filter cuts the contradiction rate well below the ungoverned
    # baseline (it catches conflicts at answer time)...
    assert rt["contradiction_rate"].mean < b2["contradiction_rate"].mean
    # ...but cannot repair durable state: constraint violations stay at the
    # ungoverned level, whereas the write-time gate (B3) drives them to zero.
    assert rt["constraint_violations"].mean > b3["constraint_violations"].mean
    assert b3["constraint_violations"].mean == 0.0

"""Tests for the extended comparison baselines (Tier-B reviewer baselines).

* ``Brag``  — RAG-only: vectors-only retrieval, answer from text, no governance.
* ``Brtcf`` — retrieval-time contradiction filter: no write gate, filter at read.

These verify the baselines build, that the canonical B-suite is unaffected, and
the headline contrast: a read-time filter can surface contradictions but cannot
prevent durable constraint violations — only the write-time gate (B3) does both.
"""

from __future__ import annotations

from ocm.evaluation.baselines import build_all_baselines, build_baseline
from ocm.evaluation.experiment import aggregate_methods, run_multiseed
from ocm.evaluation.strategies import MemoryStrategy


def test_build_all_baselines_excludes_extended():
    # The canonical helper stays B0–B4; extended baselines are opt-in.
    from ocm.core.config import Settings
    from ocm.core.container import CoreContainer

    container = CoreContainer(Settings(deterministic_test_mode=True, chroma_mode="memory"))
    assert list(build_all_baselines(container)) == ["B0", "B1", "B2", "B3", "B4"]


def test_extended_baselines_build_by_name():
    from ocm.core.config import Settings
    from ocm.core.container import CoreContainer

    container = CoreContainer(Settings(deterministic_test_mode=True, chroma_mode="memory"))
    for name in ("Brag", "Brtcf"):
        strat = build_baseline(name, container)
        assert isinstance(strat, MemoryStrategy)
        assert strat.name == name


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

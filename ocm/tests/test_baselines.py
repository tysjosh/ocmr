"""Baseline strategy + B0–B4 construction/run tests (Req 22.1–22.5).

Drives every baseline (B0–B4) over a single deterministic, offline
``CoreContainer`` to confirm each one constructs from the registry and runs a
write + query with the ablation its toggles imply:

* **B0** (vectors only) returns semantic results and no symbolic exact match.
* **B1** (graph/symbolic only) returns the symbolic owner exact match and no
  semantic-only hit.
* **B2** (graph + semantic, no governance) returns both, but does not surface
  the T1 status conflict.
* **B3** (full governance) surfaces the quarantined T1 status conflict.
* **B4** (B3 + Answer_Policy) renders an answer string onto the package.

Requirements: 22.1, 22.2, 22.3, 22.4, 22.5.
"""

from __future__ import annotations

import pytest

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.evaluation.baselines import (
    BASELINE_REGISTRY,
    BASELINE_TOGGLES,
    build_all_baselines,
    build_baseline,
)
from ocm.evaluation.strategies import MemoryStrategy, StrategyToggles
from ocm.ontology.enums import TaskStatus
from ocm.retrieval.evidence_packager import EvidencePackage


@pytest.fixture
def fresh_container() -> CoreContainer:
    """A wired, deterministic, offline container per test (hermetic)."""
    settings = Settings(
        deterministic_test_mode=True, chroma_mode="memory", extractor="mock"
    )
    return CoreContainer(settings)


def _seed_t1_scenario(strategy: MemoryStrategy) -> None:
    """Write the three Task T1 sentences through the governed pipeline."""
    strategy.write("Alice owns Project Orion. Bob is assigned to Task T1.", "src-1")
    strategy.write("Bob completed Task T1.", "src-2")
    strategy.write("Task T1 is not started.", "src-3")


def _seed_assignment_conflict(strategy: MemoryStrategy) -> None:
    """Two conflicting single-valued ASSIGNED_TO writes (T1 → Bob, then Carol).

    The second write contradicts the first on the m:1 ``ASSIGNED_TO`` relation,
    so it is quarantined with ``conflicting_ids`` referencing the first
    (accepted) assertion — which makes the conflict surfaceable in the evidence
    package for governed baselines.
    """
    strategy.write("Bob is assigned to Task T1.", "src-1")
    strategy.write("Carol is assigned to Task T1.", "src-2")


# --------------------------------------------------------------------------- #
# Registry / construction
# --------------------------------------------------------------------------- #
def test_registry_defines_b0_through_b4():
    canonical = {"B0", "B1", "B2", "B3", "B4"}
    # The canonical B-suite is always present; extended comparison baselines
    # (RAG-only, retrieval-time filter, supersession-only) are opt-ins.
    assert canonical <= set(BASELINE_REGISTRY)
    assert canonical <= set(BASELINE_TOGGLES)
    assert {"Brag", "Brtcf", "Bsup"} <= set(BASELINE_REGISTRY)


def test_toggle_matrix_matches_design():
    # B0 — vectors only.
    b0 = BASELINE_TOGGLES["B0"]
    assert (b0.use_vectors, b0.use_graph, b0.use_ontology) == (True, False, False)
    assert not any(
        [b0.use_contradiction, b0.use_quarantine, b0.use_provenance, b0.use_answer_policy]
    )
    # B1 — graph/symbolic only.
    b1 = BASELINE_TOGGLES["B1"]
    assert (b1.use_graph, b1.use_vectors) == (True, False)
    # B2 — graph + semantic, no governance.
    b2 = BASELINE_TOGGLES["B2"]
    assert (b2.use_graph, b2.use_vectors) == (True, True)
    assert not any([b2.use_contradiction, b2.use_quarantine, b2.use_provenance])
    # B3 — full governance, no policy.
    b3 = BASELINE_TOGGLES["B3"]
    assert all(
        [b3.use_graph, b3.use_vectors, b3.use_contradiction, b3.use_quarantine, b3.use_provenance]
    )
    assert not b3.use_answer_policy
    # B4 — B3 + Answer_Policy.
    b4 = BASELINE_TOGGLES["B4"]
    assert b4.use_answer_policy and all(
        [b4.use_graph, b4.use_vectors, b4.use_contradiction, b4.use_quarantine, b4.use_provenance]
    )


def test_build_all_baselines_constructs(fresh_container):
    strategies = build_all_baselines(fresh_container)
    assert list(strategies) == ["B0", "B1", "B2", "B3", "B4"]
    for name, strat in strategies.items():
        assert isinstance(strat, MemoryStrategy)
        assert strat.name == name


def test_build_unknown_baseline_raises(fresh_container):
    with pytest.raises(KeyError):
        build_baseline("B9", fresh_container)


def test_toggle_aliases():
    t = StrategyToggles(use_graph=True, use_vectors=False, use_contradiction=True)
    assert t.use_symbolic is True
    assert t.use_semantic is False
    assert t.use_contradiction_gate is True


# --------------------------------------------------------------------------- #
# Per-baseline write + query behaviour
# --------------------------------------------------------------------------- #
def test_b0_vectors_only_returns_semantic_no_symbolic(fresh_container):
    b0 = build_baseline("B0", fresh_container)
    _seed_t1_scenario(b0)
    pkg = b0.query("What is the current status of Task T1?", top_k=10)
    assert isinstance(pkg, EvidencePackage)
    assert pkg.retrieved_items, "B0 should return semantic results"
    # Vectors only: no symbolic exact match should appear.
    assert not any(item.exact_match for item in pkg.retrieved_items)


def test_b1_symbolic_only_returns_symbolic_no_semantic(fresh_container):
    b1 = build_baseline("B1", fresh_container)
    _seed_t1_scenario(b1)
    # The T1 status query yields a symbolic ASSIGNED_TO exact match.
    pkg = b1.query("What is the current status of Task T1?", top_k=10)
    assert pkg.retrieved_items, "B1 should return symbolic results"
    # Graph-only: every hit is an exact symbolic match (no semantic-only hits).
    assert all(item.exact_match for item in pkg.retrieved_items)


def test_b2_hybrid_no_conflict_surfacing(fresh_container):
    b2 = build_baseline("B2", fresh_container)
    _seed_assignment_conflict(b2)
    pkg = b2.query("Is there a conflict about who is assigned to Task T1?", top_k=10)
    # No quarantine governance: conflicts are not surfaced as conflicts.
    assert pkg.conflicts == []
    # No provenance governance: no supporting sources attached.
    assert pkg.supporting_sources == []


def test_b3_surfaces_conflict(fresh_container):
    b3 = build_baseline("B3", fresh_container)
    _seed_assignment_conflict(b3)
    pkg = b3.query("Is there a conflict about who is assigned to Task T1?", top_k=10)
    # Full governance: the quarantined single-valued conflict is surfaced.
    assert pkg.conflicts, "B3 should surface the quarantined assignment conflict"


def test_b3_attaches_provenance(fresh_container):
    b3 = build_baseline("B3", fresh_container)
    b3.write("Bob is assigned to Task T1.", "src-1")
    pkg = b3.query("What is the current status of Task T1?", top_k=10)
    # Provenance governance attaches supporting sources for accepted evidence.
    assert pkg.supporting_assertions, "expected an accepted supporting assertion"
    assert pkg.supporting_sources, "B3 should attach provenance sources"


def test_b4_renders_answer_via_policy(fresh_container):
    b4 = build_baseline("B4", fresh_container)
    _seed_t1_scenario(b4)
    pkg = b4.query("What is the current status of Task T1?", top_k=10)
    # B4 layers the Answer_Policy: a rendered answer string is present.
    assert isinstance(pkg.answer, str) and pkg.answer.strip()


def test_governed_write_keeps_t1_done_across_baselines(fresh_container):
    """Writes are governed identically; T1 stays done after the conflicting write."""
    b3 = build_baseline("B3", fresh_container)
    _seed_t1_scenario(b3)
    graph = fresh_container.graph
    task_ids = [n for n in graph.node_ids() if graph.get_entity_type(n) == "Task"]
    assert len(task_ids) == 1
    assert graph.get_entity_payload(task_ids[0])["status"] == TaskStatus.done.value

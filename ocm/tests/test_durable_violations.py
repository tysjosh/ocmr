"""Durable-write constraint violation metric (paper §IV-B, Table IV/X).

The baselines differ in *write-time* governance (``baseline_settings_overrides``):
ungoverned arms (contradiction gate off) leave mutually-contradictory state
accepted in durable memory, while the full system gates it. These tests pin:

* the per-baseline write-time settings gradient (B0/B2 ungoverned, B3 full), and
* that ``durable_constraint_violations`` counts contradictory accepted state —
  zero for the governed arm, positive for an ungoverned one.

This is what makes the constraint-violation column separate baselines instead of
tying at zero (which it would if every arm shared the governed write path).
"""

from __future__ import annotations

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.evaluation.baselines import baseline_settings_overrides, build_baseline
from ocm.evaluation.experiment import durable_constraint_violations


def test_baseline_write_time_settings_gradient():
    assert baseline_settings_overrides("B0") == {
        "enable_schema_validation": False,
        "enable_contradiction_gate": False,
    }
    assert baseline_settings_overrides("B1")["enable_contradiction_gate"] is False
    assert baseline_settings_overrides("B3") == {
        "enable_schema_validation": True,
        "enable_contradiction_gate": True,
    }


def _ingest_contradiction(method: str):
    settings = Settings(
        deterministic_test_mode=True,
        chroma_mode="memory",
        extractor="mock",
    ).model_copy(update=baseline_settings_overrides(method))
    container = CoreContainer(settings)
    strat = build_baseline(method, container)
    # A single-valued (m:1) ASSIGNED_TO reassignment: T1 -> Bob, then T1 -> Carol.
    # This is gated by the C7 contradiction gate (unlike a Task *status* flip,
    # which C10 catches regardless), so it separates governed from ungoverned.
    strat.write("Bob is assigned to Task T1.", "s1")
    strat.write("Carol is assigned to Task T1.", "s2")
    return container


def test_full_ocmr_has_zero_durable_violations():
    """B3 gates the conflicting reassignment -> no contradictory accepted state."""
    container = _ingest_contradiction("B3")
    violations, _accepted = durable_constraint_violations(container)
    assert violations == 0


def test_ungoverned_baseline_accumulates_durable_violations():
    """B0 (contradiction gate off) leaves both assignees accepted (m:1 conflict)."""
    container = _ingest_contradiction("B0")
    violations, accepted = durable_constraint_violations(container)
    assert violations >= 1

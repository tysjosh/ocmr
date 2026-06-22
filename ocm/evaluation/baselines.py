"""Baseline definitions B0–B4 as toggle presets (Req 22.1–22.5).

Each baseline is the same :class:`~ocm.evaluation.strategies.MemoryStrategy`
differing only by its :class:`~ocm.evaluation.strategies.StrategyToggles`, per
the design's B0–B4 toggle matrix:

==== =========== ========= =========== ================= ============== ============== ================
Base use_ontology use_graph use_vectors use_contradiction use_quarantine use_provenance use_answer_policy
==== =========== ========= =========== ================= ============== ============== ================
B0   ✗            ✗         ✓           ✗                 ✗              ✗              ✗
B1   ✓            ✓         ✗           ✗                 ✗              ✗              ✗
B2   ✓            ✓         ✓           ✗                 ✗              ✗              ✗
B3   ✓            ✓         ✓           ✓                 ✓              ✓              ✗
B4   ✓            ✓         ✓           ✓                 ✓              ✓              ✓
==== =========== ========= =========== ================= ============== ============== ================

* **B0** — vector retrieval only (Req 22.1).
* **B1** — graph assertions + symbolic retrieval only, no vectors (Req 22.2).
* **B2** — graph + semantic, no contradiction/quarantine/provenance governance
  (Req 22.3).
* **B3** — full hybrid + contradiction + quarantine + provenance (Req 22.4).
* **B4** — B3 + the P1–P5 Answer_Policy (Req 22.5).
* **Bsup** — latest-value supersession for ``Slot -[HAS_VALUE]-> SlotValue``
  only; no broader schema/constraint/quarantine/provenance governance. This is
  an opt-in reviewer ablation, not part of the canonical B-suite.

The :data:`BASELINE_TOGGLES` map records each preset; :data:`BASELINE_REGISTRY`
maps a baseline name to a ``factory(container) -> MemoryStrategy``. Use
:func:`build_baseline` to construct one by name, or :func:`build_all_baselines`
to construct every baseline over a shared container.

Requirements: 22.1, 22.2, 22.3, 22.4, 22.5.
"""

from __future__ import annotations

from typing import Callable, Dict

from ocm.core.container import CoreContainer
from ocm.evaluation.strategies import MemoryStrategy, StrategyToggles

#: A factory that builds a configured strategy over a wired container.
BaselineFactory = Callable[[CoreContainer], MemoryStrategy]

#: The canonical B0–B4 toggle presets (the design's toggle matrix, Req 22).
BASELINE_TOGGLES: Dict[str, StrategyToggles] = {
    # B0 — vectors only (Req 22.1).
    "B0": StrategyToggles(
        use_ontology=False,
        use_graph=False,
        use_vectors=True,
        use_contradiction=False,
        use_quarantine=False,
        use_provenance=False,
        use_answer_policy=False,
    ),
    # B1 — graph / symbolic only, no vectors (Req 22.2).
    "B1": StrategyToggles(
        use_ontology=True,
        use_graph=True,
        use_vectors=False,
        use_contradiction=False,
        use_quarantine=False,
        use_provenance=False,
        use_answer_policy=False,
    ),
    # B2 — graph + semantic, no governance (Req 22.3).
    "B2": StrategyToggles(
        use_ontology=True,
        use_graph=True,
        use_vectors=True,
        use_contradiction=False,
        use_quarantine=False,
        use_provenance=False,
        use_answer_policy=False,
    ),
    # B3 — full hybrid + contradiction + quarantine + provenance (Req 22.4).
    "B3": StrategyToggles(
        use_ontology=True,
        use_graph=True,
        use_vectors=True,
        use_contradiction=True,
        use_quarantine=True,
        use_provenance=True,
        use_answer_policy=False,
    ),
    # B4 — B3 + Answer_Policy (Req 22.5).
    "B4": StrategyToggles(
        use_ontology=True,
        use_graph=True,
        use_vectors=True,
        use_contradiction=True,
        use_quarantine=True,
        use_provenance=True,
        use_answer_policy=True,
    ),
    # Brag — RAG-only: vectors-only similarity retrieval, answers read only from
    # retrieved text (no graph-assisted structural answer), no governance. A
    # vanilla retrieval-augmented baseline; distinct from B0, which is also
    # vectors-only but derives structural answers from the graph.
    "Brag": StrategyToggles(
        use_ontology=False,
        use_graph=False,
        use_vectors=True,
        use_contradiction=False,
        use_quarantine=False,
        use_provenance=False,
        use_answer_policy=False,
        use_structured_answer=False,
    ),
    # Brtcf — retrieval-time contradiction filter: full hybrid retrieval and NO
    # write-time governance (durable memory accumulates contradictions), but
    # contradictions are detected/filtered at query time. Isolates "filter at
    # read time" against OCMR's "gate at write time".
    "Brtcf": StrategyToggles(
        use_ontology=True,
        use_graph=True,
        use_vectors=True,
        use_contradiction=False,
        use_quarantine=False,
        use_provenance=False,
        use_answer_policy=False,
        use_read_time_filter=True,
    ),
    # Bsup — latest-value supersession only for Slot HAS_VALUE. Hybrid retrieval
    # is kept so it compares directly against B2/B3 on LongMemEval, but write
    # governance is intentionally reduced to the narrow same-slot overwrite rule.
    "Bsup": StrategyToggles(
        use_ontology=True,
        use_graph=True,
        use_vectors=True,
        use_contradiction=False,
        use_quarantine=False,
        use_provenance=False,
        use_answer_policy=False,
    ),
}

#: Human-readable description per baseline (used in metrics/reporting).
BASELINE_DESCRIPTIONS: Dict[str, str] = {
    "B0": "Vector retrieval only (Req 22.1)",
    "B1": "Graph assertions + symbolic only, no vectors (Req 22.2)",
    "B2": "Graph + semantic, no contradiction/quarantine/provenance (Req 22.3)",
    "B3": "Full hybrid + contradiction + quarantine + provenance (Req 22.4)",
    "B4": "B3 + Answer_Policy (Req 22.5)",
    "Brag": "RAG-only: vectors-only retrieval, answer from text, no governance",
    "Brtcf": "Retrieval-time contradiction filter: no write gate, filter at read",
    "Bsup": "Latest-value supersession only for Slot HAS_VALUE",
}

#: Per-baseline **write-time** governance settings overrides (paper §IV-A).
#: The baselines are not only retrieval ablations — they also differ in how much
#: write-time governance gates durable state, matching the paper's "text-only /
#: ontology-only / hybrid-no-governance / OCMR" gradient. Ungoverned baselines
#: leave constraint-violating (mutually contradictory) state in durable memory,
#: which is what the durable-write constraint-violation metric measures.
#:
#: ``enable_schema_validation`` gates W5 structural checks; ``enable_contradiction_gate``
#: gates W7/C7 (contradiction quarantining). C9 domain/range (W6) always runs.
BASELINE_SETTINGS: Dict[str, Dict[str, bool]] = {
    # text-only memory: no write-time governance.
    "B0": {"enable_schema_validation": False, "enable_contradiction_gate": False},
    # ontology-only memory: typed schema, but no contradiction gating.
    "B1": {"enable_schema_validation": True, "enable_contradiction_gate": False},
    # hybrid, no governance: neither write-time gate.
    "B2": {"enable_schema_validation": False, "enable_contradiction_gate": False},
    # full OCMR: all write-time governance on.
    "B3": {"enable_schema_validation": True, "enable_contradiction_gate": True},
    "B4": {"enable_schema_validation": True, "enable_contradiction_gate": True},
    # RAG-only: no write-time governance (text-only memory behaviour).
    "Brag": {"enable_schema_validation": False, "enable_contradiction_gate": False},
    # Retrieval-time filter: typed schema on, but write-time contradiction gate
    # OFF so conflicts accumulate durably and are caught only at read time.
    "Brtcf": {"enable_schema_validation": True, "enable_contradiction_gate": False},
    # Supersession-only reviewer ablation: no typed schema, no W6/C7 governance,
    # only the narrow Slot HAS_VALUE latest-value overwrite rule.
    "Bsup": {
        "enable_schema_validation": False,
        "enable_constraint_validation": False,
        "enable_contradiction_gate": False,
        "supersession_only_has_value": True,
    },
}


def baseline_settings_overrides(name: str) -> Dict[str, bool]:
    """Return the write-time ``Settings`` overrides for baseline ``name`` (may be empty)."""
    return dict(BASELINE_SETTINGS.get(name, {}))


def _make_factory(name: str, toggles: StrategyToggles) -> BaselineFactory:
    """Build a factory closure that constructs baseline ``name``."""

    def factory(container: CoreContainer) -> MemoryStrategy:
        # B4 renders decision-support answers; treat it as high-stakes so the
        # Answer_Policy includes provenance (P4).
        high_stakes = toggles.use_answer_policy
        return MemoryStrategy(name, container, toggles, high_stakes=high_stakes)

    factory.__name__ = f"build_{name}"
    factory.__doc__ = f"Build the {name} baseline: {BASELINE_DESCRIPTIONS.get(name, name)}."
    return factory


#: Registry mapping a baseline name to its ``factory(container) -> MemoryStrategy``.
BASELINE_REGISTRY: Dict[str, BaselineFactory] = {
    name: _make_factory(name, toggles) for name, toggles in BASELINE_TOGGLES.items()
}

#: Baselines the Baseline_Runner executes against the benchmark by default
#: (B0–B3); B4 layers the Answer_Policy on B3 for the answer-quality comparison.
DEFAULT_RUN_BASELINES: tuple[str, ...] = ("B0", "B1", "B2", "B3")

#: The canonical B-suite (the design's toggle matrix). Extended comparison
#: baselines (e.g. ``Brag``, ``Brtcf``, ``Bsup``) live in the registry but are opt-in via
#: an explicit ``baselines=`` list, so they never alter the canonical suite.
CANONICAL_BASELINES: tuple[str, ...] = ("B0", "B1", "B2", "B3", "B4")


def build_baseline(name: str, container: CoreContainer) -> MemoryStrategy:
    """Construct a single baseline strategy by name over ``container``.

    Args:
        name: One of ``"B0"`` … ``"B4"``.
        container: The wired :class:`CoreContainer` to back the strategy.

    Returns:
        A configured :class:`MemoryStrategy`.

    Raises:
        KeyError: If ``name`` is not a known baseline.
    """
    try:
        factory = BASELINE_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(BASELINE_REGISTRY))
        raise KeyError(f"Unknown baseline {name!r}; known baselines: {known}") from exc
    return factory(container)


def build_all_baselines(container: CoreContainer) -> Dict[str, MemoryStrategy]:
    """Construct every canonical baseline (B0–B4) over a shared ``container``.

    Returns a name → :class:`MemoryStrategy` map in B0…B4 order. Extended
    comparison baselines (``Brag``, ``Brtcf``) are excluded here and built
    explicitly by name via :func:`build_baseline` when needed.
    """
    return {name: build_baseline(name, container) for name in CANONICAL_BASELINES}

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
}

#: Human-readable description per baseline (used in metrics/reporting).
BASELINE_DESCRIPTIONS: Dict[str, str] = {
    "B0": "Vector retrieval only (Req 22.1)",
    "B1": "Graph assertions + symbolic only, no vectors (Req 22.2)",
    "B2": "Graph + semantic, no contradiction/quarantine/provenance (Req 22.3)",
    "B3": "Full hybrid + contradiction + quarantine + provenance (Req 22.4)",
    "B4": "B3 + Answer_Policy (Req 22.5)",
}


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
    """Construct every baseline (B0–B4) over a shared ``container``.

    Returns a name → :class:`MemoryStrategy` map in B0…B4 order.
    """
    return {name: build_baseline(name, container) for name in BASELINE_TOGGLES}

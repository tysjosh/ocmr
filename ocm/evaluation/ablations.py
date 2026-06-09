"""Mechanism ablations for the evaluation harness (paper §IV-D, Table X).

The paper evaluates full OCMR against four mechanism-critical ablations:

* **w/o typed schema** — disable the W5 typed-schema validation gate.
* **w/o contradiction gate** — disable the W7/C7 contradiction gate *and* the
  quarantine routing/surfacing (contradictions flow into memory ungoverned).
* **w/o provenance scoring** — drop provenance from reranking and the answer
  package.
* **w/o hybrid routing** — collapse to a single retrieval channel (semantic
  only), removing symbolic/semantic hybrid routing.

Unlike the B0–B4 baselines (which differ only at retrieval composition), two of
these ablations are *write-time* governance changes, realized through
``Settings`` switches (``enable_schema_validation`` / ``enable_contradiction_gate``).
Each ablation therefore builds its **own** container with overridden settings so
the governed write path itself changes, then drives retrieval with matching
:class:`StrategyToggles`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.evaluation.strategies import MemoryStrategy, StrategyToggles


@dataclass(frozen=True)
class AblationSpec:
    """One experimental arm: write-time setting overrides + retrieval toggles."""

    name: str
    description: str
    settings_overrides: dict[str, Any] = field(default_factory=dict)
    toggles: StrategyToggles = field(default_factory=StrategyToggles)


#: Full OCMR uses every governance mechanism (the B3 configuration).
_FULL_TOGGLES = StrategyToggles(
    use_ontology=True,
    use_graph=True,
    use_vectors=True,
    use_contradiction=True,
    use_quarantine=True,
    use_provenance=True,
    use_answer_policy=False,
)


ABLATIONS: dict[str, AblationSpec] = {
    "full": AblationSpec(
        name="full",
        description="Full OCMR (all governance mechanisms enabled)",
        settings_overrides={},
        toggles=_FULL_TOGGLES,
    ),
    "no_schema": AblationSpec(
        name="no_schema",
        description="w/o typed schema constraints (W5 disabled)",
        settings_overrides={"enable_schema_validation": False},
        toggles=_FULL_TOGGLES,
    ),
    "no_contradiction_gate": AblationSpec(
        name="no_contradiction_gate",
        description="w/o contradiction gating / quarantine routing (W7/C7 disabled)",
        settings_overrides={"enable_contradiction_gate": False},
        toggles=StrategyToggles(
            use_ontology=True,
            use_graph=True,
            use_vectors=True,
            use_contradiction=False,
            use_quarantine=False,
            use_provenance=True,
            use_answer_policy=False,
        ),
    ),
    "no_provenance": AblationSpec(
        name="no_provenance",
        description="w/o provenance scoring in reranking / answer policy",
        settings_overrides={},
        toggles=StrategyToggles(
            use_ontology=True,
            use_graph=True,
            use_vectors=True,
            use_contradiction=True,
            use_quarantine=True,
            use_provenance=False,
            use_answer_policy=False,
        ),
    ),
    "no_hybrid": AblationSpec(
        name="no_hybrid",
        description="w/o hybrid symbolic-semantic routing (semantic channel only)",
        settings_overrides={},
        toggles=StrategyToggles(
            use_ontology=True,
            use_graph=False,  # collapse to a single (semantic) channel
            use_vectors=True,
            use_contradiction=True,
            use_quarantine=True,
            use_provenance=True,
            use_answer_policy=False,
        ),
    ),
}

#: The ablation arms reported in Table X (full first, then the four removals).
DEFAULT_ABLATIONS: tuple[str, ...] = (
    "full",
    "no_schema",
    "no_contradiction_gate",
    "no_provenance",
    "no_hybrid",
)


def build_ablation_strategy(
    name: str, settings_factory: Callable[[], Settings]
) -> MemoryStrategy:
    """Build a :class:`MemoryStrategy` for ablation ``name``.

    Constructs a dedicated :class:`CoreContainer` whose :class:`Settings` carry
    the ablation's write-time governance overrides, then wires a strategy with
    the ablation's retrieval toggles.
    """
    if name not in ABLATIONS:
        raise KeyError(f"unknown ablation {name!r}; known: {sorted(ABLATIONS)}")
    spec = ABLATIONS[name]
    settings = settings_factory().model_copy(update=spec.settings_overrides)
    container = CoreContainer(settings)
    return MemoryStrategy(spec.name, container, spec.toggles)

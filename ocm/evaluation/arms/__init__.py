"""Experimental arms: every comparison condition in one registry.

An **arm** is one experimental condition the harness can run. Historically the
arms lived in three unrelated modules and were dispatched by a name-prefix
convention (``method.startswith("B")`` selected a baseline, anything else fell
through to the ablations), which meant the arm namespaces were separated only by
a single letter and there was no single place to enumerate what could be run.

This package gathers the three families and gives them one explicit registry:

============  ==========================================  =====================================
Family        Arms                                        Module
============  ==========================================  =====================================
``baseline``  B0-B4, Brag, Brtcf, Bsup, Bmemgpt           :mod:`ocm.evaluation.arms.baselines`
``ablation``  full, no_schema, no_contradiction_gate,     :mod:`ocm.evaluation.arms.ablations`
              no_provenance, no_hybrid
``stress``    Ungoverned_Arm, Gate_Only_Arm,              :mod:`ocm.evaluation.arms.stress`
              Schema_Provenance_Arm, Full_Arm
============  ==========================================  =====================================

All three families are the *same*
:class:`~ocm.evaluation.arms.strategies.MemoryStrategy`; they differ only in
their retrieval :class:`~ocm.evaluation.arms.strategies.StrategyToggles` and
their write-time ``Settings`` overrides. What distinguishes a family is the
research question it answers:

* **baselines** — competing memory *designs* (text-only, ontology-only,
  hybrid-without-governance, RAG, MemGPT-style, ...).
* **ablations** — mechanism *removals* from full OCMR (paper Table X).
* **stress arms** — governance toggle triples for the targeted
  schema/provenance diagnostic.

Use :func:`build_arm` to construct any arm by name, and :func:`known_arms` to
enumerate them. Family-specific helpers (:func:`build_baseline`,
:func:`build_ablation_strategy`, :func:`build_stress_arm`) remain available for
callers that already know which family they want.

The RAHGM governance conditions (C1-C5) are deliberately **not** here: they are a
different paper's human-review routers, they live in
:mod:`ocm.governance.conditions`, and they are driven by
:mod:`ocm.evaluation.rahgm.replay` rather than by the OCMR arm harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.evaluation.arms.ablations import (
    ABLATIONS,
    DEFAULT_ABLATIONS,
    AblationSpec,
    build_ablation_strategy,
)
from ocm.evaluation.arms.baselines import (
    BASELINE_DESCRIPTIONS,
    BASELINE_REGISTRY,
    BASELINE_SETTINGS,
    BASELINE_TOGGLES,
    CANONICAL_BASELINES,
    DEFAULT_RUN_BASELINES,
    BaselineFactory,
    baseline_settings_overrides,
    build_all_baselines,
    build_baseline,
)
from ocm.evaluation.arms.strategies import MemoryStrategy, StrategyToggles
from ocm.evaluation.arms.stress import (
    DECISIVE_ARM,
    STRESS_ARM_DESCRIPTIONS,
    STRESS_ARMS,
    build_stress_arm,
    stress_arm_settings_factory,
)

__all__ = [
    # Registry
    "ArmSpec",
    "ARM_REGISTRY",
    "FAMILY_BASELINE",
    "FAMILY_ABLATION",
    "FAMILY_STRESS",
    "build_arm",
    "known_arms",
    "arm_spec",
    "arm_family",
    # Shared implementation
    "MemoryStrategy",
    "StrategyToggles",
    # Baselines
    "BASELINE_TOGGLES",
    "BASELINE_SETTINGS",
    "BASELINE_DESCRIPTIONS",
    "BASELINE_REGISTRY",
    "BaselineFactory",
    "DEFAULT_RUN_BASELINES",
    "CANONICAL_BASELINES",
    "baseline_settings_overrides",
    "build_baseline",
    "build_all_baselines",
    # Ablations
    "ABLATIONS",
    "DEFAULT_ABLATIONS",
    "AblationSpec",
    "build_ablation_strategy",
    # Stress arms
    "STRESS_ARMS",
    "STRESS_ARM_DESCRIPTIONS",
    "DECISIVE_ARM",
    "stress_arm_settings_factory",
    "build_stress_arm",
]


#: Family labels.
FAMILY_BASELINE = "baseline"
FAMILY_ABLATION = "ablation"
FAMILY_STRESS = "stress"


@dataclass(frozen=True)
class ArmSpec:
    """One registered experimental arm.

    * ``name`` — the arm identifier used by the harness and in checkpoint keys.
    * ``family`` — one of :data:`FAMILY_BASELINE`, :data:`FAMILY_ABLATION`,
      :data:`FAMILY_STRESS`.
    * ``description`` — human-readable summary for reports and error messages.
    * ``builder`` — ``builder(name, settings_factory, *, extractor, embeddings)``
      returning a configured :class:`MemoryStrategy`.
    """

    name: str
    family: str
    description: str
    builder: Callable[..., MemoryStrategy]


ArmBuilder = Callable[..., MemoryStrategy]


def _build_baseline_arm(
    name: str,
    settings_factory: Callable[[], Settings],
    *,
    extractor: object | None = None,
    embeddings: object | None = None,
) -> MemoryStrategy:
    """Build baseline ``name`` with its write-time ``Settings`` overrides applied."""
    settings = settings_factory().model_copy(update=baseline_settings_overrides(name))
    container = CoreContainer(settings, extractor=extractor, embeddings=embeddings)
    return build_baseline(name, container)


def _build_ablation_arm(
    name: str,
    settings_factory: Callable[[], Settings],
    *,
    extractor: object | None = None,
    embeddings: object | None = None,
) -> MemoryStrategy:
    """Build ablation ``name`` (delegates to :func:`build_ablation_strategy`)."""
    return build_ablation_strategy(
        name, settings_factory, extractor=extractor, embeddings=embeddings
    )


def _build_registry() -> Dict[str, ArmSpec]:
    """Assemble the arm registry, rejecting cross-family name collisions.

    A duplicate name across families would make :func:`build_arm` ambiguous and
    would silently alias two different experimental conditions onto one
    checkpoint key, so it fails loudly at import time instead.
    """
    registry: Dict[str, ArmSpec] = {}

    def register(name: str, family: str, description: str, builder: ArmBuilder) -> None:
        if name in registry:
            raise ValueError(
                f"duplicate arm name {name!r}: already registered in family "
                f"{registry[name].family!r}, cannot re-register in {family!r}"
            )
        registry[name] = ArmSpec(
            name=name, family=family, description=description, builder=builder
        )

    for name in BASELINE_TOGGLES:
        register(
            name,
            FAMILY_BASELINE,
            BASELINE_DESCRIPTIONS.get(name, name),
            _build_baseline_arm,
        )
    for name, spec in ABLATIONS.items():
        register(name, FAMILY_ABLATION, spec.description, _build_ablation_arm)
    for name in STRESS_ARMS:
        register(
            name,
            FAMILY_STRESS,
            STRESS_ARM_DESCRIPTIONS.get(name, name),
            build_stress_arm,
        )
    return registry


#: Every runnable arm, keyed by name, across all three families.
ARM_REGISTRY: Dict[str, ArmSpec] = _build_registry()


def arm_spec(name: str) -> ArmSpec:
    """Return the :class:`ArmSpec` for ``name``.

    Raises:
        KeyError: If ``name`` is not a registered arm. The message lists the
            known arms grouped by family, which is the failure mode the old
            ``startswith("B")`` dispatch could not report (an unknown ``B*``
            name raised from deep inside the baseline registry, and an unknown
            non-``B`` name raised from the ablation table).
    """
    try:
        return ARM_REGISTRY[name]
    except KeyError as exc:
        families: Dict[str, list[str]] = {}
        for spec in ARM_REGISTRY.values():
            families.setdefault(spec.family, []).append(spec.name)
        known = "; ".join(
            f"{family}: {', '.join(sorted(names))}"
            for family, names in sorted(families.items())
        )
        raise KeyError(f"unknown arm {name!r}; known arms — {known}") from exc


def arm_family(name: str) -> str:
    """Return the family (``baseline`` / ``ablation`` / ``stress``) of ``name``."""
    return arm_spec(name).family


def known_arms(family: Optional[str] = None) -> Tuple[str, ...]:
    """Return every registered arm name, optionally filtered to one ``family``.

    Registration order is preserved (baselines, then ablations, then stress arms)
    so listings read the same way the tables in the paper do.
    """
    return tuple(
        name
        for name, spec in ARM_REGISTRY.items()
        if family is None or spec.family == family
    )


def build_arm(
    name: str,
    settings_factory: Callable[[], Settings],
    *,
    extractor: object | None = None,
    embeddings: object | None = None,
) -> MemoryStrategy:
    """Build the :class:`MemoryStrategy` for arm ``name``, whatever its family.

    Replaces the old name-prefix dispatch with an explicit registry lookup, so
    adding an arm is a registration rather than a naming convention, and an
    unknown name fails with the full list of what *is* runnable.

    A shared ``extractor`` / ``embeddings`` (loaded once) is injected into the
    arm's container so a heavy model is not reloaded per arm.

    Raises:
        KeyError: If ``name`` is not a registered arm.
    """
    spec = arm_spec(name)
    return spec.builder(
        name, settings_factory, extractor=extractor, embeddings=embeddings
    )

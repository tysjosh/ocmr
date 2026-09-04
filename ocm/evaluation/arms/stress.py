"""Stress-diagnostic governance arms (Req 9.1-9.5).

The **arm definitions** for the Schema/Provenance stress diagnostic. Each arm is
a triple of the three *existing* ``Settings`` governance toggles applied via
``Settings.model_copy(update=...)`` — the same mechanism
:class:`~ocm.evaluation.arms.ablations.AblationSpec` already uses — so **no new
toggle** and **no new pipeline governance code** is introduced (Req 9.5, 12.2).

======================  ======  ======  =====
Arm                     W5      W6      C7
======================  ======  ======  =====
Ungoverned_Arm (B2)     off     off     off
Gate_Only_Arm           off     off     on
Schema_Provenance_Arm   on      on      off
Full_Arm (B3)           on      on      on
======================  ======  ======  =====

where ``W5 = enable_schema_validation``, ``W6 = enable_constraint_validation``,
``C7 = enable_contradiction_gate``.

This module holds only the arm *definitions* (the toggle triples, the decisive
arm, and the settings-factory / strategy builders). The workload replay, the
typed-violation reporting, and the ``DIAGNOSTIC_SCOPE_NOTE`` honesty statement
live in :mod:`ocm.evaluation.stress_ablation`, which drives these arms. Keeping
the definitions here (dependency-free w.r.t. the runners) is what lets
:mod:`ocm.evaluation.arms` register them without importing
:mod:`ocm.evaluation.experiment`.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 12.2.
"""

from __future__ import annotations

from typing import Callable, Dict

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.evaluation.arms.ablations import ABLATIONS
from ocm.evaluation.arms.strategies import MemoryStrategy

__all__ = [
    "STRESS_ARMS",
    "STRESS_ARM_DESCRIPTIONS",
    "DECISIVE_ARM",
    "stress_arm_settings_factory",
    "build_stress_arm",
]


#: The four ablation arms as triples of the existing ``Settings`` governance
#: toggles ``enable_schema_validation`` (W5) / ``enable_constraint_validation``
#: (W6, containing C1-C10 incl. C9/C2 and the C7 gate) / ``enable_contradiction_gate``
#: (C7). Applied via ``Settings.model_copy(update=STRESS_ARMS[arm])`` so each arm is
#: configured *exclusively* through existing toggles (Req 9.5, 12.2).
STRESS_ARMS: Dict[str, Dict[str, bool]] = {
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

#: Human-readable description per stress arm (used in registry listings).
STRESS_ARM_DESCRIPTIONS: Dict[str, str] = {
    "Ungoverned_Arm": "W5/W6/C7 all off (B2-equivalent governance)",
    "Gate_Only_Arm": "contradiction gate only; no schema/constraint checks (decisive row)",
    "Schema_Provenance_Arm": "schema + constraint checks, contradiction gate off",
    "Full_Arm": "all write-time governance on (B3-equivalent)",
}

#: The decisive comparison row (Req 10.4): fed the same inputs as every arm, it
#: still leaves the invalid durable state the Schema_Provenance_Arm removes.
DECISIVE_ARM: str = "Gate_Only_Arm"


def stress_arm_settings_factory(
    base_factory: Callable[[], Settings], arm: str
) -> Callable[[], Settings]:
    """Return a settings factory that applies ``arm``'s toggle triple.

    The arm overrides are applied on top of ``base_factory()`` via
    ``model_copy(update=...)`` so ``run_multiseed`` can drive the arm through the
    existing harness without a ``B*`` baseline override masking the arm toggles.
    """
    overrides = STRESS_ARMS[arm]

    def factory() -> Settings:
        return base_factory().model_copy(update=overrides)

    return factory


def build_stress_arm(
    name: str,
    settings_factory: Callable[[], Settings],
    *,
    extractor: object | None = None,
    embeddings: object | None = None,
) -> MemoryStrategy:
    """Build a :class:`MemoryStrategy` for stress arm ``name``.

    Retrieval composition is held fixed at the full-OCMR toggles (the same ones
    the ``"full"`` ablation uses) so the arms differ *only* in write-time
    governance — which is the whole point of the diagnostic. A dedicated
    :class:`CoreContainer` carries the arm's toggle triple.

    Raises:
        KeyError: If ``name`` is not a known stress arm.
    """
    if name not in STRESS_ARMS:
        raise KeyError(f"unknown stress arm {name!r}; known: {sorted(STRESS_ARMS)}")
    settings = settings_factory().model_copy(update=STRESS_ARMS[name])
    container = CoreContainer(settings, extractor=extractor, embeddings=embeddings)
    return MemoryStrategy(name, container, ABLATIONS["full"].toggles)

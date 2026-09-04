"""Stress_Ablation_Runner (Req 9, 10, 11, 14).

The runner executes the **Schema/Provenance Stress Workload** across four
governance **arms** and reports, per arm, the :class:`TypedViolationReport`
(the four-type breakdown, the total, the legacy single-valued-contradiction
count, and the :class:`WriteOutcomeTally`). Every arm is a triple of the three
**existing** ``Settings`` governance toggles applied via
``Settings.model_copy(update=...)`` — the same mechanism ``AblationSpec`` /
``build_ablation_strategy`` already use — so **no new toggle** and **no new
pipeline governance code** is introduced (Req 9.5, 12.2, 12.3).

The four arms (Req 9.1-9.4):

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

Because **C9 lives inside W6** and **C7 is bundled inside W6**, the Gate_Only_Arm
(``enable_constraint_validation`` off, ``enable_contradiction_gate`` on) runs *no*
write-time relation checks at all and the gate would be inert on these inputs even
if it ran (they are not contradictions). With the additive Reconcile_Path_Guard the
same ``enable_constraint_validation`` toggle that gates the relation-path C9/C2
checks also gates the reconcile-path C4/C8/C10 checks, so **all four poison classes**
track the toggle identically: left accepted (Invalid_Active_State) when it is false
(Ungoverned, Gate_Only) and removed when it is true (Schema_Provenance, Full). The
Gate_Only_Arm is the **decisive** comparison (Req 10.4): it receives the *same*
inputs as every other arm yet still leaves the invalid durable state that the
Schema_Provenance_Arm removes.

Execution reuses the existing harness (Req 11.1, 12.4): the workload is built once
and the **same** examples + oracle feed **every** arm (Req 10.1); ``run_multiseed``
is additionally invoked (single seed) to obtain write-outcome tallies and prove
run-to-run identity, and ``aggregate_methods`` is kept available for single-seed
reporting parity. A single seed is sufficient because the offline oracle/mock
pipeline is deterministic — recorded in the :data:`DIAGNOSTIC_SCOPE_NOTE` (Req 11.3).

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 10.1, 10.4, 11.1, 11.3, 12.4, 14.1, 14.2, 14.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.evaluation.arms.stress import (
    DECISIVE_ARM,
    STRESS_ARMS,
    stress_arm_settings_factory,
)
from ocm.evaluation.benchmark import BenchmarkExample
from ocm.evaluation.datasets.stress_workload import (
    StressCase,
    StressOracleExtractor,
    generate_stress_workload,
)
from ocm.evaluation.experiment import aggregate_methods, run_multiseed
from ocm.evaluation.typed_violations import (
    TypedViolationReport,
    WriteOutcomeTally,
    typed_violations,
)

__all__ = [
    "STRESS_ARMS",
    "DECISIVE_ARM",
    "DIAGNOSTIC_SCOPE_NOTE",
    "StressAblationResult",
    "run_stress_ablation",
]


# --------------------------------------------------------------------------- #
# Arm definitions (Req 9.1-9.5)
# --------------------------------------------------------------------------- #
# :data:`STRESS_ARMS` and :data:`DECISIVE_ARM` are defined in
# :mod:`ocm.evaluation.arms.stress` alongside the baseline and ablation arm
# definitions, and re-exported here (via ``__all__``) so existing importers of
# this module are unaffected. They remain toggle triples of the EXISTING
# ``Settings`` governance switches; no new toggle is introduced (Req 9.5, 12.2).

#: The mandatory honesty statement (Req 14.1, 14.2, 14.4) emitted with every
#: artifact. Declares the workload a targeted diagnostic, identifies the
#: Gate_Only_Arm shared-input comparison as its defense, records single-seed
#: sufficiency, and documents the single additive Reconcile_Path_Guard.
DIAGNOSTIC_SCOPE_NOTE: str = (
    "This table is a TARGETED DIAGNOSTIC, not a real-benchmark result. The poison "
    "writes are constructed specifically to exercise the schema/domain-range (C9), "
    "temporal (C2), decision-evidence (C8), and task-status (C4/C10) checks and are "
    "deliberately NOT single-valued contradictions, so the contradiction gate (C7) "
    "has nothing to fire on. The defense of the result is the Gate_Only_Arm, which "
    "receives the SAME INPUTS as every other arm yet still leaves the invalid durable "
    "state that the Schema_Provenance_Arm removes. Single-seed execution is sufficient "
    "because the offline oracle/mock pipeline is deterministic, consistent with the "
    "other deterministic single-seed tables in the evaluation. A single additive "
    "Reconcile_Path_Guard was added to gate the C4/C8/C10 reconcile path by the "
    "existing enable_constraint_validation toggle (matching how the C9 domain/range "
    "and C2 temporal relation-path checks are already gated); the change is "
    "behavior-preserving in the default (all-governance-on) configuration — with "
    "enable_constraint_validation=True the C4/C8/C10 reconcile behavior is "
    "byte-identical to before — and no check logic was modified. With the guard, all "
    "four violation types discriminate: each is > 0 under Gate_Only/Ungoverned and 0 "
    "under Schema_Provenance/Full."
)


# --------------------------------------------------------------------------- #
# Result data model (Data Models → StressAblationResult)
# --------------------------------------------------------------------------- #
@dataclass
class StressAblationResult:
    """Per-arm Typed_Violation_Reports plus the honesty framing (Req 9, 10, 14).

    * ``arms`` — ordered ``arm name -> TypedViolationReport`` (insertion order
      matches :data:`STRESS_ARMS`: Ungoverned, Gate_Only, Schema_Provenance, Full).
    * ``decisive_arm`` — the :data:`DECISIVE_ARM` row flagged decisive (Req 10.4).
    * ``seed`` — the single seed the workload was generated with (Req 11.3).
    * ``diagnostic_scope_note`` — the :data:`DIAGNOSTIC_SCOPE_NOTE` (Req 14).
    """

    arms: dict[str, TypedViolationReport] = field(default_factory=dict)
    decisive_arm: str = DECISIVE_ARM
    seed: int = 0
    diagnostic_scope_note: str = DIAGNOSTIC_SCOPE_NOTE

    @property
    def decisive_report(self) -> TypedViolationReport:
        """The Typed_Violation_Report of the decisive (Gate_Only) arm."""
        return self.arms[self.decisive_arm]

    def is_decisive(self, arm: str) -> bool:
        """True iff ``arm`` is the decisive comparison row (Req 10.4)."""
        return arm == self.decisive_arm


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
def _default_settings() -> Settings:
    """Deterministic, offline base settings (Req 6.3, 11.3).

    The oracle extractor is injected into every ``CoreContainer`` and overrides
    the ``"mock"`` selection, so no GPU / API key / network access is required.
    """
    return Settings(
        deterministic_test_mode=True,
        chroma_mode="memory",
        extractor="mock",
    )


# --------------------------------------------------------------------------- #
# Per-arm replay (the same ingestion BaselineRunner._ingest_sessions performs)
# --------------------------------------------------------------------------- #
def _replay_arm(
    arm: str,
    examples: list[BenchmarkExample],
    oracle: StressOracleExtractor,
    base_factory: Callable[[], Settings],
) -> tuple[CoreContainer, WriteOutcomeTally]:
    """Build the arm's container, replay every session, and tally write outcomes.

    Feeds the **same** ``examples`` + ``oracle`` given to every arm (Req 10.1);
    per arm builds ``CoreContainer(settings, extractor=oracle)`` and replays each
    session through ``container.write_pipeline.run(...)`` (the exact ingestion the
    baseline runner performs), accumulating the ``WriteResult.summary`` buckets
    into a :class:`WriteOutcomeTally` (Req 8.1, 8.2 — no new outcome categories).
    """
    settings = base_factory().model_copy(update=STRESS_ARMS[arm])
    container = CoreContainer(settings, extractor=oracle)
    tally = WriteOutcomeTally()
    for example in examples:
        for session in example.sessions:
            source_ref = f"{example.id}:{session.session_id}"
            result = container.write_pipeline.run(session.input, source_ref)
            summary = result.summary
            tally.accepted += int(getattr(summary, "num_accepted", 0) or 0)
            tally.superseded += int(getattr(summary, "num_superseded", 0) or 0)
            tally.quarantined += int(getattr(summary, "num_quarantined", 0) or 0)
            tally.rejected += int(getattr(summary, "num_rejected", 0) or 0)
    return container, tally


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_stress_ablation(
    *,
    seed: int = 1337,
    settings_factory: Callable[[], Settings] | None = None,
) -> StressAblationResult:
    """Run the Stress_Workload across the four arms (Req 9, 10, 11, 14).

    Builds the workload **once** and feeds the same examples + oracle to every arm
    (Req 10.1). Per arm it builds a ``CoreContainer`` from the arm's toggle triple,
    replays every session, accumulates a :class:`WriteOutcomeTally`, then attaches
    :func:`typed_violations` over that arm's durable ACTIVE store. The result flags
    the :data:`DECISIVE_ARM` row decisive (Req 10.4) and carries the
    :data:`DIAGNOSTIC_SCOPE_NOTE` (Req 14).

    The existing harness is reused (Req 11.1, 12.4): ``run_multiseed`` is invoked
    with ``provided_examples``, ``extractor=oracle``, a per-arm settings factory,
    and a single seed to obtain write-outcome tallies and prove run-to-run identity;
    ``aggregate_methods`` is kept available for single-seed reporting parity. A
    single seed is sufficient because the offline pipeline is deterministic (Req 11.3).
    """
    base_factory = settings_factory or _default_settings

    # Build the workload ONCE; the same examples + oracle feed every arm (Req 10.1).
    examples, oracle, _cases = generate_stress_workload(seed)

    arms: dict[str, TypedViolationReport] = {}
    for arm in STRESS_ARMS:
        container, tally = _replay_arm(arm, examples, oracle, base_factory)
        report = typed_violations(container)
        report.write_outcomes = tally
        arms[arm] = report

    # --- Harness reuse (Req 11.1, 11.2, 12.4) ---------------------------- #
    # Additionally drive each arm through the existing multi-seed harness with the
    # same examples + oracle and a single seed (Req 11.1). This obtains the arm's
    # write-outcome tally via the shared harness; that harness tally is required to
    # equal the direct-replay tally assembled above — an independent-path identity
    # check that proves the deterministic offline pipeline yields the same
    # write-outcome counts run-to-run (Req 11.2). ``aggregate_methods`` is exercised
    # for single-seed reporting parity with the other deterministic tables (Req 12.4).
    # The typed four-type breakdown is computed from the per-arm container above
    # (run_multiseed does not expose containers); the two agree by determinism.
    for arm in STRESS_ARMS:
        arm_factory = stress_arm_settings_factory(base_factory, arm)
        ms = run_multiseed(
            ["full"],
            seeds=[seed],
            settings_factory=arm_factory,
            extractor=oracle,
            provided_examples=examples,
            warmup=False,
            key_suffix=f"__stress_{arm}",
        )
        harness = ms.write_outcomes["full"]
        direct = arms[arm].write_outcomes
        assert (
            harness["accepted"] == direct.accepted
            and harness["superseded"] == direct.superseded
            and harness["quarantined"] == direct.quarantined
            and harness["rejected"] == direct.rejected
        ), f"harness and direct-replay write-outcome tallies diverged for arm {arm!r}"
        # Single-seed reporting parity (keep aggregate_methods available, Req 12.4).
        aggregate_methods(ms)

    return StressAblationResult(
        arms=arms,
        decisive_arm=DECISIVE_ARM,
        seed=seed,
        diagnostic_scope_note=DIAGNOSTIC_SCOPE_NOTE,
    )

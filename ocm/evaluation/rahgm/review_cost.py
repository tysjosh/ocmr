"""Reviewer-minutes model for the controlled replay (Req 11.4).

Experiment 1 is a replay study with no human reviewers, but ``R100`` — reviewer
minutes per 100 writes — is a primary outcome. This module supplies an **explicit,
inspectable cost model** so ``R100`` is computable in replay, and every table that
consumes it is labelled ``modelled: true``.

The model is intentionally simple and monotone in the same quantities the router
uses, so it cannot flatter RAHGM: a write that the router considered complex
(more failed checks, a hard contradiction, an unresolved alias, high consequence)
also costs more to adjudicate, under *every* condition. Universal review pays the
same per-item price as selective review for an identical item; its ``R100``
advantage or disadvantage comes only from how many items it queues.

This is a model, not measured human data. It is not used anywhere that human
timing is reported: Experiment 2's decision times come from the simulated analyst,
which is separately and explicitly labelled as a simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ocm.governance.features import RiskFeatures
from ocm.governance.review_queue import ExplanationDepth

#: Base minutes to adjudicate an item at each explanation depth. Depth adds
#: reading time; whether it also adds *value* is the empirical question of RQ2.
BASE_MINUTES: Mapping[ExplanationDepth, float] = {
    ExplanationDepth.minimal: 0.60,
    ExplanationDepth.evidence: 0.95,
    ExplanationDepth.full: 1.40,
}

#: Additional minutes per unresolved-or-failed check beyond the first.
MINUTES_PER_CHECK = 0.35

#: Additional minutes for a hard contradiction (two claims must be reconciled).
HARD_CONTRADICTION_MINUTES = 0.80

#: Additional minutes when entity identity is unresolved (alias adjudication).
ALIAS_MINUTES = 0.50

#: Additional minutes scaled by consequence (more care on consequential writes).
CONSEQUENCE_MINUTES = 0.40

#: Additional minutes when the transition is hard to undo.
IRREVERSIBILITY_MINUTES = 0.45


@dataclass(frozen=True)
class ReviewCostModel:
    """A deterministic reviewer-minutes model for one review item.

    ``minutes = base[depth] + 0.35·k + 0.80·[f_c = 1] + 0.50·[f_e > 0]
    + 0.40·q + 0.45·(1 − v)``
    """

    depth: ExplanationDepth = ExplanationDepth.evidence
    modelled: bool = True

    def minutes(self, features: RiskFeatures) -> float:
        """Modelled reviewer minutes for an item with these features."""
        total = BASE_MINUTES[self.depth]
        total += MINUTES_PER_CHECK * features.k
        if features.f_c >= 1.0:
            total += HARD_CONTRADICTION_MINUTES
        if features.f_e > 0.0:
            total += ALIAS_MINUTES
        total += CONSEQUENCE_MINUTES * features.consequence
        total += IRREVERSIBILITY_MINUTES * (1.0 - features.reversibility)
        return round(total, 4)

    def seconds(self, features: RiskFeatures) -> float:
        """Modelled reviewer seconds for an item with these features."""
        return round(self.minutes(features) * 60.0, 2)

    def as_dict(self) -> dict[str, object]:
        """A JSON-serializable description, including the model disclosure."""
        return {
            "depth": self.depth.value,
            "modelled": self.modelled,
            "base_minutes": BASE_MINUTES[self.depth],
            "minutes_per_check": MINUTES_PER_CHECK,
            "hard_contradiction_minutes": HARD_CONTRADICTION_MINUTES,
            "alias_minutes": ALIAS_MINUTES,
            "consequence_minutes": CONSEQUENCE_MINUTES,
            "irreversibility_minutes": IRREVERSIBILITY_MINUTES,
            "note": (
                "Reviewer minutes are produced by this explicit cost model, not "
                "measured from human participants."
            ),
        }

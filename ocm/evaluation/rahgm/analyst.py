"""Simulated analyst for Experiment 2 (Req 13.3, 14.1, 14.2).

**This is a simulation, not the paper's human study.** Paper §3.6 preregisters 80
participants with analytic experience. No human participants are available here,
so this module supplies an explicit generative analyst model that exercises the
review interface, the explanation renderer, the adjudication path, and the
feedback loop end to end.

What the simulated results can and cannot support:

* They *can* show that the harness produces the measures §3.7 specifies, that the
  three explanation depths render and are consumable, that adjudication releases
  and dismisses correctly, and that the feedback loop stays bounded under analyst
  behavior of a given quality.
* They *cannot* answer RQ2. Whether explanation depth improves human adjudication
  and calibration is an empirical question about people; here it is an assumption
  encoded in :data:`DEPTH_COMPETENCE`. Any depth effect the simulation reports is
  a restatement of that assumption, and is labelled as such.

The model is documented rather than tuned to a target: parameters are set from the
qualitative findings the paper cites — explanations can raise acceptance without
raising team performance, and reliance grows with an unbroken streak of correct
recommendations — so the simulation is a plausible stand-in rather than an
optimistic one.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from ocm.governance.features import RiskFeatures
from ocm.governance.review_queue import (
    ExplanationDepth,
    ReviewAction,
    ReviewItem,
    render_explanation,
)

#: Disclosure attached to every artifact this module contributes to.
SIMULATION_DISCLOSURE = (
    "Experiment 2 results come from a generative simulated-analyst model, not "
    "from the paper's preregistered 80-participant human study. Explanation-depth "
    "effects reflect the model's assumptions and are not human-subjects evidence."
)


@dataclass(frozen=True)
class DepthCompetence:
    """How one explanation depth affects the simulated analyst.

    Attributes:
        base_accuracy: Probability of selecting the correct action on a case with
            no complicating features.
        evidence_bonus: Extra accuracy when the case's difficulty is
            evidence-related and the depth exposes evidence.
        seconds_base: Median decision seconds before difficulty adjustments.
        seconds_per_field: Added seconds per rendered explanation field, so deeper
            explanations cost reading time.
        confidence_shift: Additive shift to self-reported confidence. Deeper
            explanations raise stated confidence more than they raise accuracy,
            which is the miscalibration the literature reports.
        workload: Baseline NASA-TLX contribution on a 0–100 scale.
    """

    base_accuracy: float
    evidence_bonus: float
    seconds_base: float
    seconds_per_field: float
    confidence_shift: float
    workload: float


#: The assumed competence profile per depth. These are *assumptions*, not
#: measurements: the minimal condition withholds the evidence needed to resolve
#: evidence-driven cases, the evidence condition supplies it, and the full
#: condition adds material that costs time and inflates stated confidence more
#: than it improves accuracy.
DEPTH_COMPETENCE: dict[ExplanationDepth, DepthCompetence] = {
    ExplanationDepth.minimal: DepthCompetence(
        base_accuracy=0.72,
        evidence_bonus=0.00,
        seconds_base=26.0,
        seconds_per_field=0.9,
        confidence_shift=-0.02,
        workload=52.0,
    ),
    ExplanationDepth.evidence: DepthCompetence(
        base_accuracy=0.79,
        evidence_bonus=0.11,
        seconds_base=34.0,
        seconds_per_field=1.1,
        confidence_shift=0.04,
        workload=48.0,
    ),
    ExplanationDepth.full: DepthCompetence(
        base_accuracy=0.81,
        evidence_bonus=0.12,
        seconds_base=49.0,
        seconds_per_field=1.4,
        confidence_shift=0.11,
        workload=58.0,
    ),
}

#: Accuracy penalty per unresolved-or-failed check beyond the first: harder cases
#: are harder for people too, at every depth.
DIFFICULTY_PENALTY_PER_CHECK = 0.055

#: Accuracy penalty scaled by consequence, modelling decision pressure.
CONSEQUENCE_PENALTY = 0.04

#: Probability of following the system recommendation without independent
#: analysis, as a function of the current unbroken correct-recommendation streak.
#: Modelled as a saturating curve: reliance grows but never becomes certainty.
AUTOMATION_BIAS_CEILING = 0.42
AUTOMATION_BIAS_RATE = 0.22


@dataclass
class AnalystProfile:
    """One simulated participant.

    Attributes:
        analyst_id: Stable identifier, stored separately from any interaction log
            in the same way the paper's protocol separates participant ids.
        skill: Additive accuracy offset capturing between-participant variation.
        speed: Multiplicative decision-time factor.
        bias_susceptibility: Individual scaling of automation bias.
        workload_offset: Additive NASA-TLX offset.
        seed: Per-participant seed.
    """

    analyst_id: str
    skill: float = 0.0
    speed: float = 1.0
    bias_susceptibility: float = 1.0
    workload_offset: float = 0.0
    seed: int = 0


def sample_profiles(n: int, *, seed: int = 1337) -> list[AnalystProfile]:
    """Draw ``n`` participant profiles with between-participant variation."""
    rng = random.Random(seed)
    profiles: list[AnalystProfile] = []
    for i in range(n):
        profiles.append(
            AnalystProfile(
                analyst_id=f"P{i + 1:03d}",
                skill=rng.gauss(0.0, 0.06),
                speed=math.exp(rng.gauss(0.0, 0.22)),
                bias_susceptibility=max(0.0, rng.gauss(1.0, 0.30)),
                workload_offset=rng.gauss(0.0, 7.0),
                seed=seed * 7919 + i,
            )
        )
    return profiles


@dataclass
class AdjudicationTrace:
    """Everything recorded for one simulated adjudication (§3.6)."""

    analyst_id: str
    item_id: str
    write_id: str | None
    scenario_index: int
    trial_index: int
    depth: ExplanationDepth
    action: ReviewAction
    correct_action: ReviewAction
    correct: bool
    seconds: float
    confidence: float
    evidence_opened: int
    action_changes: int
    followed_recommendation: bool
    recommendation_correct: bool
    streak_before: int
    consequential: bool
    k: int
    workload: float

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "analyst_id": self.analyst_id,
            "item_id": self.item_id,
            "write_id": self.write_id,
            "scenario_index": self.scenario_index,
            "trial_index": self.trial_index,
            "depth": self.depth.value,
            "action": self.action.value,
            "correct_action": self.correct_action.value,
            "correct": self.correct,
            "seconds": self.seconds,
            "confidence": self.confidence,
            "evidence_opened": self.evidence_opened,
            "action_changes": self.action_changes,
            "followed_recommendation": self.followed_recommendation,
            "recommendation_correct": self.recommendation_correct,
            "streak_before": self.streak_before,
            "consequential": self.consequential,
            "k": self.k,
            "workload": self.workload,
        }


class SimulatedAnalyst:
    """A generative model of one analyst adjudicating review items.

    Behavior depends on the explanation depth, the case's difficulty, the
    participant's profile, and the current recommendation-following streak. The
    model is stateful across trials so complacency can accumulate, which is what
    the §4.3 automation-bias analysis tests for.
    """

    def __init__(self, profile: AnalystProfile) -> None:
        """Create an analyst bound to a participant profile."""
        self.profile = profile
        self.streak = 0
        self.trial = 0
        self.traces: list[AdjudicationTrace] = []

    # -- public API --------------------------------------------------------
    def adjudicate(
        self,
        item: ReviewItem,
        depth: ExplanationDepth,
        correct_action: ReviewAction,
        *,
        scenario_index: int = 0,
    ) -> AdjudicationTrace:
        """Adjudicate one review item and record the full trace."""
        competence = DEPTH_COMPETENCE[depth]
        features = item.decision.features
        explanation = render_explanation(item, depth)
        rng = random.Random(f"{self.profile.seed}:{item.item_id}:{depth.value}")

        recommendation = item.recommended_action
        recommendation_correct = recommendation is correct_action
        streak_before = self.streak

        # Automation bias: the chance of deferring to the recommendation without
        # independent analysis, growing with an unbroken correct-recommendation
        # streak and saturating well below certainty.
        defer_probability = min(
            0.95,
            AUTOMATION_BIAS_CEILING
            * self.profile.bias_susceptibility
            * (1.0 - math.exp(-AUTOMATION_BIAS_RATE * streak_before)),
        )
        deferred = rng.random() < defer_probability

        if deferred:
            action = recommendation
            action_changes = 0
            evidence_opened = 0
        else:
            accuracy = self._accuracy(competence, features, depth)
            if rng.random() < accuracy:
                action = correct_action
            else:
                action = self._plausible_error(correct_action, item, rng)
            action_changes = 1 if rng.random() < 0.28 else 0
            evidence_opened = self._evidence_opened(depth, item, rng)

        correct = action is correct_action
        seconds = self._seconds(competence, features, explanation, deferred, rng)
        confidence = self._confidence(competence, features, correct, deferred, rng)
        workload = self._workload(competence, features, rng)

        # Streak tracks consecutive correct *recommendations*, which is what the
        # analyst can observe accumulating.
        self.streak = streak_before + 1 if recommendation_correct else 0

        trace = AdjudicationTrace(
            analyst_id=self.profile.analyst_id,
            item_id=item.item_id,
            write_id=item.write_id,
            scenario_index=scenario_index,
            trial_index=self.trial,
            depth=depth,
            action=action,
            correct_action=correct_action,
            correct=correct,
            seconds=seconds,
            confidence=confidence,
            evidence_opened=evidence_opened,
            action_changes=action_changes,
            followed_recommendation=action is recommendation,
            recommendation_correct=recommendation_correct,
            streak_before=streak_before,
            consequential=item.consequential,
            k=features.k,
            workload=workload,
        )
        self.trial += 1
        self.traces.append(trace)
        return trace

    # -- components --------------------------------------------------------
    def _accuracy(
        self,
        competence: DepthCompetence,
        features: RiskFeatures,
        depth: ExplanationDepth,
    ) -> float:
        """Probability of choosing the correct action for this case."""
        accuracy = competence.base_accuracy + self.profile.skill
        # Evidence-driven difficulty is only resolvable when the depth shows
        # evidence; this is the mechanism by which depth could matter.
        evidence_driven = features.f_v > 0.0 or features.evidence_count <= 1
        if evidence_driven and depth is not ExplanationDepth.minimal:
            accuracy += competence.evidence_bonus
        accuracy -= DIFFICULTY_PENALTY_PER_CHECK * max(0, features.k - 1)
        accuracy -= CONSEQUENCE_PENALTY * features.consequence
        return max(0.05, min(0.99, accuracy))

    @staticmethod
    def _plausible_error(
        correct_action: ReviewAction, item: ReviewItem, rng: random.Random
    ) -> ReviewAction:
        """Choose a plausible wrong action rather than a uniform random one.

        Analysts err toward the conservative options — holding or asking for more
        evidence — more often than toward committing something novel.
        """
        options = [
            ReviewAction.quarantine,
            ReviewAction.request_evidence,
            item.recommended_action,
            ReviewAction.accept,
            ReviewAction.supersede,
            ReviewAction.reject,
        ]
        weights = [0.30, 0.22, 0.20, 0.12, 0.10, 0.06]
        pool = [
            (option, weight)
            for option, weight in zip(options, weights)
            if option is not correct_action
        ]
        total = sum(w for _o, w in pool)
        draw = rng.random() * total
        cumulative = 0.0
        for option, weight in pool:
            cumulative += weight
            if draw <= cumulative:
                return option
        return pool[-1][0]

    def _seconds(
        self,
        competence: DepthCompetence,
        features: RiskFeatures,
        explanation: dict[str, Any],
        deferred: bool,
        rng: random.Random,
    ) -> float:
        """Decision time in seconds, log-normally dispersed."""
        base = competence.seconds_base
        base += competence.seconds_per_field * len(explanation)
        base += 4.5 * features.k
        base += 6.0 * features.consequence
        if deferred:
            # Deferring is fast: that is the efficiency automation bias buys and
            # the failure mode it creates.
            base *= 0.42
        base *= self.profile.speed
        return round(max(1.5, base * math.exp(rng.gauss(0.0, 0.28))), 2)

    def _confidence(
        self,
        competence: DepthCompetence,
        features: RiskFeatures,
        correct: bool,
        deferred: bool,
        rng: random.Random,
    ) -> float:
        """Self-reported confidence on a 0–1 scale.

        Confidence tracks correctness only weakly, and depth raises stated
        confidence more than it raises accuracy — the mechanism that produces
        miscalibration in the ECE measure.
        """
        confidence = 0.62 + competence.confidence_shift
        confidence += 0.14 if correct else -0.06
        confidence -= 0.03 * max(0, features.k - 1)
        if deferred:
            confidence += 0.06
        confidence += rng.gauss(0.0, 0.09)
        return round(max(0.01, min(0.99, confidence)), 4)

    def _evidence_opened(
        self, depth: ExplanationDepth, item: ReviewItem, rng: random.Random
    ) -> int:
        """How many evidence snippets the analyst opened."""
        if depth is ExplanationDepth.minimal:
            # Minimal shows no evidence to open.
            return 0
        available = len(item.evidence.supporting) + len(item.evidence.conflicting)
        if available == 0:
            return 0
        propensity = 0.65 if depth is ExplanationDepth.evidence else 0.80
        return sum(1 for _ in range(available) if rng.random() < propensity)

    def _workload(
        self,
        competence: DepthCompetence,
        features: RiskFeatures,
        rng: random.Random,
    ) -> float:
        """NASA-TLX workload on a 0–100 scale."""
        workload = competence.workload + self.profile.workload_offset
        workload += 3.2 * features.k
        workload += 6.0 * features.consequence
        workload += rng.gauss(0.0, 4.5)
        return round(max(0.0, min(100.0, workload)), 2)

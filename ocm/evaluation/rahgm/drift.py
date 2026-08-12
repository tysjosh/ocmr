"""Distribution drift: the condition under which feedback adaptation should matter.

**Not in the paper.** §3.5 evaluates adaptive RAHGM against frozen RAHGM on the same
distribution the policy was fitted on. On that setup the two are indistinguishable,
because a policy fitted on the training partition is already near-optimal on the
test partition — bounded updates have no headroom to recover. Separating C4 from C5
therefore cannot show what it was designed to show.

Adaptation should earn its keep when the *write distribution changes* after the
policy is frozen. This module constructs that condition explicitly.

The drift is a single, documented transformation with a consistent relabeling:

    A fraction ``severity`` of authoritative corrections have their source degraded
    from an authoritative scheme (``a ≥ 0.90``) to a mid-authority one
    (``tool``, ``a = 0.75``), and their gold transition changes from ``supersede``
    to ``review``.

The relabeling is not a free choice. ``h(u)`` requires authority ``≥ 0.90``, so a
degraded correction is no longer an *authoritative* correction — it is a
contradiction against an accepted value from a non-authoritative source, which is
exactly what the undrifted corpus already golds as ``review`` in
``conflict_weak_authority``. Drift therefore moves cases across an existing
decision boundary rather than inventing a new one.

This models a concrete operational event: an upstream feed that used to be a system
of record is re-plumbed through a less reliable path, so writes that previously
warranted autonomous supersession now warrant review. A frozen policy keeps
committing them; an adapting policy should learn to escalate them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

from ocm.evaluation.rahgm.corpus import (
    AUTHORITY_BY_SCHEME,
    CandidateWrite,
    Partition,
    RahgmCorpus,
    Scenario,
    WriteClass,
    build_questions,
)
from ocm.evaluation.rahgm.metrics import compute_metrics
from ocm.evaluation.rahgm.replay import (
    DevelopedPolicy,
    ScenarioReplayer,
    develop_policy,
    make_canary_evaluator,
    make_oracle_reviewer,
)
from ocm.governance.adaptation import BLOCK_SIZE, BoundedUpdater, CanaryGate
from ocm.governance.conditions import Condition
from ocm.governance.policy import Tier

#: The scheme a degraded source is rewritten to. Below the 0.90 authoritative
#: floor, so ``h(u)`` no longer holds, but still an attributed, plausible source.
DEGRADED_SCHEME = "tool"

#: Default fraction of authoritative corrections that drift.
DEFAULT_SEVERITY = 0.6

#: The two drift modes, which turn out to have opposite implications.
#:
#: ``covariate`` — feature values shift and the gold label follows. The router
#: consumes authority directly, so it reroutes without being retrained.
#:
#: ``label`` — the gold label flips while every feature stays fixed. The router
#: cannot distinguish the affected writes from ones it should still commit, so no
#: amount of feature evidence resolves it and only feedback can.
DRIFT_MODES: tuple[str, ...] = ("covariate", "label")

#: Templates used for label drift. Both sit below the ``h(u)`` authority floor and
#: are routed autonomously by the fitted policy, so flipping their gold to
#: ``review`` creates writes the frozen policy commits and should not.
LABEL_DRIFT_TEMPLATES: tuple[str, ...] = (
    "discriminating_authority_high",
    "discriminating_consequence_low",
    "discriminating_reversibility_high",
)

#: Templates eligible for degradation: the authoritative corrections whose gold
#: transition depends on clearing the authority floor.
_ELIGIBLE_PREFIXES = ("correction_", "chain_")
_ELIGIBLE_MARKERS = ("authoritative", "correction_", "revision")


def _eligible(write: CandidateWrite) -> bool:
    """Whether a write is an authoritative correction that drift can degrade."""
    if write.write_class is not WriteClass.correction:
        return False
    if write.gold_transition is not Tier.supersede:
        return False
    if write.authority < 0.90:
        return False
    return any(m in write.template for m in _ELIGIBLE_MARKERS) or write.template.startswith(
        _ELIGIBLE_PREFIXES
    )


def drift_write(write: CandidateWrite) -> CandidateWrite:
    """Degrade one authoritative correction and relabel it consistently.

    The source scheme is rewritten, the rubric authority follows the scheme, and the
    gold transition becomes ``review``. ``expected_object_after`` is cleared because
    a write that should now be held no longer changes the current value — which
    keeps the scenario's downstream questions consistent with the new labels.
    """
    tail = write.source_ref.split(":", 1)[1] if ":" in write.source_ref else write.source_ref
    return replace(
        write,
        source_ref=f"{DEGRADED_SCHEME}:{tail}",
        authority=AUTHORITY_BY_SCHEME[DEGRADED_SCHEME],
        gold_transition=Tier.review,
        expected_object_after=None,
        minimum_evidence=(
            "corroboration from a source of at least authoritative standing "
            "(the original source has been downgraded)"
        ),
        perturbations=tuple(sorted(set(write.perturbations) | {"source_authority"})),
    )


def label_drift_write(write: CandidateWrite) -> CandidateWrite:
    """Flip a write's gold transition to ``review`` leaving every feature untouched.

    Nothing observable changes: same source, same authority, same consequence, same
    reversibility, same constraint outcomes. Only the correct answer moves. This is
    the drift a feature-derived router cannot detect, because the writes that
    drifted are indistinguishable from the writes that did not.
    """
    return replace(
        write,
        gold_transition=Tier.review,
        expected_object_after=None,
        minimum_evidence=(
            "an out-of-band signal: this class of write became review-worthy "
            "without any change to its observable features"
        ),
    )


def _label_eligible(write: CandidateWrite) -> bool:
    """Whether a write is a label-drift target."""
    return (
        write.template in LABEL_DRIFT_TEMPLATES
        and write.gold_transition is Tier.supersede
    )


def drift_scenario(
    scenario: Scenario,
    *,
    severity: float,
    rng: random.Random,
    mode: str = "covariate",
) -> Scenario:
    """Apply drift to one scenario, rebuilding its questions from the new labels."""
    if mode not in DRIFT_MODES:
        raise ValueError(f"unknown drift mode {mode!r}; expected one of {DRIFT_MODES}")

    predicate = _eligible if mode == "covariate" else _label_eligible
    transform = drift_write if mode == "covariate" else label_drift_write

    eligible = [w for w in scenario.writes if predicate(w)]
    n_drift = int(round(len(eligible) * severity))
    chosen = set()
    if n_drift > 0:
        chosen = {w.write_id for w in rng.sample(eligible, n_drift)}

    writes = tuple(
        transform(w) if w.write_id in chosen else w for w in scenario.writes
    )

    # Questions are rebuilt so the gold answer reflects the drifted labels: a
    # correction that must now be held no longer defines the current value.
    class _Pool:
        incumbents = scenario.incumbents

    return replace(
        scenario, writes=writes, questions=build_questions(writes, _Pool())  # type: ignore[arg-type]
    )


def drift_scenarios(
    scenarios: Iterable[Scenario],
    *,
    severity: float = DEFAULT_SEVERITY,
    seed: int = 1337,
    mode: str = "covariate",
) -> list[Scenario]:
    """Apply drift to a sequence of scenarios deterministically."""
    rng = random.Random(seed)
    return [
        drift_scenario(s, severity=severity, rng=rng, mode=mode) for s in scenarios
    ]


# --------------------------------------------------------------------------- #
# The study
# --------------------------------------------------------------------------- #
@dataclass
class DriftArmResult:
    """One arm's outcome on the drifted stream."""

    arm: str
    mcr: float
    review_rate: float
    accuracy: float
    dvr: float
    r100: float
    early_accuracy: float
    late_accuracy: float
    early_mcr: float
    late_mcr: float
    n_deployed: int
    n_proposed: int
    cumulative_drift: float

    @property
    def recovery(self) -> float:
        """Accuracy gained from the first half of the stream to the second."""
        return self.late_accuracy - self.early_accuracy

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "arm": self.arm,
            "mcr": self.mcr,
            "review_rate": self.review_rate,
            "accuracy": self.accuracy,
            "dvr": self.dvr,
            "r100": self.r100,
            "early_accuracy": self.early_accuracy,
            "late_accuracy": self.late_accuracy,
            "recovery": self.recovery,
            "early_mcr": self.early_mcr,
            "late_mcr": self.late_mcr,
            "n_proposed": self.n_proposed,
            "n_deployed": self.n_deployed,
            "cumulative_drift": self.cumulative_drift,
        }


def _halves(records: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    """Split a record stream into first and second halves, in order."""
    midpoint = len(records) // 2
    return list(records[:midpoint]), list(records[midpoint:])


def _accuracy(records: Sequence[Any]) -> float:
    return (
        sum(1 for r in records if not r.error) / len(records) if records else float("nan")
    )


def _mcr(records: Sequence[Any]) -> float:
    consequential = [r for r in records if r.consequential]
    if not consequential:
        return float("nan")
    return sum(1 for r in consequential if r.error and not r.escalated) / len(
        consequential
    )


def run_drift_study(
    corpus: RahgmCorpus | None = None,
    *,
    developed: DevelopedPolicy | None = None,
    severity: float = DEFAULT_SEVERITY,
    seed: int = 1337,
    repeats: int = 3,
    mode: str = "covariate",
) -> dict[str, Any]:
    """Compare frozen and adaptive RAHGM on a drifted write distribution.

    The policy is developed on the *undrifted* training and development partitions,
    exactly as in Experiment 1, and then evaluated on drifted test scenarios. The
    frozen arm cannot respond; the adaptive arm receives adjudication feedback from
    the drifted stream through the bounded updater and canary gate.

    ``repeats`` concatenates the drifted test scenarios several times, because a
    single pass over 10 scenarios yields only a handful of 20-write feedback blocks —
    too few for adaptation to act. This is a longer operational horizon, not extra
    data: the same scenarios recur, as a persistent agent would revisit them.

    Returns per-arm results plus the first-half/second-half recovery contrast, which
    is the signature of adaptation actually working.
    """
    from ocm.evaluation.rahgm.corpus import generate_corpus

    corpus = corpus or generate_corpus()
    developed = developed or develop_policy(corpus)

    undrifted = corpus.partition(Partition.test)
    drifted = drift_scenarios(undrifted, severity=severity, seed=seed, mode=mode)
    stream = [s for _ in range(repeats) for s in drifted]

    writes_by_id = {w.write_id: w for s in drifted for w in s.writes}
    reviewer = make_oracle_reviewer(writes_by_id)
    canary_gate = CanaryGate(make_canary_evaluator(corpus))

    # Count writes whose gold label the drift actually moved.
    before = {w.write_id: w.gold_transition for s in undrifted for w in s.writes}
    n_relabelled = sum(
        1
        for s in drifted
        for w in s.writes
        if before.get(w.write_id) is not w.gold_transition
    )

    results: list[DriftArmResult] = []
    for arm, condition in (
        ("frozen", Condition.frozen_rahgm),
        ("adaptive", Condition.adaptive_rahgm),
    ):
        replayer = ScenarioReplayer(
            condition,
            params=developed.params,
            reviewer=reviewer,
            canary_gate=canary_gate if arm == "adaptive" else None,
            updater=BoundedUpdater() if arm == "adaptive" else None,
            collect_cases=False,
        )
        scenario_results, harness = replayer.run_corpus(stream)
        records = [r for sr in scenario_results for r in sr.records]
        violations = sum(sr.durable_violations for sr in scenario_results)
        metrics = compute_metrics(records, durable_violations=violations)

        early, late = _halves(records)
        registry = harness.registry if harness else None
        results.append(
            DriftArmResult(
                arm=arm,
                mcr=metrics.mcr,
                review_rate=metrics.review_rate,
                accuracy=metrics.accuracy,
                dvr=metrics.dvr,
                r100=metrics.r100,
                early_accuracy=_accuracy(early),
                late_accuracy=_accuracy(late),
                early_mcr=_mcr(early),
                late_mcr=_mcr(late),
                n_proposed=registry.n_proposed if registry else 0,
                n_deployed=(
                    registry.n_proposed - registry.n_rejected if registry else 0
                ),
                cumulative_drift=registry.cumulative_drift if registry else 0.0,
            )
        )

    by_arm = {r.arm: r for r in results}
    frozen, adaptive = by_arm["frozen"], by_arm["adaptive"]

    return {
        "experiment": "drift_study_adaptive_vs_frozen",
        "in_paper": False,
        "rationale": (
            "§3.5 compares adaptive and frozen RAHGM on the distribution the policy "
            "was fitted on, where bounded updates have no headroom and the two arms "
            "are indistinguishable. This study supplies the condition under which "
            "adaptation should matter: a documented shift in the write distribution "
            "after the policy is frozen."
        ),
        "drift": {
            "mode": mode,
            "severity": severity,
            "n_scenarios": len(drifted),
            "repeats": repeats,
            "n_writes_in_stream": sum(len(s.writes) for s in stream),
            "n_relabelled_per_pass": n_relabelled,
            "transformation": (
                (
                    "authoritative corrections have their source degraded below the "
                    "0.90 h(u) floor; gold changes supersede -> review, matching how "
                    "the undrifted corpus already labels a non-authoritative "
                    "contradiction. Features change, so the router can see the drift."
                )
                if mode == "covariate"
                else (
                    "gold changes supersede -> review for selected templates with "
                    "every feature held fixed. The router cannot see the drift, so "
                    "only feedback can respond to it."
                )
            ),
            "degraded_scheme": DEGRADED_SCHEME if mode == "covariate" else None,
            "degraded_authority": (
                AUTHORITY_BY_SCHEME[DEGRADED_SCHEME] if mode == "covariate" else None
            ),
            "label_drift_templates": (
                list(LABEL_DRIFT_TEMPLATES) if mode == "label" else None
            ),
            "block_size": BLOCK_SIZE,
        },
        "arms": [r.as_dict() for r in results],
        "contrast": {
            "accuracy_delta_points": 100.0 * (adaptive.accuracy - frozen.accuracy),
            "mcr_delta_points": 100.0 * (adaptive.mcr - frozen.mcr),
            "review_rate_delta_points": 100.0
            * (adaptive.review_rate - frozen.review_rate),
            "frozen_recovery_points": 100.0 * frozen.recovery,
            "adaptive_recovery_points": 100.0 * adaptive.recovery,
            "recovery_advantage_points": 100.0 * (adaptive.recovery - frozen.recovery),
            "adaptive_updates_deployed": adaptive.n_deployed,
            "adaptive_updates_proposed": adaptive.n_proposed,
        },
    }

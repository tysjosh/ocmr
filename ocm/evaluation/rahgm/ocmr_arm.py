"""Risk-adaptive escalation as an arm of OCMR's own benchmark.

This is the experiment that decides whether selective human escalation recovers
the recall cost OCMR pays for conservative quarantine. OCMR reports durable-write
violations 50.72 → 0.00 but task success 77.20 (ungoverned) → 60.00 (governed),
and names "quarantine-release policies" as future work. Everything else in
:mod:`ocm.evaluation.rahgm` measures the router on a purpose-built corpus; this
module measures it on OCMR's established benchmark, against OCMR's own baselines
and decisive metrics.

Arms
----
``B3``
    OCMR as published: a quarantine is terminal. No review, no release.
``B3R``
    OCMR plus the risk-adaptive router. Writes the router escalates enter a review
    queue; an analyst adjudicates and may **release** them into durable memory.
``B3Q``
    Every OCMR quarantine is reviewed. This is not a proposal — it bounds how much
    recall any release policy could recover, and prices universal review of the
    quarantine stream.

Why the runner's protocol matters
---------------------------------
:class:`~ocm.evaluation.runner.BaselineRunner` ingests one example's sessions and
then immediately answers that example's questions before moving on. Releasing a
held write therefore helps the example that produced it, and cannot retroactively
corrupt an already-scored answer. Without that interleaving, release on a shared
store would merely move an error from one example to another.

Reviewers
---------
Two, reported separately, because they answer different questions.

``identity_reviewer``
    Deployable. Releases a held write when its conflicting incumbent was authored
    by a *different* benchmark example — a cross-context identifier collision,
    which OCMR's own audit identifies as the dominant cause of its false
    quarantines. Upholds a conflict that arises within one example, which is
    genuine contradiction. Uses only information a review item carries.

``oracle_reviewer``
    A ceiling, not a proposal. Releases exactly the writes OCMR's audit labels
    false quarantines (a quarantine inside an example whose questions expect no
    conflict). Bounds what any reviewer could achieve on this benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ocm.core.container import CoreContainer
from ocm.evaluation.baselines import baseline_settings_overrides, build_baseline
from ocm.evaluation.benchmark import BenchmarkExample
from ocm.evaluation.experiment import (
    _default_settings,
    _seed_everything,
    decisive_metrics,
    durable_constraint_violations,
)
from ocm.evaluation.metrics import MetricsReporter
from ocm.evaluation.runner import BaselineRunner
from ocm.evaluation.typed_violations import typed_violations
from ocm.governance.features import FeatureExtractor, WriteContext
from ocm.governance.policy import EscalationPolicy, PolicyParameters, Tier
from ocm.governance.review_queue import ReviewAction, ReviewItem, ReviewQueue
from ocm.governance.router import GovernedCommitManager, RiskAdaptiveRouter
from ocm.memory.contracts import CandidateAssertion

#: The arms this module runs, in reporting order.
ARMS: tuple[str, ...] = ("B0", "B2", "B3", "B3R", "B3Q")

#: Arms that install the escalation router.
GOVERNED_ARMS: frozenset[str] = frozenset({"B3R", "B3Q"})

#: Human-readable descriptions for the results table.
ARM_DESCRIPTIONS: dict[str, str] = {
    "B0": "Text/vector memory, no write-time governance",
    "B2": "Hybrid retrieval, no write-time governance",
    "B3": "OCMR: write-time governance, quarantine is terminal",
    "B3R": "OCMR + risk-adaptive escalation and review-and-release",
    "B3Q": "OCMR + review every quarantine (recall ceiling)",
}


#: A reviewer decides what to do with one held write. It receives the review item
#: and the adjudication context, and returns a :class:`ReviewAction`.
Reviewer = Callable[[ReviewItem, "ReviewContext"], ReviewAction]


@dataclass
class ReviewContext:
    """What a reviewer may consult when adjudicating a held write.

    Attributes:
        example: The benchmark example being ingested.
        authored_by: ``assertion_id -> example_id`` for every accepted assertion,
            so a reviewer can tell whether a conflict comes from another context.
        expects_conflict: Whether this example's questions expect a conflict.
            Only the oracle reviewer may read this.
    """

    example: BenchmarkExample
    authored_by: dict[str, str]
    expects_conflict: bool


def identity_reviewer(item: ReviewItem, context: ReviewContext) -> ReviewAction:
    """Release writes held because of a cross-context identifier collision.

    When the conflicting incumbent was authored by a different example, the two
    assertions are about different logical entities that happen to share an id, so
    the "contradiction" is spurious and the proposed write is the correct current
    value for *this* context. An analyst resolving identity would release it.

    A conflict raised within a single example is a genuine contradiction and is
    upheld. This reviewer reads only the review item and assertion authorship —
    never the benchmark's expected-conflict labels.
    """
    conflicting = list(item.decision.features.incumbent_ids) or list(
        item.ocmr_verdict.conflicting_ids
    )
    cross_context = any(
        context.authored_by.get(cid, context.example.id) != context.example.id
        for cid in conflicting
    )
    if not cross_context:
        return ReviewAction.quarantine
    return ReviewAction.supersede if conflicting else ReviewAction.accept


def oracle_reviewer(item: ReviewItem, context: ReviewContext) -> ReviewAction:
    """Release exactly the writes OCMR's audit labels false quarantines.

    A ceiling: it consults the benchmark's ``expected_conflict`` labels, which no
    deployable reviewer has. Reported to bound the recoverable recall, never as a
    proposed policy.
    """
    if context.expects_conflict:
        return ReviewAction.quarantine
    conflicting = list(item.decision.features.incumbent_ids) or list(
        item.ocmr_verdict.conflicting_ids
    )
    return ReviewAction.supersede if conflicting else ReviewAction.accept


def release_all_reviewer(item: ReviewItem, context: ReviewContext) -> ReviewAction:
    """Release every held write, exercising no judgment whatsoever.

    A **control, not a policy.** OCMR scores task success as the fraction of
    expected answer tokens found in a haystack built from the answer plus retrieved
    item text, so admitting more assertions mechanically enlarges the haystack. If
    this control matches a reviewer that adjudicates, then the measured recall
    recovery is an artifact of admitting more content rather than of admitting the
    *right* content, and the review step is contributing nothing.
    """
    conflicting = list(item.decision.features.incumbent_ids) or list(
        item.ocmr_verdict.conflicting_ids
    )
    return ReviewAction.supersede if conflicting else ReviewAction.accept


def uphold_all_reviewer(item: ReviewItem, context: ReviewContext) -> ReviewAction:
    """Hold every escalated write — the lower control.

    Behaviourally equivalent to OCMR's terminal quarantine, so it isolates the cost
    of *routing* writes to review from the effect of releasing them. Any difference
    between this and B3 is attributable to the router changing OCMR's own decisions
    rather than to review.
    """
    return ReviewAction.quarantine


def make_random_reviewer(release_probability: float, seed: int = 20260812) -> Reviewer:
    """A reviewer that releases at a fixed rate, exercising no judgment.

    This is the control the integrity-retention claim actually needs. ``identity``
    keeps 28% of the contradiction gain while recovering 95% of the recall cost,
    but retention and recovery are *both* monotone in release volume: hold nothing
    and you keep everything at no recall, release everything and you keep nothing
    at full recall. Any reviewer that releases some intermediate fraction lands
    somewhere in between **without discriminating at all**.

    Sweeping ``release_probability`` traces that no-skill frontier. A reviewer only
    demonstrates judgment by sitting *above* it: retaining more integrity than a
    coin flip that releases the same number of writes. Without this comparison,
    "28% kept at 95% recovered" is not evidence of adjudication.

    Args:
        release_probability: Chance of releasing any given escalated write.
        seed: Fixed so the control is reproducible across arms and seeds.
    """
    import random as _random

    def reviewer(item: ReviewItem, context: ReviewContext) -> ReviewAction:
        # Keyed on the write's identity rather than call order, so the same write
        # gets the same verdict in every arm and the comparison stays paired.
        conflicting = list(item.decision.features.incumbent_ids) or list(
            item.ocmr_verdict.conflicting_ids
        )
        candidate = item.candidate
        key = (
            f"{context.example.id}|{candidate.subject_id}|{candidate.predicate}"
            f"|{candidate.object_id}|{sorted(conflicting)}"
        )
        rng = _random.Random(f"{seed}|{key}")
        if rng.random() >= release_probability:
            return ReviewAction.quarantine
        return ReviewAction.supersede if conflicting else ReviewAction.accept

    reviewer.__name__ = f"random_{release_probability:.2f}_reviewer"
    reviewer.__doc__ = (
        f"Control: releases each escalated write with probability "
        f"{release_probability:.2f}, without judgment."
    )
    return reviewer


#: Release rates for the no-skill frontier. 0.0 and 1.0 are omitted because
#: ``uphold_all`` and ``release_all`` already cover those endpoints exactly.
RANDOM_RELEASE_RATES: tuple[float, ...] = (0.25, 0.5, 0.75)

#: Reviewer registry. ``release_all``, ``uphold_all`` and the ``random_*`` rows are
#: controls that bound the measurement; only ``identity`` is deployable, and
#: ``oracle`` is a ceiling.
REVIEWERS: dict[str, Reviewer] = {
    "identity": identity_reviewer,
    "oracle": oracle_reviewer,
    "release_all": release_all_reviewer,
    "uphold_all": uphold_all_reviewer,
    **{
        f"random{int(rate * 100)}": make_random_reviewer(rate)
        for rate in RANDOM_RELEASE_RATES
    },
}


# --------------------------------------------------------------------------- #
# The governed runner
# --------------------------------------------------------------------------- #
@dataclass
class ArmResult:
    """One arm's outcome on OCMR's benchmark."""

    arm: str
    task_success: float
    contradiction_rate: float
    constraint_violations: float
    typed_violations: int
    n_records: int
    writes_accepted: int
    writes_superseded: int
    writes_quarantined: int
    writes_rejected: int
    n_escalated: int
    n_released: int
    n_upheld: int
    review_rate_per_100_writes: float
    n_candidates: int
    #: Volume-proof cross-checks. OCMR's task success is answer-token recall over a
    #: haystack that includes retrieved text, so admitting more assertions raises it
    #: mechanically. These penalize admitting *wrong* content, so they distinguish a
    #: genuine recall gain from a volume artifact.
    hallucination_rate: float | None = None
    factual_precision: float | None = None
    supporting_precision: float | None = None
    conflict_surfacing_rate: float | None = None
    per_category: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "arm": self.arm,
            "description": ARM_DESCRIPTIONS.get(self.arm, self.arm),
            "task_success": self.task_success,
            "contradiction_rate": self.contradiction_rate,
            "constraint_violations": self.constraint_violations,
            "typed_violations": self.typed_violations,
            "n_records": self.n_records,
            "n_candidates": self.n_candidates,
            "writes": {
                "accepted": self.writes_accepted,
                "superseded": self.writes_superseded,
                "quarantined": self.writes_quarantined,
                "rejected": self.writes_rejected,
            },
            "escalated": self.n_escalated,
            "released": self.n_released,
            "upheld": self.n_upheld,
            "review_rate_per_100_writes": self.review_rate_per_100_writes,
            "volume_proof_checks": {
                "memory_induced_hallucination_rate": self.hallucination_rate,
                "factual_precision": self.factual_precision,
                "supporting_evidence_precision": self.supporting_precision,
                "conflict_surfacing_rate": self.conflict_surfacing_rate,
                "note": (
                    "Task success is answer-token recall over a haystack that "
                    "includes retrieved text, so it rises with the volume of "
                    "admitted memory. These metrics penalize admitting incorrect "
                    "content and so separate a real gain from a volume artifact."
                ),
            },
            "per_category_task_success": dict(self.per_category),
        }


class GovernedArmRunner(BaselineRunner):
    """Runs OCMR's benchmark with the escalation router installed on an arm.

    Subclasses :class:`BaselineRunner` so scoring, logging, and record shape are
    byte-identical to OCMR's published harness. The only additions are installing
    the router on the container and adjudicating the review queue after each
    example is ingested.
    """

    def __init__(
        self,
        *,
        policy: EscalationPolicy | None = None,
        reviewer: Reviewer = identity_reviewer,
        escalate_all_quarantines: bool = False,
        extractor: Any = None,
        embeddings: Any = None,
        **kwargs: Any,
    ) -> None:
        """Create a governed runner.

        Args:
            policy: The escalation policy. Defaults to the registered prior.
            reviewer: How held writes are adjudicated.
            escalate_all_quarantines: When ``True`` the router is bypassed and
                every OCMR quarantine is escalated — the ``B3Q`` ceiling arm.
            extractor: Optional W1 extractor shared across arms. Injecting one
                Qwen instance (ideally disk-cached) is what lets this reproduce
                OCMR's published numbers instead of the offline mock's.
            embeddings: Optional embedding provider shared across arms.
        """
        super().__init__(**kwargs)
        self.policy = policy or EscalationPolicy(PolicyParameters())
        self.reviewer = reviewer
        self.escalate_all_quarantines = escalate_all_quarantines
        self.extractor = extractor
        self.embeddings = embeddings
        self.telemetry: dict[str, Any] = {}

    def _container(self, settings: Any) -> CoreContainer:
        """Build a container, injecting the shared extractor/embeddings when given.

        Every arm must share one extractor instance so governance comparisons are
        not confounded by extraction differences — the same invariant OCMR's own
        harness maintains.
        """
        kwargs: dict[str, Any] = {}
        if self.extractor is not None:
            kwargs["extractor"] = self.extractor
        if self.embeddings is not None:
            kwargs["embeddings"] = self.embeddings
        return CoreContainer(settings, **kwargs)

    def run(
        self, examples: list[BenchmarkExample], baselines: Iterable[str] = ("B3R",)
    ) -> list[dict]:
        """Run each arm over ``examples``, returning OCMR-shaped result records."""
        records: list[dict] = []
        self.telemetry = {}

        for arm in baselines:
            base = "B3" if arm in GOVERNED_ARMS else arm
            settings = self._settings_factory().model_copy(
                update=baseline_settings_overrides(base)
            )
            container = self._container(settings)
            strategy = build_baseline(base, container)

            queue: ReviewQueue | None = None
            governed: GovernedCommitManager | None = None
            authored_by: dict[str, str] = {}
            counts = {
                "candidates": 0,
                "accepted": 0,
                "superseded": 0,
                "quarantined": 0,
                "rejected": 0,
                "escalated": 0,
                "released": 0,
                "upheld": 0,
            }

            if arm in GOVERNED_ARMS:
                queue, governed = self._install(container, arm)

            for example in examples:
                write_counts = self._ingest_sessions(strategy, example)
                for key in ("candidates", "accepted", "superseded", "quarantined", "rejected"):
                    counts[key] += int(write_counts.get(key, 0) or 0)

                if queue is not None:
                    escalated, released, upheld = self._adjudicate(
                        queue, example, authored_by
                    )
                    counts["escalated"] += escalated
                    counts["released"] += released
                    counts["upheld"] += upheld

                # Record authorship after adjudication so a release is visible to
                # later examples as this example's assertion.
                self._record_authorship(container, example, authored_by)

                for q_index, question in enumerate(example.questions):
                    records.append(
                        self._run_question(
                            arm,
                            strategy,
                            example,
                            q_index,
                            question,
                            write_quarantined=write_counts["quarantined"],
                        )
                    )

            violations, accepted_count = durable_constraint_violations(container)
            self.telemetry[arm] = {
                "counts": counts,
                "durable_violations": violations,
                "accepted_count": accepted_count,
                "typed_violations": typed_violations(container).total,
                "n_routing_decisions": len(governed.decisions) if governed else 0,
            }
        return records

    # -- installation ------------------------------------------------------
    def _install(
        self, container: CoreContainer, arm: str
    ) -> tuple[ReviewQueue, GovernedCommitManager]:
        """Install the router and review queue on ``container``."""
        inner = container.commit_manager
        queue = ReviewQueue(
            commit_manager=inner,
            quarantine_store=container.quarantine_store,
            graph=container.graph,
            repo=container.repo,
        )
        router = RiskAdaptiveRouter(
            self.policy,
            feature_extractor=FeatureExtractor(settings=container.settings),
            condition=arm,
            settings=container.settings,
        )
        governed = GovernedCommitManager(
            inner=inner,
            router=router,
            graph=container.graph,
        )

        if self.escalate_all_quarantines:
            # B3Q: bypass the router's selectivity. Every write OCMR would
            # quarantine is escalated instead, which is what "review the whole
            # quarantine stream" means.
            def commit(candidate: CandidateAssertion, vr: Any, *, created_at: Any = None) -> Any:
                action = vr.recommended_action or ("accept" if vr.valid else "reject")
                if action != "quarantine":
                    return inner.commit(candidate, vr, created_at=created_at)
                decision = router.decide(
                    candidate, vr, container.graph, WriteContext()
                )
                from dataclasses import replace as _replace

                decision = _replace(decision, tier=Tier.review)
                governed.decisions.append(decision)
                outcome = inner.commit(candidate, vr, created_at=created_at)
                queue.enqueue(candidate, decision, outcome, ocmr_verdict=vr)
                return outcome

            governed.commit = commit  # type: ignore[method-assign]
        else:
            def review_hook(
                candidate: CandidateAssertion, decision: Any, outcome: Any
            ) -> None:
                queue.enqueue(candidate, decision, outcome)

            governed.review_hook = review_hook

        container.commit_manager = governed
        container.write_pipeline.commit_manager = governed
        return queue, governed

    # -- adjudication ------------------------------------------------------
    def _adjudicate(
        self,
        queue: ReviewQueue,
        example: BenchmarkExample,
        authored_by: dict[str, str],
    ) -> tuple[int, int, int]:
        """Adjudicate every pending review item for the example just ingested."""
        expects_conflict = any(
            bool(getattr(q, "expected_conflict", False)) for q in example.questions
        )
        context = ReviewContext(
            example=example, authored_by=authored_by, expects_conflict=expects_conflict
        )
        pending = queue.pending()
        released = 0
        for item in pending:
            action = self.reviewer(item, context)
            record = queue.adjudicate(
                item.item_id, action, analyst_id="benchmark-reviewer"
            )
            if record.released:
                released += 1
        return len(pending), released, len(pending) - released

    @staticmethod
    def _record_authorship(
        container: CoreContainer, example: BenchmarkExample, authored_by: dict[str, str]
    ) -> None:
        """Attribute every currently-accepted assertion to its first author."""
        try:
            accepted = container.repo.list_assertions("accepted")
        except Exception:  # pragma: no cover - defensive
            return
        for assertion in accepted:
            authored_by.setdefault(assertion.id, example.id)


# --------------------------------------------------------------------------- #
# Fitting the router on OCMR's own benchmark
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BenchmarkCase:
    """One candidate write from OCMR's benchmark with its escalation label.

    The escalation label is kept **separate** from ``case.gold_tier`` deliberately.
    In RAHGM's tier vocabulary "hold without review" and "hold for review" are the
    same durable transition, so a tier cannot express whether a human was needed.
    Folding the label into the tier makes every quarantine look review-worthy and
    turns any subsequent fit into a quarantine detector.
    """

    case: Any
    quarantined: bool
    false_quarantine: bool
    example_id: str

    @property
    def should_escalate(self) -> bool:
        """Whether a human should have seen this write."""
        return self.false_quarantine


def collect_benchmark_cases(
    examples: Sequence[BenchmarkExample],
    *,
    extractor: Any = None,
    embeddings: Any = None,
    settings_factory: Any = None,
) -> list["BenchmarkCase"]:
    """Collect routing cases from OCMR's benchmark, labelled review-worthy or not.

    A write is **review-worthy** when OCMR quarantines it *and* the owning example's
    questions expect no conflict — OCMR's own false-quarantine heuristic, and the
    population its §V-I audit counts. A quarantine inside a conflict-expecting
    example is correct autonomous enforcement and needs no human.

    Learning this distinction is precisely the selectivity claim: escalate the false
    quarantines, leave the genuine ones held. Features come from a pass through
    plain OCMR, so the labels never leak into the inputs.
    """
    from ocm.evaluation.rahgm.replay import ocmr_verdict  # noqa: F401  (parity check)
    from ocm.governance.conditions import AutonomousOcmrRouter
    from ocm.governance.policy import RoutingCase, compute_guards

    settings = (settings_factory or _default_settings)().model_copy(
        update=baseline_settings_overrides("B3")
    )
    container_kwargs: dict[str, Any] = {}
    if extractor is not None:
        container_kwargs["extractor"] = extractor
    if embeddings is not None:
        container_kwargs["embeddings"] = embeddings
    container = CoreContainer(settings, **container_kwargs)
    strategy = build_baseline("B3", container)
    features = FeatureExtractor(settings=container.settings)
    router = AutonomousOcmrRouter(feature_extractor=features, settings=container.settings)
    governed = GovernedCommitManager(
        inner=container.commit_manager, router=router, graph=container.graph
    )
    container.commit_manager = governed
    container.write_pipeline.commit_manager = governed

    cases: list[BenchmarkCase] = []
    for example in examples:
        expects_conflict = any(
            bool(getattr(q, "expected_conflict", False)) for q in example.questions
        )
        seen = len(governed.decisions)
        for session in example.sessions:
            strategy.write(session.input, f"{example.id}:{session.session_id}")
        for decision in governed.decisions[seen:]:
            quarantined = decision.tier is Tier.review
            cases.append(
                BenchmarkCase(
                    case=RoutingCase(
                        features=decision.features,
                        guards=decision.guards,
                        gold_tier=decision.tier,
                        consequential=bool(quarantined and expects_conflict),
                    ),
                    quarantined=quarantined,
                    # OCMR's own false-quarantine heuristic.
                    false_quarantine=quarantined and not expects_conflict,
                    example_id=example.id,
                )
            )
    return cases


def fit_policy_on_benchmark(
    examples: Sequence[BenchmarkExample],
    *,
    dev_fraction: float = 0.4,
    iterations: int = 3000,
    extractor: Any = None,
    embeddings: Any = None,
    settings_factory: Any = None,
) -> dict[str, Any]:
    """Fit the escalation policy on a held-out split of OCMR's benchmark.

    Examples are split by category-stratified index so the development and test
    halves share no trajectory, matching OCMR's own no-overlap protocol.

    Returns the fitted parameters plus the development diagnostics, so a caller can
    see whether the features can separate false from genuine quarantines at all
    before reading any test-set result.
    """
    from ocm.governance.policy import (
        build_training_samples,
        fit_policy,
        select_thresholds,
    )

    by_category: dict[str, list[BenchmarkExample]] = {}
    for example in examples:
        by_category.setdefault(example.category, []).append(example)

    dev: list[BenchmarkExample] = []
    for category_examples in by_category.values():
        cut = max(1, int(round(len(category_examples) * dev_fraction)))
        dev.extend(category_examples[:cut])

    dev_ids = {e.id for e in dev}
    test = [e for e in examples if e.id not in dev_ids]

    from ocm.governance.policy import TrainingSample

    dev_cases = collect_benchmark_cases(
        dev, extractor=extractor, embeddings=embeddings, settings_factory=settings_factory
    )
    n_quarantined = sum(1 for c in dev_cases if c.quarantined)
    n_false = sum(1 for c in dev_cases if c.false_quarantine)

    # Labels come from the explicit escalation flag, not from the tier. A genuine
    # quarantine is a negative: OCMR was right to hold it and no human was needed.
    samples = [
        TrainingSample(
            features=c.case.features,
            label=1 if c.should_escalate else 0,
            consequential=c.case.consequential,
        )
        for c in dev_cases
    ]
    fit = fit_policy(samples, iterations=iterations)

    # Threshold selection needs the same label, so the routing cases are rebuilt
    # with gold_tier expressing "should escalate" rather than "what OCMR did".
    from ocm.governance.policy import RoutingCase as _RoutingCase

    selection_cases = [
        _RoutingCase(
            features=c.case.features,
            guards=c.case.guards,
            gold_tier=Tier.review if c.should_escalate else Tier.accept,
            consequential=c.case.consequential,
        )
        for c in dev_cases
    ]
    selection = select_thresholds(selection_cases, fit.params)
    params = fit.params.with_thresholds(selection.tau_l, selection.tau_h).project()

    return {
        "params": params,
        "fit": fit.as_dict(),
        "thresholds": selection.as_dict(),
        "n_dev_examples": len(dev),
        "n_test_examples": len(test),
        "n_dev_cases": len(dev_cases),
        "n_dev_quarantined": n_quarantined,
        "n_dev_false_quarantine": n_false,
        "n_dev_genuine_quarantine": n_quarantined - n_false,
        "dev_examples": dev,
        "test_examples": test,
    }


def separability_report(
    examples: Sequence[BenchmarkExample],
    params: PolicyParameters,
    *,
    extractor: Any = None,
    embeddings: Any = None,
    settings_factory: Any = None,
) -> dict[str, Any]:
    """Can the constraint-failure features tell a false quarantine from a genuine one?

    Restricted to writes OCMR quarantines, since those are the only ones a release
    policy could act on. Precision is the share of escalated quarantines that were
    genuinely false; the base rate is what a policy escalating *every* quarantine
    would score, so precision at or below the base rate means the features carry no
    discriminating signal.
    """
    policy = EscalationPolicy(params)
    cases = collect_benchmark_cases(
        examples, extractor=extractor, embeddings=embeddings, settings_factory=settings_factory
    )
    quarantined = [c for c in cases if c.quarantined]
    escalated = [
        c
        for c in quarantined
        if policy.route(c.case.features, c.case.guards)[0] is Tier.review
    ]
    n_false = sum(1 for c in quarantined if c.false_quarantine)
    tp = sum(1 for c in escalated if c.false_quarantine)

    precision = tp / len(escalated) if escalated else 0.0
    recall = tp / n_false if n_false else 0.0
    base_rate = n_false / len(quarantined) if quarantined else 0.0
    return {
        "n_cases": len(cases),
        "n_quarantined": len(quarantined),
        "n_false_quarantine": n_false,
        "n_genuine_quarantine": len(quarantined) - n_false,
        "base_rate_false": base_rate,
        "n_escalated": len(escalated),
        "escalation_share_of_quarantines": (
            len(escalated) / len(quarantined) if quarantined else 0.0
        ),
        "precision": precision,
        "recall": recall,
        "lift_over_base_rate": precision - base_rate,
    }


# --------------------------------------------------------------------------- #
# The experiment
# --------------------------------------------------------------------------- #
def _per_category(records: Sequence[dict], arm: str) -> dict[str, float]:
    """Task success per benchmark category for one arm."""
    out: dict[str, list[float]] = {}
    for record in records:
        if record.get("baseline_name") != arm:
            continue
        out.setdefault(str(record.get("category")), []).append(
            float(record.get("score", 0.0))
        )
    return {
        category: 100.0 * sum(scores) / len(scores)
        for category, scores in sorted(out.items())
        if scores
    }


def run_ocmr_escalation_arm(
    *,
    per_category: int = 25,
    seed: int = 1337,
    arms: Sequence[str] = ARMS,
    reviewer: str = "identity",
    params: PolicyParameters | None = None,
    examples: list[BenchmarkExample] | None = None,
    extractor: Any = None,
    embeddings: Any = None,
    settings_factory: Any = None,
) -> dict[str, Any]:
    """Run OCMR's benchmark with and without risk-adaptive escalation.

    Args:
        per_category: Trajectories per benchmark category (OCMR uses 25).
        seed: Benchmark seed.
        arms: Which arms to run.
        reviewer: ``"identity"`` (deployable) or ``"oracle"`` (ceiling).
        params: Escalation policy parameters.
        examples: Pre-generated benchmark examples.

    Returns:
        A report dict with one entry per arm plus the B3-to-B3R contrast that
        decides whether escalation recovers OCMR's recall cost.
    """
    from ocm.evaluation.benchmark import BenchmarkGenerator

    _seed_everything(seed)
    examples = examples or BenchmarkGenerator(seed=seed).generate(
        per_category=per_category
    )
    reviewer_fn = REVIEWERS[reviewer]
    policy = EscalationPolicy(params or PolicyParameters())

    results: dict[str, ArmResult] = {}
    for arm in arms:
        runner = GovernedArmRunner(
            policy=policy,
            reviewer=reviewer_fn,
            escalate_all_quarantines=(arm == "B3Q"),
            settings_factory=settings_factory or _default_settings,
            extractor=extractor,
            embeddings=embeddings,
        )
        records = runner.run(examples, baselines=[arm])
        telemetry = runner.telemetry[arm]
        counts = telemetry["counts"]

        accepted_count = telemetry["accepted_count"] or 1
        violation_rate = 100.0 * telemetry["durable_violations"] / accepted_count
        decisive = decisive_metrics(records, constraint_violation_rate=violation_rate)
        n_writes = counts["candidates"] or 1

        reported = MetricsReporter().compute(records).get(arm, {})
        answer = reported.get("answer", {}) or {}
        retrieval = reported.get("retrieval", {}) or {}

        results[arm] = ArmResult(
            arm=arm,
            task_success=float(decisive.get("task_success", float("nan"))),
            contradiction_rate=float(decisive.get("contradiction_rate", float("nan"))),
            constraint_violations=float(
                decisive.get("constraint_violations", float("nan"))
            ),
            typed_violations=int(telemetry["typed_violations"]),
            n_records=len(records),
            writes_accepted=counts["accepted"],
            writes_superseded=counts["superseded"],
            writes_quarantined=counts["quarantined"],
            writes_rejected=counts["rejected"],
            n_escalated=counts["escalated"],
            n_released=counts["released"],
            n_upheld=counts["upheld"],
            review_rate_per_100_writes=100.0 * counts["escalated"] / n_writes,
            n_candidates=counts["candidates"],
            hallucination_rate=answer.get("memory_induced_hallucination_rate"),
            factual_precision=answer.get("factual_precision"),
            supporting_precision=retrieval.get("supporting_evidence_precision"),
            conflict_surfacing_rate=answer.get("conflict_surfacing_rate"),
            per_category=_per_category(records, arm),
        )

    contrast: dict[str, Any] | None = None
    if "B3" in results and "B3R" in results:
        b3, b3r = results["B3"], results["B3R"]
        ungoverned = results.get("B0") or results.get("B2")
        recoverable = (
            ungoverned.task_success - b3.task_success if ungoverned else float("nan")
        )
        recovered = b3r.task_success - b3.task_success
        contrast = {
            "task_success_b3": b3.task_success,
            "task_success_b3r": b3r.task_success,
            "task_success_ungoverned": (
                ungoverned.task_success if ungoverned else None
            ),
            "recall_gap_to_close": recoverable,
            "recall_recovered": recovered,
            "fraction_of_gap_recovered": (
                recovered / recoverable
                if recoverable and recoverable == recoverable and recoverable != 0
                else float("nan")
            ),
            "violation_delta": b3r.constraint_violations - b3.constraint_violations,
            "contradiction_delta": b3r.contradiction_rate - b3.contradiction_rate,
            "review_rate_per_100_writes": b3r.review_rate_per_100_writes,
            "released": b3r.n_released,
            "upheld": b3r.n_upheld,
        }

    return {
        "experiment": "ocmr_benchmark_escalation_arm",
        "in_paper": False,
        "claim": (
            "Selective escalation with review-and-release recovers the task-success "
            "cost OCMR pays for conservative quarantine, without reintroducing "
            "durable-state violations."
        ),
        "protocol": {
            "per_category": per_category,
            "seed": seed,
            "reviewer": reviewer,
            "reviewer_note": (
                "identity: deployable, reads only the review item and assertion "
                "authorship. oracle: ceiling, reads the benchmark's "
                "expected_conflict labels."
            ),
            "n_examples": len(examples),
            "policy": policy.params.as_dict(),
        },
        "arms": {name: result.as_dict() for name, result in results.items()},
        "contrast": contrast,
    }

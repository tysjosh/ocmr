"""Routing ablation and risk–coverage analysis (paper §4.2, Table 4).

Isolates the contribution of each routing signal by disabling it and re-routing the
*same* held-out cases. Because :class:`~ocm.governance.policy.RoutingCase` bundles
the features, the guards, and the gold label, the ablation is a pure re-scoring: no
replay is repeated, so every variant sees byte-identical inputs and the differences
are attributable to the routing signal alone.

Variants (Table 4):

* **Full RAHGM** — the fitted policy.
* **Quarantine-only escalation** — escalate exactly when OCMR would quarantine.
  This is the autonomous-OCMR routing signal, and its false-quarantine rate is the
  problem the paper set out to fix.
* **Scalar threshold** — a single tuned confidence threshold with no failure
  pattern, no consequence, and no reversibility.
* **Failure pattern only** — ``f(u)`` and ``k`` with the consequence, reversibility,
  and authority terms zeroed.
* **Reversibility only** — the reversibility discount with the failure pattern
  zeroed.
* **Without consequence** / **without authority** — leave-one-signal-out.

Each variant's thresholds are re-selected on the development partition under its
own signal, so a variant is never handicapped by thresholds tuned for a different
score.

Two further variants are **not in the paper**. They quantify remedies for two
specification gaps this evaluation surfaced in eq. (3):

* *Centered reversibility.* Because ``β_v ≥ 0`` and ``v ∈ [0,1]``, the term
  ``−β_v·v`` spans ``[−β_v, 0]``. Reversibility is therefore a one-sided discount:
  the least reversible write is only as risky as a write carrying no reversibility
  information at all, and irreversibility never *adds* risk. Centering the term at
  ``v = ½`` makes it two-sided.
* *Bounded discounts.* Eq. (3) bounds ``β_v`` and ``β_a`` below at zero but not
  above, and nothing relates them to ``β_q``. When a fitted discount exceeds the
  consequence weight, a well-attributed irreversible write can be discounted into
  autonomous commitment however consequential it is. Capping each discount at
  ``β_q`` measures the cost of closing that gap.

Requirements: 13.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

from ocm.evaluation.rahgm.corpus import Partition, RahgmCorpus
from ocm.evaluation.rahgm.metrics import WriteRecord, risk_coverage_auc
from ocm.governance.policy import (
    MCR_CEILING,
    EscalationPolicy,
    PolicyParameters,
    RoutingCase,
    Tier,
    select_thresholds,
)

#: Confidence threshold grid for the scalar-threshold variant.
_SCALAR_GRID = tuple(round(0.50 + 0.01 * i, 2) for i in range(51))


@dataclass(frozen=True)
class AblationVariant:
    """One routing variant: how to derive its parameters and how to route."""

    key: str
    name: str
    #: Transform the fitted parameters into this variant's parameters.
    transform: Callable[[PolicyParameters], PolicyParameters]
    #: When set, routing ignores the score entirely and uses this rule instead.
    rule: Callable[[RoutingCase], Tier] | None = None
    retune_thresholds: bool = True
    #: MCR ceiling used when re-selecting thresholds on development (eq. 5).
    mcr_ceiling: float = MCR_CEILING
    note: str = ""


def _zero_modifiers(params: PolicyParameters) -> PolicyParameters:
    """Failure pattern only: drop consequence, reversibility, and authority."""
    return replace(params, beta_q=0.0, beta_v=0.0, beta_a=0.0)


def _zero_failures(params: PolicyParameters) -> PolicyParameters:
    """Reversibility only: drop the failure pattern, consequence, and authority."""
    return replace(
        params, beta_f=(0.0, 0.0, 0.0, 0.0, 0.0), beta_k=0.0, beta_q=0.0, beta_a=0.0
    )


def _quarantine_only(case: RoutingCase) -> Tier:
    """Escalate exactly when OCMR's own verdict would quarantine.

    Reconstructed from the guards and features rather than from a stored verdict:
    a prohibited write rejects, any hard failure or unresolved check quarantines,
    and anything clean commits. This is OCMR's conservative behavior, whose false
    quarantines the paper measures.
    """
    if case.guards.g:
        return Tier.reject
    if case.features.k > 0:
        return Tier.review
    return Tier.accept


@dataclass
class VariantResult:
    """One variant's ablation outcome."""

    key: str
    name: str
    mcr: float
    review_rate: float
    risk_coverage_auc: float
    routing_accuracy: float
    false_escalation_rate: float
    false_quarantine_rate: float
    queue_precision: float
    queue_recall: float
    params: dict[str, Any] | None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "key": self.key,
            "name": self.name,
            "mcr": self.mcr,
            "review_rate": self.review_rate,
            "risk_coverage_auc": self.risk_coverage_auc,
            "routing_accuracy": self.routing_accuracy,
            "false_escalation_rate": self.false_escalation_rate,
            "false_quarantine_rate": self.false_quarantine_rate,
            "queue_precision": self.queue_precision,
            "queue_recall": self.queue_recall,
            "params": self.params,
            "note": self.note,
        }


def build_variants() -> list[AblationVariant]:
    """The Table 4 variant list, in reporting order."""
    return [
        AblationVariant(
            key="full",
            name="Full RAHGM",
            transform=lambda p: p,
            note="failure pattern, consequence, reversibility, and authority",
        ),
        AblationVariant(
            key="quarantine_only",
            name="Quarantine-only escalation",
            transform=lambda p: p,
            rule=_quarantine_only,
            retune_thresholds=False,
            note="escalate exactly when OCMR would quarantine (autonomous OCMR)",
        ),
        AblationVariant(
            key="scalar_threshold",
            name="Scalar threshold",
            transform=lambda p: p,
            rule=None,
            retune_thresholds=False,
            note="tuned scalar confidence threshold; no failure pattern",
        ),
        AblationVariant(
            key="failure_pattern_only",
            name="Failure pattern only",
            transform=_zero_modifiers,
            note="f(u) and k only",
        ),
        AblationVariant(
            key="reversibility_only",
            name="Reversibility only",
            transform=_zero_failures,
            note="reversibility discount only",
        ),
        AblationVariant(
            key="without_consequence",
            name="Without consequence",
            transform=lambda p: replace(p, beta_q=0.0),
            note="leave-one-out: consequence",
        ),
        AblationVariant(
            key="without_authority",
            name="Without authority",
            transform=lambda p: replace(p, beta_a=0.0),
            note="leave-one-out: authority",
        ),
        # The three variants below are not in the paper. The first two are
        # *rejected* hypotheses, retained because they are the natural fixes a
        # reader would propose; the third is the one that works.
        AblationVariant(
            key="centered_reversibility",
            name="Centered reversibility (rejected)",
            transform=lambda p: replace(p, reversibility_center=0.5),
            note=(
                "rejected: makes the reversibility term two-sided, but does not "
                "beat tuning tau_l on the unmodified equation"
            ),
        ),
        AblationVariant(
            key="bounded_discounts",
            name="Bounded discounts (rejected)",
            transform=_bound_discounts,
            note=(
                "rejected: caps each displayed discount at the consequence weight, "
                "but does not beat tuning tau_l on the unmodified equation"
            ),
        ),
        AblationVariant(
            key="tightened_threshold",
            name="Tightened MCR ceiling (proposed)",
            transform=lambda p: p,
            mcr_ceiling=0.01,
            note=(
                "eq. (3) unchanged; eq. (5) selected at a 0.01 MCR ceiling instead "
                "of 0.02 to leave out-of-sample headroom"
            ),
        ),
    ]


def _bound_discounts(params: PolicyParameters) -> PolicyParameters:
    """Cap each displayed discount at the consequence weight.

    Eq. (3) requires ``β_v, β_a ≥ 0`` but sets no upper bound relative to ``β_q``.
    When a discount exceeds the consequence weight, a well-attributed or cheaply
    reversible write can be discounted into autonomous commitment however
    consequential it is. This variant measures what capping them costs.
    """
    cap = params.beta_q
    return replace(
        params, beta_v=min(params.beta_v, cap), beta_a=min(params.beta_a, cap)
    )


def _records_from_cases(
    cases: Sequence[RoutingCase],
    router: Callable[[RoutingCase], tuple[Tier, float]],
) -> list[WriteRecord]:
    """Re-route cases and wrap them as records so metric code is shared.

    The ablation measures *routing*, so the final transition is taken to be the
    routed tier for autonomous tiers, and the gold transition for escalated cases
    that a reviewer would resolve. That is the same perfect-reviewer control
    Experiment 1 uses, applied consistently across variants.
    """
    records: list[WriteRecord] = []
    for index, case in enumerate(cases):
        tier, risk = router(case)
        escalated = tier is Tier.review
        final = case.gold_tier if escalated else tier
        records.append(
            WriteRecord(
                write_id=f"ablation-{index:05d}",
                scenario_id="ablation",
                partition="test",
                write_class="",
                template="",
                gold=case.gold_tier,
                routed=tier,
                final=final,
                escalated=escalated,
                consequential=case.consequential,
                risk=risk,
            )
        )
    return records


def _summarize(
    variant: AblationVariant,
    records: Sequence[WriteRecord],
    params: PolicyParameters | None,
) -> VariantResult:
    """Compute the Table 4 columns for one variant."""
    n = len(records) or 1
    consequential = [r for r in records if r.consequential]
    missed = [r for r in consequential if r.error and not r.escalated]
    escalated = [r for r in records if r.escalated]
    review_worthy = [r for r in records if r.review_worthy]
    true_positives = [r for r in escalated if r.review_worthy]

    return VariantResult(
        key=variant.key,
        name=variant.name,
        mcr=len(missed) / len(consequential) if consequential else 0.0,
        review_rate=len(escalated) / n,
        risk_coverage_auc=risk_coverage_auc(records),
        routing_accuracy=sum(1 for r in records if not r.routing_error) / n,
        false_escalation_rate=sum(1 for r in records if r.false_escalation) / n,
        false_quarantine_rate=sum(1 for r in records if r.false_quarantine) / n,
        queue_precision=len(true_positives) / len(escalated) if escalated else 0.0,
        queue_recall=len(true_positives) / len(review_worthy) if review_worthy else 0.0,
        params=params.as_dict() if params else None,
        note=variant.note,
    )


def _tune_scalar_threshold(
    dev_cases: Sequence[RoutingCase],
) -> float:
    """Tune the scalar-threshold variant on development cases.

    The scalar signal available without a failure pattern is the escalation
    probability's crudest proxy: consequence alone would leak the rubric, so the
    variant uses ``1 − reversibility·authority``, a single opaque number combining
    the two scalar quantities a confidence score would plausibly encode. The
    threshold maximizing F₂ under the MCR ceiling is selected.
    """
    best_threshold = 0.5
    best = (-1.0, 1.0)
    for threshold in _SCALAR_GRID:
        tp = fp = fn = 0
        consequential = missed = 0
        for case in dev_cases:
            if case.guards.g:
                tier = Tier.reject
            else:
                score = _scalar_score(case)
                tier = Tier.review if score >= threshold else Tier.accept
            escalates = tier is Tier.review
            if case.review_worthy:
                tp += 1 if escalates else 0
                fn += 0 if escalates else 1
            elif escalates:
                fp += 1
            if case.consequential:
                consequential += 1
                if not escalates and tier is not case.gold_tier:
                    missed += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        denominator = 4.0 * precision + recall
        f2 = (5.0 * precision * recall / denominator) if denominator else 0.0
        mcr = missed / consequential if consequential else 0.0
        if mcr <= 0.02 and (f2, -mcr) > best:
            best = (f2, -mcr)
            best_threshold = threshold
    return best_threshold


def _scalar_score(case: RoutingCase) -> float:
    """The scalar-threshold variant's single opaque score."""
    features = case.features
    return 1.0 - (features.reversibility * features.authority)


def run_ablation(
    corpus: RahgmCorpus,
    params: PolicyParameters,
    *,
    dev_cases: Sequence[RoutingCase] | None = None,
    test_cases: Sequence[RoutingCase] | None = None,
) -> dict[str, Any]:
    """Run the routing ablation and populate Table 4.

    Args:
        corpus: The evaluation corpus.
        params: The fitted full-RAHGM parameters.
        dev_cases: Development cases, collected if omitted.
        test_cases: Held-out test cases, collected if omitted.

    Returns:
        A report dict with one entry per variant plus the strongest-baseline
        comparison the paper reports.
    """
    from ocm.evaluation.rahgm.replay import collect_routing_cases

    dev_cases = list(
        dev_cases
        if dev_cases is not None
        else collect_routing_cases(corpus.partition(Partition.dev))
    )
    test_cases = list(
        test_cases
        if test_cases is not None
        else collect_routing_cases(corpus.partition(Partition.test))
    )

    scalar_threshold = _tune_scalar_threshold(dev_cases)
    results: list[VariantResult] = []

    for variant in build_variants():
        variant_params: PolicyParameters | None = variant.transform(params).project()

        if variant.key == "quarantine_only":
            router: Callable[[RoutingCase], tuple[Tier, float]] = lambda case: (
                _quarantine_only(case),
                1.0 if case.features.k > 0 else 0.0,
            )
            variant_params = None
        elif variant.key == "scalar_threshold":
            def router(case: RoutingCase, threshold: float = scalar_threshold) -> tuple[Tier, float]:
                score = _scalar_score(case)
                if case.guards.g:
                    return Tier.reject, score
                return (Tier.review if score >= threshold else Tier.accept), score
            variant_params = None
        else:
            if variant.retune_thresholds:
                selection = select_thresholds(
                    dev_cases, variant_params, mcr_ceiling=variant.mcr_ceiling
                )
                variant_params = variant_params.with_thresholds(
                    selection.tau_l, selection.tau_h
                ).project()
            policy = EscalationPolicy(variant_params)

            def router(case: RoutingCase, policy: EscalationPolicy = policy) -> tuple[Tier, float]:
                tier, _rule, risk = policy.route(case.features, case.guards)
                return tier, risk

        records = _records_from_cases(test_cases, router)
        results.append(_summarize(variant, records, variant_params))

    full = next(r for r in results if r.key == "full")
    # The strongest baseline is chosen among variants that actually route
    # differently from the full policy. A variant with an identical review rate and
    # MCR is the same operating point reached by a different parameterization, so
    # naming it "the strongest baseline" would be an artifact of tie-breaking.
    others = [
        r
        for r in results
        if r.key != "full"
        and not (
            abs(r.review_rate - full.review_rate) < 1e-9
            and abs(r.mcr - full.mcr) < 1e-9
        )
    ]
    strongest = min(others, key=lambda r: r.risk_coverage_auc) if others else None

    return {
        "experiment": "experiment1_routing_ablation",
        "n_test_cases": len(test_cases),
        "n_dev_cases": len(dev_cases),
        "scalar_threshold": scalar_threshold,
        "variants": [r.as_dict() for r in results],
        "full_vs_strongest_baseline": (
            {
                "strongest_baseline": strongest.key,
                "auc_full": full.risk_coverage_auc,
                "auc_baseline": strongest.risk_coverage_auc,
                "mcr_delta_points": 100.0 * (full.mcr - strongest.mcr),
                "review_rate_delta_points": 100.0
                * (full.review_rate - strongest.review_rate),
            }
            if strongest
            else None
        ),
        "leave_one_out": {
            "removing_failure_pattern_mcr_delta_points": 100.0
            * (
                next(r for r in results if r.key == "reversibility_only").mcr - full.mcr
            ),
            "removing_reversibility_review_rate_delta_points": 100.0
            * (
                next(r for r in results if r.key == "failure_pattern_only").review_rate
                - full.review_rate
            ),
            "removing_consequence_mcr_delta_points": 100.0
            * (next(r for r in results if r.key == "without_consequence").mcr - full.mcr),
            "removing_authority_review_rate_delta_points": 100.0
            * (
                next(r for r in results if r.key == "without_authority").review_rate
                - full.review_rate
            ),
        },
    }

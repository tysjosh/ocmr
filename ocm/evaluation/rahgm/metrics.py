"""RAHGM measures — eq. (10) and eq. (11) of paper §3.7.

Primary and integrity metrics over test writes ``i = 1..N``:

* ``MCR = Σ cᵢ·eᵢ·(1 − zᵢ) / Σ cᵢ`` — missed consequential conflicts: a
  consequential case whose final transition is wrong **and** which was not
  escalated. Escalating a hard case is not a miss even if it stays unresolved.
* ``R100 = 100·Σ tᵢ / N`` — reviewer minutes per 100 writes.
* ``DVR = (1/N)·Σ vᵢ`` — durable-state violation rate, with ``vᵢ`` sourced from
  OCMR's existing ``typed_violations`` report so the two papers measure durable
  integrity identically.

Secondary: false escalations, false quarantines (the OCMR failure this work
targets), review-queue precision/recall, correction quality, and the
risk–coverage AUC.

Calibration, eq. (11): ``BS = (1/N)Σ(pᵢ − yᵢ)²`` and
``ECE = Σ_b (|I_b|/N)·|acc(I_b) − conf(I_b)|``.

Requirements: 11.1, 11.2, 11.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ocm.governance.policy import Tier

#: Default number of equal-width bins for ECE.
DEFAULT_ECE_BINS = 10


# --------------------------------------------------------------------------- #
# Per-write record
# --------------------------------------------------------------------------- #
@dataclass
class WriteRecord:
    """One replayed write and everything the metrics need from it.

    Attributes:
        write_id / scenario_id / partition: Provenance.
        write_class / template: Corpus family, for breakdowns.
        gold: The correct transition (the reference standard).
        routed: The tier the condition's router chose.
        final: The transition durable memory actually received. Differs from
            ``routed`` when a reviewer released or rejected a held write.
        escalated: ``zᵢ`` — whether the write was presented for review.
        consequential: ``cᵢ``.
        risk: The escalation probability ``r(u)`` (0.0 for conditions with no score).
        review_minutes: ``tᵢ`` — reviewer minutes, from the review-cost model or
            from the simulated analyst's decision time.
        released: Whether a held write was later released into memory.
        creates_violation: Whether committing this write leaves invalid durable
            state, per the corpus construction.
    """

    write_id: str
    scenario_id: str
    partition: str
    write_class: str
    template: str
    gold: Tier
    routed: Tier
    final: Tier
    escalated: bool
    consequential: bool
    risk: float = 0.0
    review_minutes: float = 0.0
    released: bool = False
    creates_violation: bool = False
    #: OCMR's own verdict for this write, before the router rewrote it. Retained so
    #: an analysis can tell whether the *constraint* outcome changed even when the
    #: routed tier did not — which is how state corruption becomes visible.
    ocmr_action: str | None = None
    ocmr_failed_check: str | None = None

    # -- derived -----------------------------------------------------------
    @property
    def error(self) -> bool:
        """``eᵢ`` — whether the final transition differs from the gold transition."""
        return self.final is not self.gold

    @property
    def routing_error(self) -> bool:
        """Whether the *router* chose a tier other than the gold transition.

        Escalating a case whose gold is ``review`` is correct routing; escalating a
        routine accept is a false escalation.
        """
        return self.routed is not self.gold

    @property
    def false_escalation(self) -> bool:
        """Escalated a write that ground truth says needed no human."""
        return self.escalated and self.gold in (Tier.accept, Tier.supersede)

    @property
    def false_quarantine(self) -> bool:
        """A valid update that never reached durable memory.

        This is the OCMR failure the paper targets: 835 of 1,198 quarantined
        writes were valid updates conservatively held back. A write counts here
        when its gold transition would have admitted it but the final state did
        not.
        """
        return self.gold in (Tier.accept, Tier.supersede) and self.final in (
            Tier.review,
            Tier.reject,
        )

    @property
    def review_worthy(self) -> bool:
        """Whether ground truth says a human should have seen this write."""
        return self.gold is Tier.review

    @property
    def correct_correction(self) -> bool:
        """Correction quality: an authoritative correction that did supersede."""
        return self.gold is Tier.supersede and self.final is Tier.supersede

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "write_id": self.write_id,
            "scenario_id": self.scenario_id,
            "partition": self.partition,
            "write_class": self.write_class,
            "template": self.template,
            "gold": self.gold.value,
            "routed": self.routed.value,
            "final": self.final.value,
            "escalated": self.escalated,
            "consequential": self.consequential,
            "risk": self.risk,
            "review_minutes": self.review_minutes,
            "released": self.released,
            "error": self.error,
            "false_escalation": self.false_escalation,
            "false_quarantine": self.false_quarantine,
            "ocmr_action": self.ocmr_action,
            "ocmr_failed_check": self.ocmr_failed_check,
        }


# --------------------------------------------------------------------------- #
# Aggregate metrics
# --------------------------------------------------------------------------- #
@dataclass
class ReplayMetrics:
    """The full metric set for one condition over one set of write records."""

    n_writes: int
    mcr: float
    r100: float
    dvr: float
    review_rate: float
    false_escalation_rate: float
    false_quarantine_rate: float
    accuracy: float
    routing_accuracy: float
    queue_precision: float
    queue_recall: float
    correction_quality: float
    risk_coverage_auc: float
    durable_violations: int
    n_consequential: int
    n_escalated: int
    n_released: int
    review_minutes_total: float
    typed_violations: dict[str, int] = field(default_factory=dict)
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "n_writes": self.n_writes,
            "mcr": self.mcr,
            "r100": self.r100,
            "dvr": self.dvr,
            "review_rate": self.review_rate,
            "false_escalation_rate": self.false_escalation_rate,
            "false_quarantine_rate": self.false_quarantine_rate,
            "accuracy": self.accuracy,
            "routing_accuracy": self.routing_accuracy,
            "queue_precision": self.queue_precision,
            "queue_recall": self.queue_recall,
            "correction_quality": self.correction_quality,
            "risk_coverage_auc": self.risk_coverage_auc,
            "durable_violations": self.durable_violations,
            "n_consequential": self.n_consequential,
            "n_escalated": self.n_escalated,
            "n_released": self.n_released,
            "review_minutes_total": self.review_minutes_total,
            "typed_violations": dict(self.typed_violations),
            "per_class": {k: dict(v) for k, v in self.per_class.items()},
        }


def compute_metrics(
    records: Sequence[WriteRecord],
    *,
    durable_violations: int = 0,
    typed_violations: dict[str, int] | None = None,
) -> ReplayMetrics:
    """Compute eq. (10) and the secondary outcomes over ``records``.

    Args:
        records: Per-write records from the replay, test partition only.
        durable_violations: ``Σ vᵢ`` — the durable-state violation count from
            ``typed_violations``, summed across scenarios.
        typed_violations: Optional per-type breakdown, carried into the report.

    Returns:
        The :class:`ReplayMetrics` bundle. An empty record set yields an all-zero
        report rather than raising, so a condition that queued nothing is still
        comparable.
    """
    records = list(records)
    n = len(records)
    if n == 0:
        return ReplayMetrics(
            n_writes=0,
            mcr=0.0,
            r100=0.0,
            dvr=0.0,
            review_rate=0.0,
            false_escalation_rate=0.0,
            false_quarantine_rate=0.0,
            accuracy=0.0,
            routing_accuracy=0.0,
            queue_precision=0.0,
            queue_recall=0.0,
            correction_quality=0.0,
            risk_coverage_auc=0.0,
            durable_violations=durable_violations,
            n_consequential=0,
            n_escalated=0,
            n_released=0,
            review_minutes_total=0.0,
            typed_violations=dict(typed_violations or {}),
        )

    consequential = [r for r in records if r.consequential]
    missed = [r for r in consequential if r.error and not r.escalated]
    mcr = len(missed) / len(consequential) if consequential else 0.0

    minutes = sum(r.review_minutes for r in records)
    r100 = 100.0 * minutes / n

    escalated = [r for r in records if r.escalated]
    review_worthy = [r for r in records if r.review_worthy]
    true_positives = [r for r in escalated if r.review_worthy]

    queue_precision = len(true_positives) / len(escalated) if escalated else 0.0
    queue_recall = len(true_positives) / len(review_worthy) if review_worthy else 0.0

    corrections = [r for r in records if r.gold is Tier.supersede]
    correction_quality = (
        sum(1 for r in corrections if r.correct_correction) / len(corrections)
        if corrections
        else 0.0
    )

    return ReplayMetrics(
        n_writes=n,
        mcr=mcr,
        r100=r100,
        dvr=durable_violations / n,
        review_rate=len(escalated) / n,
        false_escalation_rate=sum(1 for r in records if r.false_escalation) / n,
        false_quarantine_rate=sum(1 for r in records if r.false_quarantine) / n,
        accuracy=sum(1 for r in records if not r.error) / n,
        routing_accuracy=sum(1 for r in records if not r.routing_error) / n,
        queue_precision=queue_precision,
        queue_recall=queue_recall,
        correction_quality=correction_quality,
        risk_coverage_auc=risk_coverage_auc(records),
        durable_violations=durable_violations,
        n_consequential=len(consequential),
        n_escalated=len(escalated),
        n_released=sum(1 for r in records if r.released),
        review_minutes_total=minutes,
        typed_violations=dict(typed_violations or {}),
        per_class=_per_class(records),
    )


def _per_class(records: Sequence[WriteRecord]) -> dict[str, dict[str, float]]:
    """Accuracy, review rate, and false-quarantine rate per write class."""
    out: dict[str, dict[str, float]] = {}
    classes = sorted({r.write_class for r in records})
    for name in classes:
        subset = [r for r in records if r.write_class == name]
        count = len(subset)
        out[name] = {
            "n": float(count),
            "accuracy": sum(1 for r in subset if not r.error) / count,
            "review_rate": sum(1 for r in subset if r.escalated) / count,
            "false_quarantine_rate": sum(1 for r in subset if r.false_quarantine) / count,
        }
    return out


# --------------------------------------------------------------------------- #
# Risk–coverage curve
# --------------------------------------------------------------------------- #
def risk_coverage_auc(records: Sequence[WriteRecord]) -> float:
    """Area under the risk–coverage curve (lower is better).

    Writes are sorted by ascending escalation risk. At coverage ``c`` the system
    has committed the ``c`` fraction it considered least risky; the curve value is
    the error rate among those writes, judged against the gold transition. A
    routing signal that ranks genuinely dangerous writes highest keeps the error
    rate near zero over most of the coverage range, so its AUC is small.

    Returns ``0.0`` when every write is riskless (nothing to rank) — the correct
    degenerate value, since no ordering can do better than no errors.
    """
    records = list(records)
    if not records:
        return 0.0

    # Ties are broken by routing error so a condition cannot benefit from an
    # arbitrary ordering among equally scored writes.
    ordered = sorted(records, key=lambda r: (r.risk, r.routing_error))

    coverages: list[float] = []
    error_rates: list[float] = []
    errors = 0
    for i, record in enumerate(ordered, start=1):
        if record.routing_error:
            errors += 1
        coverages.append(i / len(ordered))
        error_rates.append(errors / i)

    area = 0.0
    prev_coverage = 0.0
    prev_error = error_rates[0]
    for coverage, error in zip(coverages, error_rates):
        area += 0.5 * (error + prev_error) * (coverage - prev_coverage)
        prev_coverage, prev_error = coverage, error
    return area


# --------------------------------------------------------------------------- #
# Calibration (eq. 11)
# --------------------------------------------------------------------------- #
def brier_score(confidences: Sequence[float], correctness: Sequence[float]) -> float:
    """``BS = (1/N)Σ(pᵢ − yᵢ)²`` of eq. (11)."""
    pairs = list(zip(confidences, correctness))
    if not pairs:
        return float("nan")
    return sum((float(p) - float(y)) ** 2 for p, y in pairs) / len(pairs)


@dataclass(frozen=True)
class CalibrationBin:
    """One ECE bin."""

    lower: float
    upper: float
    count: int
    accuracy: float
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "accuracy": self.accuracy,
            "confidence": self.confidence,
        }


def calibration_bins(
    confidences: Sequence[float],
    correctness: Sequence[float],
    *,
    bins: int = DEFAULT_ECE_BINS,
) -> list[CalibrationBin]:
    """Equal-width calibration bins over ``[0, 1]``."""
    pairs = [(float(p), float(y)) for p, y in zip(confidences, correctness)]
    out: list[CalibrationBin] = []
    if not pairs:
        return out
    width = 1.0 / bins
    for b in range(bins):
        lower = b * width
        upper = 1.0 if b == bins - 1 else (b + 1) * width
        members = [
            (p, y)
            for p, y in pairs
            if (lower <= p < upper) or (b == bins - 1 and p == 1.0)
        ]
        if not members:
            out.append(CalibrationBin(lower, upper, 0, float("nan"), float("nan")))
            continue
        accuracy = sum(y for _p, y in members) / len(members)
        confidence = sum(p for p, _y in members) / len(members)
        out.append(CalibrationBin(lower, upper, len(members), accuracy, confidence))
    return out


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[float],
    *,
    bins: int = DEFAULT_ECE_BINS,
) -> float:
    """``ECE = Σ_b (|I_b|/N)·|acc(I_b) − conf(I_b)|`` of eq. (11)."""
    pairs = list(zip(confidences, correctness))
    if not pairs:
        return float("nan")
    total = len(pairs)
    ece = 0.0
    for bucket in calibration_bins(confidences, correctness, bins=bins):
        if bucket.count == 0:
            continue
        ece += (bucket.count / total) * abs(bucket.accuracy - bucket.confidence)
    return ece


# --------------------------------------------------------------------------- #
# Preregistered success criteria (Req 12.5)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SuccessCriteria:
    """The three preregistered criteria of §3.7.

    The policy is considered successful only if ``R100`` is lower than under
    universal review, ``DVR_C5 − DVR_C2 ≤ 0.005``, and ``MCR_C5 < MCR_C3``.
    """

    r100_below_universal: bool
    dvr_within_tolerance: bool
    mcr_below_fixed: bool
    r100_adaptive: float
    r100_universal: float
    dvr_delta: float
    mcr_adaptive: float
    mcr_fixed: float

    r100_fixed: float = float("nan")

    @property
    def met(self) -> bool:
        """Whether all three criteria hold, read strictly as preregistered."""
        return (
            self.r100_below_universal
            and self.dvr_within_tolerance
            and self.mcr_below_fixed
        )

    @property
    def mcr_tied_at_floor(self) -> bool:
        """Whether the MCR criterion failed only because both arms reached zero.

        The preregistered criterion is a strict inequality, so it cannot be met when
        the fixed-threshold baseline also attains zero missed consequential
        conflicts. That is a floor effect, not evidence against the policy, and it
        is reported as such rather than folded into a bare pass/fail.
        """
        return (
            not self.mcr_below_fixed
            and self.mcr_adaptive == self.mcr_fixed
            and self.mcr_adaptive == 0.0
        )

    def interpretation(self) -> str:
        """A plain-language reading of the outcome."""
        if self.met:
            return "All three preregistered criteria are met."
        if self.mcr_tied_at_floor and self.r100_below_universal and self.dvr_within_tolerance:
            ratio = (
                self.r100_fixed / self.r100_adaptive
                if self.r100_adaptive
                else float("nan")
            )
            return (
                "Two of three criteria are met. The MCR criterion is a strict "
                "inequality and cannot be satisfied because both the adaptive "
                "policy and the fixed-threshold baseline attain zero missed "
                "consequential conflicts on this corpus. The substantive result is "
                "that RAHGM matches the baseline's MCR floor at "
                f"{ratio:.2f}x lower review demand "
                f"(R100 {self.r100_adaptive:.1f} vs {self.r100_fixed:.1f})."
            )
        failures = [
            name
            for name, ok in (
                ("R100(C5) < R100(C1)", self.r100_below_universal),
                ("DVR(C5) - DVR(C2) <= 0.005", self.dvr_within_tolerance),
                ("MCR(C5) < MCR(C3)", self.mcr_below_fixed),
            )
            if not ok
        ]
        return "Criteria not met: " + "; ".join(failures) + "."

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "met": self.met,
            "mcr_tied_at_floor": self.mcr_tied_at_floor,
            "interpretation": self.interpretation(),
            "r100_below_universal": self.r100_below_universal,
            "dvr_within_tolerance": self.dvr_within_tolerance,
            "mcr_below_fixed": self.mcr_below_fixed,
            "r100_adaptive": self.r100_adaptive,
            "r100_universal": self.r100_universal,
            "r100_fixed": self.r100_fixed,
            "dvr_delta": self.dvr_delta,
            "mcr_adaptive": self.mcr_adaptive,
            "mcr_fixed": self.mcr_fixed,
            "criteria": [
                "R100(C5) < R100(C1)",
                "DVR(C5) - DVR(C2) <= 0.005",
                "MCR(C5) < MCR(C3)",
            ],
        }


def evaluate_success_criteria(
    metrics_by_condition: dict[str, ReplayMetrics],
    *,
    dvr_tolerance: float = 0.005,
) -> SuccessCriteria:
    """Evaluate the §3.7 success criteria from the five-arm metric table."""
    adaptive = metrics_by_condition["adaptive_rahgm"]
    universal = metrics_by_condition["universal_review"]
    autonomous = metrics_by_condition["autonomous_ocmr"]
    fixed = metrics_by_condition["fixed_threshold"]

    dvr_delta = adaptive.dvr - autonomous.dvr
    return SuccessCriteria(
        r100_below_universal=adaptive.r100 < universal.r100,
        dvr_within_tolerance=dvr_delta <= dvr_tolerance,
        mcr_below_fixed=adaptive.mcr < fixed.mcr,
        r100_adaptive=adaptive.r100,
        r100_universal=universal.r100,
        dvr_delta=dvr_delta,
        mcr_adaptive=adaptive.mcr,
        mcr_fixed=fixed.mcr,
        r100_fixed=fixed.r100,
    )

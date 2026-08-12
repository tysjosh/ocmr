"""Annotation simulators and Krippendorff's alpha (Req 9.8).

The paper's corpus and quarantine audit are labelled by two independent human
annotators with a third adjudicating disagreements, reporting Krippendorff's alpha
before adjudication. No human annotators are available here, so this module
provides two **rubric-based annotator simulators** that apply the labelling rubric
with independent noise, plus an adjudicator.

This substitution is disclosed in every emitted artifact. The reported alpha
characterizes the *rubric's* determinacy under perturbation — how reliably a
label can be recovered from the case features — not human agreement. It is not a
substitute for the paper's inter-annotator study.

Requirements: 9.6, 9.8, 13.1, 14.1.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Sequence

#: Disclosure attached to every alpha this module computes.
ALPHA_DISCLOSURE = (
    "Krippendorff's alpha is computed over two rubric-based annotator simulators "
    "with independent label noise, not over human annotators. It measures how "
    "determinately the labelling rubric recovers a label from case features."
)


# --------------------------------------------------------------------------- #
# Krippendorff's alpha (nominal)
# --------------------------------------------------------------------------- #
def krippendorff_alpha(
    ratings: Sequence[Sequence[Hashable | None]],
) -> float:
    """Krippendorff's alpha for nominal data.

    Args:
        ratings: One row per unit, one column per annotator. ``None`` marks a
            missing rating; units with fewer than two ratings are skipped.

    Returns:
        ``alpha = 1 − D_o / D_e``. Returns ``1.0`` when every rater agrees
        everywhere and there is no observed disagreement, and ``nan`` when alpha
        is undefined (fewer than two usable units, or no variation to explain).

    The coincidence-matrix formulation is used, which handles unequal numbers of
    raters per unit correctly.
    """
    units = [[r for r in row if r is not None] for row in ratings]
    units = [u for u in units if len(u) >= 2]
    if len(units) < 1:
        return float("nan")

    categories = sorted({str(v) for unit in units for v in unit})
    if len(categories) <= 1:
        # Every rating is identical: there is no disagreement to explain, and no
        # expected disagreement either. Perfect agreement is the honest reading.
        return 1.0

    index = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    # Coincidence matrix.
    coincidence = [[0.0] * k for _ in range(k)]
    for unit in units:
        m = len(unit)
        for i, a in enumerate(unit):
            for j, b in enumerate(unit):
                if i == j:
                    continue
                coincidence[index[str(a)]][index[str(b)]] += 1.0 / (m - 1)

    totals = [sum(row) for row in coincidence]
    grand = sum(totals)
    if grand <= 0:
        return float("nan")

    observed_disagreement = sum(
        coincidence[i][j] for i in range(k) for j in range(k) if i != j
    )
    expected_disagreement = sum(
        totals[i] * totals[j] / (grand - 1.0)
        for i in range(k)
        for j in range(k)
        if i != j
    )
    if expected_disagreement <= 0:
        return float("nan")
    return 1.0 - (observed_disagreement / expected_disagreement)


# --------------------------------------------------------------------------- #
# Annotator simulators
# --------------------------------------------------------------------------- #
@dataclass
class AnnotatorSimulator:
    """Applies a labelling rubric with independent, seeded noise.

    Attributes:
        annotator_id: Identifier retained in the agreement table.
        error_rate: Probability of emitting a label other than the rubric's.
        seed: Per-annotator seed, so two simulators disagree independently.
    """

    annotator_id: str
    error_rate: float = 0.08
    seed: int = 0

    def label(
        self,
        unit_id: str,
        rubric_label: str,
        alternatives: Sequence[str],
    ) -> str:
        """Return this annotator's label for one unit.

        With probability ``error_rate`` an alternative label is chosen instead of
        the rubric's, modelling the residual subjectivity a human annotator would
        introduce on borderline cases.
        """
        rng = random.Random(f"{self.seed}:{self.annotator_id}:{unit_id}")
        if not alternatives or rng.random() >= self.error_rate:
            return rubric_label
        others = [a for a in alternatives if a != rubric_label]
        return rng.choice(others) if others else rubric_label


@dataclass
class AgreementReport:
    """Inter-annotator agreement over one label field."""

    field: str
    alpha: float
    n_units: int
    n_disagreements: int
    annotators: tuple[str, ...]
    simulated: bool = True
    disclosure: str = ALPHA_DISCLOSURE

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view including the simulation disclosure."""
        return {
            "field": self.field,
            "krippendorff_alpha": self.alpha,
            "n_units": self.n_units,
            "n_disagreements": self.n_disagreements,
            "annotators": list(self.annotators),
            "simulated": self.simulated,
            "disclosure": self.disclosure,
        }


def dual_annotate(
    units: Sequence[tuple[str, str]],
    alternatives: Sequence[str],
    *,
    field: str,
    error_rate: float = 0.08,
    seed: int = 0,
) -> tuple[list[str], AgreementReport]:
    """Label units with two simulated annotators and adjudicate disagreements.

    Args:
        units: ``(unit_id, rubric_label)`` pairs.
        alternatives: The full label vocabulary.
        field: Name of the label field, for the report.
        error_rate: Per-annotator error probability.
        seed: Base seed.

    Returns:
        ``(adjudicated_labels, agreement_report)``. The adjudicator resolves a
        disagreement to the rubric label, which is the role the paper's third
        annotator plays; agreement is reported *before* adjudication.
    """
    a = AnnotatorSimulator("annotator_a", error_rate=error_rate, seed=seed)
    b = AnnotatorSimulator("annotator_b", error_rate=error_rate, seed=seed + 7919)

    ratings: list[list[str | None]] = []
    adjudicated: list[str] = []
    disagreements = 0

    for unit_id, rubric_label in units:
        label_a = a.label(unit_id, rubric_label, alternatives)
        label_b = b.label(unit_id, rubric_label, alternatives)
        ratings.append([label_a, label_b])
        if label_a != label_b:
            disagreements += 1
        adjudicated.append(label_a if label_a == label_b else rubric_label)

    report = AgreementReport(
        field=field,
        alpha=krippendorff_alpha(ratings),
        n_units=len(units),
        n_disagreements=disagreements,
        annotators=(a.annotator_id, b.annotator_id),
    )
    return adjudicated, report

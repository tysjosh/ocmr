"""Experiment 4 — end-to-end and cross-domain effects (paper §4.5, Table 7).

Each condition's resulting memory state is used by the same downstream analytic
agent on held-out questions. The point is that governance is not an end in itself:
a policy that protects durable integrity while withholding valid corrections
produces a memory that answers questions *staler*, not better.

Three outcomes per condition:

* **answer accuracy** — the fraction of questions whose current value in memory
  matches the value a fully correct replay would have produced;
* **unsupported conclusions** — answers drawn from an accepted assertion with no
  attributable source. An abstention is *not* an unsupported conclusion;
* **stale-value propagation** — answers that return the incumbent a valid
  correction should have replaced. This is where the false-quarantine cost becomes
  visible in analytic output.

Results are broken out by scenario family so the largest gain and the largest cost
can be named, and reviewer minutes per 100 writes are reported alongside so an
accuracy gain is never read without its oversight price.

Requirements: 11.2, 13.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ocm.evaluation.rahgm.corpus import Partition, RahgmCorpus, generate_corpus
from ocm.evaluation.rahgm.metrics import ReplayMetrics
from ocm.evaluation.rahgm.replay import ConditionResult, run_experiment1
from ocm.governance.conditions import CONDITION_NAMES, Condition

#: The conditions Table 7 reports (universal review is omitted in the paper's
#: table, but it is included here as the oversight-cost upper bound).
TABLE7_ORDER: tuple[Condition, ...] = (
    Condition.universal_review,
    Condition.autonomous_ocmr,
    Condition.fixed_threshold,
    Condition.frozen_rahgm,
    Condition.adaptive_rahgm,
)


@dataclass
class EndToEndResult:
    """One condition's downstream analytic outcome."""

    condition: Condition
    name: str
    n_questions: int
    answer_accuracy: float
    unsupported_rate: float
    stale_propagation_rate: float
    abstention_rate: float
    ambiguous_rate: float
    r100: float
    dvr: float
    review_rate: float

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "condition": self.condition.value,
            "name": self.name,
            "n_questions": self.n_questions,
            "answer_accuracy": self.answer_accuracy,
            "unsupported_rate": self.unsupported_rate,
            "stale_propagation_rate": self.stale_propagation_rate,
            "abstention_rate": self.abstention_rate,
            "ambiguous_rate": self.ambiguous_rate,
            "r100": self.r100,
            "dvr": self.dvr,
            "review_rate": self.review_rate,
        }


def _summarize(
    condition: Condition, result: ConditionResult
) -> EndToEndResult:
    """Compute the Table 7 columns for one condition."""
    answers = result.answers
    n = len(answers) or 1
    return EndToEndResult(
        condition=condition,
        name=CONDITION_NAMES[condition],
        n_questions=len(answers),
        answer_accuracy=sum(1 for a in answers if a["correct"]) / n,
        unsupported_rate=sum(1 for a in answers if a["unsupported"]) / n,
        stale_propagation_rate=sum(1 for a in answers if a["stale"]) / n,
        abstention_rate=sum(1 for a in answers if a["abstained"]) / n,
        ambiguous_rate=sum(1 for a in answers if a["ambiguous"]) / n,
        r100=result.metrics.r100,
        dvr=result.metrics.dvr,
        review_rate=result.metrics.review_rate,
    )


def _by_family(
    result: ConditionResult,
) -> dict[str, dict[str, float]]:
    """Per-template-family accuracy and false-quarantine breakdown.

    The corpus template family is the closest analogue of the paper's scenario
    families (planning, temporal reasoning, entity resolution, evidence
    integration, contradiction-heavy), because each template is built to exercise
    exactly one of those capabilities.
    """
    out: dict[str, dict[str, float]] = {}
    families = sorted({r.template for r in result.records})
    for family in families:
        subset = [r for r in result.records if r.template == family]
        count = len(subset) or 1
        out[family] = {
            "n": float(len(subset)),
            "transition_accuracy": sum(1 for r in subset if not r.error) / count,
            "review_rate": sum(1 for r in subset if r.escalated) / count,
            "false_quarantine_rate": sum(1 for r in subset if r.false_quarantine) / count,
        }
    return out


#: Template families grouped onto the paper's capability names, so §4.5 can report
#: effects "separately for planning, temporal reasoning, entity resolution,
#: evidence integration, and contradiction-heavy scenarios".
CAPABILITY_FAMILIES: dict[str, tuple[str, ...]] = {
    "entity_resolution": ("conflict_alias_ambiguity",),
    "temporal_reasoning": (
        "conflict_undated_update",
        "conflict_terminal_status_flip",
        "correction_person_status",
        "routine_first_status",
    ),
    "evidence_integration": (
        "conflict_unsupported_final_decision",
        "conflict_weak_authority",
        "routine_evidence_link",
    ),
    "contradiction_heavy": (
        "correction_slot_value",
        "correction_assignment",
        "correction_project_status",
    ),
    "planning": (
        "conflict_irreversible_deletion",
        "routine_membership",
        "routine_participation",
        "routine_about",
    ),
    "malformed": (
        "reject_unregistered_predicate",
        "reject_unattributed",
        "reject_domain_range",
    ),
}


def _by_capability(result: ConditionResult) -> dict[str, dict[str, float]]:
    """Aggregate template families onto the paper's capability names."""
    out: dict[str, dict[str, float]] = {}
    for capability, families in CAPABILITY_FAMILIES.items():
        subset = [r for r in result.records if r.template in families]
        if not subset:
            continue
        count = len(subset)
        out[capability] = {
            "n": float(count),
            "transition_accuracy": sum(1 for r in subset if not r.error) / count,
            "review_rate": sum(1 for r in subset if r.escalated) / count,
            "false_quarantine_rate": sum(1 for r in subset if r.false_quarantine) / count,
        }
    return out


def run_experiment4(
    corpus: RahgmCorpus | None = None,
    *,
    experiment1: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the end-to-end analysis and populate Table 7 (§4.5).

    Reuses Experiment 1's replay rather than repeating it: the memory states
    Experiment 1 produced *are* the states the downstream agent must query, and
    re-deriving them would risk the two experiments disagreeing.

    Args:
        corpus: The evaluation corpus.
        experiment1: An existing Experiment 1 report; run if omitted.

    Returns:
        A report dict with Table 7, the per-capability breakdown, and the
        relative-to-autonomous-OCMR comparison §4.5 reports.
    """
    corpus = corpus or generate_corpus()
    experiment1 = experiment1 or run_experiment1(corpus)
    results: dict[str, ConditionResult] = experiment1["_results"]

    table7: list[EndToEndResult] = []
    capability: dict[str, dict[str, dict[str, float]]] = {}
    family: dict[str, dict[str, dict[str, float]]] = {}

    for condition in TABLE7_ORDER:
        result = results.get(condition.value)
        if result is None:
            continue
        table7.append(_summarize(condition, result))
        capability[condition.value] = _by_capability(result)
        family[condition.value] = _by_family(result)

    by_condition = {r.condition.value: r for r in table7}
    autonomous = by_condition.get(Condition.autonomous_ocmr.value)
    adaptive = by_condition.get(Condition.adaptive_rahgm.value)

    comparison: dict[str, Any] | None = None
    if autonomous is not None and adaptive is not None:
        gains = _capability_deltas(
            capability.get(Condition.adaptive_rahgm.value, {}),
            capability.get(Condition.autonomous_ocmr.value, {}),
        )
        ranked = sorted(gains.items(), key=lambda kv: -kv[1]["transition_accuracy_delta"])
        comparison = {
            "answer_accuracy_delta_points": 100.0
            * (adaptive.answer_accuracy - autonomous.answer_accuracy),
            "stale_propagation_delta_points": 100.0
            * (adaptive.stale_propagation_rate - autonomous.stale_propagation_rate),
            "unsupported_delta_points": 100.0
            * (adaptive.unsupported_rate - autonomous.unsupported_rate),
            "dvr_delta": adaptive.dvr - autonomous.dvr,
            "analyst_minutes_per_100_writes": adaptive.r100,
            "capability_deltas": gains,
            "largest_gain": (
                {"capability": ranked[0][0], **ranked[0][1]} if ranked else None
            ),
            "largest_cost": (
                {"capability": ranked[-1][0], **ranked[-1][1]} if ranked else None
            ),
        }

    return {
        "experiment": "experiment4_end_to_end",
        "n_conditions": len(table7),
        "table7": [r.as_dict() for r in table7],
        "by_capability": capability,
        "by_template_family": family,
        "adaptive_vs_autonomous": comparison,
        "notes": [
            "Answer accuracy is read from the final accepted memory state; an "
            "abstention counts as incorrect but not as an unsupported conclusion.",
            "Reviewer minutes come from the explicit review-cost model, so an "
            "accuracy gain is always reported next to its oversight price.",
        ],
    }


def _capability_deltas(
    adaptive: dict[str, dict[str, float]], autonomous: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    """Per-capability deltas between adaptive RAHGM and autonomous OCMR."""
    out: dict[str, dict[str, float]] = {}
    for capability, values in adaptive.items():
        baseline = autonomous.get(capability)
        if baseline is None:
            continue
        out[capability] = {
            "n": values["n"],
            "transition_accuracy_delta": 100.0
            * (values["transition_accuracy"] - baseline["transition_accuracy"]),
            "false_quarantine_delta": 100.0
            * (values["false_quarantine_rate"] - baseline["false_quarantine_rate"]),
            "review_rate_delta": 100.0
            * (values["review_rate"] - baseline["review_rate"]),
        }
    return out

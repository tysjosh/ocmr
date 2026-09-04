"""Durable-state correctness metric + the Bevi evidence-weighted baseline.

Covers three things:

1. :func:`~ocm.evaluation.durable_state.durable_state_outcomes` bucket logic,
   especially the quarantine-aware ``stale`` rule that keeps a governed arm from
   being scored as silently wrong when it correctly declines a write.
2. The ``Bevi`` write policy's three decision paths (candidate more / less /
   equally confident than the incumbent).
3. The per-arm failure signatures across the 2x2 perturbation of the LongMemEval
   oracle. This is the regression lock on the appendix table: it pins that
   ``Bevi`` is *not* redundant with ``Bsup``, and that B3 under a declared
   authoritative-update policy collapses onto last-writer-wins.
"""

from __future__ import annotations

import pytest

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.evaluation.arms import baseline_settings_overrides, known_arms
from ocm.evaluation.datasets.longmemeval_adapter import (
    build_from_kupdate_oracle,
    sample_annotations,
    sample_instances,
)
from ocm.evaluation.durable_state import (
    ABSTAINED,
    CORRECT,
    MISSING,
    SPLIT,
    STALE,
    durable_state_outcomes,
    normalize_value,
    resolve_gold_keys,
)
from ocm.extraction.base import ExtractionResult

GOLD = "San Francisco"
STALE_VALUE = "New York"
SLOT = "q:residence"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _SlotExtractor:
    """Emits one ``Slot -[HAS_VALUE]-> SlotValue`` write per source_ref."""

    version = "test-slot-1"

    def __init__(self, writes: dict[str, tuple[str, float, str]]) -> None:
        self._writes = writes

    def extract(self, text: str, source_ref: str) -> ExtractionResult:
        write = self._writes.get(source_ref)
        if write is None:
            return ExtractionResult(extractor_version=self.version)
        value, confidence, intent = write
        return ExtractionResult(
            entities=[
                {"type": "Slot", "name": SLOT},
                {"type": "SlotValue", "name": value, "fields": {"value": value}},
            ],
            relations=[
                {
                    "subject": SLOT,
                    "predicate": "HAS_VALUE",
                    "object": value,
                    "confidence": confidence,
                    "write_intent": intent,
                }
            ],
            extractor_version=self.version,
        )


def _settings(arm: str | None = None, *, authoritative: bool = False) -> Settings:
    base = Settings(
        deterministic_test_mode=True,
        chroma_mode="memory",
        extractor="mock",
        authoritative_update_supersede=authoritative,
    )
    return base.model_copy(update=baseline_settings_overrides(arm)) if arm else base


def _replay_pair(arm: str, first, second, *, authoritative: bool = False):
    """Write ``first`` then ``second`` through ``arm``; return (container, gold)."""
    extractor = _SlotExtractor({"s0": first, "s1": second})
    container = CoreContainer(
        _settings(arm, authoritative=authoritative), extractor=extractor
    )
    for ref in ("s0", "s1"):
        container.write_pipeline.run("text", ref)
    gold = resolve_gold_keys(container, {(SLOT, "HAS_VALUE"): GOLD})
    return container, gold


def _bucket(container, gold) -> str:
    report = durable_state_outcomes(container, gold)
    assert report.total == 1
    return report.outcomes[next(iter(gold))]


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #
def test_normalize_value_is_case_and_whitespace_insensitive():
    assert normalize_value("  San   Francisco ") == normalize_value("san francisco")


def test_missing_key_counts_as_missing_not_dropped():
    """An unextracted fact stays in the denominator as ``missing`` (Req: metric scope)."""
    container = CoreContainer(_settings())
    gold = resolve_gold_keys(container, {(SLOT, "HAS_VALUE"): GOLD})
    report = durable_state_outcomes(container, gold)
    assert report.total == 1
    assert report.missing == 1
    assert report.scored == 0
    # Governance-scoped rates must not divide by zero.
    assert report.dsc_gov == 0.0
    assert report.ssr_gov == 0.0


def test_ungoverned_arm_holding_both_values_scores_split_not_correct():
    """Hoarding must not be rewarded: gold present alongside stale is a split.

    This is the flaw the metric exists to close — token-containment task success
    scores such a store 1.0 because the gold token appears somewhere.
    """
    container, gold = _replay_pair(
        "B2", (STALE_VALUE, 0.85, "new_fact"), (GOLD, 0.85, "new_fact")
    )
    report = durable_state_outcomes(container, gold)
    assert report.split == 1
    assert report.correct == 0
    assert report.stale == 0  # split and stale are disjoint, never double-counted


def test_governed_hold_scores_abstained_not_stale():
    """A quarantined conflict leaves the incumbent accepted but *flagged*.

    Without the quarantine check this would score ``stale``, penalising the
    governed arm for behaving exactly as designed.
    """
    container, gold = _replay_pair(
        "B3", (STALE_VALUE, 0.85, "new_fact"), (GOLD, 0.85, "new_fact")
    )
    report = durable_state_outcomes(container, gold)
    assert report.abstained == 1
    assert report.stale == 0


# --------------------------------------------------------------------------- #
# Bevi operator
# --------------------------------------------------------------------------- #
def test_bevi_is_registered_as_a_baseline_arm():
    assert "Bevi" in known_arms("baseline")
    assert baseline_settings_overrides("Bevi")["toki_operator"] == "evidence"


def test_bevi_higher_confidence_candidate_supersedes():
    container, gold = _replay_pair(
        "Bevi", (STALE_VALUE, 0.60, "update"), (GOLD, 0.90, "update")
    )
    assert _bucket(container, gold) == CORRECT


def test_bevi_lower_confidence_candidate_loses_and_is_not_flagged():
    """The incumbent wins, and the loss is a log line, not a surfaced conflict.

    Scoring ``stale`` (not ``abstained``) is the point: Bevi never abstains, so
    its wrong answers must be visible as silent staleness.
    """
    container, gold = _replay_pair(
        "Bevi", (STALE_VALUE, 0.90, "update"), (GOLD, 0.60, "update")
    )
    report = durable_state_outcomes(container, gold)
    assert report.outcomes[next(iter(gold))] == STALE
    assert report.abstained == 0
    assert report.ssr == 100.0


def test_bevi_equal_confidence_degrades_to_last_writer_wins():
    """On a confidence tie the newer write wins, so Bevi becomes LWW.

    Both real datasets write every value at 0.85, so this is the behaviour Bevi
    exhibits on the *unperturbed* workload — which is why the confidence-inverted
    cell is required for the arm to be informative.
    """
    container, gold = _replay_pair(
        "Bevi", (STALE_VALUE, 0.85, "update"), (GOLD, 0.85, "update")
    )
    assert _bucket(container, gold) == CORRECT

    # Same confidences, reversed arrival: the stale value now wins on recency.
    container, gold = _replay_pair(
        "Bevi", (GOLD, 0.85, "update"), (STALE_VALUE, 0.85, "update")
    )
    assert _bucket(container, gold) == STALE


def test_unknown_toki_operator_fails_loudly():
    """A typo in a settings override must not masquerade as an ungoverned arm."""
    settings = _settings().model_copy(
        update={
            "enable_schema_validation": False,
            "enable_constraint_validation": False,
            "enable_contradiction_gate": False,
            "toki_operator": "nonsense",
        }
    )
    container = CoreContainer(
        settings, extractor=_SlotExtractor({"s0": (GOLD, 0.85, "new_fact")})
    )
    with pytest.raises(ValueError, match="unknown toki_operator"):
        container.write_pipeline.run("text", "s0")


# --------------------------------------------------------------------------- #
# Perturbation axes
# --------------------------------------------------------------------------- #
def test_perturbation_modes_are_validated():
    with pytest.raises(ValueError, match="order must be one of"):
        build_from_kupdate_oracle(
            sample_instances(), sample_annotations(), order="sideways"
        )
    with pytest.raises(ValueError, match="confidence must be one of"):
        build_from_kupdate_oracle(
            sample_instances(), sample_annotations(), confidence="vibes"
        )


def test_permuted_order_writes_the_stale_value_last():
    _examples, oracle = build_from_kupdate_oracle(
        sample_instances(), sample_annotations(), order="permuted"
    )
    written = [
        relation["object"]
        for idx in range(3)
        for relation in getattr(
            oracle._writes.get(f"ku_0001:s{idx}"), "relations", []
        )
    ]
    assert written[-1] == STALE_VALUE
    assert written[0] == GOLD


def test_inverted_confidence_makes_the_gold_value_less_confident():
    _examples, oracle = build_from_kupdate_oracle(
        sample_instances(), sample_annotations(), confidence="inverted"
    )
    by_value = {
        relation["object"]: relation["confidence"]
        for idx in range(3)
        for relation in getattr(
            oracle._writes.get(f"ku_0001:s{idx}"), "relations", []
        )
    }
    assert by_value[GOLD] < by_value[STALE_VALUE]
    # Both stay above the 0.8 contradiction threshold so the C7 gate still fires;
    # otherwise the perturbation would be confounded with gate disengagement.
    assert min(by_value.values()) > 0.8


def test_oracle_carries_the_gold_current_value():
    _examples, oracle = build_from_kupdate_oracle(
        sample_instances(), sample_annotations()
    )
    assert oracle.gold_current_values == {("ku_0001:residence", "HAS_VALUE"): GOLD}


# --------------------------------------------------------------------------- #
# The appendix table: per-arm failure signature across the 2x2
# --------------------------------------------------------------------------- #
#: ``(order, confidence) -> bucket`` per arm. Locks in the measured signatures.
EXPECTED_SIGNATURES: dict[tuple[str, bool], dict[tuple[str, str], str]] = {
    # Last-writer-wins is right whenever recency identifies the current value,
    # and silently wrong once arrival order is permuted. Confidence is ignored.
    ("Bsup", False): {
        ("aligned", "aligned"): CORRECT,
        ("aligned", "inverted"): CORRECT,
        ("permuted", "aligned"): STALE,
        ("permuted", "inverted"): STALE,
    },
    # Evidence-weighted additionally fails when the stale side is more confident.
    # The aligned/inverted cell is what makes Bevi non-redundant with Bsup.
    ("Bevi", False): {
        ("aligned", "aligned"): CORRECT,
        ("aligned", "inverted"): STALE,
        ("permuted", "aligned"): STALE,
        ("permuted", "inverted"): STALE,
    },
    # Under a declared authoritative-update policy every later write supersedes
    # unconditionally, so B3 reduces *exactly* to last-writer-wins.
    ("B3", True): {
        ("aligned", "aligned"): CORRECT,
        ("aligned", "inverted"): CORRECT,
        ("permuted", "aligned"): STALE,
        ("permuted", "inverted"): STALE,
    },
    # With the policy off, governance never leaves a silently stale value: it
    # abstains where recency would have been right, and protects the correct
    # value from a late stale write where recency would have been wrong.
    ("B3", False): {
        ("aligned", "aligned"): ABSTAINED,
        ("aligned", "inverted"): ABSTAINED,
        ("permuted", "aligned"): CORRECT,
        ("permuted", "inverted"): CORRECT,
    },
}


def _replay_cell(arm: str, order: str, confidence: str, *, authoritative: bool) -> str:
    examples, oracle = build_from_kupdate_oracle(
        sample_instances(), sample_annotations(), order=order, confidence=confidence
    )
    container = CoreContainer(
        _settings(arm, authoritative=authoritative), extractor=oracle
    )
    for example in examples:
        for session in example.sessions:
            container.write_pipeline.run(
                session.input, f"{example.id}:{session.session_id}"
            )
    gold = resolve_gold_keys(container, oracle.gold_current_values)
    return _bucket(container, gold)


@pytest.mark.parametrize(("arm", "authoritative"), sorted(EXPECTED_SIGNATURES))
def test_arm_failure_signature_across_perturbation_grid(arm, authoritative):
    expected = EXPECTED_SIGNATURES[(arm, authoritative)]
    observed = {
        cell: _replay_cell(arm, *cell, authoritative=authoritative)
        for cell in expected
    }
    assert observed == expected


def test_only_governed_arm_avoids_silent_staleness():
    """The headline claim: B3 (policy off) is the only arm with SSR == 0.

    Every operator that must elect a winner goes silently stale in at least one
    cell; the governed arm converts those cases into declared abstention or
    protects the correct value outright.
    """
    goes_stale = {
        key: any(bucket == STALE for bucket in signature.values())
        for key, signature in EXPECTED_SIGNATURES.items()
    }
    assert goes_stale == {
        ("Bsup", False): True,
        ("Bevi", False): True,
        ("B3", True): True,  # the authoritative-update policy reintroduces it
        ("B3", False): False,  # the only configuration that never goes stale
    }

    governed = EXPECTED_SIGNATURES[("B3", False)]
    assert STALE not in governed.values()
    assert SPLIT not in governed.values()
    assert MISSING not in governed.values()

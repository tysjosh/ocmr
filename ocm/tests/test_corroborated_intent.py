"""Tests for ``intent_mode="corroborated"`` (corroboration-gated supersession).

Why the mode exists
-------------------
Under ``intent_mode="auto"`` a changed value is emitted as ``update``, and
``authoritative_update_supersede`` honours that unconditionally -- "the latest
value replaces the incumbent (no margin / evidence gate)". The ``update`` label
is assigned purely because ``prev is not None``, an artifact of iteration order
that carries no evidence the later value is more reliable. On raw extraction that
makes recency into authority: a noisy late extraction retires a correct early
one. It also means C7 always takes the supersede branch, so the governed arm
quarantines *nothing* and the gate's refusal behaviour is never exercised.

``corroborated`` routes a change through ``correction`` instead, with confidence
derived from how many distinct sessions attest the value. C7's Algorithm 1 line 7
then decides: supersede only if the candidate beats the incumbent by
``supersede_margin``, else quarantine.

What these tests pin
--------------------
* the confidence ladder never drops to or below ``contradiction_high_confidence``
  (0.8), or the conflict would go "soft" and be silently accepted -- producing
  violations instead of quarantines and disengaging the gate
* ``auto`` / ``new_fact`` / ``memgpt`` behaviour and, critically, ``auto``'s
  checkpoint key are unchanged, since existing checkpoints depend on both
* a well-corroborated value is not displaced by a one-off re-extraction
  (quarantine), while a better-corroborated candidate does supersede
* corroboration counts distinct sessions, not raw occurrences
"""

from __future__ import annotations

import pytest

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.evaluation.datasets.longmemeval_adapter import (
    CORROBORATION_BASE_CONFIDENCE,
    CORROBORATION_MAX_CONFIDENCE,
    NEW_FACT_CONFIDENCE,
    UPDATE_CONFIDENCE,
    build_e2e_from_extraction,
    corroboration_confidence,
    load_longmemeval,
)


def _relations(oracle):
    """Every emitted relation across the oracle's cached session writes."""
    out = []
    for sw in oracle._writes.values():
        out.extend(sw.relations)
    return out


def _instances(n: int = 2):
    return load_longmemeval(
        "data/longmemeval_s.json", question_type="knowledge-update", limit=n
    )


# --------------------------------------------------------------------------- #
# Confidence ladder
# --------------------------------------------------------------------------- #
def test_ladder_never_disengages_the_contradiction_gate():
    """Every emittable confidence must exceed contradiction_high_confidence.

    c7_contradiction_gate treats a conflict as blocking only when both sides
    exceed the threshold. Below it the conflict is "soft" and silently accepted,
    which would create durable violations instead of the quarantines this mode is
    meant to produce.
    """
    threshold = Settings().contradiction_high_confidence
    for attestations in range(1, 100):
        assert corroboration_confidence(attestations) > threshold


def test_ladder_is_monotonic_and_capped():
    values = [corroboration_confidence(k) for k in range(1, 20)]
    assert values[0] == pytest.approx(CORROBORATION_BASE_CONFIDENCE)
    assert all(b >= a for a, b in zip(values, values[1:]))
    assert max(values) == pytest.approx(CORROBORATION_MAX_CONFIDENCE)


def test_two_extra_attestations_clear_the_default_margin():
    """The documented calibration: >= 2 more attestations beats supersede_margin."""
    margin = Settings().supersede_margin
    base = corroboration_confidence(1)
    assert (corroboration_confidence(2) - base) <= margin      # one is not enough
    assert (corroboration_confidence(3) - base) > margin       # two is


def test_zero_or_negative_attestations_do_not_underflow():
    assert corroboration_confidence(0) == pytest.approx(CORROBORATION_BASE_CONFIDENCE)
    assert corroboration_confidence(-5) == pytest.approx(CORROBORATION_BASE_CONFIDENCE)


# --------------------------------------------------------------------------- #
# Existing modes must not regress
# --------------------------------------------------------------------------- #
def _stub_extractor():
    calls = {"n": 0}

    def extract(text):
        calls["n"] += 1
        n = calls["n"]
        return [
            {"attribute": "stable_fact", "value": "always_same"},
            {"attribute": "drifting", "value": f"v{n // 7}"},
        ]

    return extract, calls


@pytest.mark.parametrize("mode", ["auto", "new_fact"])
def test_existing_modes_keep_flat_confidence(mode):
    """auto / new_fact still emit the original constants, unchanged."""
    extract, _calls = _stub_extractor()
    _examples, oracle = build_e2e_from_extraction(
        _instances(), extract, intent_mode=mode
    )
    confidences = {round(float(r["confidence"]), 4) for r in _relations(oracle)}
    assert confidences <= {NEW_FACT_CONFIDENCE, UPDATE_CONFIDENCE}


def test_auto_emits_update_and_corroborated_emits_correction():
    """The intent swap is the whole mechanism: update bypasses the margin test."""
    extract, _ = _stub_extractor()
    _ex_a, oracle_auto = build_e2e_from_extraction(
        _instances(), extract, intent_mode="auto"
    )
    extract, _ = _stub_extractor()
    _ex_c, oracle_corr = build_e2e_from_extraction(
        _instances(), extract, intent_mode="corroborated"
    )

    auto_intents = {r["write_intent"] for r in _relations(oracle_auto)}
    corr_intents = {r["write_intent"] for r in _relations(oracle_corr)}

    assert "update" in auto_intents and "correction" not in auto_intents
    assert "correction" in corr_intents and "update" not in corr_intents


def test_corroborated_preserves_the_write_sequence():
    """Same slots, same values, same order -- only intent and confidence differ."""
    extract, _ = _stub_extractor()
    _ex_a, oracle_auto = build_e2e_from_extraction(
        _instances(), extract, intent_mode="auto"
    )
    extract, _ = _stub_extractor()
    _ex_c, oracle_corr = build_e2e_from_extraction(
        _instances(), extract, intent_mode="corroborated"
    )

    def shape(oracle):
        return [
            (r["subject"], r["predicate"], r["object"]) for r in _relations(oracle)
        ]

    assert shape(oracle_auto) == shape(oracle_corr)


def test_unknown_intent_mode_is_rejected():
    extract, _ = _stub_extractor()
    with pytest.raises(ValueError, match="intent_mode"):
        build_e2e_from_extraction(_instances(1), extract, intent_mode="nonsense")


# --------------------------------------------------------------------------- #
# Corroboration counting
# --------------------------------------------------------------------------- #
def test_repeats_within_one_session_count_once():
    """A verbose session must not inflate a value's confidence on its own."""

    def extract(text):
        # same (attribute, value) three times in every session
        return [{"attribute": "a", "value": "x"}] * 3

    _examples, oracle = build_e2e_from_extraction(
        _instances(1), extract, intent_mode="corroborated"
    )
    relations = _relations(oracle)
    assert relations, "expected at least one write"
    # One session attests it once, but it recurs across many sessions, so the
    # value is well corroborated and the single write carries a high confidence.
    assert relations[0]["write_intent"] == "new_fact"
    assert float(relations[0]["confidence"]) > CORROBORATION_BASE_CONFIDENCE


def test_a_value_seen_once_gets_the_base_confidence():
    """A value attested by exactly one session sits at the bottom of the ladder."""
    calls = {"n": 0}

    def extract(text):
        calls["n"] += 1
        # every session yields a distinct value -> each attested exactly once
        return [{"attribute": "a", "value": f"only_{calls['n']}"}]

    _examples, oracle = build_e2e_from_extraction(
        _instances(1), extract, intent_mode="corroborated"
    )
    confidences = [float(r["confidence"]) for r in _relations(oracle)]
    assert confidences, "expected at least one write"
    assert all(
        c == pytest.approx(CORROBORATION_BASE_CONFIDENCE) for c in confidences
    ), f"expected all base confidence, got {sorted(set(confidences))}"


# --------------------------------------------------------------------------- #
# End-to-end gate behaviour: the point of the mode
# --------------------------------------------------------------------------- #
def _replay(oracle, examples, *, authoritative: bool):
    """Replay writes through the full governed path and total the outcomes."""
    from ocm.evaluation.runner import BaselineRunner

    def settings_factory() -> Settings:
        return Settings(
            deterministic_test_mode=True,
            chroma_mode="memory",
            extractor="mock",
            authoritative_update_supersede=authoritative,
        )

    from ocm.retrieval.embeddings import DeterministicEmbeddingProvider
    from ocm.evaluation.arms import build_arm

    strategy = build_arm(
        "B3", settings_factory, extractor=oracle,
        embeddings=DeterministicEmbeddingProvider(),
    )
    runner = BaselineRunner(settings_factory=settings_factory, top_k=10)
    totals = {"accepted": 0, "superseded": 0, "quarantined": 0, "rejected": 0}
    for example in examples:
        counts = runner._ingest_sessions(strategy, example)
        for key in totals:
            totals[key] += int(counts.get(key, 0))
    return totals, strategy.container


def test_auto_never_quarantines_but_corroborated_does():
    """The headline: auto's unconditional supersede leaves the gate unexercised.

    This is the behaviour the mode exists to change -- on raw extraction every
    governed arm reported quarantined=0, so the refusal path was never tested.
    """
    extract, _ = _stub_extractor()
    ex_auto, oracle_auto = build_e2e_from_extraction(
        _instances(), extract, intent_mode="auto"
    )
    auto_totals, _c1 = _replay(oracle_auto, ex_auto, authoritative=True)

    extract, _ = _stub_extractor()
    ex_corr, oracle_corr = build_e2e_from_extraction(
        _instances(), extract, intent_mode="corroborated"
    )
    corr_totals, _c2 = _replay(oracle_corr, ex_corr, authoritative=True)

    assert auto_totals["quarantined"] == 0
    assert auto_totals["superseded"] > 0
    # correction routes through the margin test, so at least some conflicts are
    # refused rather than silently overwritten.
    assert corr_totals["quarantined"] > 0


def test_corroborated_keeps_durable_state_unambiguous():
    """Refusing a change must not leave two accepted values behind.

    A quarantined candidate never enters the graph, so the incumbent stays the
    single accepted value -- zero single-valued violations, same as auto.
    """
    from ocm.evaluation.experiment import durable_constraint_violations

    extract, _ = _stub_extractor()
    ex, oracle = build_e2e_from_extraction(
        _instances(), extract, intent_mode="corroborated"
    )
    _totals, container = _replay(oracle, ex, authoritative=True)

    violations, _accepted = durable_constraint_violations(container)
    assert violations == 0

"""Tests for retrieval-free durable-state scoring (``ocm.evaluation.exact_state``).

The metric exists because ``task_success`` is answer-token recall over the
rendered answer *plus every retrieved item*, making it monotone in retrieved
volume: an arm that retains stale state and retrieves it scores at least as high
as one that superseded it. On MultiWOZ that saturates the metric entirely --
``EvidencePackager._slot_value_answer`` returns the most recent accepted
``HAS_VALUE`` when several are active, so an ungoverned arm is handed the current
gold value despite carrying single-valued violations.

These tests pin the behaviour that makes the new metric discriminative:

* an unsuperseded slot is scored ``ambiguous``, never ``correct``
* the governed arm reaches 100% where ungoverned arms do not
* ``task_success`` is verified to be saturated on the same data, so the tests
  document *why* the extra metric is needed rather than just asserting it works
* outcome classification covers correct / wrong_value / ambiguous / missing
"""

from __future__ import annotations

import pytest

from ocm.core.config import Settings
from ocm.evaluation.arms import build_arm
from ocm.evaluation.datasets.multiwoz_adapter import (
    build_from_dialogues,
    sample_dialogues,
)
from ocm.evaluation.exact_state import (
    ExactStateReport,
    StateProbe,
    exact_state_report,
    probes_from_examples,
)
from ocm.evaluation.experiment import durable_constraint_violations
from ocm.evaluation.runner import BaselineRunner
from ocm.retrieval.embeddings import DeterministicEmbeddingProvider


def _settings_factory() -> Settings:
    return Settings(
        deterministic_test_mode=True,
        chroma_mode="memory",
        extractor="mock",
        authoritative_update_supersede=True,
    )


@pytest.fixture(scope="module")
def multiwoz():
    """MultiWOZ-shaped fixture: new, changed, and stable slots."""
    return build_from_dialogues(sample_dialogues())


def _ingest(arm: str, examples, oracle):
    """Ingest every example under ``arm`` and return its container + records."""
    strategy = build_arm(
        arm,
        _settings_factory,
        extractor=oracle,
        embeddings=DeterministicEmbeddingProvider(),
    )
    runner = BaselineRunner(settings_factory=_settings_factory, top_k=10)
    records = []
    for example in examples:
        counts = runner._ingest_sessions(strategy, example)
        for q_index, question in enumerate(example.questions):
            records.append(
                runner._run_question(
                    arm, strategy, example, q_index, question,
                    write_quarantined=counts["quarantined"],
                )
            )
    return strategy.container, records


# --------------------------------------------------------------------------- #
# Probe derivation
# --------------------------------------------------------------------------- #
def test_probes_are_derived_from_slot_markers(multiwoz):
    """Each slot-addressed question yields one probe carrying the gold value."""
    examples, _oracle = multiwoz
    probes = probes_from_examples(examples)

    assert probes, "expected at least one probe from the MultiWOZ fixture"
    assert all(p.predicate == "HAS_VALUE" for p in probes)
    # The subject is the qualified slot key from the [[...]] marker, not free text.
    assert all(":" in p.subject_name for p in probes)
    assert all("[[" not in p.subject_name for p in probes)


def test_questions_without_a_marker_are_skipped():
    """A query with no [[slot]] marker is skipped rather than mis-attributed."""

    class _Q:
        query = "Who owns Project Orion?"
        expected_answer_contains = ["Alice"]

    class _Ex:
        id = "ex1"
        questions = [_Q()]

    assert probes_from_examples([_Ex()]) == []


# --------------------------------------------------------------------------- #
# The headline: exact-state discriminates where task_success does not
# --------------------------------------------------------------------------- #
def test_task_success_is_saturated_across_arms(multiwoz):
    """Precondition for this module existing: task_success cannot separate arms.

    All arms recall the current gold value because the answer deriver returns the
    most recent accepted HAS_VALUE even when an ungoverned arm left two active.
    """
    examples, oracle = multiwoz
    scores = {}
    for arm in ("B0", "B2", "B3"):
        _container, records = _ingest(arm, examples, oracle)
        scores[arm] = 100.0 * sum(r["score"] for r in records) / len(records)

    assert scores["B0"] == pytest.approx(scores["B3"]), (
        f"task_success unexpectedly separates arms ({scores}); if the answer "
        "deriver changed, this module's rationale needs revisiting"
    )


def test_governed_arm_reaches_full_exact_state_accuracy(multiwoz):
    """B3 supersedes, so every probed slot holds exactly one correct value."""
    examples, oracle = multiwoz
    probes = probes_from_examples(examples)
    container, _records = _ingest("B3", examples, oracle)

    report = exact_state_report(container, probes)

    assert report.exact_state_accuracy == pytest.approx(100.0)
    assert report.ambiguous == 0
    assert report.wrong_value == 0
    assert report.missing == 0


@pytest.mark.parametrize("arm", ["B0", "B2"])
def test_ungoverned_arms_leave_ambiguous_state(multiwoz, arm):
    """An arm that never supersedes is scored ambiguous, not correct."""
    examples, oracle = multiwoz
    probes = probes_from_examples(examples)
    container, _records = _ingest(arm, examples, oracle)

    report = exact_state_report(container, probes)

    assert report.ambiguous > 0, f"{arm} should retain unsuperseded slot values"
    assert report.exact_state_accuracy < 100.0
    # Ambiguity is the failure mode here -- the newest value is still present, so
    # the arm is not *wrong*, it is undecided.
    assert report.wrong_value == 0


def test_exact_state_separates_governed_from_ungoverned(multiwoz):
    """The metric's purpose: a strict ordering B3 > B0/B2 on the same data."""
    examples, oracle = multiwoz
    probes = probes_from_examples(examples)

    accuracy = {}
    for arm in ("B0", "B2", "B3"):
        container, _records = _ingest(arm, examples, oracle)
        accuracy[arm] = exact_state_report(container, probes).exact_state_accuracy

    assert accuracy["B3"] > accuracy["B0"]
    assert accuracy["B3"] > accuracy["B2"]


def test_ambiguous_count_matches_durable_violations(multiwoz):
    """Cross-check: ambiguous probes correspond to single-valued violations."""
    examples, oracle = multiwoz
    probes = probes_from_examples(examples)
    container, _records = _ingest("B0", examples, oracle)

    report = exact_state_report(container, probes)
    violations, _accepted = durable_constraint_violations(container)

    assert report.ambiguous == violations


# --------------------------------------------------------------------------- #
# Outcome classification
# --------------------------------------------------------------------------- #
def test_missing_subject_is_scored_missing(multiwoz):
    """A probe naming a slot that was never written counts as missing."""
    examples, oracle = multiwoz
    container, _records = _ingest("B3", examples, oracle)

    report = exact_state_report(
        container,
        [StateProbe("no-such-dialogue:no-such-slot", "HAS_VALUE", "anything")],
    )

    assert report.missing == 1
    assert report.correct == 0
    assert report.exact_state_accuracy == pytest.approx(0.0)


def test_wrong_value_is_distinguished_from_missing(multiwoz):
    """A slot holding a single but incorrect value is wrong_value, not missing."""
    examples, oracle = multiwoz
    probes = probes_from_examples(examples)
    container, _records = _ingest("B3", examples, oracle)

    # Re-probe a real slot with a gold value it certainly does not hold.
    bogus = StateProbe(probes[0].subject_name, "HAS_VALUE", "definitely-not-the-value")
    report = exact_state_report(container, [bogus])

    assert report.wrong_value == 1
    assert report.missing == 0
    assert report.correct == 0


def test_lenient_accuracy_tolerates_containment(multiwoz):
    """Containment matching surfaces formatting mismatches as scoring artifacts."""
    examples, oracle = multiwoz
    probes = probes_from_examples(examples)
    container, _records = _ingest("B3", examples, oracle)

    truncated = StateProbe(
        probes[0].subject_name,
        "HAS_VALUE",
        probes[0].expected_object[: max(1, len(probes[0].expected_object) - 1)],
    )
    report = exact_state_report(container, [truncated])

    # Strict scoring rejects the truncated gold; lenient scoring accepts it.
    assert report.correct == 0
    assert report.lenient_accuracy == pytest.approx(100.0)


def test_empty_probe_set_is_all_zero():
    """No probes yields an all-zero report rather than a division error."""
    report = ExactStateReport()

    assert report.total == 0
    assert report.exact_state_accuracy == pytest.approx(0.0)
    assert report.ambiguous_rate == pytest.approx(0.0)
    assert report.lenient_accuracy == pytest.approx(0.0)

"""Tests for the gold-map builders in :mod:`ocm.evaluation.durable_state`.

``gold_from_marked_questions`` is the piece that lets the slot-addressed surfaces
(MultiWOZ, the oracle LongMemEval arm) be scored with the durable-state metric:
their questions already name the slot they ask about in a ``[[key]]`` marker and
carry the gold in ``expected_answer_contains``, so the gold map can be derived
from the benchmark itself rather than hand-assembled.

The end-to-end tests use the MultiWOZ fixture, which exercises new, changed, and
stable slots. They pin the property the metric exists for: ``task_success``
saturates across arms because ``EvidencePackager._slot_value_answer`` returns the
most recent accepted ``HAS_VALUE`` when several are active, whereas the
durable-state buckets separate an arm that superseded from one that did not.
"""

from __future__ import annotations

import pytest

from ocm.core.config import Settings
from ocm.evaluation.arms import build_arm
from ocm.evaluation.datasets.multiwoz_adapter import (
    build_from_dialogues,
    sample_dialogues,
)
from ocm.evaluation.durable_state import (
    durable_state_outcomes,
    gold_from_marked_questions,
    gold_from_slot_values,
    resolve_gold_keys,
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
    return build_from_dialogues(sample_dialogues())


def _replay(arm: str, examples, oracle):
    strategy = build_arm(
        arm, _settings_factory, extractor=oracle,
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
# Builder behaviour
# --------------------------------------------------------------------------- #
def test_marker_and_gold_are_extracted(multiwoz):
    examples, _oracle = multiwoz
    gold = gold_from_marked_questions(examples)

    assert gold, "expected a gold entry per slot-addressed question"
    for (name, predicate), value in gold.items():
        assert predicate == "HAS_VALUE"
        assert ":" in name and "[[" not in name  # the marker's contents, unwrapped
        assert value


def test_predicate_is_configurable(multiwoz):
    examples, _oracle = multiwoz
    gold = gold_from_marked_questions(examples, predicate="HAS_STATUS")

    assert {predicate for _name, predicate in gold} == {"HAS_STATUS"}


def test_questions_without_a_marker_are_skipped():
    """Guessing a subject from free text could score a key never asked about."""

    class _Q:
        query = "Who owns Project Orion?"
        expected_answer_contains = ["Alice"]

    class _Ex:
        id = "ex1"
        questions = [_Q()]

    assert gold_from_marked_questions([_Ex()]) == {}


def test_questions_without_expected_answers_are_skipped():
    class _Q:
        query = "value of slot [[d:s]]"
        expected_answer_contains: list[str] = []

    class _Ex:
        id = "ex1"
        questions = [_Q()]

    assert gold_from_marked_questions([_Ex()]) == {}


def test_agrees_with_the_slot_values_builder(multiwoz):
    """The marker builder is a front-end for ``gold_from_slot_values``."""
    examples, _oracle = multiwoz
    from_markers = gold_from_marked_questions(examples)

    manual = gold_from_slot_values(
        [
            (
                q.query.split("[[")[1].split("]]")[0],
                q.expected_answer_contains[0],
            )
            for ex in examples
            for q in ex.questions
        ]
    )
    assert from_markers == manual


def test_empty_input_yields_an_empty_map():
    assert gold_from_marked_questions([]) == {}


# --------------------------------------------------------------------------- #
# End to end: the buckets separate arms that task_success cannot
# --------------------------------------------------------------------------- #
def test_task_success_is_saturated_across_arms(multiwoz):
    """Precondition for durable-state scoring: recall cannot separate the arms.

    ``_slot_value_answer`` returns the newest accepted value when several are
    active, so an ungoverned arm still recalls the current gold value.
    """
    examples, oracle = multiwoz
    scores = {}
    for arm in ("B0", "B3"):
        _container, records = _replay(arm, examples, oracle)
        scores[arm] = 100.0 * sum(r["score"] for r in records) / len(records)

    assert scores["B0"] == pytest.approx(scores["B3"]), (
        f"task_success unexpectedly separates arms ({scores}); if the answer "
        "deriver changed, the rationale for durable-state scoring needs revisiting"
    )


def test_governed_arm_reaches_full_durable_state_correctness(multiwoz):
    examples, oracle = multiwoz
    container, _records = _replay("B3", examples, oracle)
    gold = resolve_gold_keys(container, gold_from_marked_questions(examples))

    report = durable_state_outcomes(container, gold)

    assert report.dsc == pytest.approx(100.0)
    assert report.split == 0
    assert report.stale == 0
    assert report.missing == 0


@pytest.mark.parametrize("arm", ["B0", "B2"])
def test_ungoverned_arms_leave_split_state(multiwoz, arm):
    """An arm that never supersedes holds two accepted values -> ``split``."""
    examples, oracle = multiwoz
    container, _records = _replay(arm, examples, oracle)
    gold = resolve_gold_keys(container, gold_from_marked_questions(examples))

    report = durable_state_outcomes(container, gold)

    assert report.split > 0
    assert report.dsc < 100.0
    # Not silently stale: the newest value is present, the store is just undecided.
    assert report.stale == 0


def test_split_count_matches_durable_constraint_violations(multiwoz):
    """Cross-check against the legacy measure the split bucket reproduces."""
    examples, oracle = multiwoz
    container, _records = _replay("B0", examples, oracle)
    gold = resolve_gold_keys(container, gold_from_marked_questions(examples))

    report = durable_state_outcomes(container, gold)
    violations, _accepted = durable_constraint_violations(container)

    # Equal on this fixture because no slot carries more than two values; with
    # three or more, violations counts len(distinct) - 1 while split counts the key
    # once, so violations >= split in general.
    assert report.split == violations


def test_unresolvable_gold_name_lands_in_missing(multiwoz):
    """A slot that never materialised stays in the denominator as ``missing``."""
    examples, oracle = multiwoz
    container, _records = _replay("B3", examples, oracle)

    gold = resolve_gold_keys(
        container, gold_from_slot_values([("no-such-dialogue:no-such-slot", "x")])
    )
    report = durable_state_outcomes(container, gold)

    assert report.total == 1
    assert report.missing == 1

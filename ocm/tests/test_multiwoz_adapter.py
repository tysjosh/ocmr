"""End-to-end smoke test for the MultiWOZ adapter (real-data governance).

Drives the full governed write pipeline with the oracle extractor over a small
MultiWOZ-shaped fixture and checks the headline behaviour the paper claims on
real dialogue-state slots:

* governed (gate on): a changed slot **supersedes** → one accepted value, zero
  durable constraint violations;
* ungoverned (gate off): both values are accepted → a durable single-valued
  violation accumulates.
"""

from __future__ import annotations

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.evaluation.datasets.multiwoz_adapter import (
    build_from_dialogues,
    sample_dialogues,
)
from ocm.evaluation.experiment import durable_constraint_violations


def _ingest(container, example, oracle):
    """Write every turn of one example through the governed pipeline."""
    for session in example.sessions:
        source_ref = f"{example.id}:{session.session_id}"
        container.write_pipeline.run(session.input, source_ref)


def _accepted_has_value(container) -> dict[str, set[str]]:
    """Map slot subject -> set of accepted SlotValue object ids (HAS_VALUE edges)."""
    out: dict[str, set[str]] = {}
    for s, o, _k, _d in container.graph.find_edges_by_predicate("HAS_VALUE"):
        out.setdefault(s, set()).add(o)
    return out


def test_build_from_dialogues_shapes_examples_and_oracle():
    examples, oracle = build_from_dialogues(sample_dialogues())
    assert [e.id for e in examples] == ["mwz-0001", "mwz-0002", "mwz-0003"]
    # mwz-0001 has one slot that changed -> its question is conflict-flagged.
    d1 = examples[0]
    assert d1.questions[0].expected_answer_contains == ["south"]
    assert d1.questions[0].expected_conflict is True
    # The oracle knows the per-turn writes.
    assert oracle.extract("", "mwz-0001:t0").relations[0]["write_intent"] == "new_fact"
    assert oracle.extract("", "mwz-0001:t1").relations[0]["write_intent"] == "correction"


def test_governed_supersedes_changed_slot_zero_violations():
    examples, oracle = build_from_dialogues(sample_dialogues())
    container = CoreContainer(
        Settings(deterministic_test_mode=True, chroma_mode="memory"),
        extractor=oracle,
    )
    for ex in examples:
        _ingest(container, ex, oracle)

    accepted = _accepted_has_value(container)
    # Every slot subject holds exactly ONE accepted value under governance.
    assert accepted, "expected HAS_VALUE assertions to be committed"
    assert all(len(objs) == 1 for objs in accepted.values())
    # And no durable single-valued constraint violations remain.
    violations, _ = durable_constraint_violations(container)
    assert violations == 0


def test_ungoverned_accumulates_violation_on_changed_slot():
    examples, oracle = build_from_dialogues(sample_dialogues())
    container = CoreContainer(
        Settings(
            deterministic_test_mode=True,
            chroma_mode="memory",
            enable_contradiction_gate=False,  # ungoverned: no write-time gate
        ),
        extractor=oracle,
    )
    for ex in examples:
        _ingest(container, ex, oracle)

    # With the gate off, the changed slots keep both values -> violations > 0.
    violations, _ = durable_constraint_violations(container)
    assert violations >= 1

"""Foundation tests for the LongMemEval knowledge-update adapter (Arm A / oracle).

Exercises the shared foundation both evaluation arms build on:

* subset loading / filtering (``knowledge-update`` vs ``abstention``);
* the oracle adapter shaping examples + gold writes from a value trajectory;
* the headline governance behaviour on open-domain knowledge updates — governed
  supersede → one accepted value, zero durable violations; ungoverned → both
  values retained → a single-valued violation;
* the offline annotation helpers (evidence-turn selection + validation).
"""

from __future__ import annotations

import json

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.evaluation.datasets.longmemeval_adapter import (
    build_from_kupdate_oracle,
    is_abstention,
    load_longmemeval,
    sample_annotations,
    sample_instances,
)
from ocm.evaluation.datasets.longmemeval_annotate import (
    annotate_instances,
    evidence_turns,
    validate_annotation,
)
from ocm.evaluation.experiment import durable_constraint_violations


def _accepted_has_value(container) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for s, o, _k, _d in container.graph.find_edges_by_predicate("HAS_VALUE"):
        out.setdefault(s, set()).add(o)
    return out


def _ingest(container, example):
    for session in example.sessions:
        container.write_pipeline.run(session.input, f"{example.id}:{session.session_id}")


# -- loading / filtering ---------------------------------------------------- #
def test_load_longmemeval_filters_by_type_and_abstention(tmp_path):
    data = [
        {"question_id": "a", "question_type": "knowledge-update", "question": "q"},
        {"question_id": "b", "question_type": "temporal-reasoning", "question": "q"},
        {"question_id": "c_abs", "question_type": "knowledge-update", "question": "q"},
    ]
    p = tmp_path / "lme.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    # Default: knowledge-update, drop abstention.
    ku = load_longmemeval(str(p))
    assert [i["question_id"] for i in ku] == ["a"]
    # Only abstention questions (any type).
    abst = load_longmemeval(str(p), question_type=None, abstention=True)
    assert [i["question_id"] for i in abst] == ["c_abs"]
    # All types, indifferent to abstention.
    allq = load_longmemeval(str(p), question_type=None, abstention=None)
    assert len(allq) == 3


def test_is_abstention():
    assert is_abstention({"question_id": "x_abs"}) is True
    assert is_abstention({"question_id": "x"}) is False


# -- oracle adapter shaping ------------------------------------------------- #
def test_build_from_kupdate_oracle_shapes_examples_and_intents():
    examples, oracle = build_from_kupdate_oracle(sample_instances(), sample_annotations())
    assert [e.id for e in examples] == ["ku_0001"]
    ex = examples[0]
    # The recall question targets the qualified slot key and expects the latest value.
    assert "[[ku_0001:residence]]" in ex.questions[0].query
    assert ex.questions[0].expected_answer_contains == ["San Francisco"]
    # First value is a new_fact (session a); the change is an update (session c).
    assert oracle.extract("", "ku_0001:s0").relations[0]["write_intent"] == "new_fact"
    assert oracle.extract("", "ku_0001:s2").relations[0]["write_intent"] == "update"
    # The filler session (s1) carries no writes.
    assert oracle.extract("", "ku_0001:s1").relations == []


# -- governance behaviour --------------------------------------------------- #
def test_governed_supersedes_knowledge_update_zero_violations():
    examples, oracle = build_from_kupdate_oracle(sample_instances(), sample_annotations())
    container = CoreContainer(
        Settings(deterministic_test_mode=True, chroma_mode="memory",
                 authoritative_update_supersede=True),
        extractor=oracle,
    )
    for ex in examples:
        _ingest(container, ex)
    accepted = _accepted_has_value(container)
    assert accepted and all(len(objs) == 1 for objs in accepted.values())
    assert durable_constraint_violations(container)[0] == 0


def test_ungoverned_accumulates_violation_on_knowledge_update():
    examples, oracle = build_from_kupdate_oracle(sample_instances(), sample_annotations())
    container = CoreContainer(
        Settings(deterministic_test_mode=True, chroma_mode="memory",
                 enable_contradiction_gate=False),
        extractor=oracle,
    )
    for ex in examples:
        _ingest(container, ex)
    assert durable_constraint_violations(container)[0] >= 1


# -- annotation helpers ----------------------------------------------------- #
def test_evidence_turns_prefers_has_answer_flags():
    inst = {
        "question_id": "k",
        "haystack_session_ids": ["s0", "s1"],
        "answer_session_ids": ["s1"],
        "haystack_sessions": [
            [{"role": "user", "content": "irrelevant"}],
            [{"role": "user", "content": "the evidence", "has_answer": True}],
        ],
    }
    assert evidence_turns(inst) == [("s1", "the evidence")]


def test_evidence_turns_falls_back_to_answer_sessions():
    inst = {
        "question_id": "k",
        "haystack_session_ids": ["s0", "s1"],
        "answer_session_ids": ["s1"],
        "haystack_sessions": [
            [{"role": "user", "content": "irrelevant"}],
            [{"role": "user", "content": "ev1"}, {"role": "assistant", "content": "ev2"}],
        ],
    }
    assert evidence_turns(inst) == [("s1", "ev1"), ("s1", "ev2")]


def test_validate_annotation_flags_answer_mismatch():
    inst = sample_instances()[0]
    good = sample_annotations()["ku_0001"]
    assert validate_annotation(inst, good) == []
    bad = {**good, "current_value": "Boston", "trajectory": [
        {"session_id": "sess_a", "value": "Boston"}]}
    assert any("disagrees" in p for p in validate_annotation(inst, bad))


def test_annotate_instances_skips_invalid():
    insts = sample_instances()

    def good_fn(inst):
        return sample_annotations()[inst["question_id"]]

    def bad_fn(inst):
        return {"attribute": "residence", "trajectory": [], "current_value": "X"}

    assert set(annotate_instances(insts, good_fn)) == {"ku_0001"}
    assert annotate_instances(insts, bad_fn) == {}

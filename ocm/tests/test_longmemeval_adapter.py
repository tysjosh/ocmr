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


# -- suite runner (knowledge-update arm) ------------------------------------ #
def test_run_longmemeval_suite_governed_beats_ungoverned():
    from ocm.evaluation.datasets.longmemeval_adapter import run_longmemeval_suite

    report = run_longmemeval_suite(
        sample_instances(), sample_annotations(),
        baselines=("B0", "B2", "B3"), seeds=(1337,),
    )
    assert report["dataset"] == "longmemeval"
    assert report["subset"] == "knowledge-update"
    dm = report["decisive_metrics"]
    v = lambda m: dm[m]["constraint_violations"]["mean"]
    ts = lambda m: dm[m]["task_success"]["mean"]
    # Governed full system supersedes the changed fact to zero violations;
    # ungoverned arms keep both values and accumulate a single-valued violation.
    assert v("B3") == 0.0
    assert v("B0") > 0.0 and v("B2") > 0.0
    # B3 supersedes where ungoverned arms do not.
    assert report["write_outcomes"]["B3"]["superseded"] > report["write_outcomes"]["B0"]["superseded"]
    # Recall preserved: the governed arm recalls the current (latest) value.
    assert ts("B3") > 0.0


# -- abstention arm --------------------------------------------------------- #
def test_build_abstention_examples_has_no_grounded_writes():
    from ocm.evaluation.datasets.longmemeval_adapter import build_abstention_examples

    insts = [{
        "question_id": "ku_0001_abs",
        "question_type": "knowledge-update",
        "question": "What did the user say about owning a yacht?",
        "haystack_session_ids": ["s0"],
        "haystack_sessions": [[{"role": "user", "content": "I like sailing."}]],
        "answer_session_ids": [],
    }]
    examples, oracle = build_abstention_examples(insts)
    assert examples[0].category == "abstention"
    # No grounded HAS_VALUE write for the ungrounded fact.
    assert oracle.extract("", "ku_0001_abs:s0").relations == []


def test_evaluate_abstention_governed_abstains_when_ungrounded():
    from ocm.evaluation.datasets.longmemeval_adapter import evaluate_abstention

    insts = [{
        "question_id": "ku_0001_abs",
        "question_type": "knowledge-update",
        "question": "What is the user's frequent-flyer number?",
        "haystack_session_ids": ["s0"],
        "haystack_sessions": [[{"role": "user", "content": "I flew to Paris last week."}]],
        "answer_session_ids": [],
    }]
    res = evaluate_abstention(insts, baselines=("B3",))
    # With no grounded value, the governed system must abstain (100% on plumbing).
    assert res["B3"]["abstention_accuracy"] == 100.0
    assert res["B3"]["n"] == 1

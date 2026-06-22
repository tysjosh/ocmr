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
from types import SimpleNamespace

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


# -- concrete LLM annotator (model-agnostic, tested with a fake chat_fn) ----- #
def test_parse_annotation_json_tolerates_prose():
    from ocm.evaluation.datasets.longmemeval_annotate import parse_annotation_json

    text = 'Sure! Here is the label:\n{"attribute": "residence", "values": ["NY", "SF"], "current_value": "SF"}\nDone.'
    obj = parse_annotation_json(text)
    assert obj == {"attribute": "residence", "values": ["NY", "SF"], "current_value": "SF"}
    assert parse_annotation_json("no json here") is None


def test_align_values_to_sessions_preserves_order():
    from ocm.evaluation.datasets.longmemeval_annotate import align_values_to_sessions

    inst = {
        "question_id": "k",
        "haystack_session_ids": ["s0", "s1", "s2"],
        "haystack_sessions": [
            [{"role": "user", "content": "I live in New York"}],
            [{"role": "user", "content": "weather chat"}],
            [{"role": "user", "content": "moved to San Francisco"}],
        ],
    }
    traj = align_values_to_sessions(inst, ["New York", "San Francisco"])
    assert traj == [
        {"session_id": "s0", "value": "New York"},
        {"session_id": "s2", "value": "San Francisco"},
    ]
    # Ungrounded values are skipped.
    assert align_values_to_sessions(inst, ["Boston"]) == []


def test_build_llm_annotate_fn_end_to_end_with_fake_model():
    from ocm.evaluation.datasets.longmemeval_adapter import (
        build_from_kupdate_oracle,
        sample_instances,
    )
    from ocm.evaluation.datasets.longmemeval_annotate import (
        annotate_instances,
        build_llm_annotate_fn,
    )

    # A fake "model" that returns the gold JSON for the fixture question.
    def fake_chat(prompt: str) -> str:
        return '{"attribute": "residence", "values": ["New York", "San Francisco"], "current_value": "San Francisco"}'

    insts = sample_instances()
    anns = annotate_instances(insts, build_llm_annotate_fn(fake_chat))
    assert set(anns) == {"ku_0001"}
    assert anns["ku_0001"]["attribute"] == "residence"
    assert [t["value"] for t in anns["ku_0001"]["trajectory"]] == ["New York", "San Francisco"]
    # And the produced annotation drives the oracle adapter correctly.
    examples, oracle = build_from_kupdate_oracle(insts, anns)
    assert oracle.extract("", "ku_0001:s0").relations[0]["write_intent"] == "new_fact"
    assert oracle.extract("", "ku_0001:s2").relations[0]["write_intent"] == "update"


def test_build_llm_annotate_fn_rejects_single_value():
    from ocm.evaluation.datasets.longmemeval_adapter import sample_instances
    from ocm.evaluation.datasets.longmemeval_annotate import build_llm_annotate_fn

    def one_value(prompt: str) -> str:
        return '{"attribute": "residence", "values": ["New York"], "current_value": "New York"}'

    fn = build_llm_annotate_fn(one_value)
    assert fn(sample_instances()[0]) is None  # a knowledge update needs >= 2 values


# -- Arm B: end-to-end from text (real extraction, fake model) -------------- #
def _fake_fact_chat(prompt: str) -> str:
    """A deterministic stand-in for an LLM fact extractor, keyed off content."""
    text = prompt.lower()
    if "new york" in text:
        return '[{"attribute": "residence", "value": "New York"}]'
    if "san francisco" in text:
        return '[{"attribute": "residence", "value": "San Francisco"}]'
    return "[]"


def test_parse_facts_json_array_and_facts_object():
    from ocm.evaluation.datasets.longmemeval_adapter import parse_facts_json

    assert parse_facts_json('ok: [{"attribute":"a","value":"1"}] done') == [
        {"attribute": "a", "value": "1"}]
    assert parse_facts_json('{"facts": [{"attribute":"a","value":"1"}]}') == [
        {"attribute": "a", "value": "1"}]
    assert parse_facts_json("nothing") == []


def test_normalize_attribute():
    from ocm.evaluation.datasets.longmemeval_adapter import normalize_attribute

    assert normalize_attribute("Where I Live!") == "where_i_live"
    assert normalize_attribute("residence") == "residence"


def test_build_e2e_belief_tracks_intent_and_caches_writes():
    from ocm.evaluation.datasets.longmemeval_adapter import (
        build_e2e_from_extraction,
        build_fact_extract_fn,
    )

    fx = build_fact_extract_fn(_fake_fact_chat)
    examples, oracle = build_e2e_from_extraction(sample_instances(), fx)
    # First mention is a new_fact; the later changed value is an update.
    assert oracle.extract("", "ku_0001:s0").relations[0]["write_intent"] == "new_fact"
    assert oracle.extract("", "ku_0001:s2").relations[0]["write_intent"] == "update"
    # The filler session yields no facts.
    assert oracle.extract("", "ku_0001:s1").relations == []
    # The recall question is the natural question (no slot marker) + gold answer.
    assert examples[0].questions[0].query == "Where does the user currently live?"
    assert examples[0].questions[0].expected_answer_contains == ["San Francisco"]


def test_build_e2e_new_fact_mode_does_not_emit_update():
    from ocm.evaluation.datasets.longmemeval_adapter import (
        build_e2e_from_extraction,
        build_fact_extract_fn,
    )

    fx = build_fact_extract_fn(_fake_fact_chat)
    _, oracle = build_e2e_from_extraction(sample_instances(), fx, intent_mode="new_fact")
    # Conservative mode: the changed value is still a new_fact (gate will quarantine).
    assert oracle.extract("", "ku_0001:s2").relations[0]["write_intent"] == "new_fact"


def test_run_longmemeval_e2e_governed_beats_ungoverned_on_violations():
    from ocm.evaluation.datasets.longmemeval_adapter import (
        build_fact_extract_fn,
        run_longmemeval_e2e,
    )

    fx = build_fact_extract_fn(_fake_fact_chat)
    report = run_longmemeval_e2e(
        sample_instances(), fx, baselines=("B0", "B2", "B3"), seeds=(1337,)
    )
    assert report["arm"] == "end_to_end"
    dm = report["decisive_metrics"]
    v = lambda m: dm[m]["constraint_violations"]["mean"]
    # Governed supersede → 0 durable violations; ungoverned keep both values.
    assert v("B3") == 0.0
    assert v("B0") > 0.0 and v("B2") > 0.0
    assert report["write_outcomes"]["B3"]["superseded"] >= 1


def _abs_instance() -> list[dict]:
    return [{
        "question_id": "ku_0002_abs",
        "question_type": "knowledge-update",
        "question": "What is the user's frequent-flyer number?",
        "haystack_session_ids": ["s0"],
        "haystack_sessions": [[
            {"role": "user", "content": "I adopted a cat named Fluffy."},
            {"role": "assistant", "content": "That's sweet."},
        ]],
        "answer_session_ids": [],
    }]


def test_build_abstention_e2e_extracts_full_haystack_facts():
    from ocm.evaluation.datasets.longmemeval_adapter import (
        build_abstention_e2e_from_extraction,
        build_fact_extract_fn,
    )

    def pet_fact_chat(prompt: str) -> str:
        return '[{"attribute": "pet_name", "value": "Fluffy"}]'

    fx = build_fact_extract_fn(pet_fact_chat)
    examples, oracle = build_abstention_e2e_from_extraction(_abs_instance(), fx)
    assert examples[0].category == "abstention_e2e"
    assert examples[0].questions[0].query == "What is the user's frequent-flyer number?"
    assert examples[0].questions[0].expected_answer_contains == []
    rel = oracle.extract("", "ku_0002_abs:s0").relations[0]
    assert rel["predicate"] == "HAS_VALUE"
    assert rel["write_intent"] == "new_fact"


def test_evaluate_abstention_e2e_abstains_when_extraction_empty():
    from ocm.evaluation.datasets.longmemeval_adapter import (
        build_fact_extract_fn,
        evaluate_abstention_e2e,
    )

    fx = build_fact_extract_fn(lambda _prompt: "[]")
    report = evaluate_abstention_e2e(
        _abs_instance(), fx, baselines=("B3",), seeds=(1337,)
    )
    assert report["subset"] == "abstention"
    assert report["arm"] == "end_to_end"
    metric = report["abstention_metrics"]["B3"]["abstention_accuracy"]
    assert metric["mean"] == 100.0
    assert report["counts"]["B3"]["abstained"] == 1
    assert report["counts"]["B3"]["non_abstained"] == 0


def test_package_abstention_depends_on_final_answer_not_support():
    from ocm.evaluation.datasets.longmemeval_adapter import _package_is_abstention

    assert _package_is_abstention(
        SimpleNamespace(answer=None, supporting_assertions=[object()])
    )
    assert not _package_is_abstention(
        SimpleNamespace(answer="Fluffy", supporting_assertions=[])
    )


def test_evaluate_abstention_e2e_counts_noisy_support_as_diagnostic():
    from ocm.evaluation.datasets.longmemeval_adapter import (
        build_fact_extract_fn,
        evaluate_abstention_e2e,
    )

    def pet_fact_chat(prompt: str) -> str:
        text = prompt.lower()
        if "fluffy" in text:
            return '[{"attribute": "pet_name", "value": "Fluffy"}]'
        return "[]"

    fx = build_fact_extract_fn(pet_fact_chat)
    report = evaluate_abstention_e2e(
        _abs_instance(), fx, baselines=("B0",), seeds=(1337,)
    )
    metric = report["abstention_metrics"]["B0"]["abstention_accuracy"]
    false_answer = report["abstention_metrics"]["B0"]["false_answer_rate"]
    support = report["abstention_metrics"]["B0"]["supporting_response_rate"]
    assert metric["mean"] == 100.0
    assert false_answer["mean"] == 0.0
    assert support["mean"] == 100.0
    assert report["counts"]["B0"]["abstained"] == 1
    assert report["counts"]["B0"]["non_abstained"] == 0
    assert report["counts"]["B0"]["supporting_responses"] == 1
    assert report["write_outcomes"]["B0"]["accepted"] >= 1

"""MemGPT-style baseline (Bmemgpt) — LLM-managed self-editing memory.

Verifies the paradigm contrast the paper draws between LLM-managed memory and
constraint-governed admission:

* When the LLM correctly decides ``update`` on a changed fact, the incumbent is
  superseded and the store stays single-valued (zero durable violations) — like
  OCMR on trusted updates.
* When the LLM incorrectly decides ``insert`` on a changed fact, BOTH values
  stay active — a durable single-valued violation. This is the failure mode
  OCMR's contradiction gate prevents by construction, and it is why LLM-managed
  memory carries no consistency guarantee.

Also covers the decision prompt/parsing helpers.
"""

from __future__ import annotations

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.evaluation.arms import baseline_settings_overrides
from ocm.evaluation.datasets.longmemeval_adapter import (
    build_e2e_from_extraction,
    build_memgpt_decide_fn,
    parse_memgpt_action,
)
from ocm.evaluation.experiment import durable_constraint_violations


def _fact_extract(text: str):
    """Deterministic per-session fact extractor for the fixture."""
    if "New York" in text:
        return [{"attribute": "residence", "value": "New York"}]
    if "San Francisco" in text:
        return [{"attribute": "residence", "value": "San Francisco"}]
    return []


def _instances():
    return [{
        "question_id": "ku_mem_0001",
        "question_type": "knowledge-update",
        "question": "Where does the user currently live?",
        "answer": "San Francisco",
        "haystack_session_ids": ["s0", "s1"],
        "haystack_sessions": [
            [{"role": "user", "content": "I live in New York."}],
            [{"role": "user", "content": "I moved to San Francisco."}],
        ],
        "answer_session_ids": ["s0", "s1"],
    }]


def _ingest(container, example):
    for s in example.sessions:
        container.write_pipeline.run(s.input, f"{example.id}:{s.session_id}")


def test_memgpt_correct_update_supersedes_zero_violations():
    # decide_fn returns "update" on the changed fact -> overwrite -> single value.
    def decide(*, attribute, new_value, current_value):
        return "insert" if not current_value else "update"

    examples, oracle = build_e2e_from_extraction(
        _instances(), _fact_extract, intent_mode="memgpt", decide_fn=decide
    )
    # s0 inserts New York; s1 updates -> San Francisco supersedes.
    assert oracle.extract("", "ku_mem_0001:s0").relations[0]["write_intent"] == "new_fact"
    assert oracle.extract("", "ku_mem_0001:s1").relations[0]["write_intent"] == "update"

    container = CoreContainer(
        Settings(deterministic_test_mode=True, chroma_mode="memory",
                 **baseline_settings_overrides("Bmemgpt")),
        extractor=oracle,
    )
    _ingest(container, examples[0])
    assert durable_constraint_violations(container)[0] == 0


def test_memgpt_wrong_insert_leaves_violation():
    # decide_fn always "insert" -> the changed fact does NOT overwrite ->
    # two active values for the same slot -> a durable violation.
    def decide(*, attribute, new_value, current_value):
        return "insert"

    examples, oracle = build_e2e_from_extraction(
        _instances(), _fact_extract, intent_mode="memgpt", decide_fn=decide
    )
    assert oracle.extract("", "ku_mem_0001:s1").relations[0]["write_intent"] == "new_fact"

    container = CoreContainer(
        Settings(deterministic_test_mode=True, chroma_mode="memory",
                 **baseline_settings_overrides("Bmemgpt")),
        extractor=oracle,
    )
    _ingest(container, examples[0])
    # MemGPT's insert error leaves both values active — the guarantee OCMR keeps.
    assert durable_constraint_violations(container)[0] >= 1


def test_governed_b3_has_no_violation_on_same_stream():
    # The same changed fact under full governance (authoritative update) is
    # always superseded, regardless of the (here wrong) memgpt-style intent.
    def decide(*, attribute, new_value, current_value):
        return "insert"

    examples, oracle = build_e2e_from_extraction(
        _instances(), _fact_extract, intent_mode="memgpt", decide_fn=decide
    )
    container = CoreContainer(
        Settings(deterministic_test_mode=True, chroma_mode="memory",
                 authoritative_update_supersede=True),  # full governed path
        extractor=oracle,
    )
    _ingest(container, examples[0])
    assert durable_constraint_violations(container)[0] == 0


def test_build_memgpt_decide_fn_short_circuits_and_parses():
    calls = {"n": 0}

    def fake_chat(prompt: str) -> str:
        calls["n"] += 1
        return '{"action": "update"}'

    decide = build_memgpt_decide_fn(fake_chat)
    # No current value -> insert without calling the model.
    assert decide(attribute="residence", new_value="NY", current_value=None) == "insert"
    assert calls["n"] == 0
    # With a current value -> model decides.
    assert decide(attribute="residence", new_value="SF", current_value="NY") == "update"
    assert calls["n"] == 1


def test_parse_memgpt_action_fallback_is_insert():
    assert parse_memgpt_action('{"action":"update"}') == "update"
    assert parse_memgpt_action('noise {"action":"skip"} more') == "skip"
    assert parse_memgpt_action("unparseable") == "insert"  # safe default

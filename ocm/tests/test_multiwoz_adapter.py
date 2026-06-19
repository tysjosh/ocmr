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
from ocm.evaluation.baselines import build_baseline
from ocm.evaluation.datasets.multiwoz_adapter import (
    build_from_dialogues,
    normalize_hf_multiwoz,
    normalize_raw_multiwoz,
    run_multiwoz_suite,
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
    # mwz-0001's first slot recalls the current (post-change) value; MultiWOZ
    # models supersession, not contradiction-surfacing, so expected_conflict=False.
    d1 = examples[0]
    assert d1.questions[0].expected_answer_contains == ["south"]
    assert d1.questions[0].expected_conflict is False
    # The oracle knows the per-turn writes.
    assert oracle.extract("", "mwz-0001:t0").relations[0]["write_intent"] == "new_fact"
    assert oracle.extract("", "mwz-0001:t1").relations[0]["write_intent"] == "update"


def test_governed_supersedes_changed_slot_zero_violations():
    examples, oracle = build_from_dialogues(sample_dialogues())
    container = CoreContainer(
        Settings(deterministic_test_mode=True, chroma_mode="memory",
                 authoritative_update_supersede=True),
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


def test_authoritative_update_supersedes_serial_changes():
    # A slot changed three times: the latest value must win (supersede each
    # time), not quarantine after the first change. Zero violations, no
    # quarantines, and the current value is retained.
    dialogues = [{
        "dialogue_id": "d",
        "turns": [
            {"utterance": "centre", "state": {"hotel-area": "centre"}},
            {"utterance": "south", "state": {"hotel-area": "south"}},
            {"utterance": "north", "state": {"hotel-area": "north"}},
            {"utterance": "east", "state": {"hotel-area": "east"}},
        ],
    }]
    examples, oracle = build_from_dialogues(dialogues)
    container = CoreContainer(
        Settings(deterministic_test_mode=True, chroma_mode="memory",
                 authoritative_update_supersede=True),
        extractor=oracle,
    )
    ex = examples[0]
    superseded = quarantined = 0
    for s in ex.sessions:
        r = container.write_pipeline.run(s.input, f"{ex.id}:{s.session_id}")
        superseded += len(r.superseded)
        quarantined += len(r.quarantined)
    assert superseded == 3 and quarantined == 0
    # The accepted current value is the latest ("east").
    values = {
        (container.graph.get_entity_payload(o) or {}).get("value")
        for _s, o, _k, _d in container.graph.find_edges_by_predicate("HAS_VALUE")
    }
    assert values == {"east"}
    assert durable_constraint_violations(container)[0] == 0


def test_without_policy_serial_updates_quarantine():
    # Default (policy off): an ``update`` conflict is conservatively quarantined,
    # so serial changes after the first do not supersede. This guards the
    # default behavior and documents why MultiWOZ enables the policy.
    dialogues = [{
        "dialogue_id": "d",
        "turns": [
            {"utterance": "centre", "state": {"hotel-area": "centre"}},
            {"utterance": "south", "state": {"hotel-area": "south"}},
        ],
    }]
    examples, oracle = build_from_dialogues(dialogues)
    container = CoreContainer(
        Settings(deterministic_test_mode=True, chroma_mode="memory"),  # policy OFF
        extractor=oracle,
    )
    ex = examples[0]
    quarantined = 0
    for s in ex.sessions:
        r = container.write_pipeline.run(s.input, f"{ex.id}:{s.session_id}")
        quarantined += len(r.quarantined)
    assert quarantined >= 1  # the change is quarantined without the policy


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


def test_normalize_raw_multiwoz_extracts_user_state():
    # Official MultiWOZ 2.2 JSON shape: speaker USER/SYSTEM, frames[].state.slot_values.
    raw = {
        "dialogue_id": "MUL0001.json",
        "turns": [
            {"speaker": "USER", "utterance": "centre please",
             "frames": [{"state": {"slot_values": {"hotel-area": ["centre"]}}}]},
            {"speaker": "SYSTEM", "utterance": "ok", "frames": []},
            {"speaker": "USER", "utterance": "make it south",
             "frames": [{"state": {"slot_values": {"hotel-area": ["south"]}}}]},
        ],
    }
    norm = normalize_raw_multiwoz(raw)
    assert norm["dialogue_id"] == "MUL0001.json"
    assert [t["state"] for t in norm["turns"]] == [
        {"hotel-area": "centre"}, {"hotel-area": "south"}
    ]


def test_normalize_hf_multiwoz_extracts_user_state():
    # A faithful mini HF multi_woz_v22-shaped record (columnar turns).
    hf = {
        "dialogue_id": "PMUL0001.json",
        "turns": {
            "speaker": [0, 1, 0],
            "utterance": ["a hotel in the centre", "sure", "make it south"],
            "frames": [
                {"state": [
                    {"slots_values": {"slots_values_name": ["hotel-area"],
                                       "slots_values_list": [["centre"]]}}
                ]},
                {"state": []},
                {"state": [
                    {"slots_values": {"slots_values_name": ["hotel-area"],
                                       "slots_values_list": [["south"]]}}
                ]},
            ],
        },
    }
    norm = normalize_hf_multiwoz(hf)
    assert norm["dialogue_id"] == "PMUL0001.json"
    # Only the two USER turns are kept, with their cumulative state.
    assert [t["state"] for t in norm["turns"]] == [
        {"hotel-area": "centre"}, {"hotel-area": "south"}
    ]


def test_run_multiwoz_suite_governed_beats_ungoverned_on_violations():
    report = run_multiwoz_suite(sample_dialogues(), baselines=("B0", "B2", "B3"), seeds=(1337,))
    dm = report["decisive_metrics"]
    v = lambda m: dm[m]["constraint_violations"]["mean"]
    ts = lambda m: dm[m]["task_success"]["mean"]
    # Ungoverned arms accumulate single-valued violations; the governed full
    # system supersedes changed slots to zero.
    assert v("B3") == 0.0
    assert v("B0") > 0.0 and v("B2") > 0.0
    # B3 supersedes (changed slots) where ungoverned arms do not.
    assert report["write_outcomes"]["B3"]["superseded"] > report["write_outcomes"]["B0"]["superseded"]
    # The HAS_VALUE answer-derivation rule makes task success meaningful: the
    # governed arm recalls the current slot value (no tradeoff on supersession).
    assert ts("B3") > 0.0


def test_slot_key_exact_match_disambiguates_substring_keys():
    # Two slots where one key is a substring of the other, set to different
    # values; the [[key]] marker must resolve each to its OWN value, not the
    # substring sibling's.
    dialogues = [{
        "dialogue_id": "d",
        "turns": [
            {"utterance": "set both",
             "state": {"hotel-area": "centre", "hotel-area-code": "CB1"}},
        ],
    }]
    examples, oracle = build_from_dialogues(dialogues)
    container = CoreContainer(
        Settings(deterministic_test_mode=True, chroma_mode="memory"), extractor=oracle
    )
    ex = examples[0]
    for s in ex.sessions:
        container.write_pipeline.run(s.input, f"{ex.id}:{s.session_id}")
    # Query the shorter key explicitly; answer must be its value, not "CB1".
    pkg = build_baseline("B3", container).query(
        "What is the current value of slot [[d:hotel-area]]?", top_k=10
    )
    assert pkg.answer is not None and "centre" in pkg.answer
    assert "CB1" not in pkg.answer

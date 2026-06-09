"""Unit tests for the deterministic offline Mock_Extractor (task 10.1).

These tests pin the behaviors W1 relies on and that downstream stages
(normalizer, resolver, write pipeline, Task T1 integration test) depend on:

* The extractor satisfies the :class:`Extractor` protocol and returns a
  Pydantic-validated :class:`ExtractionResult` (Req 3.1, 3.2).
* Identical input + identical config yields byte-identical output (Req 3.5).
* The Task T1 scenario sentences extract the expected entities, events, and
  relations (supports the 13.9 integration test).
* It runs offline with no API key / network (Req 3.7) — guaranteed by having
  no network imports and no external calls.
"""

from __future__ import annotations

from ocm.extraction.base import Extractor
from ocm.extraction.mock_extractor import MockExtractor
from ocm.memory.contracts import ExtractionResult


def _rels(result: ExtractionResult) -> set[tuple[str, str, str]]:
    return {(r["subject"], r["predicate"], r["object"]) for r in result.relations}


def _entities(result: ExtractionResult) -> set[tuple[str, str]]:
    return {(e["type"], e["name"]) for e in result.entities}


def test_satisfies_extractor_protocol_and_version() -> None:
    ex = MockExtractor()
    assert isinstance(ex, Extractor)
    assert ex.version == "mock-1"


def test_returns_validated_extraction_result() -> None:
    result = MockExtractor().extract("Alice owns Project Orion.", "doc::1")
    assert isinstance(result, ExtractionResult)
    assert result.extractor_version == "mock-1"


def test_deterministic_byte_identical_output() -> None:
    text = "Alice owns Project Orion. Bob is assigned to Task T1."
    a = MockExtractor().extract(text, "doc::1").model_dump_json()
    b = MockExtractor().extract(text, "doc::1").model_dump_json()
    assert a == b


def test_owns_relation_extraction() -> None:
    result = MockExtractor().extract("Alice owns Project Orion.", "doc::1")
    assert ("Person", "Alice") in _entities(result)
    assert ("Project", "Orion") in _entities(result)
    assert ("Alice", "OWNS", "Orion") in _rels(result)


def test_assigned_to_extraction_in_registry_direction() -> None:
    result = MockExtractor().extract("Bob is assigned to Task T1.", "doc::1")
    assert ("Person", "Bob") in _entities(result)
    assert ("Task", "T1") in _entities(result)
    # ASSIGNED_TO is Task -> Person per the relation registry.
    assert ("T1", "ASSIGNED_TO", "Bob") in _rels(result)


def test_completion_sets_done_with_completion_event_and_results_in() -> None:
    result = MockExtractor().extract("Bob completed Task T1.", "doc::1")
    task = next(e for e in result.entities if e["type"] == "Task")
    assert task["fields"]["status"] == "done"
    assert len(result.events) == 1
    event_name = result.events[0]["name"]
    # A done task is backed by a RESULTS_IN completion event (constraint C4).
    assert (event_name, "RESULTS_IN", "T1") in _rels(result)
    assert ("Bob", "PARTICIPATES_IN", event_name) in _rels(result)


def test_status_statement_not_started_maps_to_todo() -> None:
    result = MockExtractor().extract("Task T1 is not started.", "doc::1")
    task = next(e for e in result.entities if e["type"] == "Task")
    assert task["fields"]["status"] == "todo"


def test_correction_keyword_sets_write_intent() -> None:
    result = MockExtractor().extract("Actually, Alice owns Project Orion.", "doc::1")
    owns = next(r for r in result.relations if r["predicate"] == "OWNS")
    assert owns["write_intent"] == "correction"


def test_default_confidence_is_high() -> None:
    result = MockExtractor().extract("Alice owns Project Orion.", "doc::1")
    assert all(r["confidence"] == 0.95 for r in result.relations)


def test_t1_scenario_three_writes() -> None:
    ex = MockExtractor()
    w1 = ex.extract("Alice owns Project Orion. Bob is assigned to Task T1.", "s::1")
    w2 = ex.extract("Bob completed Task T1.", "s::2")
    w3 = ex.extract("Task T1 is not started.", "s::3")

    assert ("Alice", "OWNS", "Orion") in _rels(w1)
    assert ("T1", "ASSIGNED_TO", "Bob") in _rels(w1)

    done_task = next(e for e in w2.entities if e["type"] == "Task")
    assert done_task["fields"]["status"] == "done"
    assert w2.events  # completion event present

    todo_task = next(e for e in w3.entities if e["type"] == "Task")
    assert todo_task["fields"]["status"] == "todo"

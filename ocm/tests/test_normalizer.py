"""Unit tests for the W2 Normalizer (task 10.4).

These tests pin the value-level normalization behaviors that downstream stages
(entity resolution, assertion building, schema/constraint validation) depend on:

* Status synonyms map to canonical enum values, type-aware, including the
  headline "completed" -> "done" rule for Tasks while "completed" stays
  "completed" for Projects (Req 4.3).
* Priority synonyms map to canonical enum values, including "high priority" ->
  "high" (Req 4.4).
* Relation names canonicalize to registered predicate identifiers, e.g.
  "assigned to" -> "ASSIGNED_TO", "owns" -> "OWNS" (Req 4.5).
* Confidence values are parsed and clamped into [0, 1] (Req 4.6).
* Distinct entities are preserved as distinct and never merged on the basis of
  normalization (Req 4.7).
"""

from __future__ import annotations

from ocm.memory.contracts import ExtractionResult
from ocm.resolution.normalizer import Normalizer


def _result(
    *,
    entities: list[dict] | None = None,
    relations: list[dict] | None = None,
    claims: list[dict] | None = None,
) -> ExtractionResult:
    """Build a minimal ExtractionResult with only the lists under test."""
    return ExtractionResult(
        entities=entities or [],
        events=[],
        claims=claims or [],
        documents=[],
        decisions=[],
        relations=relations or [],
        extractor_version="test-1",
    )


def _entity(entity_type: str, name: str, **fields) -> dict:
    """Build an entity dict with status/priority nested under "fields"."""
    return {"type": entity_type, "name": name, "fields": dict(fields)}


# --------------------------------------------------------------------------- #
# Status synonym mapping (Req 4.3)
# --------------------------------------------------------------------------- #

def test_task_completed_maps_to_done() -> None:
    """A Task status of "completed" canonicalizes to "done" (Req 4.3)."""
    result = Normalizer().normalize(
        _result(entities=[_entity("Task", "T1", status="completed")])
    )
    assert result.entities[0]["fields"]["status"] == "done"


def test_project_completed_stays_completed() -> None:
    """Status mapping is type-aware: a Project stays "completed" (Req 4.3)."""
    result = Normalizer().normalize(
        _result(entities=[_entity("Project", "Orion", status="completed")])
    )
    assert result.entities[0]["fields"]["status"] == "completed"


def test_task_not_started_maps_to_todo() -> None:
    """"not started" canonicalizes to the Task "todo" enum value (Req 4.3)."""
    result = Normalizer().normalize(
        _result(entities=[_entity("Task", "T1", status="not started")])
    )
    assert result.entities[0]["fields"]["status"] == "todo"


def test_task_in_progress_synonym_maps_to_in_progress() -> None:
    """"in progress" canonicalizes to the "in_progress" enum value (Req 4.3)."""
    result = Normalizer().normalize(
        _result(entities=[_entity("Task", "T1", status="in progress")])
    )
    assert result.entities[0]["fields"]["status"] == "in_progress"


def test_status_normalized_at_top_level_too() -> None:
    """Status synonyms are mapped whether nested in "fields" or top-level (Req 4.3)."""
    result = Normalizer().normalize(
        _result(entities=[{"type": "Task", "name": "T1", "status": "completed"}])
    )
    assert result.entities[0]["status"] == "done"


# --------------------------------------------------------------------------- #
# Priority synonym mapping (Req 4.4)
# --------------------------------------------------------------------------- #

def test_priority_high_priority_maps_to_high() -> None:
    """"high priority" canonicalizes to "high" (Req 4.4)."""
    result = Normalizer().normalize(
        _result(entities=[_entity("Task", "T1", priority="high priority")])
    )
    assert result.entities[0]["fields"]["priority"] == "high"


def test_priority_in_range_value_preserved() -> None:
    """An already-canonical priority value is preserved (Req 4.4)."""
    result = Normalizer().normalize(
        _result(entities=[_entity("Task", "T1", priority="urgent")])
    )
    assert result.entities[0]["fields"]["priority"] == "urgent"


# --------------------------------------------------------------------------- #
# Confidence parsing / clamping (Req 4.6)
# --------------------------------------------------------------------------- #

def test_confidence_above_one_clamps_to_one() -> None:
    """A numeric confidence above 1.0 clamps to 1.0 (Req 4.6)."""
    result = Normalizer().normalize(
        _result(relations=[{"subject": "a", "predicate": "OWNS",
                            "object": "b", "confidence": 1.5}])
    )
    assert result.relations[0]["confidence"] == 1.0


def test_confidence_below_zero_clamps_to_zero() -> None:
    """A negative numeric confidence clamps to 0.0 (Req 4.6)."""
    result = Normalizer().normalize(
        _result(relations=[{"subject": "a", "predicate": "OWNS",
                            "object": "b", "confidence": -3}])
    )
    assert result.relations[0]["confidence"] == 0.0


def test_confidence_in_range_preserved() -> None:
    """An in-range confidence value is preserved unchanged (Req 4.6)."""
    result = Normalizer().normalize(
        _result(relations=[{"subject": "a", "predicate": "OWNS",
                            "object": "b", "confidence": 0.42}])
    )
    assert result.relations[0]["confidence"] == 0.42


def test_confidence_percentage_string_parsed() -> None:
    """A percentage string like "80%" parses to 0.8 (Req 4.6)."""
    result = Normalizer().normalize(
        _result(relations=[{"subject": "a", "predicate": "OWNS",
                            "object": "b", "confidence": "80%"}])
    )
    assert result.relations[0]["confidence"] == 0.8


def test_confidence_textual_term_parsed() -> None:
    """A textual confidence term like "high" parses to its numeric value (Req 4.6)."""
    result = Normalizer().normalize(
        _result(relations=[{"subject": "a", "predicate": "OWNS",
                            "object": "b", "confidence": "high"}])
    )
    assert result.relations[0]["confidence"] == 0.9


# --------------------------------------------------------------------------- #
# Predicate canonicalization (Req 4.5)
# --------------------------------------------------------------------------- #

def test_predicate_assigned_to_canonicalized() -> None:
    """The relation name "assigned to" canonicalizes to "ASSIGNED_TO" (Req 4.5)."""
    result = Normalizer().normalize(
        _result(relations=[{"subject": "T1", "predicate": "assigned to",
                            "object": "Bob", "confidence": 0.9}])
    )
    assert result.relations[0]["predicate"] == "ASSIGNED_TO"


def test_predicate_owns_canonicalized() -> None:
    """The relation name "owns" canonicalizes to "OWNS" (Req 4.5)."""
    result = Normalizer().normalize(
        _result(relations=[{"subject": "Alice", "predicate": "owns",
                            "object": "Orion", "confidence": 0.9}])
    )
    assert result.relations[0]["predicate"] == "OWNS"


# --------------------------------------------------------------------------- #
# Distinct entities preserved (Req 4.7)
# --------------------------------------------------------------------------- #

def test_distinct_entities_are_not_merged() -> None:
    """Two entities with different names remain two distinct entries (Req 4.7)."""
    result = Normalizer().normalize(
        _result(entities=[
            _entity("Person", "alice"),
            _entity("Person", "bob"),
        ])
    )
    assert len(result.entities) == 2
    names = [e["name"] for e in result.entities]
    assert names == ["Alice", "Bob"]


def test_normalization_does_not_collapse_identical_canonical_names() -> None:
    """Entities whose names canonicalize to the same form are still preserved (Req 4.7)."""
    result = Normalizer().normalize(
        _result(entities=[
            _entity("Person", "alice smith"),
            _entity("Person", "Alice  Smith"),
        ])
    )
    # Both canonicalize to "Alice Smith" (case + collapsed whitespace) but the
    # Normalizer must NOT merge them; merging is the Entity Resolver's job.
    assert len(result.entities) == 2
    assert all(e["name"] == "Alice Smith" for e in result.entities)

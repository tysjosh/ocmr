"""Unit tests for the W5 ``SchemaValidator`` (structural validation only).

These tests pin down the structural behavior the Schema_Validator guarantees on
a :class:`CandidateAssertion` before constraint checks (W6) run:

* A well-formed candidate — registered predicate (``OWNS``), subject ``Person``
  and object ``Project`` both present in the Graph_Store, confidence in range —
  passes validation (Req 7.2, 7.5, 7.6).
* An unregistered predicate fails with ``failed_check`` naming the registered
  predicate check (Req 7.2).
* A candidate whose subject/object are absent from the graph fails the entity
  reference check (Req 7.5).
* W5 is **structural only**: it does NOT reject a domain/range type mismatch
  against resolved entity types — that graph-level check is deferred to
  constraint C9 at W6. An ``OWNS`` candidate whose subject is actually a ``Task``
  still passes W5 structurally, documenting the structural-vs-graph-level
  boundary (Req 7.6).

Out-of-enum rejection is covered for the ontology models in
``test_schema_validation.py`` (Req 26.1); here we anchor the structural schema
boundary the validator enforces at write time.

Requirements: 7.2, 7.6, 26.1.
"""

from __future__ import annotations

import pytest

from ocm.memory.contracts import CandidateAssertion
from ocm.memory.graph_store import GraphStore
from ocm.ontology.models import Person, Project, Task
from ocm.validation.schema_validator import (
    CHECK_ENTITY_REFERENCES,
    CHECK_REGISTERED_PREDICATE,
    SchemaValidator,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def validator() -> SchemaValidator:
    return SchemaValidator()


def _graph_with_person_and_project() -> GraphStore:
    """A graph holding a ``Person`` (person-1) and a ``Project`` (project-1)."""
    graph = GraphStore()
    graph.add_entity("Person", Person(id="person-1", name="Alice"))
    graph.add_entity("Project", Project(id="project-1", name="Orion"))
    return graph


def _owns_candidate(**overrides) -> CandidateAssertion:
    base = dict(
        subject_id="person-1",
        predicate="OWNS",
        object_id="project-1",
        confidence=0.9,
        source_ref="src-1",
    )
    base.update(overrides)
    return CandidateAssertion(**base)


# ---------------------------------------------------------------------------
# Accept a valid candidate (Req 7.2, 7.5, 7.6)
# ---------------------------------------------------------------------------
def test_valid_candidate_passes(validator: SchemaValidator):
    graph = _graph_with_person_and_project()
    candidate = _owns_candidate()

    result = validator.validate(candidate, graph)

    assert result.valid is True
    assert result.failed_check is None
    assert result.reason is None


# ---------------------------------------------------------------------------
# Reject an unregistered predicate (Req 7.2)
# ---------------------------------------------------------------------------
def test_unregistered_predicate_is_rejected(validator: SchemaValidator):
    graph = _graph_with_person_and_project()
    candidate = _owns_candidate(predicate="FROBNICATES")

    result = validator.validate(candidate, graph)

    assert result.valid is False
    assert result.failed_check == CHECK_REGISTERED_PREDICATE
    assert result.recommended_action == "reject"
    assert "FROBNICATES" in result.reason


# ---------------------------------------------------------------------------
# Reject when subject/object entities are absent from the graph (Req 7.5)
# ---------------------------------------------------------------------------
def test_missing_entity_references_are_rejected(validator: SchemaValidator):
    graph = GraphStore()  # empty graph: neither endpoint exists
    candidate = _owns_candidate()

    result = validator.validate(candidate, graph)

    assert result.valid is False
    assert result.failed_check == CHECK_ENTITY_REFERENCES
    assert result.recommended_action == "reject"
    assert "person-1" in result.reason
    assert "project-1" in result.reason


def test_missing_object_reference_is_rejected(validator: SchemaValidator):
    graph = GraphStore()
    graph.add_entity("Person", Person(id="person-1", name="Alice"))
    # object project-1 is absent
    candidate = _owns_candidate()

    result = validator.validate(candidate, graph)

    assert result.valid is False
    assert result.failed_check == CHECK_ENTITY_REFERENCES
    assert "project-1" in result.reason
    assert "person-1" not in result.reason


# ---------------------------------------------------------------------------
# W5 is structural only: domain/range type mismatch is NOT rejected here.
# The resolved-type domain/range check is deferred to constraint C9 at W6.
# (Req 7.6)
# ---------------------------------------------------------------------------
def test_domain_range_type_mismatch_passes_structural_validation(
    validator: SchemaValidator,
):
    # OWNS declares source types {Person, Organization}, but here the subject
    # entity is actually a Task. W5 must NOT reject this — the resolved-type
    # domain/range check belongs to C9 at W6, not the structural schema stage.
    graph = GraphStore()
    graph.add_entity("Task", Task(id="person-1", title="Mislabeled subject"))
    graph.add_entity("Project", Project(id="project-1", name="Orion"))
    candidate = _owns_candidate()

    result = validator.validate(candidate, graph)

    assert result.valid is True, (
        "W5 performs structural validation only and must not reject a "
        "domain/range type mismatch (that is C9's responsibility at W6)."
    )
    assert result.failed_check is None

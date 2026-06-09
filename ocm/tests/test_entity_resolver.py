"""Unit tests for the Entity_Resolver (W3, task 10.6).

These tests exercise every branch of the conservative resolution priority order
(Req 5.8) implemented by ``ocm.resolution.entity_resolver.EntityResolver``,
against the real :class:`GraphStore` and ontology models (no mocks, Req 26.4):

* **Exact ID match → resolved_existing (Req 5.1).** A mention carrying a known
  id resolves to that entity directly.
* **Exact normalized name + type match → resolved_existing (Req 5.2).** Name
  matching is case-/punctuation-insensitive via ``normalize_name``.
* **Alias + type match → resolved_existing (Req 5.3).** A mention by an existing
  entity's alias resolves to it.
* **No match → created_new (Req 5.5).** With a deterministic IdGenerator a new
  id is minted; without one the outcome is ``unresolved``.
* **Uncertain match → possible_match (Req 5.6).** Token-overlap ("Bob" vs
  "Bob Smith") and ambiguous exact-name duplicates both surface candidates;
  ``build_possibly_same_as`` turns the outcome into POSSIBLY_SAME_AS dicts.
* **Priority ordering (Req 5.8).** Exact id beats name: a mention whose id
  points at one entity but whose name matches another resolves by id.

Requirements: 5.1, 5.2, 5.3, 5.5, 5.6, 5.8.
"""

from __future__ import annotations

import pytest

from ocm.core.ids import IdGenerator
from ocm.memory.graph_store import GraphStore
from ocm.ontology.enums import ResolutionStatus
from ocm.ontology.models import Person, Task
from ocm.resolution.entity_resolver import (
    POSSIBLY_SAME_AS,
    POSSIBLY_SAME_AS_CONFIDENCE,
    EntityResolver,
)


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def ids() -> IdGenerator:
    """A deterministic IdGenerator so minted ids are reproducible (Req 27.5)."""
    return IdGenerator(deterministic=True, seed=0)


@pytest.fixture
def resolver() -> EntityResolver:
    return EntityResolver()


def _person(person_id: str, name: str, *, aliases: list[str] | None = None) -> Person:
    return Person(id=person_id, name=name, aliases=aliases or [])


# ---------------------------------------------------------------------------
# 1. Exact ID match -> resolved_existing (Req 5.1)
# ---------------------------------------------------------------------------
def test_exact_id_match_resolves_existing(resolver: EntityResolver, ids: IdGenerator):
    graph = GraphStore()
    graph.add_entity("Person", _person("per_1", "Alice"))

    outcome = resolver.resolve(
        {"type": "Person", "name": "Alice", "id": "per_1"}, graph, ids
    )

    assert outcome.resolution_status == ResolutionStatus.resolved_existing
    assert outcome.entity_id == "per_1"
    assert outcome.candidate_matches == []


def test_unknown_id_falls_through_to_name_match(
    resolver: EntityResolver, ids: IdGenerator
):
    # An id that is not in the graph must not short-circuit; resolution falls
    # through to the next priority steps.
    graph = GraphStore()
    graph.add_entity("Person", _person("per_1", "Alice"))

    outcome = resolver.resolve(
        {"type": "Person", "name": "Alice", "id": "per_does_not_exist"}, graph, ids
    )

    # Name+type still matches the seeded entity.
    assert outcome.resolution_status == ResolutionStatus.resolved_existing
    assert outcome.entity_id == "per_1"


# ---------------------------------------------------------------------------
# 2. Exact normalized name + type match -> resolved_existing (Req 5.2)
# ---------------------------------------------------------------------------
def test_exact_name_and_type_match_resolves_existing(
    resolver: EntityResolver, ids: IdGenerator
):
    graph = GraphStore()
    graph.add_entity("Person", _person("per_1", "Alice Smith"))

    # Case- and punctuation-insensitive normalized match.
    outcome = resolver.resolve(
        {"type": "Person", "name": "  alice   smith! "}, graph, ids
    )

    assert outcome.resolution_status == ResolutionStatus.resolved_existing
    assert outcome.entity_id == "per_1"
    assert outcome.candidate_matches == []


def test_name_match_requires_matching_type(
    resolver: EntityResolver, ids: IdGenerator
):
    # A same-name entity of a *different* type is not a match; with no other
    # candidate a new entity is minted (Req 5.5).
    graph = GraphStore()
    graph.add_entity("Task", Task(id="tas_1", title="Alpha"))

    outcome = resolver.resolve({"type": "Person", "name": "Alpha"}, graph, ids)

    assert outcome.resolution_status == ResolutionStatus.created_new
    assert outcome.entity_id is not None
    assert outcome.entity_id != "tas_1"


# ---------------------------------------------------------------------------
# 3. Alias + type match -> resolved_existing (Req 5.3)
# ---------------------------------------------------------------------------
def test_alias_and_type_match_resolves_existing(
    resolver: EntityResolver, ids: IdGenerator
):
    graph = GraphStore()
    graph.add_entity("Person", _person("per_1", "Robert Smith", aliases=["Bobby"]))

    # Mention uses the alias as its name; alias step (Req 5.3) resolves it.
    outcome = resolver.resolve({"type": "Person", "name": "Bobby"}, graph, ids)

    assert outcome.resolution_status == ResolutionStatus.resolved_existing
    assert outcome.entity_id == "per_1"
    assert outcome.candidate_matches == []


def test_alias_on_mention_matches_existing_name(
    resolver: EntityResolver, ids: IdGenerator
):
    # The mention's aliases (not its name) intersect the existing entity's name.
    graph = GraphStore()
    graph.add_entity("Person", _person("per_1", "Bobby"))

    outcome = resolver.resolve(
        {"type": "Person", "name": "Robert Smith", "aliases": ["Bobby"]},
        graph,
        ids,
    )

    assert outcome.resolution_status == ResolutionStatus.resolved_existing
    assert outcome.entity_id == "per_1"


# ---------------------------------------------------------------------------
# 5. No match -> created_new (Req 5.5)
# ---------------------------------------------------------------------------
def test_no_match_creates_new_with_minted_id(
    resolver: EntityResolver, ids: IdGenerator
):
    graph = GraphStore()
    graph.add_entity("Person", _person("per_1", "Alice"))

    outcome = resolver.resolve({"type": "Person", "name": "Charlie"}, graph, ids)

    assert outcome.resolution_status == ResolutionStatus.created_new
    assert outcome.entity_id is not None
    assert outcome.entity_id.startswith("per_")
    assert outcome.entity_id != "per_1"
    assert outcome.candidate_matches == []


def test_no_match_without_id_generator_is_unresolved(resolver: EntityResolver):
    # Without an IdGenerator a no-match cannot mint an id -> unresolved (Req 5.7).
    graph = GraphStore()

    outcome = resolver.resolve({"type": "Person", "name": "Charlie"}, graph, ids=None)

    assert outcome.resolution_status == ResolutionStatus.unresolved
    assert outcome.entity_id is None
    assert outcome.candidate_matches == []


def test_created_new_ids_are_deterministic(resolver: EntityResolver):
    # Two fresh deterministic generators over identical input reproduce the id.
    def run() -> str:
        graph = GraphStore()
        outcome = resolver.resolve(
            {"type": "Person", "name": "Charlie"},
            graph,
            IdGenerator(deterministic=True, seed=0),
            source_ref="src-1",
        )
        return outcome.entity_id

    assert run() == run()


# ---------------------------------------------------------------------------
# 6. Uncertain match -> possible_match (Req 5.6)
# ---------------------------------------------------------------------------
def test_token_overlap_is_possible_match(resolver: EntityResolver, ids: IdGenerator):
    # "Bob" is a token-subset of existing "Bob Smith": uncertain, never merged.
    graph = GraphStore()
    graph.add_entity("Person", _person("per_1", "Bob Smith"))

    outcome = resolver.resolve({"type": "Person", "name": "Bob"}, graph, ids)

    assert outcome.resolution_status == ResolutionStatus.possible_match
    # A new entity is minted (no silent merge) and the candidate is surfaced.
    assert outcome.entity_id is not None
    assert outcome.entity_id != "per_1"
    assert outcome.candidate_matches == ["per_1"]


def test_ambiguous_exact_name_duplicates_are_possible_match(
    resolver: EntityResolver, ids: IdGenerator
):
    # Two existing entities share the exact normalized name -> genuinely
    # ambiguous, so both are surfaced as candidates rather than auto-picking one.
    graph = GraphStore()
    graph.add_entity("Person", _person("per_1", "John Doe"))
    graph.add_entity("Person", _person("per_2", "john doe"))

    outcome = resolver.resolve({"type": "Person", "name": "John Doe"}, graph, ids)

    assert outcome.resolution_status == ResolutionStatus.possible_match
    assert outcome.candidate_matches == ["per_1", "per_2"]
    assert outcome.entity_id not in {"per_1", "per_2"}


def test_uncertain_without_id_generator_is_unresolved_but_surfaces_candidates(
    resolver: EntityResolver,
):
    # No IdGenerator: cannot mint a new entity, but candidates are still reported.
    graph = GraphStore()
    graph.add_entity("Person", _person("per_1", "Bob Smith"))

    outcome = resolver.resolve({"type": "Person", "name": "Bob"}, graph, ids=None)

    assert outcome.resolution_status == ResolutionStatus.unresolved
    assert outcome.entity_id is None
    assert outcome.candidate_matches == ["per_1"]


def test_build_possibly_same_as_emits_relation_per_candidate(
    resolver: EntityResolver, ids: IdGenerator
):
    graph = GraphStore()
    graph.add_entity("Person", _person("per_1", "John Doe"))
    graph.add_entity("Person", _person("per_2", "john doe"))

    outcome = resolver.resolve(
        {"type": "Person", "name": "John Doe"}, graph, ids, source_ref="src-99"
    )
    relations = EntityResolver.build_possibly_same_as(outcome, source_ref="src-99")

    assert len(relations) == 2
    for rel, candidate in zip(relations, ["per_1", "per_2"]):
        assert rel["subject"] == outcome.entity_id
        assert rel["predicate"] == POSSIBLY_SAME_AS
        assert rel["object"] == candidate
        assert rel["confidence"] == POSSIBLY_SAME_AS_CONFIDENCE
        assert rel["source_ref"] == "src-99"


def test_build_possibly_same_as_empty_for_non_possible_match(
    resolver: EntityResolver, ids: IdGenerator
):
    # A resolved_existing outcome yields no POSSIBLY_SAME_AS links.
    graph = GraphStore()
    graph.add_entity("Person", _person("per_1", "Alice"))
    outcome = resolver.resolve({"type": "Person", "name": "Alice"}, graph, ids)

    assert EntityResolver.build_possibly_same_as(outcome) == []


# ---------------------------------------------------------------------------
# 8. Priority ordering: exact id beats name (Req 5.8)
# ---------------------------------------------------------------------------
def test_exact_id_beats_name_match(resolver: EntityResolver, ids: IdGenerator):
    # Seed two entities: per_1 named "Alice", per_2 named "Bob". The mention
    # carries id=per_1 but name "Bob" (which exactly matches per_2). The exact
    # id step must win, resolving to per_1 (Req 5.8 priority order).
    graph = GraphStore()
    graph.add_entity("Person", _person("per_1", "Alice"))
    graph.add_entity("Person", _person("per_2", "Bob"))

    outcome = resolver.resolve(
        {"type": "Person", "name": "Bob", "id": "per_1"}, graph, ids
    )

    assert outcome.resolution_status == ResolutionStatus.resolved_existing
    assert outcome.entity_id == "per_1"


def test_exact_name_beats_alias_match(resolver: EntityResolver, ids: IdGenerator):
    # per_1 has "Bobby" as its exact name; per_2 has "Bobby" only as an alias.
    # The exact-name step resolves to per_1 before the alias step is consulted.
    graph = GraphStore()
    graph.add_entity("Person", _person("per_1", "Bobby"))
    graph.add_entity("Person", _person("per_2", "Robert", aliases=["Bobby"]))

    outcome = resolver.resolve({"type": "Person", "name": "Bobby"}, graph, ids)

    assert outcome.resolution_status == ResolutionStatus.resolved_existing
    assert outcome.entity_id == "per_1"


# ---------------------------------------------------------------------------
# Defensive: a typeless mention is unresolved (Req 5.7)
# ---------------------------------------------------------------------------
def test_missing_type_is_unresolved(resolver: EntityResolver, ids: IdGenerator):
    graph = GraphStore()
    outcome = resolver.resolve({"name": "Alice"}, graph, ids)

    assert outcome.resolution_status == ResolutionStatus.unresolved
    assert outcome.entity_id is None
    assert outcome.candidate_matches == []

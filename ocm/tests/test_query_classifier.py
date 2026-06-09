"""Unit tests for the Query Classifier (R0) (Req 14.1, 14.2)."""

from __future__ import annotations

import pytest

from ocm.retrieval.query_classifier import QueryClassification, QueryClassifier


@pytest.fixture()
def classifier() -> QueryClassifier:
    return QueryClassifier()


# -- Req 14.1: classification into one of the six query types ---------------


@pytest.mark.parametrize(
    "query, expected_type",
    [
        ("Who owns Project Orion?", "direct_fact"),
        ("Who is assigned to Task T1?", "direct_fact"),
        ("What is the current status of Task T1?", "direct_fact"),
        ("What happened before the launch Event?", "temporal"),
        ("When did the kickoff happen?", "temporal"),
        ("What does Event E2 precede?", "temporal"),
        ("What are the next steps for Project Orion?", "planning"),
        ("Show me the todo items for Task T1.", "planning"),
        ("Is there a conflict about T1?", "contradiction_check"),
        ("Is it true that Alice owns Orion?", "contradiction_check"),
        ("Where did this come from?", "provenance_request"),
        ("What is the evidence for this claim?", "provenance_request"),
        ("How do we know the status of T1?", "provenance_request"),
        ("Tell me about the project.", "open_ended"),
        ("Summarize everything you know.", "open_ended"),
    ],
)
def test_classify_query_type(classifier, query, expected_type):
    result = classifier.classify(query)
    assert isinstance(result, QueryClassification)
    assert result.query_type == expected_type


# -- Req 14.2: result shape -------------------------------------------------


def test_result_contains_all_fields(classifier):
    result = classifier.classify("Who owns Project Orion?")
    assert isinstance(result.query_type, str)
    assert isinstance(result.entities, list)
    assert isinstance(result.predicates, list)
    assert isinstance(result.needs_semantic_fallback, bool)


# -- entity extraction ------------------------------------------------------


def test_extracts_capitalized_entity(classifier):
    result = classifier.classify("Who owns Project Orion?")
    assert any("Orion" in e for e in result.entities)


def test_extracts_identifier_entity(classifier):
    result = classifier.classify("Who is assigned to Task T1?")
    assert "T1" in result.entities


def test_extracts_quoted_entity(classifier):
    result = classifier.classify('Who owns "Project Orion"?')
    assert "Project Orion" in result.entities


def test_drops_leading_interrogative_from_entities(classifier):
    result = classifier.classify("Who owns Project Orion?")
    assert "Who" not in result.entities
    assert "Who owns Project Orion" not in result.entities


# -- predicate extraction ---------------------------------------------------


def test_owns_predicate(classifier):
    result = classifier.classify("Who owns Project Orion?")
    assert "OWNS" in result.predicates


def test_assigned_to_predicate(classifier):
    result = classifier.classify("Who is assigned to Task T1?")
    assert "ASSIGNED_TO" in result.predicates


def test_precedes_predicate(classifier):
    result = classifier.classify("What happened before Event E2?")
    assert "PRECEDES" in result.predicates


def test_contradicts_predicate(classifier):
    result = classifier.classify("Is there a conflict about T1?")
    assert "CONTRADICTS" in result.predicates


def test_evidence_for_predicate(classifier):
    result = classifier.classify("What is the evidence for this claim?")
    assert "EVIDENCE_FOR" in result.predicates


# -- needs_semantic_fallback ------------------------------------------------


def test_structural_lookup_skips_semantic_fallback(classifier):
    # direct_fact with entity + predicate => symbolic suffices.
    result = classifier.classify("Who owns Project Orion?")
    assert result.query_type == "direct_fact"
    assert result.entities and result.predicates
    assert result.needs_semantic_fallback is False


def test_open_ended_needs_semantic_fallback(classifier):
    result = classifier.classify("Tell me about the project.")
    assert result.query_type == "open_ended"
    assert result.needs_semantic_fallback is True


def test_contradiction_check_needs_semantic_fallback(classifier):
    result = classifier.classify("Is there a conflict about T1?")
    assert result.query_type == "contradiction_check"
    assert result.needs_semantic_fallback is True


def test_direct_fact_without_predicate_needs_fallback(classifier):
    # Entity but no recognizable predicate => cannot be a pure symbolic lookup.
    result = classifier.classify("Who is Alice?")
    assert result.needs_semantic_fallback is True


# -- edge cases -------------------------------------------------------------


def test_empty_query_is_open_ended(classifier):
    result = classifier.classify("")
    assert result.query_type == "open_ended"
    assert result.entities == []
    assert result.predicates == []
    assert result.needs_semantic_fallback is True

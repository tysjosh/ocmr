"""Unit tests for the Semantic Retriever (R2, Req 16.1–16.5).

Exercised hermetically with the :class:`DeterministicEmbeddingProvider` and an
in-memory :class:`VectorIndex` (Chroma fallback), so no model download or
network access is required.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ocm.retrieval.embeddings import DeterministicEmbeddingProvider
from ocm.retrieval.semantic_retriever import (
    SemanticHit,
    SemanticRetriever,
    status_filter,
)
from ocm.retrieval.vector_index import (
    MEMORY_TYPE_ASSERTION,
    MEMORY_TYPE_CLAIM,
    STATUS_ACCEPTED,
    STATUS_QUARANTINED,
    VectorIndex,
)


def _classification(query_type: str) -> SimpleNamespace:
    """A minimal QueryClassification stand-in carrying just ``query_type``."""
    return SimpleNamespace(
        query_type=query_type, entities=[], predicates=[], needs_semantic_fallback=True
    )


@pytest.fixture
def populated_index() -> VectorIndex:
    """An in-memory index with two accepted claims, an accepted assertion, and
    one quarantined claim — all about the same topic so they co-rank."""
    index = VectorIndex(DeterministicEmbeddingProvider(), chroma_mode="memory")
    index.add("clm_owner_accepted", "Alice owns Project Orion", MEMORY_TYPE_CLAIM, STATUS_ACCEPTED)
    index.add("asr_owner_accepted", "Alice OWNS Project Orion", MEMORY_TYPE_ASSERTION, STATUS_ACCEPTED)
    index.add("doc_unrelated", "Quarterly budget figures", MEMORY_TYPE_CLAIM, STATUS_ACCEPTED)
    index.add("clm_owner_quarantined", "Bob owns Project Orion", MEMORY_TYPE_CLAIM, STATUS_QUARANTINED)
    return index


def test_retrieve_returns_semantic_hits(populated_index: VectorIndex) -> None:
    """Req 16.1: embeds the query and returns ranked SemanticHit results."""
    retriever = SemanticRetriever(populated_index)
    hits = retriever.retrieve("who owns Project Orion", _classification("direct_fact"), top_k=10)

    assert hits, "expected at least one semantic hit"
    assert all(isinstance(hit, SemanticHit) for hit in hits)
    # SemanticHit shape: memory_id, memory_type, status, similarity, text.
    first = hits[0]
    assert isinstance(first.memory_id, str)
    assert first.memory_type in {"claim", "assertion", "document", "event"}
    assert 0.0 <= first.similarity <= 1.0
    # Ranked nearest-first (descending similarity).
    sims = [hit.similarity for hit in hits]
    assert sims == sorted(sims, reverse=True)


def test_normal_query_excludes_quarantined(populated_index: VectorIndex) -> None:
    """Req 16.2 + 16.5: a non-conflict query returns only accepted items."""
    retriever = SemanticRetriever(populated_index)
    hits = retriever.retrieve("who owns Project Orion", _classification("direct_fact"), top_k=10)

    statuses = {hit.status for hit in hits}
    assert statuses == {STATUS_ACCEPTED}
    ids = {hit.memory_id for hit in hits}
    assert "clm_owner_quarantined" not in ids
    assert "clm_owner_accepted" in ids


def test_conflict_query_includes_quarantined(populated_index: VectorIndex) -> None:
    """Req 16.3 + 16.4: a contradiction_check query also surfaces quarantined items."""
    retriever = SemanticRetriever(populated_index)
    hits = retriever.retrieve(
        "is there a conflict about who owns Project Orion",
        _classification("contradiction_check"),
        top_k=10,
    )

    ids = {hit.memory_id for hit in hits}
    assert "clm_owner_quarantined" in ids, "conflict query must include the quarantined item"
    # Accepted items remain present alongside the quarantined one.
    assert "clm_owner_accepted" in ids
    assert {hit.status for hit in hits} >= {STATUS_ACCEPTED, STATUS_QUARANTINED}


def test_include_conflicts_flag_overrides_query_type(populated_index: VectorIndex) -> None:
    """Req 16.3: include_conflicts=True surfaces quarantined items even for a
    non-conflict query type."""
    retriever = SemanticRetriever(populated_index)
    hits = retriever.retrieve(
        "who owns Project Orion",
        _classification("direct_fact"),
        top_k=10,
        include_conflicts=True,
    )
    assert "clm_owner_quarantined" in {hit.memory_id for hit in hits}


def test_status_filter_shapes() -> None:
    """The where-filter widens for conflict queries and is accepted-only otherwise."""
    assert status_filter(_classification("direct_fact")) == {"status": "accepted"}
    assert status_filter(_classification("contradiction_check")) == {
        "status": {"$in": ["accepted", "quarantined"]}
    }
    assert status_filter(_classification("direct_fact"), include_conflicts=True) == {
        "status": {"$in": ["accepted", "quarantined"]}
    }


def test_top_k_zero_returns_empty(populated_index: VectorIndex) -> None:
    """A non-positive top_k yields no hits."""
    retriever = SemanticRetriever(populated_index)
    assert retriever.retrieve("anything", _classification("open_ended"), top_k=0) == []

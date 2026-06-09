"""Consolidated retrieval unit tests for Req 26.5.

Requirement 26.5 calls for focused unit tests covering the core retrieval
behaviours end-of-stage. This module gathers those required cases into one
hermetic, offline suite (deterministic embeddings + in-memory vector index +
in-memory SQLite repository), complementing — not duplicating — the
component-level suites (``test_symbolic_retriever.py``,
``test_semantic_retriever.py``, ``test_retrieval_pipeline.py``):

* Symbolic retrieval returns the correct owner (Req 15.1).
* Semantic retrieval returns a relevant claim (Req 16.1).
* A conflict query retrieves a quarantined contradiction (Req 16.3).
* The reranker penalizes a contradicted assertion (Req 17.3).
* An evidence package includes provenance sources (Req 18.3).

Requirements: 15.1, 16.1, 16.3, 17.3, 18.3, 26.5.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ocm.core.ids import IdGenerator
from ocm.memory.graph_store import GraphStore
from ocm.memory.provenance_tracker import ProvenanceTracker
from ocm.memory.sqlite_repository import SQLiteRepository
from ocm.ontology.enums import AssertionStatus
from ocm.ontology.models import Assertion
from ocm.retrieval.embeddings import DeterministicEmbeddingProvider
from ocm.retrieval.evidence_packager import EvidencePackager
from ocm.retrieval.query_classifier import QueryClassifier
from ocm.retrieval.reranker import RankedItem, Reranker
from ocm.retrieval.semantic_retriever import SemanticHit, SemanticRetriever
from ocm.retrieval.symbolic_retriever import OWNS, SymbolicRetriever
from ocm.retrieval.vector_index import (
    MEMORY_TYPE_CLAIM,
    STATUS_ACCEPTED,
    STATUS_QUARANTINED,
    VectorIndex,
)

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _accepted(aid: str, subj: str, pred: str, obj: str, conf: float = 0.9) -> Assertion:
    return Assertion(
        id=aid,
        subject_id=subj,
        predicate=pred,
        object_id=obj,
        confidence=conf,
        status=AssertionStatus.accepted,
        source_ref=f"doc::{aid}",
        created_at=_NOW,
        extractor_version="test-1",
    )


def _seed_owner_graph() -> GraphStore:
    """Alice OWNS Project Orion."""
    g = GraphStore()
    g.add_entity("Person", {"id": "p_alice", "name": "Alice"})
    g.add_entity("Project", {"id": "proj_orion", "name": "Orion"})
    g.add_assertion(_accepted("a_owns", "p_alice", OWNS, "proj_orion"))
    return g


# --------------------------------------------------------------------------
# Req 15.1 — symbolic retrieval returns the correct owner
# --------------------------------------------------------------------------
def test_symbolic_retrieval_returns_correct_owner() -> None:
    """Seed Alice OWNS Orion; classify "who owns Orion?" -> owner Alice (Req 15.1)."""
    graph = _seed_owner_graph()
    classification = QueryClassifier().classify("Who owns Orion?")

    # The classifier recognized this as a structural ownership lookup.
    assert classification.query_type == "direct_fact"
    assert OWNS in classification.predicates
    assert "Orion" in classification.entities

    hits = SymbolicRetriever().retrieve(classification, graph)

    assert len(hits) == 1
    hit = hits[0]
    assert hit.predicate == OWNS
    assert hit.subject_id == "p_alice"  # the owner
    assert hit.object_id == "proj_orion"
    assert hit.assertion_id == "a_owns"
    assert hit.exact_match is True


# --------------------------------------------------------------------------
# Req 16.1 — semantic retrieval returns a relevant claim
# --------------------------------------------------------------------------
def test_semantic_retrieval_returns_relevant_claim() -> None:
    """An in-memory index returns a topically relevant claim for a query (Req 16.1)."""
    index = VectorIndex(DeterministicEmbeddingProvider(), chroma_mode="memory")
    index.add("clm_owner", "Alice owns Project Orion", MEMORY_TYPE_CLAIM, STATUS_ACCEPTED)
    index.add("clm_budget", "Quarterly budget figures", MEMORY_TYPE_CLAIM, STATUS_ACCEPTED)

    classification = QueryClassifier().classify("who owns Project Orion")
    hits = SemanticRetriever(index).retrieve("who owns Project Orion", classification, top_k=5)

    assert hits, "expected at least one semantic hit"
    assert all(isinstance(h, SemanticHit) for h in hits)
    ids = {h.memory_id for h in hits}
    assert "clm_owner" in ids
    # The owner claim is the most relevant (ranked first).
    assert hits[0].memory_id == "clm_owner"
    assert hits[0].memory_type == "claim"
    assert 0.0 <= hits[0].similarity <= 1.0


# --------------------------------------------------------------------------
# Req 16.3 — a conflict query retrieves a quarantined contradiction
# --------------------------------------------------------------------------
def test_conflict_query_retrieves_quarantined_contradiction() -> None:
    """A contradiction_check query surfaces the quarantined contradiction (Req 16.3)."""
    index = VectorIndex(DeterministicEmbeddingProvider(), chroma_mode="memory")
    index.add("clm_accepted", "Alice owns Project Orion", MEMORY_TYPE_CLAIM, STATUS_ACCEPTED)
    index.add("clm_quarantined", "Bob owns Project Orion", MEMORY_TYPE_CLAIM, STATUS_QUARANTINED)

    retriever = SemanticRetriever(index)

    # A normal query keeps the quarantined contradiction hidden (Req 16.2/16.5).
    normal = retriever.retrieve(
        "who owns Project Orion",
        QueryClassifier().classify("who owns Project Orion"),
        top_k=10,
    )
    assert "clm_quarantined" not in {h.memory_id for h in normal}

    # A conflict query surfaces the quarantined contradiction alongside accepted.
    conflict_classification = QueryClassifier().classify(
        "is there a conflict about who owns Project Orion"
    )
    assert conflict_classification.query_type == "contradiction_check"

    conflict = retriever.retrieve(
        "is there a conflict about who owns Project Orion",
        conflict_classification,
        top_k=10,
    )
    ids = {h.memory_id for h in conflict}
    assert "clm_quarantined" in ids
    assert "clm_accepted" in ids
    assert {h.status for h in conflict} >= {STATUS_ACCEPTED, STATUS_QUARANTINED}


# --------------------------------------------------------------------------
# Req 17.3 — the reranker penalizes a contradicted assertion
# --------------------------------------------------------------------------
def test_reranker_penalizes_contradicted_assertion() -> None:
    """Two identical hits: the contradicted one scores strictly lower (Req 17.3)."""
    clean = SemanticHit(
        memory_id="asr_clean",
        memory_type="assertion",
        status=STATUS_ACCEPTED,
        similarity=0.8,
        text="Alice OWNS Project Orion",
    )
    contradicted = SemanticHit(
        memory_id="asr_contradicted",
        memory_type="assertion",
        status=STATUS_ACCEPTED,
        similarity=0.8,
        text="Alice OWNS Project Orion",
    )

    reranker = Reranker()
    ranked = reranker.rerank(
        semantic=[clean, contradicted],
        contradicted_ids=["asr_contradicted"],
        now=_NOW,
    )

    by_id = {item.memory_id: item for item in ranked}
    assert by_id["asr_contradicted"].contradicted is True
    assert by_id["asr_clean"].contradicted is False
    # The only differing signal is the contradiction penalty -> strictly lower.
    assert by_id["asr_contradicted"].score < by_id["asr_clean"].score
    # And the clean item ranks ahead of the contradicted one.
    assert ranked[0].memory_id == "asr_clean"


# --------------------------------------------------------------------------
# Req 18.3 — an evidence package includes provenance sources
# --------------------------------------------------------------------------
def test_evidence_package_includes_sources() -> None:
    """A packaged result attaches provenance for its supporting assertions (Req 18.3)."""
    repo = SQLiteRepository(":memory:")
    ids = IdGenerator(deterministic=True)
    provenance = ProvenanceTracker(repo, ids)
    provenance.record(
        subject_id="a_owns",
        source_ref="doc::a_owns",
        created_at=_NOW,
        extractor_version="test-1",
        supporting_evidence_ids=["ev1"],
    )

    supporting = RankedItem(
        memory_id="a_owns",
        memory_type="assertion",
        status=STATUS_ACCEPTED,
        score=0.9,
        confidence=0.9,
        predicate=OWNS,
        subject_id="p_alice",
        object_id="proj_orion",
        exact_match=True,
    )

    classification = QueryClassifier().classify("Who owns Orion?")
    pkg = EvidencePackager().package(
        query="Who owns Orion?",
        classification=classification,
        ranked=[supporting],
        provenance_tracker=provenance,
    )

    assert {sa.id for sa in pkg.supporting_assertions} == {"a_owns"}
    assert pkg.supporting_sources, "expected supporting_sources to be populated"
    assert any(p.subject_id == "a_owns" for p in pkg.supporting_sources)
    assert any(p.source_ref == "doc::a_owns" for p in pkg.supporting_sources)

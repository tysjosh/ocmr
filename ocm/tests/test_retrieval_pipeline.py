"""End-to-end tests for R4 — :mod:`ocm.retrieval.evidence_packager` — and the
:class:`~ocm.retrieval.retrieval_pipeline.RetrievalPipeline` (R0→R4).

Wires the real stages (Query Classifier, Symbolic Retriever, Semantic Retriever,
Reranker, Evidence Packager) over a seeded ``GraphStore`` + an in-memory
``VectorIndex`` backed by :class:`DeterministicEmbeddingProvider`, with
provenance and quarantine stored in an in-memory SQLite repository. Confirms:

* an owner query surfaces the owner (``answer``), a supporting assertion with
  its confidence, and provenance sources (Req 18.1, 18.2, 18.3);
* a contradiction_check query surfaces conflicts (Req 18.4);
* every query records a per-query research log (Req 25.2).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ocm.core.config import Settings
from ocm.core.ids import IdGenerator
from ocm.core.logging import ResearchLogger
from ocm.memory.graph_store import GraphStore
from ocm.memory.provenance_tracker import ProvenanceTracker
from ocm.memory.quarantine_store import QuarantineStore
from ocm.ontology.enums import AssertionStatus, Severity
from ocm.ontology.models import Assertion
from ocm.retrieval.embeddings import DeterministicEmbeddingProvider
from ocm.retrieval.evidence_packager import EvidencePackage, EvidencePackager
from ocm.retrieval.query_classifier import QueryClassifier
from ocm.retrieval.reranker import Reranker
from ocm.retrieval.retrieval_pipeline import RetrievalPipeline
from ocm.retrieval.semantic_retriever import SemanticRetriever
from ocm.retrieval.symbolic_retriever import (
    ASSIGNED_TO,
    OWNS,
    SymbolicRetriever,
)
from ocm.retrieval.vector_index import VectorIndex

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


def _quarantined(aid: str, subj: str, pred: str, obj: str, conf: float = 0.7) -> Assertion:
    return Assertion(
        id=aid,
        subject_id=subj,
        predicate=pred,
        object_id=obj,
        confidence=conf,
        status=AssertionStatus.quarantined,
        source_ref=f"doc::{aid}",
        created_at=_NOW,
        extractor_version="test-1",
    )


def _seed_graph() -> GraphStore:
    """Alice OWNS Project Orion; Carol OWNS Project Atlas; T1 ASSIGNED_TO Bob."""
    g = GraphStore()
    g.add_entity("Person", {"id": "p_alice", "name": "Alice"})
    g.add_entity("Person", {"id": "p_bob", "name": "Bob"})
    g.add_entity("Person", {"id": "p_carol", "name": "Carol"})
    g.add_entity("Person", {"id": "p_dave", "name": "Dave"})
    g.add_entity("Project", {"id": "proj_orion", "name": "Project Orion"})
    g.add_entity("Project", {"id": "proj_atlas", "name": "Project Atlas"})
    g.add_entity("Task", {"id": "task_t1", "title": "Task T1"})

    g.add_assertion(_accepted("a_owns_orion", "p_alice", OWNS, "proj_orion"))
    g.add_assertion(_accepted("a_owns_atlas", "p_carol", OWNS, "proj_atlas"))
    g.add_assertion(_accepted("a_assigned", "task_t1", ASSIGNED_TO, "p_bob"))
    return g


@pytest.fixture
def pipeline_env():
    """Wire a full retrieval pipeline over an in-memory repo + vector index."""
    try:
        from ocm.memory.sqlite_repository import SQLiteRepository
    except Exception:  # pragma: no cover
        pytest.skip("StorageRepository not implemented yet (task 3.2)")

    repo = SQLiteRepository(":memory:")
    ids = IdGenerator(deterministic=True)
    graph = _seed_graph()

    # Vector index (in-memory) with deterministic, offline embeddings.
    provider = DeterministicEmbeddingProvider()
    vectors = VectorIndex(provider, chroma_mode="memory", graph=graph)
    for assertion in (
        _accepted("a_owns_orion", "p_alice", OWNS, "proj_orion"),
        _accepted("a_owns_atlas", "p_carol", OWNS, "proj_atlas"),
        _accepted("a_assigned", "task_t1", ASSIGNED_TO, "p_bob"),
    ):
        vectors.embed_assertion(assertion)
    # A quarantined conflicting assertion: Dave OWNS Project Atlas.
    vectors.embed_assertion(_quarantined("q_atlas", "p_dave", OWNS, "proj_atlas"))

    # Provenance for the accepted assertions (Req 18.3).
    provenance = ProvenanceTracker(repo, ids)
    provenance.record(
        subject_id="a_owns_orion",
        source_ref="doc::a_owns_orion",
        created_at=_NOW,
        extractor_version="test-1",
        supporting_evidence_ids=["ev1"],
    )
    provenance.record(
        subject_id="a_owns_atlas",
        source_ref="doc::a_owns_atlas",
        created_at=_NOW,
        extractor_version="test-1",
    )

    # An unresolved conflict over Project Atlas ownership (Req 18.4).
    quarantine = QuarantineStore(repo, ids)
    quarantine.add(
        candidate_payload={"id": "q_atlas", "subject_id": "p_dave", "predicate": OWNS,
                           "object_id": "proj_atlas"},
        reason="Conflicting OWNS for Project Atlas",
        severity=Severity.high,
        conflicting_ids=["a_owns_atlas"],
        created_at=_NOW,
    )

    logger = ResearchLogger()
    pipeline = RetrievalPipeline(
        classifier=QueryClassifier(),
        symbolic_retriever=SymbolicRetriever(),
        semantic_retriever=SemanticRetriever(vectors),
        reranker=Reranker(),
        evidence_packager=EvidencePackager(),
        graph=graph,
        provenance_tracker=provenance,
        quarantine_store=quarantine,
        research_logger=logger,
        settings=Settings(**{"deterministic_test_mode": True, "chroma_mode": "memory"}),
        ids=ids,
    )
    return pipeline, logger


def test_owner_query_surfaces_owner_with_support_and_sources(pipeline_env) -> None:
    """'Who owns Project Orion?' -> owner answer + supporting assertion + sources."""
    pipeline, logger = pipeline_env
    pkg = pipeline.query("Who owns Project Orion?", top_k=5)

    assert isinstance(pkg, EvidencePackage)
    # R4 derived the owner from the exact symbolic OWNS hit (Req 18.5).
    assert pkg.answer == "Alice"
    # Supporting assertion carries id + confidence (Req 18.2).
    supporting_ids = {sa.id for sa in pkg.supporting_assertions}
    assert "a_owns_orion" in supporting_ids
    owner_sa = next(sa for sa in pkg.supporting_assertions if sa.id == "a_owns_orion")
    assert 0.0 <= owner_sa.confidence <= 1.0
    # Confidence is derived from the top supporting assertion.
    assert pkg.confidence == pytest.approx(owner_sa.confidence)
    # Provenance for the supporting assertion is attached (Req 18.3).
    assert any(p.subject_id == "a_owns_orion" for p in pkg.supporting_sources)
    # The full ranked candidate set is carried for the caller (Req 18.1).
    assert any(item.memory_id == "a_owns_orion" for item in pkg.retrieved_items)

    # Per-query research log recorded (Req 25.2).
    query_records = [r for r in logger.records if r["kind"] == "query"]
    assert len(query_records) == 1
    rec = query_records[0]
    assert rec["query_type"] == "direct_fact"
    assert rec["symbolic_results_count"] >= 1
    assert "a_owns_orion" in rec["top_k_ids"]
    assert "latency_ms" in rec


def test_contradiction_query_surfaces_conflicts(pipeline_env) -> None:
    """A contradiction_check query surfaces unresolved conflicts (Req 18.4)."""
    pipeline, logger = pipeline_env
    pkg = pipeline.query(
        "Is there a conflict about who owns Project Atlas?", top_k=5
    )

    assert isinstance(pkg, EvidencePackage)
    assert len(pkg.conflicts) >= 1
    # The quarantined assertion and/or the contradicted accepted assertion show up.
    conflict_ids = {c.memory_id for c in pkg.conflicts}
    assert ("q_atlas" in conflict_ids) or ("a_owns_atlas" in conflict_ids)

    rec = [r for r in logger.records if r["kind"] == "query"][-1]
    assert rec["query_type"] == "contradiction_check"
    assert rec["conflicts_returned"] == len(pkg.conflicts)


def test_unknown_query_reports_missing_information(pipeline_env) -> None:
    """A query with no matching memory reports missing_information (Req 18.5)."""
    pipeline, _logger = pipeline_env
    pkg = pipeline.query("Who owns Project Nonexistent Zephyr?", top_k=5)

    # Either nothing matched or nothing was accepted-supported: missing info set.
    if not pkg.supporting_assertions:
        assert pkg.missing_information
        assert pkg.confidence == 0.0

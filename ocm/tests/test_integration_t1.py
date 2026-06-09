"""End-to-end Task T1 integration test — write path → retrieval path (R0→R4).

Wires the **full** OCM stack — the Write Pipeline (W1–W8) and the Retrieval
Pipeline (R0–R4) — over a single in-memory SQLite repository, a shared
``GraphStore``, and a single in-memory ``VectorIndex`` (deterministic, offline
embeddings). The write side embeds accepted assertions (Commit Manager
``embed_hook``) and accepted claims / documents / events (WritePipeline
``memory_embed_hook``) into the same vector index the Semantic Retriever reads,
so the retrieval pipeline answers from exactly what the write pipeline durably
committed.

Scenario — the three sequential Task T1 writes (the MockExtractor-recognized
sentences):

1. "Alice owns Project Orion. Bob is assigned to Task T1." → OWNS + ASSIGNED_TO
   accepted.
2. "Bob completed Task T1." → a completion Event + RESULTS_IN / PARTICIPATES_IN
   accepted and Task T1 reconciled to ``done``.
3. "Task T1 is not started." (high confidence) → the contradiction gate
   prevents silent acceptance: it is quarantined as a status contradiction and
   T1 stays ``done``.

Then the Retrieval Pipeline is queried — "What is the current status of Task
T1?" — and we assert the durable accepted state and that the quarantined
conflict is surfaced:

* the Task T1 graph node status is ``done`` (the current accepted status, the
  contradiction gate having blocked the "not started" overwrite, Req 28.5);
* the status query returns a structured ``EvidencePackage`` carrying **both**
  symbolic and semantic results, with supporting assertions (ids + confidence),
  provenance, and a conflicts field (Req 28.7, 28.8);
* the quarantined "not started" status contradiction is durably retrievable
  from the Quarantine_Store, and a contradiction_check query surfaces it
  (Req 28.8).

Requirements: 28.5, 28.7, 28.8.
"""

from __future__ import annotations

import pytest

from ocm.core.config import Settings
from ocm.core.ids import IdGenerator
from ocm.core.logging import ResearchLogger
from ocm.extraction.mock_extractor import MockExtractor
from ocm.memory.assertion_builder import AssertionBuilder
from ocm.memory.commit_manager import CommitManager
from ocm.memory.graph_store import GraphStore
from ocm.memory.provenance_tracker import ProvenanceTracker
from ocm.memory.quarantine_store import QuarantineStore
from ocm.memory.sqlite_repository import SQLiteRepository
from ocm.memory.write_pipeline import WritePipeline
from ocm.ontology.enums import AssertionStatus, QuarantineStatus, TaskStatus
from ocm.resolution.entity_resolver import EntityResolver
from ocm.resolution.normalizer import Normalizer
from ocm.retrieval.embeddings import DeterministicEmbeddingProvider
from ocm.retrieval.evidence_packager import EvidencePackage, EvidencePackager
from ocm.retrieval.query_classifier import QueryClassifier
from ocm.retrieval.reranker import Reranker
from ocm.retrieval.retrieval_pipeline import RetrievalPipeline
from ocm.retrieval.semantic_retriever import SemanticRetriever
from ocm.retrieval.symbolic_retriever import SymbolicRetriever
from ocm.retrieval.vector_index import VectorIndex
from ocm.validation.constraints import ConstraintValidator
from ocm.validation.schema_validator import SchemaValidator


@pytest.fixture
def stack():
    """Wire the full write + retrieval stack over shared in-memory stores."""
    settings = Settings(
        deterministic_test_mode=True, chroma_mode="memory", extractor="mock"
    )
    repo = SQLiteRepository(":memory:")
    ids = IdGenerator(deterministic=True)
    graph = GraphStore()
    provenance = ProvenanceTracker(repo, ids)
    quarantine = QuarantineStore(repo, ids)

    # One in-memory vector index, shared by the write (embed) and read (query)
    # paths. Graph wired so assertion embeddings render with entity names.
    vector_index = VectorIndex(
        DeterministicEmbeddingProvider(), chroma_mode="memory", graph=graph
    )

    commit = CommitManager(
        repo=repo,
        graph=graph,
        ids=ids,
        quarantine_store=quarantine,
        provenance_tracker=provenance,
        embed_hook=vector_index.embed_assertion,  # Req 13.5
    )
    research_logger = ResearchLogger()
    write_pipeline = WritePipeline(
        extractor=MockExtractor(),
        normalizer=Normalizer(),
        resolver=EntityResolver(),
        assertion_builder=AssertionBuilder(),
        schema_validator=SchemaValidator(),
        constraint_validator=ConstraintValidator(settings),
        commit_manager=commit,
        repo=repo,
        graph=graph,
        ids=ids,
        provenance_tracker=provenance,
        quarantine_store=quarantine,
        memory_embed_hook=vector_index.embed_memory,  # Req 16.6
        research_logger=research_logger,
        settings=settings,
    )

    retrieval_pipeline = RetrievalPipeline(
        classifier=QueryClassifier(),
        symbolic_retriever=SymbolicRetriever(),
        semantic_retriever=SemanticRetriever(vector_index),
        reranker=Reranker(),
        evidence_packager=EvidencePackager(),
        graph=graph,
        provenance_tracker=provenance,
        quarantine_store=quarantine,
        research_logger=research_logger,
        settings=settings,
        ids=ids,
    )

    return {
        "write": write_pipeline,
        "retrieval": retrieval_pipeline,
        "graph": graph,
        "quarantine": quarantine,
        "repo": repo,
    }


def _accepted_predicates(result) -> set[str]:
    return {o.candidate.predicate for o in result.accepted}


def _task_t1_id(graph: GraphStore) -> str:
    """Resolve the single Task node id (the scenario creates exactly one)."""
    task_ids = [n for n in graph.node_ids() if graph.get_entity_type(n) == "Task"]
    assert len(task_ids) == 1, f"expected exactly one Task node, got {task_ids}"
    return task_ids[0]


def test_t1_end_to_end_status_query_and_conflict_surfacing(stack):
    write = stack["write"]
    retrieval = stack["retrieval"]
    graph = stack["graph"]
    quarantine = stack["quarantine"]

    # ---- Write 1: ownership + assignment accepted -----------------------
    r1 = write.run("Alice owns Project Orion. Bob is assigned to Task T1.", "src-1")
    assert {"OWNS", "ASSIGNED_TO"} <= _accepted_predicates(r1)

    # ---- Write 2: completion → RESULTS_IN/PARTICIPATES_IN, T1 -> done ----
    r2 = write.run("Bob completed Task T1.", "src-2")
    preds2 = _accepted_predicates(r2)
    assert "RESULTS_IN" in preds2
    assert "PARTICIPATES_IN" in preds2

    t1_id = _task_t1_id(graph)
    assert graph.get_entity_payload(t1_id)["status"] == TaskStatus.done.value

    # ---- Write 3: "not started" at high confidence → quarantined --------
    r3 = write.run("Task T1 is not started.", "src-3")
    assert len(r3.quarantined) == 1
    assert r3.quarantined[0].decision == "quarantined"
    # Req 28.5 — the contradiction gate prevented a silent overwrite: T1 stays done.
    assert graph.get_entity_payload(t1_id)["status"] == TaskStatus.done.value

    # ---- The quarantined contradiction is durably retrievable (Req 28.8) -
    unresolved = quarantine.list(QuarantineStatus.unresolved)
    status_conflicts = [
        q for q in unresolved if "status contradiction" in q.reason and t1_id in q.conflicting_ids
    ]
    assert len(status_conflicts) == 1
    # The quarantined flip is a HAS_STATUS assertion whose object is the
    # StatusValue node (``status:todo``) and whose conflicting_ids point at the
    # *accepted* HAS_STATUS assertion (paired surfacing) plus the Task entity.
    assert status_conflicts[0].candidate_payload["predicate"] == "HAS_STATUS"
    assert status_conflicts[0].candidate_payload["object_id"] == f"status:{TaskStatus.todo.value}"
    accepted_status_edge = graph.out_edges(t1_id, "HAS_STATUS")
    assert len(accepted_status_edge) == 1
    accepted_status_aid = accepted_status_edge[0][3]["assertion_id"]
    assert accepted_status_aid in status_conflicts[0].conflicting_ids

    # ====================================================================
    # Retrieval — "What is the current status of Task T1?"
    # ====================================================================
    pkg = retrieval.query("What is the current status of Task T1?", top_k=10)

    assert isinstance(pkg, EvidencePackage)
    # Req 28.7 — both symbolic and semantic results feed the ranked candidate
    # set: the ASSIGNED_TO edge (symbolic, exact match) and claims/assertions
    # about T1 (semantic) all surface.
    assert pkg.retrieved_items, "status query returned no retrieved items"
    assert any(item.exact_match for item in pkg.retrieved_items), (
        "expected at least one exact (symbolic) match in the ranked set"
    )
    assert any(not item.exact_match for item in pkg.retrieved_items), (
        "expected at least one semantic (non-exact) match in the ranked set"
    )

    # Req 28.8 — the package carries supporting assertions (ids + confidence)
    # and provenance for what it reports.
    assert pkg.supporting_assertions, "status query produced no supporting assertions"
    for sa in pkg.supporting_assertions:
        assert sa.id
        assert 0.0 <= sa.confidence <= 1.0
    assert pkg.confidence == pytest.approx(pkg.supporting_assertions[0].confidence)
    assert pkg.supporting_sources, "expected provenance sources for the supporting assertions"

    # The package reflects the durable accepted T1 state: the assignment edge
    # for T1 (subject) is among the supporting evidence, and the graph node is
    # done. (No "not started" assertion was ever accepted/embedded.)
    assert any(item.subject_id == t1_id for item in pkg.retrieved_items if item.exact_match)
    assert graph.get_entity_payload(t1_id)["status"] == TaskStatus.done.value

    # ====================================================================
    # Retrieval — contradiction query surfaces the quarantined conflict
    # ====================================================================
    conflict_pkg = retrieval.query("Is there a conflict about Task T1?", top_k=10)
    assert isinstance(conflict_pkg, EvidencePackage)
    # The status contradiction is recorded as a HAS_STATUS flip whose
    # conflicting_ids reference the accepted HAS_STATUS assertion (and the Task
    # entity), so it surfaces inline in the package's conflicts for the queried
    # entity as a paired {accepted, quarantined} record.
    assert conflict_pkg.conflicts, "the conflict query must surface the status contradiction"
    paired = [c for c in conflict_pkg.conflicts if "status contradiction" in (c.reason or "").lower()]
    assert paired, "expected a status-contradiction conflict item"
    assert paired[0].accepted and "done" in paired[0].accepted.lower()
    assert paired[0].quarantined and "todo" in paired[0].quarantined.lower()
    # And it remains durably retrievable from the Quarantine_Store.
    assert any(
        "status contradiction" in q.reason and t1_id in q.conflicting_ids
        for q in quarantine.list(QuarantineStatus.unresolved)
    )


def test_t1_only_accepted_assertions_are_graph_edges(stack):
    """Sanity: every accepted assertion (and only those) is an edge (Req 10.5)."""
    write = stack["write"]
    graph = stack["graph"]
    repo = stack["repo"]

    write.run("Alice owns Project Orion. Bob is assigned to Task T1.", "src-1")
    write.run("Bob completed Task T1.", "src-2")
    write.run("Task T1 is not started.", "src-3")

    for a in repo.list_assertions():
        if a.status == AssertionStatus.accepted:
            assert graph.has_assertion(a.subject_id, a.object_id, a.predicate)

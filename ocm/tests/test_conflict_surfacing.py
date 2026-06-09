"""Regression tests for conflict surfacing, answer derivation, and relevance.

Covers the research-critical audit fixes:

* **#7** — a status-flip contradiction surfaces *inline* in the evidence package
  for the exact entity queried (not hidden behind a contradiction_check query).
* **#8** — deterministic answers are derived for status / temporal / decision
  query types (not just OWNS / ASSIGNED_TO).
* **#6** — quarantined items are surfaced as conflicts only when they are
  *relevant* to the query (id/entity overlap), not merely near in vector space.
"""

from __future__ import annotations

from types import SimpleNamespace

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.retrieval.embeddings import DeterministicEmbeddingProvider
from ocm.retrieval.evidence_packager import EvidencePackager
from ocm.retrieval.reranker import RankedItem
from ocm.retrieval.semantic_retriever import SemanticRetriever
from ocm.retrieval.vector_index import (
    MEMORY_TYPE_CLAIM,
    STATUS_ACCEPTED,
    STATUS_QUARANTINED,
    VectorIndex,
)


def _container() -> CoreContainer:
    return CoreContainer(
        Settings(deterministic_test_mode=True, chroma_mode="memory", extractor="mock")
    )


# --------------------------------------------------------------------------- #
# #7 — status-flip contradiction surfaces inline on the queried entity
# --------------------------------------------------------------------------- #
def test_status_query_surfaces_status_flip_conflict_inline():
    c = _container()
    c.write_pipeline.run("Alice owns Project Orion. Bob is assigned to Task T1.", "s1")
    c.write_pipeline.run("Bob completed Task T1.", "s2")  # T1 -> done
    c.write_pipeline.run("Task T1 is not started.", "s3")  # contradicts -> quarantined

    pkg = c.retrieval_pipeline.query("What is the current status of Task T1?", top_k=10)

    # #7: the known contradiction is surfaced inline, not silently collapsed.
    assert pkg.conflicts, "the status query must surface the status contradiction"
    assert any(
        "status contradiction" in (cf.reason or "").lower() for cf in pkg.conflicts
    )
    # #8: a deterministic status answer is produced (the accepted status).
    assert pkg.answer is not None and "done" in pkg.answer.lower()


def test_unrelated_query_does_not_surface_the_status_conflict():
    """Precision: a query about a different entity must not surface the T1 conflict."""
    c = _container()
    c.write_pipeline.run("Alice owns Project Orion. Bob is assigned to Task T1.", "s1")
    c.write_pipeline.run("Bob completed Task T1.", "s2")
    c.write_pipeline.run("Task T1 is not started.", "s3")

    pkg = c.retrieval_pipeline.query("Who owns Project Orion?", top_k=10)

    # The T1 status contradiction is unrelated to the ownership question.
    assert not any(
        "status contradiction" in (cf.reason or "").lower() for cf in pkg.conflicts
    )
    assert pkg.answer == "Alice"


# --------------------------------------------------------------------------- #
# #8 — deterministic answers for more query types
# --------------------------------------------------------------------------- #
def test_temporal_query_answers_preceding_events():
    c = _container()
    c.write_pipeline.run("Event Kickoff precedes Event Review.", "s1")

    pkg = c.retrieval_pipeline.query("What happened before Review?", top_k=10)

    assert pkg.answer is not None and "Kickoff" in pkg.answer


def test_decision_query_answers_from_decision_summary():
    c = _container()
    c.write_pipeline.run("We decided to launch Project Orion.", "s1")

    pkg = c.retrieval_pipeline.query("What was decided about Orion?", top_k=10)

    assert pkg.answer is not None and "launch" in pkg.answer.lower()


# --------------------------------------------------------------------------- #
# #6 — quarantined-item relevance (semantic retriever precision)
# --------------------------------------------------------------------------- #
def test_semantic_retriever_filters_irrelevant_quarantined_items():
    index = VectorIndex(DeterministicEmbeddingProvider(), chroma_mode="memory")
    index.add("acc", "Alice owns Project Orion", MEMORY_TYPE_CLAIM, STATUS_ACCEPTED)
    index.add("q_rel", "Bob owns Project Orion", MEMORY_TYPE_CLAIM, STATUS_QUARANTINED)
    index.add("q_unrel", "Mallory finished Task Zeta", MEMORY_TYPE_CLAIM, STATUS_QUARANTINED)

    classification = SimpleNamespace(
        query_type="contradiction_check",
        entities=["Project Orion"],
        predicates=[],
        needs_semantic_fallback=True,
    )
    hits = SemanticRetriever(index).retrieve(
        "is there a conflict about Project Orion", classification, top_k=10
    )
    ids = {h.memory_id for h in hits}

    assert "acc" in ids
    assert "q_rel" in ids  # quarantined but about the queried entity -> kept
    assert "q_unrel" not in ids  # quarantined and unrelated -> filtered (precision)


# --------------------------------------------------------------------------- #
# #6 / #7 — packager only surfaces quarantine records relevant to the query
# --------------------------------------------------------------------------- #
class _StubGraph:
    """Minimal graph exposing the surface the packager's relevance check uses."""

    def __init__(self, ids):
        self._ids = list(ids)

    def node_ids(self):
        return list(self._ids)

    def get_entity_type(self, node_id):
        return "Task"

    def get_entity_payload(self, node_id):
        return {"id": node_id, "title": node_id, "name": node_id}


class _StubQuarantine:
    def __init__(self, records):
        self._records = records

    def list(self, status=None):
        return list(self._records)


def test_packager_surfaces_only_relevant_quarantine_conflicts():
    graph = _StubGraph(["task_t1", "task_t2"])
    quarantine = _StubQuarantine(
        [
            SimpleNamespace(
                id="q1", conflicting_ids=["task_t1"], reason="status contradiction on T1",
                status="unresolved", severity="medium",
            ),
            SimpleNamespace(
                id="q2", conflicting_ids=["task_t2"], reason="status contradiction on T2",
                status="unresolved", severity="medium",
            ),
        ]
    )
    ranked = [
        RankedItem(
            memory_id="a1",
            memory_type="assertion",
            status="accepted",
            score=0.9,
            subject_id="task_t1",
            predicate="ASSIGNED_TO",
            object_id="p_bob",
            exact_match=True,
        )
    ]
    classification = SimpleNamespace(
        query_type="direct_fact",
        entities=["task_t1"],
        predicates=[],
        needs_semantic_fallback=False,
    )

    pkg = EvidencePackager().package(
        "status of task_t1",
        classification,
        ranked,
        graph=graph,
        quarantine_store=quarantine,
    )
    conflict_ids = {c.memory_id for c in pkg.conflicts}

    assert "q1" in conflict_ids  # relevant: conflicting_ids overlaps task_t1
    assert "q2" not in conflict_ids  # irrelevant: about task_t2

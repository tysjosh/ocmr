"""Property 3 — Quarantined/rejected never appear in accepted retrieval.

Feature: ontology-constrained-memory, Property 3.

For **any** write stream and **any** default (non-conflict) query, every item
that surfaces as accepted supporting evidence in the retrieval pipeline's
``EvidencePackage`` is genuinely accepted: it carries ``status == "accepted"``
in the vector-index/graph result *and* — when it is an assertion — its row in
the source-of-truth ``Storage_Repository`` has ``status == accepted``.
Quarantined or rejected candidates (which are never embedded and never added to
the accepted-only graph) therefore never leak into a default result set.

This wires the full stack end to end against an offline, deterministic
configuration: the ``MockExtractor`` + ``WritePipeline`` (W1–W8) over an
in-memory SQLite repository + ``GraphStore``, with accepted assertions embedded
into an in-memory ``VectorIndex`` (deterministic, offline embeddings) via the
commit-manager / write-pipeline embed hooks, then read back through the
``RetrievalPipeline`` (R0–R4).

Validates: Requirements 10.3, 10.4, 10.5, 16.2, 16.5.
"""

from __future__ import annotations

from datetime import datetime, timezone

from hypothesis import given
from hypothesis import strategies as st

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
from ocm.ontology.enums import AssertionStatus
from ocm.resolution.entity_resolver import EntityResolver
from ocm.resolution.normalizer import Normalizer
from ocm.retrieval.embeddings import DeterministicEmbeddingProvider
from ocm.retrieval.evidence_packager import EvidencePackager
from ocm.retrieval.query_classifier import QueryClassifier
from ocm.retrieval.reranker import Reranker
from ocm.retrieval.retrieval_pipeline import RetrievalPipeline
from ocm.retrieval.semantic_retriever import SemanticRetriever
from ocm.retrieval.symbolic_retriever import SymbolicRetriever
from ocm.retrieval.vector_index import VectorIndex
from ocm.tests.markers import pbt_property
from ocm.validation.constraints import ConstraintValidator
from ocm.validation.schema_validator import SchemaValidator

# A fixed write timestamp keeps the run hermetic and clock-independent.
_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)

# Small, fixed vocabularies so the generated stream reuses the same entities and
# regularly produces conflicts (e.g. two assignees for one task) that route to
# quarantine — exactly the candidates that must never appear in accepted memory.
_PEOPLE = ["Alice", "Bob", "Carol", "Dave"]
_PROJECTS = ["Orion", "Atlas", "Nova"]
_TASKS = ["T1", "T2", "T3"]

# Write-op templates the MockExtractor recognizes. None contain a "correction"
# trigger word, so candidates route to accept or quarantine (never supersede).
_OWN = st.builds(
    lambda p, pr: f"{p} owns {pr}.",
    st.sampled_from(_PEOPLE),
    st.sampled_from(_PROJECTS),
)
_ASSIGN = st.builds(
    lambda p, t: f"{p} is assigned to {t}.",
    st.sampled_from(_PEOPLE),
    st.sampled_from(_TASKS),
)
_COMPLETE = st.builds(
    lambda p, t: f"{p} completed {t}.",
    st.sampled_from(_PEOPLE),
    st.sampled_from(_TASKS),
)
_NOT_STARTED = st.builds(lambda t: f"Task {t} is not started.", st.sampled_from(_TASKS))

_WRITE_OP = st.one_of(_OWN, _ASSIGN, _COMPLETE, _NOT_STARTED)
_WRITE_STREAM = st.lists(_WRITE_OP, min_size=1, max_size=6)

# Default, NON-conflict queries (classify as direct_fact, not contradiction_check).
_QUERY = st.one_of(
    st.builds(lambda pr: f"Who owns {pr}?", st.sampled_from(_PROJECTS)),
    st.builds(lambda t: f"Who is assigned to {t}?", st.sampled_from(_TASKS)),
    st.builds(lambda t: f"What is the status of Task {t}?", st.sampled_from(_TASKS)),
)


def _build_stack():
    """Wire the full write + retrieval stack over hermetic, offline components."""
    settings = Settings(deterministic_test_mode=True, chroma_mode="memory")
    repo = SQLiteRepository(":memory:")
    ids = IdGenerator(deterministic=True)
    graph = GraphStore()
    provenance = ProvenanceTracker(repo, ids)
    quarantine = QuarantineStore(repo, ids)

    # In-memory vector index with deterministic, offline embeddings; only the
    # commit manager (accepted assertions) and write pipeline (accepted claims /
    # documents / events) ever embed into it.
    vector_index = VectorIndex(
        DeterministicEmbeddingProvider(), chroma_mode="memory", graph=graph
    )

    commit = CommitManager(
        repo=repo,
        graph=graph,
        ids=ids,
        quarantine_store=quarantine,
        provenance_tracker=provenance,
        embed_hook=vector_index.embed_assertion,
    )
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
        memory_embed_hook=vector_index.embed_memory,
        research_logger=ResearchLogger(),
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
        research_logger=ResearchLogger(),
        settings=settings,
        ids=ids,
    )
    return repo, write_pipeline, retrieval_pipeline


@pbt_property(3, "Quarantined/rejected never appear in accepted retrieval")
@given(write_stream=_WRITE_STREAM, query=_QUERY)
def test_accepted_only_default_retrieval(write_stream: list[str], query: str) -> None:
    """Every accepted-memory result of a default query is genuinely accepted."""
    repo, write_pipeline, retrieval_pipeline = _build_stack()

    # Run the generated write stream through the full W1–W8 pipeline.
    for idx, text in enumerate(write_stream):
        write_pipeline.run(text, f"src-{idx}", created_at=_NOW)

    # Default (non-conflict) retrieval — include_conflicts defaults to False.
    pkg = retrieval_pipeline.query(query, top_k=5)

    # Source-of-truth: assertion ids whose stored status is NOT accepted
    # (quarantined assertions are never persisted, but superseded/anything else
    # must also never surface as accepted supporting evidence).
    non_accepted_ids = {
        a.id for a in repo.list_assertions() if a.status != AssertionStatus.accepted
    }

    # Req 16.2 / 16.5: a default query never surfaces quarantined (or any
    # non-accepted) items — every retrieved candidate is accepted-status.
    for item in pkg.retrieved_items:
        assert item.status == AssertionStatus.accepted.value, (
            f"default query surfaced a non-accepted item: "
            f"{item.memory_id} (status={item.status!r})"
        )

    # Req 10.3 / 10.4 / 10.5: every supporting assertion is genuinely accepted.
    supporting_ids = [sa.id for sa in pkg.supporting_assertions]
    for sid in supporting_ids:
        assertion = repo.get_assertion(sid)
        if assertion is not None:  # claims/docs/events have no assertion row
            assert assertion.status == AssertionStatus.accepted, (
                f"supporting assertion {sid} has status "
                f"{assertion.status!r}, expected accepted"
            )

    # No quarantined/superseded/rejected id ever appears as accepted support.
    assert set(supporting_ids).isdisjoint(non_accepted_ids), (
        f"non-accepted ids leaked into supporting evidence: "
        f"{set(supporting_ids) & non_accepted_ids}"
    )

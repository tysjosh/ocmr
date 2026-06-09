"""End-to-end Write Pipeline (W1–W8) checks over the Task T1 scenario.

Exercises the full pipeline with the offline ``MockExtractor`` against an
in-memory SQLite repository + graph, walking the three Task T1 writes:

1. "Alice owns Project Orion. Bob is assigned to Task T1." → OWNS + ASSIGNED_TO
   accepted.
2. "Bob completed Task T1." → a completion Event + RESULTS_IN/PARTICIPATES_IN
   accepted and Task T1 reconciled to ``done``.
3. "Task T1 is not started." (high confidence) → quarantined as a status
   contradiction rather than silently overwriting the accepted ``done`` status.

Requirements: 3.1, 10.1, 10.6, 10.7, 13.5, 16.6, 19.2, 25.1.
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
from ocm.ontology.enums import AssertionStatus, TaskStatus
from ocm.resolution.entity_resolver import EntityResolver
from ocm.resolution.normalizer import Normalizer
from ocm.validation.constraints import ConstraintValidator
from ocm.validation.schema_validator import SchemaValidator


@pytest.fixture
def pipeline():
    settings = Settings(deterministic_test_mode=True, chroma_mode="memory")
    repo = SQLiteRepository(":memory:")
    ids = IdGenerator(deterministic=True)
    graph = GraphStore()
    provenance = ProvenanceTracker(repo, ids)
    quarantine = QuarantineStore(repo, ids)
    embedded: list[str] = []
    memory_embedded: list[tuple[str, str]] = []
    commit = CommitManager(
        repo=repo,
        graph=graph,
        ids=ids,
        quarantine_store=quarantine,
        provenance_tracker=provenance,
        embed_hook=lambda a: embedded.append(a.id),
    )
    research_logger = ResearchLogger()
    wp = WritePipeline(
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
        memory_embed_hook=lambda t, m: memory_embedded.append((t, m.id)),
        research_logger=research_logger,
        settings=settings,
    )
    wp._embedded = embedded  # type: ignore[attr-defined]
    wp._memory_embedded = memory_embedded  # type: ignore[attr-defined]
    return wp


def _accepted_predicates(result) -> set[str]:
    return {o.candidate.predicate for o in result.accepted}


def test_write1_accepts_owns_and_assigned_to(pipeline):
    result = pipeline.run(
        "Alice owns Project Orion. Bob is assigned to Task T1.", "src-1"
    )
    preds = _accepted_predicates(result)
    assert "OWNS" in preds
    assert "ASSIGNED_TO" in preds
    assert result.rejected == []
    # Accepted assertions were embedded via the commit-manager hook (Req 13.5).
    assert pipeline._embedded  # type: ignore[attr-defined]
    # Claims were embedded via the memory hook (Req 16.6).
    assert any(t == "Claim" for t, _ in pipeline._memory_embedded)  # type: ignore[attr-defined]


def test_full_t1_scenario_quarantines_status_contradiction(pipeline):
    # Write 1 — ownership + assignment.
    r1 = pipeline.run("Alice owns Project Orion. Bob is assigned to Task T1.", "src-1")
    assert {"OWNS", "ASSIGNED_TO"} <= _accepted_predicates(r1)

    # Write 2 — completion: RESULTS_IN + PARTICIPATES_IN accepted, T1 -> done.
    r2 = pipeline.run("Bob completed Task T1.", "src-2")
    preds2 = _accepted_predicates(r2)
    assert "RESULTS_IN" in preds2
    assert "PARTICIPATES_IN" in preds2

    graph = pipeline.graph
    # Find the Task T1 node and assert it is now done.
    task_ids = [n for n in graph.node_ids() if graph.get_entity_type(n) == "Task"]
    assert len(task_ids) == 1
    t1_id = task_ids[0]
    assert graph.get_entity_payload(t1_id)["status"] == TaskStatus.done.value

    # Write 3 — "not started" at high confidence: quarantined, not overwritten.
    r3 = pipeline.run("Task T1 is not started.", "src-3")
    assert len(r3.quarantined) == 1
    assert r3.quarantined[0].decision == "quarantined"
    assert r3.summary.num_quarantined == 1
    # The accepted status is unchanged (no silent overwrite, Req 10.6).
    assert graph.get_entity_payload(t1_id)["status"] == TaskStatus.done.value

    # The quarantine is durable and surfaced for conflict queries.
    conflicts = pipeline.quarantine_store.list()
    assert any("status contradiction" in q.reason for q in conflicts)

    # Only accepted assertions are edges in the graph (Req 10.5).
    for a in pipeline.repo.list_assertions():
        if a.status == AssertionStatus.accepted:
            assert graph.has_assertion(a.subject_id, a.object_id, a.predicate)


def test_failed_extraction_is_handled_gracefully(pipeline):
    # Empty input yields no candidates; the run still returns a summary.
    result = pipeline.run("", "src-empty")
    assert result.summary.num_candidates == 0
    assert result.accepted == []

"""Property test for deterministic IDs across runs (task 10.8).

Feature: ontology-constrained-memory, Property 8.

This module validates correctness Property 8 over the *real* Write Pipeline
(W1-W8) wired end to end with the offline :class:`MockExtractor`, an in-memory
:class:`SQLiteRepository`, a fresh :class:`GraphStore`, and a deterministic
:class:`IdGenerator` (``deterministic=True``), under
``Settings(deterministic_test_mode=True, ...)`` (Req 27.5).

Property 8 -- *Deterministic IDs across runs*: under
``deterministic_test_mode``, two independent runs over a fixed input batch
produce **identical** entity ID sequences and identical accepted-assertion ID
sequences (Req 27.5, 3.5). Because the Mock_Extractor is a pure function of its
input (Req 3.5) and the IdGenerator derives ids from
``entity_type|normalized_name|source_ref`` plus a per-run counter that is reset
on each fresh ``IdGenerator`` construction (Req 27.5), running the same batch of
texts through two freshly-built pipeline stacks -- in the same order, with a
fixed ``created_at`` so timestamps never depend on the wall clock -- must
reproduce byte-identical id sequences.

The generated batches mix recognizable patterns the Mock_Extractor understands
(ownership, assignment, completion, task-status, decisions) with arbitrary free
text, over a small vocabulary so multiple writes touch overlapping entities and
the id streams are non-trivial. Batch size, sentence choice, and names are
varied with Hypothesis across >= 100 iterations.

Validates: Requirements 27.5, 3.5.
"""

from __future__ import annotations

from datetime import datetime, timezone

from hypothesis import given
from hypothesis import strategies as st

from ocm.core.config import Settings
from ocm.core.ids import IdGenerator
from ocm.extraction.mock_extractor import MockExtractor
from ocm.memory.assertion_builder import AssertionBuilder
from ocm.memory.commit_manager import CommitManager
from ocm.memory.graph_store import GraphStore
from ocm.memory.provenance_tracker import ProvenanceTracker
from ocm.memory.quarantine_store import QuarantineStore
from ocm.memory.sqlite_repository import SQLiteRepository
from ocm.memory.write_pipeline import WritePipeline
from ocm.resolution.entity_resolver import EntityResolver
from ocm.resolution.normalizer import Normalizer
from ocm.tests.markers import pbt_property
from ocm.validation.constraints import ConstraintValidator
from ocm.validation.schema_validator import SchemaValidator

#: Fixed wall-clock-independent timestamp so every write produces stable
#: timestamps and the only source of ids is the deterministic IdGenerator.
FIXED_TS = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

# Small vocabularies keep the entity space tight so successive writes in a batch
# touch overlapping entities (resolve-existing vs create-new), exercising the
# per-run id counter rather than always minting brand-new names.
_PEOPLE = ["Alice", "Bob", "Carol"]
_PROJECTS = ["Orion", "Apollo"]
_TASKS = ["T1", "T2"]


def _sentence_strategy() -> st.SearchStrategy[str]:
    """Sentences drawn from Mock_Extractor-recognizable patterns + free text."""
    person = st.sampled_from(_PEOPLE)
    project = st.sampled_from(_PROJECTS)
    task = st.sampled_from(_TASKS)
    return st.one_of(
        st.builds(lambda p, x: f"{p} owns Project {x}", person, project),
        st.builds(lambda p, t: f"{p} is assigned to Task {t}", person, task),
        st.builds(lambda p, t: f"{p} completed Task {t}", person, task),
        st.builds(lambda t: f"Task {t} is not started", task),
        st.builds(lambda t: f"Task {t} is in progress", task),
        st.builds(lambda t: f"Task {t} is blocked", task),
        # Arbitrary free text: becomes a Claim, never destabilizes ids.
        st.text(min_size=0, max_size=24),
    )


@st.composite
def input_batches(draw: st.DrawFn) -> list[str]:
    """A fixed batch of write texts (each 1-3 sentences joined by '. ')."""
    n_texts = draw(st.integers(min_value=1, max_value=6))
    batch: list[str] = []
    for _ in range(n_texts):
        sentences = draw(st.lists(_sentence_strategy(), min_size=1, max_size=3))
        batch.append(". ".join(sentences))
    return batch


def _build_pipeline() -> WritePipeline:
    """Build a fresh, hermetic, deterministic Write Pipeline stack.

    Every stack gets its own ``:memory:`` repository, graph, and a freshly
    constructed deterministic :class:`IdGenerator` (per-run counter reset to 0),
    so two stacks are fully independent yet reproduce the same id sequence.
    """
    settings = Settings(
        deterministic_test_mode=True, chroma_mode="memory", extractor="mock"
    )
    repo = SQLiteRepository(":memory:")
    graph = GraphStore()
    ids = IdGenerator(deterministic=True)
    provenance = ProvenanceTracker(repo, ids)
    quarantine = QuarantineStore(repo, ids)
    commit = CommitManager(
        repo=repo,
        graph=graph,
        ids=ids,
        quarantine_store=quarantine,
        provenance_tracker=provenance,
    )
    return WritePipeline(
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
        settings=settings,
    )


def _run_batch(batch: list[str]) -> tuple[list[str], list[str]]:
    """Run ``batch`` through a fresh stack; return (entity_ids, assertion_ids).

    Each text is written with a stable, index-derived ``source_ref`` and a
    fixed ``created_at`` so the only variability would come from id generation.
    Returns the entity-id sequence (insertion order from the repository) and the
    accepted-assertion-id sequence (in batch + per-write order).
    """
    pipeline = _build_pipeline()
    try:
        assertion_ids: list[str] = []
        for i, text in enumerate(batch):
            result = pipeline.run(text, f"src-{i}", created_at=FIXED_TS)
            assertion_ids.extend(
                o.assertion_id for o in result.accepted if o.assertion_id is not None
            )
        entity_ids = [payload["id"] for _, payload in pipeline.repo.list_entities()]
        return entity_ids, assertion_ids
    finally:
        pipeline.repo.close()


@pbt_property(8, "Deterministic IDs across runs")
@given(batch=input_batches())
def test_deterministic_ids_across_runs(batch: list[str]) -> None:
    """Two runs over a fixed batch produce identical entity/assertion id sequences.

    Validates: Requirements 27.5, 3.5
    """
    entities_run1, assertions_run1 = _run_batch(batch)
    entities_run2, assertions_run2 = _run_batch(batch)

    assert entities_run1 == entities_run2, (
        "entity id sequence differed across deterministic runs:\n"
        f"  run 1: {entities_run1}\n  run 2: {entities_run2}"
    )
    assert assertions_run1 == assertions_run2, (
        "accepted assertion id sequence differed across deterministic runs:\n"
        f"  run 1: {assertions_run1}\n  run 2: {assertions_run2}"
    )

"""Property test for provenance coverage (task 8.8).

Feature: ontology-constrained-memory, Property 4.

This module validates correctness Property 4 over the *real* governance stack
wired end to end:

    Constraint_Validator (W6, C7) -> Contradiction_Checker (W7) -> Commit_Manager (W8)

against a live :class:`GraphStore` / :class:`SQLiteRepository(":memory:")`.

Property 4 — *Every accepted assertion has provenance*: after processing an
arbitrary stream of valid candidate assertions, **every** assertion that ends up
``accepted`` in the repository has at least one provenance record, and that
record's ``subject_id`` matches the accepted assertion's id (Req 12.1, 12.2,
12.4). The Commit_Manager records provenance on the accept path keyed by the
assertion id, so the durable ``provenance`` table must always cover the accepted
set.

The stream mixes two well-typed predicates over seeded, differently-typed
entities so domain/range (C9) passes and multiple accepts happen:

* ``OWNS`` (Person/Organization -> Project, m:n) — every such candidate is
  accepted, guaranteeing a healthy accepted population.
* ``ASSIGNED_TO`` (Task -> Person, m:1) — the single-valued relation, where a
  second distinct high-confidence assignee for the same Task is quarantined
  rather than accepted; the property must still hold for the survivors.

All candidates use ``write_intent="new_fact"`` so conflicts route to quarantine
(never supersede), keeping the accepted set populated purely by the accept path.
Entity counts, predicate choice, endpoints, and confidences are varied with
Hypothesis across >= 100 iterations.

Validates: Requirements 12.1, 12.2, 12.4.
"""

from __future__ import annotations

from datetime import datetime, timezone

from hypothesis import given
from hypothesis import strategies as st

from ocm.core.config import Settings
from ocm.core.ids import IdGenerator
from ocm.memory.commit_manager import CommitManager
from ocm.memory.contracts import CandidateAssertion
from ocm.memory.graph_store import GraphStore
from ocm.memory.provenance_tracker import ProvenanceTracker
from ocm.memory.quarantine_store import QuarantineStore
from ocm.memory.sqlite_repository import SQLiteRepository
from ocm.ontology.enums import AssertionStatus, PersonStatus
from ocm.ontology.models import Person, Project, Task
from ocm.tests.markers import pbt_property
from ocm.validation.constraints import ConstraintValidator

TS = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

# Any in-range confidence is a *valid* candidate (Req 1.6). Varying across the
# whole [0, 1] range exercises both the soft (low-confidence) and hard
# (high-confidence) contradiction paths for ASSIGNED_TO.
confidence = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)


@st.composite
def assertion_streams(draw: st.DrawFn):
    """Generate ``(num_persons, num_projects, num_tasks, steps)``.

    Each step is ``(kind, subject_index, object_index, confidence)`` where
    ``kind`` is ``"OWNS"`` (Person[subject] -> Project[object]) or
    ``"ASSIGNED_TO"`` (Task[subject] -> Person[object]). Indices are taken
    modulo the relevant seeded population so they always reference an existing,
    well-typed entity (domain/range therefore passes).
    """
    num_persons = draw(st.integers(min_value=1, max_value=4))
    num_projects = draw(st.integers(min_value=1, max_value=4))
    num_tasks = draw(st.integers(min_value=1, max_value=4))

    step = st.tuples(
        st.sampled_from(["OWNS", "ASSIGNED_TO"]),
        st.integers(min_value=0, max_value=3),  # subject index (mod later)
        st.integers(min_value=0, max_value=3),  # object index (mod later)
        confidence,
    )
    steps = draw(st.lists(step, min_size=1, max_size=20))
    return num_persons, num_projects, num_tasks, steps


def _build_stack(num_persons: int, num_projects: int, num_tasks: int):
    """Build a fresh, hermetic governance stack seeded with typed entities.

    Each Hypothesis example gets its own in-memory repository/graph so state
    never leaks between iterations. People are all ``active`` so the inactive
    assignee constraint (C5) never spuriously quarantines a write.
    """
    repo = SQLiteRepository(":memory:")
    graph = GraphStore()
    ids = IdGenerator(deterministic=True)
    settings = Settings(
        deterministic_test_mode=True, chroma_mode="memory", extractor="mock"
    )

    for i in range(num_persons):
        person = Person(id=f"per_{i}", name=f"P{i}", status=PersonStatus.active)
        repo.upsert_entity("Person", person)
        graph.add_entity("Person", person)
    for i in range(num_projects):
        project = Project(id=f"proj_{i}", name=f"Project {i}")
        repo.upsert_entity("Project", project)
        graph.add_entity("Project", project)
    for i in range(num_tasks):
        task = Task(id=f"task_{i}", title=f"Task {i}")
        repo.upsert_entity("Task", task)
        graph.add_entity("Task", task)

    validator = ConstraintValidator(settings)
    manager = CommitManager(
        repo=repo,
        graph=graph,
        ids=ids,
        quarantine_store=QuarantineStore(repo, ids),
        provenance_tracker=ProvenanceTracker(repo, ids),
    )
    return repo, graph, validator, manager, settings


@pbt_property(4, "Every accepted assertion has provenance")
@given(stream=assertion_streams())
def test_provenance_coverage(stream) -> None:
    """Every accepted assertion has >=1 provenance record with matching subject_id.

    Validates: Requirements 12.1, 12.2, 12.4
    """
    num_persons, num_projects, num_tasks, steps = stream
    repo, graph, validator, manager, settings = _build_stack(
        num_persons, num_projects, num_tasks
    )

    try:
        # Process the whole stream: build each well-typed candidate, validate,
        # then commit per the verdict (accept / quarantine).
        for i, (kind, subj_idx, obj_idx, conf) in enumerate(steps):
            if kind == "OWNS":
                subject_id = f"per_{subj_idx % num_persons}"
                object_id = f"proj_{obj_idx % num_projects}"
            else:  # ASSIGNED_TO
                subject_id = f"task_{subj_idx % num_tasks}"
                object_id = f"per_{obj_idx % num_persons}"

            candidate = CandidateAssertion(
                subject_id=subject_id,
                predicate=kind,
                object_id=object_id,
                confidence=conf,
                source_ref=f"doc://stream#{i}",
                write_intent="new_fact",
                extractor_version="mock-1",
            )
            verdict = validator.validate(candidate, graph, settings=settings)
            manager.commit(candidate, verdict, created_at=TS)

        accepted = list(repo.list_assertions(status=AssertionStatus.accepted.value))

        # The first valid write always succeeds (well-typed, active assignee),
        # so the accepted set is never empty — the property is non-vacuous.
        assert accepted, "expected at least one accepted assertion"

        # Property 4: every accepted assertion has >=1 provenance record, and
        # that record's subject_id matches the assertion id (Req 12.1, 12.4).
        for assertion in accepted:
            records = manager.provenance_tracker.for_subject(assertion.id)
            assert len(records) >= 1, (
                f"accepted assertion {assertion.id} "
                f"({assertion.subject_id} -[{assertion.predicate}]-> "
                f"{assertion.object_id}) has no provenance record"
            )
            assert all(r.subject_id == assertion.id for r in records), (
                f"provenance for {assertion.id} has a mismatched subject_id: "
                f"{[r.subject_id for r in records]}"
            )
    finally:
        repo.close()

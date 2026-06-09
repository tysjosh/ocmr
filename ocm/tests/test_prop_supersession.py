"""Property test for supersession integrity (task 8.7).

Feature: ontology-constrained-memory, Property 7.

This module validates correctness Property 7 over the *real* governance stack
wired end to end:

    Constraint_Validator (W6, C7) -> Contradiction_Checker (W7) -> Commit_Manager (W8)

against a live :class:`GraphStore` / :class:`SQLiteRepository(":memory:")`.

The scenario: an accepted high-confidence single-valued ``ASSIGNED_TO`` edge
``t1 -> personA`` (B) already exists. A high-confidence ``correction`` arrives
that reassigns the same task, ``t1 -> personB`` (A). The governance layer must
detect the single-valued conflict (C7 via W7), recommend ``supersede``, and the
Commit Manager must:

* accept the new correcting assertion A,
* flip the prior assertion B to ``superseded``,
* add a ``SUPERSEDES`` edge new -> old,
* leave **exactly one** accepted ``ASSIGNED_TO`` edge, and
* preserve provenance for **both** the old and the new assertion.

Person ids and confidences (> 0.8) are varied with Hypothesis across >= 100
iterations to show the invariant holds for the whole high-confidence input
space, not just one example.

Validates: Requirements 10.2, 12.3, 2.13.
"""

from __future__ import annotations

from datetime import datetime, timezone

from hypothesis import given
from hypothesis import strategies as st

from ocm.core.ids import IdGenerator
from ocm.core.config import Settings
from ocm.memory.commit_manager import SUPERSEDES, CommitManager
from ocm.memory.contracts import CandidateAssertion
from ocm.memory.graph_store import GraphStore
from ocm.memory.manual_write import manual_write
from ocm.memory.provenance_tracker import ProvenanceTracker
from ocm.memory.quarantine_store import QuarantineStore
from ocm.memory.sqlite_repository import SQLiteRepository
from ocm.ontology.enums import AssertionStatus, PersonStatus
from ocm.ontology.models import Person, Task
from ocm.tests.markers import pbt_property
from ocm.validation.constraints import ConstraintValidator

TS = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

# Confidences strictly above the default contradiction_high_confidence (0.8) so
# both the seeded assignment and the correction are "high confidence" and the
# single-valued ASSIGNED_TO conflict is treated as hard (Req 9.5). The
# correction must also *dominate* the incumbent by a margin (Algorithm 1's
# delta) to be routed to supersede, so the seed is drawn from a lower band than
# the correction.
seed_conf_strategy = st.floats(
    min_value=0.81, max_value=0.90, allow_nan=False, allow_infinity=False
)
correction_conf_strategy = st.floats(
    min_value=0.91, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Distinct person id suffixes so the correction always points at a *different*
# assignee than the seed (otherwise there is no conflict to supersede).
person_suffix = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8
)


def _build_stack():
    """Construct a fresh, hermetic governance stack for one Hypothesis example.

    Each example gets its own in-memory repository/graph so state never leaks
    between iterations. Returns the pieces the test wires together.
    """
    repo = SQLiteRepository(":memory:")
    ids = IdGenerator(deterministic=True)
    graph = GraphStore()
    settings = Settings(
        deterministic_test_mode=True, chroma_mode="memory", extractor="mock"
    )
    validator = ConstraintValidator(settings)
    manager = CommitManager(
        repo=repo,
        graph=graph,
        ids=ids,
        quarantine_store=QuarantineStore(repo, ids),
        provenance_tracker=ProvenanceTracker(repo, ids),
    )
    return repo, ids, graph, settings, validator, manager


@pbt_property(7, "Supersession preserves provenance and leaves exactly one accepted")
@given(
    suffix_a=person_suffix,
    suffix_b=person_suffix,
    seed_conf=seed_conf_strategy,
    correction_conf=correction_conf_strategy,
)
def test_supersession_integrity(
    suffix_a: str, suffix_b: str, seed_conf: float, correction_conf: float
) -> None:
    """A correction that dominates the incumbent supersedes it, preserving both
    provenances.

    Validates: Requirements 10.2, 12.3, 2.13
    """
    # Ensure the two people are distinct so the correction reassigns the task.
    person_a = f"per_{suffix_a}"
    person_b = f"per_{suffix_b}x"  # appended char guarantees A != B
    assert person_a != person_b

    repo, ids, graph, settings, validator, manager = _build_stack()
    try:
        # --- seed the world: Task t1 + two active people --------------------
        task = ("Task", Task(id="t1", title="Ship OCM"))
        people = (
            ("Person", Person(id=person_a, name="Ada", status=PersonStatus.active)),
            ("Person", Person(id=person_b, name="Bob", status=PersonStatus.active)),
        )
        for etype, ent in (task, *people):
            repo.upsert_entity(etype, ent)
            graph.add_entity(etype, ent)

        # --- accept the initial single-valued ASSIGNED_TO: t1 -> A (this is B)
        seed = CandidateAssertion(
            subject_id="t1",
            predicate="ASSIGNED_TO",
            object_id=person_a,
            confidence=seed_conf,
            source_ref="doc://seed#1",
            extractor_version="mock-1",
        )
        old_assertion = manual_write([], seed, repo, graph, ids, created_at=TS)
        old_id = old_assertion.id
        assert repo.get_assertion(old_id).status is AssertionStatus.accepted

        # --- submit a high-confidence correction: t1 -> B (this is A) -------
        correction = CandidateAssertion(
            subject_id="t1",
            predicate="ASSIGNED_TO",
            object_id=person_b,
            confidence=correction_conf,
            source_ref="doc://notes#2",
            write_intent="correction",
            extractor_version="mock-1",
        )

        # Governance should detect the conflict and recommend supersede (Req 10.2).
        vr = validator.validate(correction, graph, settings=settings)
        assert vr.recommended_action == "supersede"
        assert old_id in vr.conflicting_ids

        outcome = manager.commit(correction, vr, created_at=TS)

        # --- Property 7 assertions -----------------------------------------
        # New assertion A accepted, old assertion B superseded (Req 10.2).
        assert outcome.decision == "superseded"
        assert outcome.superseded_assertion_id == old_id
        new_id = outcome.assertion_id
        assert repo.get_assertion(new_id).status is AssertionStatus.accepted
        assert repo.get_assertion(old_id).status is AssertionStatus.superseded

        # SUPERSEDES(A -> B) edge exists (Req 2.13).
        assert graph.has_assertion(new_id, old_id, SUPERSEDES)

        # Exactly one accepted ASSIGNED_TO edge remains, pointing at B (per_b).
        assigned_edges = graph.find_edges_by_predicate("ASSIGNED_TO")
        assert len(assigned_edges) == 1
        assert assigned_edges[0][1] == person_b
        assert not graph.has_assertion("t1", person_a, "ASSIGNED_TO")
        assert graph.has_assertion("t1", person_b, "ASSIGNED_TO")

        # Provenance preserved for BOTH the old and the new assertion (Req 12.3).
        assert len(manager.provenance_tracker.for_subject(old_id)) >= 1
        assert len(manager.provenance_tracker.for_subject(new_id)) >= 1
    finally:
        repo.close()

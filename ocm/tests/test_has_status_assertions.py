"""Regression tests for status-as-assertions (HAS_STATUS), the full fix for #7.

A Task's status is promoted to a first-class ``HAS_STATUS(task -> StatusValue)``
assertion so a status flip becomes an assertion-to-assertion contradiction:

* the **first** status is accepted as a HAS_STATUS assertion;
* a **legal** transition (or a correction) supersedes the prior status assertion;
* an **illegal** flip is quarantined as a *status contradiction* whose
  ``conflicting_ids`` point at the accepted HAS_STATUS assertion (and the Task
  entity), so a plain status query surfaces a paired ``{accepted, quarantined}``
  conflict instead of silently collapsing it;
* the denormalized ``Task.status`` attribute stays in sync for back-compat.
"""

from __future__ import annotations

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.ontology.enums import AssertionStatus, TaskStatus


def _container() -> CoreContainer:
    return CoreContainer(
        Settings(deterministic_test_mode=True, chroma_mode="memory", extractor="mock")
    )


def _task_id(graph) -> str:
    tasks = [n for n in graph.node_ids() if graph.get_entity_type(n) == "Task"]
    assert len(tasks) == 1
    return tasks[0]


# --------------------------------------------------------------------------- #
# First status -> accepted HAS_STATUS assertion + synced attribute
# --------------------------------------------------------------------------- #
def test_first_status_is_accepted_as_has_status_assertion():
    c = _container()
    c.write_pipeline.run("Task T1 is in progress.", "s1")

    graph = c.graph
    t1 = _task_id(graph)
    edges = graph.out_edges(t1, "HAS_STATUS")
    assert len(edges) == 1
    _s, obj, _k, _data = edges[0]
    assert obj == f"status:{TaskStatus.in_progress.value}"
    # Backward-compatible denormalized attribute is kept in sync.
    assert graph.get_entity_payload(t1)["status"] == TaskStatus.in_progress.value
    # The StatusValue node exists and is typed.
    assert graph.get_entity_type(obj) == "StatusValue"


# --------------------------------------------------------------------------- #
# Legal transition -> supersede the prior HAS_STATUS assertion
# --------------------------------------------------------------------------- #
def test_legal_transition_supersedes_prior_status_assertion():
    c = _container()
    c.write_pipeline.run("Task T1 is not started.", "s1")  # todo
    graph = c.graph
    t1 = _task_id(graph)
    old_aid = graph.out_edges(t1, "HAS_STATUS")[0][3]["assertion_id"]

    r2 = c.write_pipeline.run("Task T1 is in progress.", "s2")  # todo -> in_progress (legal)

    assert r2.superseded, "a legal transition should supersede the prior status"
    edges = graph.out_edges(t1, "HAS_STATUS")
    assert len(edges) == 1, "only one accepted HAS_STATUS edge may remain (m:1)"
    assert edges[0][1] == f"status:{TaskStatus.in_progress.value}"
    assert graph.get_entity_payload(t1)["status"] == TaskStatus.in_progress.value
    # The old status assertion row is durably flipped to superseded.
    old = c.repo.get_assertion(old_aid)
    assert old is not None and old.status == AssertionStatus.superseded


# --------------------------------------------------------------------------- #
# Illegal flip -> quarantine pointing at the accepted status assertion
# --------------------------------------------------------------------------- #
def test_illegal_flip_quarantines_pointing_at_accepted_status_assertion():
    c = _container()
    c.write_pipeline.run("Alice owns Project Orion. Bob is assigned to Task T1.", "s1")
    c.write_pipeline.run("Bob completed Task T1.", "s2")  # -> done
    graph = c.graph
    t1 = _task_id(graph)
    accepted_aid = graph.out_edges(t1, "HAS_STATUS")[0][3]["assertion_id"]

    r3 = c.write_pipeline.run("Task T1 is not started.", "s3")  # done -> todo (illegal)

    assert len(r3.quarantined) == 1
    q = r3.quarantined[0]
    assert "status contradiction" in (q.reason or "").lower()
    # The accepted status is never overwritten.
    assert graph.get_entity_payload(t1)["status"] == TaskStatus.done.value
    # The durable quarantine record points at the accepted HAS_STATUS assertion.
    record = next(r for r in c.quarantine_store.list() if r.id == q.quarantine_id)
    assert accepted_aid in record.conflicting_ids
    assert t1 in record.conflicting_ids


# --------------------------------------------------------------------------- #
# Status query surfaces the paired {accepted, quarantined} conflict inline
# --------------------------------------------------------------------------- #
def test_status_query_surfaces_paired_conflict():
    c = _container()
    c.write_pipeline.run("Alice owns Project Orion. Bob is assigned to Task T1.", "s1")
    c.write_pipeline.run("Bob completed Task T1.", "s2")
    c.write_pipeline.run("Task T1 is not started.", "s3")

    pkg = c.retrieval_pipeline.query("What is the current status of Task T1?", top_k=10)

    paired = [c for c in pkg.conflicts if "status contradiction" in (c.reason or "").lower()]
    assert paired, "the status query must surface the status contradiction inline"
    item = paired[0]
    assert item.accepted and "done" in item.accepted.lower()
    assert item.quarantined and "todo" in item.quarantined.lower()
    # The deterministic answer still reports the accepted status.
    assert pkg.answer is not None and "done" in pkg.answer.lower()


# --------------------------------------------------------------------------- #
# Idempotent restate -> no new assertion, no quarantine
# --------------------------------------------------------------------------- #
def test_restating_same_status_is_a_noop():
    c = _container()
    c.write_pipeline.run("Task T1 is in progress.", "s1")
    graph = c.graph
    t1 = _task_id(graph)
    aid = graph.out_edges(t1, "HAS_STATUS")[0][3]["assertion_id"]

    r2 = c.write_pipeline.run("Task T1 is in progress.", "s2")

    assert not r2.quarantined
    edges = graph.out_edges(t1, "HAS_STATUS")
    assert len(edges) == 1
    assert edges[0][3]["assertion_id"] == aid  # unchanged, no churn


# --------------------------------------------------------------------------- #
# Correction intent supersedes even an otherwise-illegal flip
# --------------------------------------------------------------------------- #
def test_correction_intent_supersedes_status():
    c = _container()
    c.write_pipeline.run("Alice owns Project Orion. Bob is assigned to Task T1.", "s1")
    c.write_pipeline.run("Bob completed Task T1.", "s2")  # -> done
    graph = c.graph
    t1 = _task_id(graph)

    # A correction may revise even the terminal 'done' status.
    r3 = c.write_pipeline.run(
        "Task T1 is in progress.", "s3", write_intent="correction"
    )

    assert r3.superseded, "a correction should supersede the accepted status"
    assert not r3.quarantined
    assert graph.get_entity_payload(t1)["status"] == TaskStatus.in_progress.value
    assert len(graph.out_edges(t1, "HAS_STATUS")) == 1


# --------------------------------------------------------------------------- #
# Generalization to Project status (the planning scenario)
# --------------------------------------------------------------------------- #
def _project_id(graph) -> str:
    projects = [n for n in graph.node_ids() if graph.get_entity_type(n) == "Project"]
    assert len(projects) == 1
    return projects[0]


def test_project_first_status_is_accepted_as_has_status():
    c = _container()
    c.write_pipeline.run("Project Orion is active.", "s1")

    graph = c.graph
    pid = _project_id(graph)
    edges = graph.out_edges(pid, "HAS_STATUS")
    assert len(edges) == 1
    assert edges[0][1] == "status:active"
    assert graph.get_entity_payload(pid)["status"] == "active"


def test_project_status_flip_as_new_fact_quarantines():
    """Planning scenario: active then cancelled (bare new_fact) surfaces a conflict."""
    c = _container()
    c.write_pipeline.run("Project Orion is active.", "s1")
    graph = c.graph
    pid = _project_id(graph)
    accepted_aid = graph.out_edges(pid, "HAS_STATUS")[0][3]["assertion_id"]

    r2 = c.write_pipeline.run("Project Orion was cancelled.", "s2")  # new_fact

    assert len(r2.quarantined) == 1
    assert "status contradiction" in (r2.quarantined[0].reason or "").lower()
    # Accepted status is not silently overwritten.
    assert graph.get_entity_payload(pid)["status"] == "active"
    record = next(r for r in c.quarantine_store.list() if r.id == r2.quarantined[0].quarantine_id)
    assert accepted_aid in record.conflicting_ids
    assert pid in record.conflicting_ids

    # The conflict surfaces inline for a status query about the project.
    pkg = c.retrieval_pipeline.query("What is the status of Project Orion?", top_k=10)
    paired = [cf for cf in pkg.conflicts if "status contradiction" in (cf.reason or "").lower()]
    assert paired, "project status conflict must surface inline"
    assert paired[0].accepted and "active" in paired[0].accepted.lower()
    assert paired[0].quarantined and "cancelled" in paired[0].quarantined.lower()


def test_project_status_update_supersedes():
    c = _container()
    c.write_pipeline.run("Project Orion is active.", "s1")
    graph = c.graph
    pid = _project_id(graph)

    r2 = c.write_pipeline.run(
        "Project Orion was cancelled.", "s2", write_intent="update"
    )

    assert r2.superseded, "an update should supersede the prior project status"
    assert not r2.quarantined
    edges = graph.out_edges(pid, "HAS_STATUS")
    assert len(edges) == 1
    assert edges[0][1] == "status:cancelled"
    assert graph.get_entity_payload(pid)["status"] == "cancelled"


# --------------------------------------------------------------------------- #
# Generalization to Person status
# --------------------------------------------------------------------------- #
def _person_id(graph, name_contains: str | None = None) -> str:
    people = [n for n in graph.node_ids() if graph.get_entity_type(n) == "Person"]
    assert people
    return people[0]


def test_person_first_status_is_accepted_as_has_status():
    c = _container()
    c.write_pipeline.run("Mallory is active.", "s1")

    graph = c.graph
    pid = _person_id(graph)
    edges = graph.out_edges(pid, "HAS_STATUS")
    assert len(edges) == 1
    assert edges[0][1] == "status:active"
    assert graph.get_entity_payload(pid)["status"] == "active"


def test_person_status_flip_as_new_fact_quarantines():
    c = _container()
    c.write_pipeline.run("Mallory is active.", "s1")
    graph = c.graph
    pid = _person_id(graph)
    accepted_aid = graph.out_edges(pid, "HAS_STATUS")[0][3]["assertion_id"]

    r2 = c.write_pipeline.run("Mallory is inactive.", "s2")  # new_fact

    assert len(r2.quarantined) == 1
    assert "status contradiction" in (r2.quarantined[0].reason or "").lower()
    assert graph.get_entity_payload(pid)["status"] == "active"  # not overwritten
    record = next(r for r in c.quarantine_store.list() if r.id == r2.quarantined[0].quarantine_id)
    assert accepted_aid in record.conflicting_ids


def test_person_status_correction_supersedes():
    c = _container()
    c.write_pipeline.run("Mallory is active.", "s1")
    graph = c.graph
    pid = _person_id(graph)

    r2 = c.write_pipeline.run(
        "Mallory is inactive.", "s2", write_intent="correction"
    )

    assert r2.superseded
    assert not r2.quarantined
    assert graph.get_entity_payload(pid)["status"] == "inactive"
    assert len(graph.out_edges(pid, "HAS_STATUS")) == 1


# --------------------------------------------------------------------------- #
# Generalization to Decision status (draft -> final gated by C8)
# --------------------------------------------------------------------------- #
def _decision_id(graph) -> str:
    decisions = [n for n in graph.node_ids() if graph.get_entity_type(n) == "Decision"]
    assert len(decisions) == 1
    return decisions[0]


def test_draft_decision_accepted_as_has_status():
    c = _container()
    c.write_pipeline.run("We decided to launch Project Orion.", "s1")

    graph = c.graph
    did = _decision_id(graph)
    edges = graph.out_edges(did, "HAS_STATUS")
    assert len(edges) == 1
    assert edges[0][1] == "status:draft"
    assert graph.get_entity_payload(did)["status"] == "draft"


def test_final_decision_without_evidence_quarantines_and_retracts():
    """A first-time final decision lacking evidence is quarantined (C8) and the
    mirrored node is retracted — no accepted Decision node lingers."""
    c = _container()
    r = c.write_pipeline.run("We finalized the decision to cancel Project Atlas.", "s1")

    assert r.quarantined, "final decision without evidence must be quarantined (C8)"
    assert "status contradiction" in (r.quarantined[0].reason or "").lower()
    assert not any(
        c.graph.get_entity_type(n) == "Decision" for n in c.graph.node_ids()
    )


def test_cross_session_draft_then_final_without_evidence_keeps_draft():
    """Stable decision identity: a draft (s1) then a final-without-evidence (s2)
    for the *same* decision keeps the accepted draft and quarantines only the
    failed final upgrade."""
    c = _container()
    c.write_pipeline.run("We decided to launch Project Orion.", "s1")  # draft
    graph = c.graph
    did = _decision_id(graph)
    draft_aid = graph.out_edges(did, "HAS_STATUS")[0][3]["assertion_id"]

    r2 = c.write_pipeline.run("We finalized the decision to launch Project Orion.", "s2")

    # The same decision entity is reused (topic-based identity).
    assert _decision_id(graph) == did
    # The draft remains accepted; only the final upgrade is quarantined.
    assert r2.quarantined
    assert graph.get_entity_payload(did)["status"] == "draft"
    assert len(graph.out_edges(did, "HAS_STATUS")) == 1
    record = next(r for r in c.quarantine_store.list() if r.id == r2.quarantined[0].quarantine_id)
    assert draft_aid in record.conflicting_ids


def test_cross_session_draft_then_final_with_evidence_supersedes():
    """With supporting evidence present, finalizing supersedes the draft status."""
    from ocm.ontology.enums import AssertionStatus
    from ocm.ontology.models import Assertion, Event

    c = _container()
    c.write_pipeline.run("We decided to launch Project Orion.", "s1")  # draft
    graph = c.graph
    did = _decision_id(graph)

    # Seed an EVIDENCE_FOR edge (Event -> Decision) so C8's evidence floor holds.
    ev = Event(
        id="evt_evidence",
        type="event",
        timestamp_start=__import__("datetime").datetime(2024, 1, 1, tzinfo=__import__("datetime").timezone.utc),
        description="supporting event",
    )
    c.repo.upsert_entity("Event", ev)
    graph.add_entity("Event", ev.model_dump(mode="json"))
    edge = Assertion(
        id="ast_evidence_for_dec",
        subject_id="evt_evidence",
        predicate="EVIDENCE_FOR",
        object_id=did,
        confidence=0.95,
        status=AssertionStatus.accepted,
        source_ref="seed",
        created_at=ev.timestamp_start,
    )
    c.repo.upsert_assertion(edge)
    graph.add_assertion(edge)

    r2 = c.write_pipeline.run("We finalized the decision to launch Project Orion.", "s2")

    assert r2.superseded, "finalizing with evidence should supersede the draft status"
    assert not r2.quarantined
    edges = graph.out_edges(did, "HAS_STATUS")
    assert len(edges) == 1
    assert edges[0][1] == "status:final"
    assert graph.get_entity_payload(did)["status"] == "final"


# --------------------------------------------------------------------------- #
# Generalization to Organization status
# --------------------------------------------------------------------------- #
def _org_id(graph) -> str:
    orgs = [n for n in graph.node_ids() if graph.get_entity_type(n) == "Organization"]
    assert len(orgs) == 1
    return orgs[0]


def test_org_first_status_accepted_as_has_status():
    c = _container()
    c.write_pipeline.run("Organization Acme is active.", "s1")

    graph = c.graph
    oid = _org_id(graph)
    edges = graph.out_edges(oid, "HAS_STATUS")
    assert len(edges) == 1
    assert edges[0][1] == "status:active"
    assert graph.get_entity_payload(oid)["status"] == "active"
    # The org-status sentence must not also mint a spurious Person "Acme".
    assert not any(graph.get_entity_type(n) == "Person" for n in graph.node_ids())


def test_org_status_flip_as_new_fact_quarantines():
    c = _container()
    c.write_pipeline.run("Organization Acme is active.", "s1")
    graph = c.graph
    oid = _org_id(graph)
    accepted_aid = graph.out_edges(oid, "HAS_STATUS")[0][3]["assertion_id"]

    r2 = c.write_pipeline.run("Organization Acme is inactive.", "s2")  # new_fact

    assert len(r2.quarantined) == 1
    assert "status contradiction" in (r2.quarantined[0].reason or "").lower()
    assert graph.get_entity_payload(oid)["status"] == "active"  # not overwritten
    record = next(r for r in c.quarantine_store.list() if r.id == r2.quarantined[0].quarantine_id)
    assert accepted_aid in record.conflicting_ids


def test_org_status_update_supersedes():
    c = _container()
    c.write_pipeline.run("Organization Acme is active.", "s1")
    graph = c.graph
    oid = _org_id(graph)

    r2 = c.write_pipeline.run(
        "Organization Acme is inactive.", "s2", write_intent="update"
    )

    assert r2.superseded
    assert not r2.quarantined
    assert graph.get_entity_payload(oid)["status"] == "inactive"
    assert len(graph.out_edges(oid, "HAS_STATUS")) == 1

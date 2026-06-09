"""Regression tests for the post-build audit fixes (issues #1–#5).

Each test pins a previously-identified inconsistency so it cannot silently
regress:

1. Superseded assertions are re-tagged in the Vector_Index and drop out of
   accepted-only semantic retrieval (Req 10.3, 10.5, 16.2).
2. The agent loop does not write *question* turns back as memory (no spurious
   facts mined from interrogative phrasing) (Req 20.3).
3. The Mock_Extractor parses "X is inactive" so an ASSIGNED_TO to an inactive
   person is quarantined by C5 (Req 8.6).
4. The Mock_Extractor marks finalized decisions ``final`` and the write pipeline
   quarantines a final decision lacking evidence via C8 (Req 8.9).
5. The Mock_Extractor parses "X precedes Y" so a PRECEDES cycle is rejected by
   C3 (Req 8.4).
"""

from __future__ import annotations

import pytest

from ocm.agent.loop import AgentLoop
from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.extraction.mock_extractor import MockExtractor
from ocm.ontology.enums import AssertionStatus


def _container() -> CoreContainer:
    return CoreContainer(
        Settings(deterministic_test_mode=True, chroma_mode="memory", extractor="mock")
    )


def _rels(result) -> set[tuple[str, str, str]]:
    return {(r["subject"], r["predicate"], r["object"]) for r in result.relations}


# --------------------------------------------------------------------------- #
# #1 — superseded assertions are re-tagged in the vector index
# --------------------------------------------------------------------------- #
def test_supersede_retags_vector_index_and_excludes_from_accepted_retrieval():
    c = _container()
    c.write_pipeline.run("Bob is assigned to Task T1.", "s1")
    result = c.write_pipeline.run("Actually, Carol is assigned to Task T1.", "s2")

    # A supersession occurred (correction over a single-valued ASSIGNED_TO).
    assert result.superseded, "expected the correction to supersede the prior assignment"

    superseded = list(c.repo.list_assertions(status=AssertionStatus.superseded.value))
    assert superseded, "expected a superseded assertion row"
    old_id = superseded[0].id

    # The vector index metadata for the old assertion is now 'superseded'.
    meta = c.vector_index._get_metadata(old_id)
    assert meta is not None
    assert meta["status"] == AssertionStatus.superseded.value

    # A default accepted-only semantic query no longer returns the superseded item.
    hits = c.vector_index.query(
        "assigned to Task T1", where={"status": AssertionStatus.accepted.value}
    )
    assert old_id not in {h.memory_id for h in hits}


# --------------------------------------------------------------------------- #
# #2 — the agent loop does not write question turns as memory
# --------------------------------------------------------------------------- #
def test_agent_loop_does_not_write_question_turns():
    c = _container()
    loop = AgentLoop(c)

    result = loop.run_turn("Who owns Project Orion?")
    assert result.committed is False
    assert result.write_result is None

    # No spurious Person "Who" was mined from the question.
    person_names = {
        (c.graph.get_entity_payload(n) or {}).get("name")
        for n in c.graph.node_ids()
        if c.graph.get_entity_type(n) == "Person"
    }
    assert "Who" not in person_names

    # A declarative turn still writes memory.
    statement = loop.run_turn("Alice owns Project Orion.")
    assert statement.committed is True
    assert statement.write_result is not None
    assert any(o.candidate.predicate == "OWNS" for o in statement.write_result.accepted)


# --------------------------------------------------------------------------- #
# #3 — inactive person status + C5 quarantine
# --------------------------------------------------------------------------- #
def test_mock_extractor_parses_inactive_person_status():
    result = MockExtractor().extract("Mallory is inactive.", "doc::1")
    person = next(e for e in result.entities if e["type"] == "Person" and e["name"] == "Mallory")
    assert person["fields"]["status"] == "inactive"


def test_assigning_to_inactive_person_is_quarantined_by_c5():
    c = _container()
    c.write_pipeline.run("Mallory is inactive.", "s1")
    result = c.write_pipeline.run("Mallory is assigned to Task T7.", "s2")

    assert result.quarantined, "ASSIGNED_TO an inactive person should be quarantined (C5)"
    assert not result.accepted
    assert any("inactive" in (o.reason or "").lower() for o in result.quarantined)


# --------------------------------------------------------------------------- #
# #4 — final decisions + C8 evidence floor
# --------------------------------------------------------------------------- #
def test_mock_extractor_marks_finalized_decision_final():
    result = MockExtractor().extract("We finalized the decision to cancel Project Atlas.", "doc::1")
    assert result.decisions, "expected a decision to be extracted"
    assert result.decisions[0]["status"] == "final"


def test_final_decision_without_evidence_is_quarantined_by_c8():
    c = _container()
    result = c.write_pipeline.run("We finalized the decision to cancel Project Atlas.", "s1")

    assert result.quarantined, "a final decision lacking evidence should be quarantined (C8)"
    # The decision is not retained as an accepted graph node.
    assert not any(c.graph.get_entity_type(n) == "Decision" for n in c.graph.node_ids())


def test_draft_decision_is_accepted_without_evidence():
    c = _container()
    c.write_pipeline.run("We decided to launch Project Orion.", "s1")
    # A draft decision is persisted as an accepted graph node (C8 does not apply).
    assert any(c.graph.get_entity_type(n) == "Decision" for n in c.graph.node_ids())


# --------------------------------------------------------------------------- #
# #5 — PRECEDES extraction + C3 acyclicity
# --------------------------------------------------------------------------- #
def test_mock_extractor_parses_precedes_relation():
    result = MockExtractor().extract("Event Kickoff precedes Event Review.", "doc::1")
    assert ("Kickoff", "PRECEDES", "Review") in _rels(result)
    event_names = {e["name"] for e in result.events}
    assert {"Kickoff", "Review"} <= event_names


def test_precedes_cycle_is_rejected_by_c3():
    c = _container()
    first = c.write_pipeline.run("Event Kickoff precedes Event Review.", "s1")
    assert any(o.candidate.predicate == "PRECEDES" for o in first.accepted)

    # The cycle-closing edge is rejected; the PRECEDES projection stays acyclic.
    second = c.write_pipeline.run("Event Review precedes Event Kickoff.", "s2")
    assert second.rejected, "a PRECEDES cycle-closing edge should be rejected (C3)"
    assert c.graph.is_acyclic("PRECEDES")
    assert len(c.graph.find_edges_by_predicate("PRECEDES")) == 1

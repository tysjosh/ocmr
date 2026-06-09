"""Agent loop and Answer Policy unit tests (task 16.3).

Two halves, matching the two units under test:

* :class:`~ocm.agent.loop.AgentLoop` — assert that a single turn drives **both**
  memory operations: it queries memory during the turn (returning an
  :class:`EvidencePackage`, Req 20.2) and commits new memory at the end of the
  turn (returning a :class:`WriteResult` with an accepted ``OWNS``, Req 20.3).
  A follow-up turn confirms memory written on an earlier turn is recalled on a
  later one (the ``commit → receive`` next-turn edge).

* :class:`~ocm.agent.answer_policy.AnswerPolicy` — assert the P1–P5 rendering
  contract by constructing :class:`EvidencePackage` instances directly:
  - P1 (Req 21.1): lead with accepted supporting assertions / a derived answer.
  - P2/P3 (Req 21.2, 21.3): two conflicts render on separate labeled lines and
    are never merged.
  - P4 (Req 21.4): provenance (``source_ref``) appears only when
    ``high_stakes=True``.
  - P5 (Req 21.5): ``missing_information`` renders a missing-evidence section,
    and an empty package states no accepted assertions support the query.

Requirements: 20.2, 20.3, 21.2, 21.3, 21.4, 21.5.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ocm.agent.answer_policy import AnswerPolicy
from ocm.agent.loop import AgentLoop, TurnResult
from ocm.agent.memory_tool import MemoryTool
from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.memory.write_pipeline import WriteResult
from ocm.ontology.models import Provenance
from ocm.retrieval.evidence_packager import (
    ConflictItem,
    EvidencePackage,
    SupportingAssertion,
)
from ocm.retrieval.reranker import RankedItem


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def loop() -> AgentLoop:
    """An AgentLoop over a hermetic, deterministic, offline container."""
    settings = Settings(
        deterministic_test_mode=True, chroma_mode="memory", extractor="mock"
    )
    container = CoreContainer(settings)
    return AgentLoop(container)


def _accepted_predicates(result: WriteResult) -> set[str]:
    return {o.candidate.predicate for o in result.accepted}


# --------------------------------------------------------------------------- #
# AgentLoop — a turn triggers query then write (Req 20.2, 20.3)
# --------------------------------------------------------------------------- #
def test_run_turn_triggers_query_then_write(loop: AgentLoop) -> None:
    """One turn both retrieves (Req 20.2) and commits new memory (Req 20.3)."""
    result = loop.run_turn("Alice owns Project Orion.")

    assert isinstance(result, TurnResult)

    # retrieve happened: the turn produced a structured EvidencePackage (Req 20.2).
    assert isinstance(result.evidence, EvidencePackage)

    # commit happened: the turn wrote new memory and accepted the OWNS fact (Req 20.3).
    assert result.committed is True
    assert isinstance(result.write_result, WriteResult)
    assert "OWNS" in _accepted_predicates(result.write_result)

    # the write was tagged with the turn's source_ref provenance (Req 20.3).
    assert result.source_ref
    for outcome in result.write_result.accepted:
        assert outcome.candidate.source_ref == result.source_ref


def test_run_turn_uses_memory_tool_seam() -> None:
    """The loop drives memory only through the MemoryTool seam (Req 20.1)."""
    settings = Settings(
        deterministic_test_mode=True, chroma_mode="memory", extractor="mock"
    )
    container = CoreContainer(settings)
    tool = MemoryTool(container)

    calls = {"query": 0, "write": 0}
    real_query, real_write = tool.query, tool.write

    def spy_query(*args, **kwargs):
        calls["query"] += 1
        return real_query(*args, **kwargs)

    def spy_write(*args, **kwargs):
        calls["write"] += 1
        return real_write(*args, **kwargs)

    tool.query = spy_query  # type: ignore[assignment]
    tool.write = spy_write  # type: ignore[assignment]

    AgentLoop(tool).run_turn("Alice owns Project Orion.")

    # Exactly one retrieve and one commit per turn (Req 20.2, 20.3).
    assert calls["query"] == 1
    assert calls["write"] == 1


def test_second_turn_recalls_owner_written_earlier(loop: AgentLoop) -> None:
    """Memory written on an earlier turn is recalled on a later one (Req 20.2)."""
    loop.run_turn("Alice owns Project Orion.")

    # The retrieve in turn 2 sees turn 1's committed memory (retrieve precedes
    # turn 2's own write).
    second = loop.run_turn("Who owns Project Orion?")

    assert isinstance(second.evidence, EvidencePackage)
    assert second.evidence.retrieved_items, "expected the prior OWNS fact to be recalled"
    owns_items = [i for i in second.evidence.retrieved_items if i.predicate == "OWNS"]
    assert owns_items or (second.answer and "Alice" in second.answer)


# --------------------------------------------------------------------------- #
# AnswerPolicy P1 — prefer accepted supporting assertions (Req 21.1)
# --------------------------------------------------------------------------- #
def test_p1_leads_with_supporting_assertions() -> None:
    """P1: the rendered answer leads with accepted supporting assertions."""
    pkg = EvidencePackage(
        confidence=0.9,
        supporting_assertions=[SupportingAssertion(id="a-1", confidence=0.9)],
        retrieved_items=[
            RankedItem(
                memory_id="a-1",
                memory_type="assertion",
                status="accepted",
                score=0.9,
                text="Alice owns Project Orion",
            )
        ],
    )
    out = AnswerPolicy().render(pkg)

    assert out.startswith("Answer")
    assert "Alice owns Project Orion" in out
    assert "[a-1]" in out


# --------------------------------------------------------------------------- #
# AnswerPolicy P2 / P3 — surface conflicts, kept separate (Req 21.2, 21.3)
# --------------------------------------------------------------------------- #
def test_p2_p3_conflicts_surfaced_on_separate_lines() -> None:
    """P2/P3: two conflicts render on separate labeled lines, never merged."""
    pkg = EvidencePackage(
        conflicts=[
            ConflictItem(memory_id="a-1", text="Task T1 is done", status="accepted"),
            ConflictItem(
                memory_id="a-2", text="Task T1 is not started", status="quarantined"
            ),
        ],
    )
    out = AnswerPolicy().render(pkg)

    # P2 — conflicts are surfaced explicitly (never dropped).
    assert "Conflicts detected" in out

    # P3 — each conflicting claim is its own labeled line (not merged).
    claim_lines = [line for line in out.splitlines() if line.startswith("- Claim")]
    assert len(claim_lines) == 2
    assert any("Claim 1" in line and "Task T1 is done" in line for line in claim_lines)
    assert any(
        "Claim 2" in line and "Task T1 is not started" in line for line in claim_lines
    )

    # The two opposing claims never appear merged onto a single line.
    assert not any(
        "Task T1 is done" in line and "Task T1 is not started" in line
        for line in out.splitlines()
    )


# --------------------------------------------------------------------------- #
# AnswerPolicy P4 — include provenance when high-stakes (Req 21.4)
# --------------------------------------------------------------------------- #
def _package_with_provenance() -> EvidencePackage:
    return EvidencePackage(
        confidence=0.9,
        supporting_assertions=[SupportingAssertion(id="a-1", confidence=0.9)],
        supporting_sources=[
            Provenance(
                id="prov-1",
                subject_id="a-1",
                source_ref="src-42",
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )
        ],
        retrieved_items=[
            RankedItem(
                memory_id="a-1",
                memory_type="assertion",
                status="accepted",
                score=0.9,
                text="Alice owns Project Orion",
            )
        ],
    )


def test_p4_high_stakes_includes_provenance() -> None:
    """P4: high_stakes=True attaches the supporting source_ref provenance."""
    out = AnswerPolicy().render(_package_with_provenance(), high_stakes=True)
    assert "Provenance:" in out
    assert "src-42" in out


def test_p4_non_high_stakes_omits_provenance() -> None:
    """P4: high_stakes=False omits provenance from the answer."""
    out = AnswerPolicy().render(_package_with_provenance(), high_stakes=False)
    assert "Provenance:" not in out
    assert "src-42" not in out


# --------------------------------------------------------------------------- #
# AnswerPolicy P5 — state missing evidence (Req 21.5)
# --------------------------------------------------------------------------- #
def test_p5_renders_missing_information_section() -> None:
    """P5: a package with missing_information renders a missing-evidence section."""
    pkg = EvidencePackage(
        confidence=0.9,
        supporting_assertions=[SupportingAssertion(id="a-1", confidence=0.9)],
        missing_information=[
            "No provenance records were found for the supporting assertions."
        ],
        retrieved_items=[
            RankedItem(
                memory_id="a-1",
                memory_type="assertion",
                status="accepted",
                score=0.9,
                text="Alice owns Project Orion",
            )
        ],
    )
    out = AnswerPolicy().render(pkg)

    assert "Missing evidence:" in out
    assert "No provenance records were found" in out


def test_p5_no_support_states_no_accepted_assertions() -> None:
    """P5: an empty package states no accepted assertions support the query."""
    out = AnswerPolicy().render(EvidencePackage())
    assert "No accepted assertions support this query." in out

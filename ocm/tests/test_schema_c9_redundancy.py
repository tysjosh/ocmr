"""W5 (typed schema) is subsumed by C9 (graph domain/range) at the outcome level.

This pins the *ablation finding* reported in the paper (§V-C): disabling the
W5 schema gate (``enable_schema_validation=False``, the ``no_schema`` ablation)
does **not** change what the governed write path admits, because every W5
rejection class is also caught downstream by constraint C9 at W6 — only the
attributed check name differs (``schema.*`` vs ``C9``), never the accept/reject
outcome.

Concretely W5's checks are each backstopped or enforced upstream:

* required fields / dangling entity refs -> C9 "cannot resolve entity type";
* unregistered predicate -> C9 "unknown predicate";
* confidence bounds -> enforced upstream by ``CandidateAssertion`` (confloat);
* status enum / static signature -> defensive no-ops on a real candidate.

So ``no_schema`` ties full OCMR on the decisive metrics by construction, not by
benchmark accident. These tests guard that claim so it can be cited.
"""

from __future__ import annotations

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.memory.contracts import CandidateAssertion, ExtractionResult
from ocm.memory.graph_store import GraphStore
from ocm.validation.constraints import c9_graph_domain_range
from ocm.validation.schema_validator import SchemaValidator


def _candidate(subject_id: str, predicate: str, object_id: str) -> CandidateAssertion:
    return CandidateAssertion(
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        confidence=0.9,
        source_ref="s",
    )


def test_w5_rejections_are_backstopped_by_c9():
    """Every candidate W5 rejects is also rejected by C9 (outcome-equivalent)."""
    graph = GraphStore()  # empty graph: all entity references dangle
    sv = SchemaValidator()

    cases = [
        # dangling entity references (W5 check 5) -> C9 cannot resolve types
        _candidate("per_x", "OWNS", "prj_y"),
        # unregistered predicate (W5 check 2) -> C9 unknown predicate
        _candidate("per_x", "COMPLETED", "tas_y"),
    ]
    for c in cases:
        w5 = sv.validate(c, graph)
        c9 = c9_graph_domain_range(c, graph)
        # W5 rejects (schema on)...
        assert w5.valid is False
        assert str(w5.failed_check).startswith("schema.")
        # ...and C9 independently rejects the same candidate (schema off).
        assert c9.valid is False
        assert c9.failed_check == "C9"


class _StubExtractor:
    """Emits a fixed extraction (reversed ASSIGNED_TO + unknown predicate)."""

    version = "stub-1"

    def extract(self, text: str, source_ref: str = "") -> ExtractionResult:
        return ExtractionResult(
            entities=[
                {"type": "Person", "name": "Bob", "fields": {}},
                {"type": "Task", "name": "T1", "fields": {}},
            ],
            events=[],
            claims=[],
            documents=[],
            decisions=[],
            relations=[
                # reversed direction (Person subject) — only valid object is Person
                {"subject": "Bob", "predicate": "ASSIGNED_TO", "object": "T1",
                 "confidence": 0.9, "write_intent": "new_fact"},
                # unregistered predicate
                {"subject": "Bob", "predicate": "COMPLETED", "object": "T1",
                 "confidence": 0.9, "write_intent": "new_fact"},
            ],
            extractor_version="stub-1",
        )


def _run(enable_schema: bool):
    settings = Settings(
        deterministic_test_mode=True,
        chroma_mode="memory",
        enable_schema_validation=enable_schema,
    )
    container = CoreContainer(settings, extractor=_StubExtractor())
    return container.write_pipeline.run("Bob is assigned to and completed T1.", "s1")


def test_no_schema_matches_full_admitted_set_end_to_end():
    """`no_schema` (W5 off) admits exactly what full OCMR (W5 on) admits."""
    full = _run(enable_schema=True)
    no_schema = _run(enable_schema=False)

    def counts(r):
        return (
            r.summary.num_accepted,
            r.summary.num_superseded,
            r.summary.num_quarantined,
            r.summary.num_rejected,
        )

    # Identical write outcomes: the schema ablation changes which gate fires,
    # never whether the candidate is admitted (the §V-C redundancy finding).
    assert counts(full) == counts(no_schema)
    # And specifically: both bad candidates are rejected under both settings.
    assert full.summary.num_accepted == 0
    assert no_schema.summary.num_accepted == 0

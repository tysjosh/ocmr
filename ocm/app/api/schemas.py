"""Request/response models for the OCM API service (Req 19.1).

The ``API_Service`` is a FastAPI app whose routers are thin: they validate the
request model, call into the :class:`~ocm.core.container.CoreContainer`, and
serialize the result. These Pydantic v2 models are the request/response bodies
for the five endpoints (task 15.2), reusing the ontology and pipeline-contract
models wherever possible so the API shape mirrors the internal contracts.

Endpoints (design "API Design"):

1. ``POST /memory/write``           — :class:`WriteRequest`  -> :class:`WriteResponse`
2. ``POST /memory/query``           — :class:`QueryRequest`  -> :class:`QueryResponse`
3. ``POST /memory/validate``        — :class:`ValidateRequest` -> :class:`ValidateResponse`
4. ``GET  /memory/entity/{id}``     — :class:`EntityResponse`
5. ``GET  /memory/conflicts``       — :class:`ConflictsResponse`

Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 19.6.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ocm.memory.contracts import (
    CandidateAssertion,
    WriteOutcome,
    WriteSummary,
)
from ocm.ontology.enums import Severity, WriteIntent
from ocm.ontology.models import Assertion, Provenance, QuarantineRecord
from ocm.retrieval.evidence_packager import ConflictItem, SupportingAssertion
from ocm.retrieval.reranker import RankedItem

__all__ = [
    "WriteRequest",
    "WriteResponse",
    "QueryRequest",
    "QueryResponse",
    "ValidateRequest",
    "ValidateResponse",
    "EntityResponse",
    "ConflictsResponse",
]


# --------------------------------------------------------------------------- #
# 1. POST /memory/write (Req 19.2)
# --------------------------------------------------------------------------- #
class WriteRequest(BaseModel):
    """Body for ``POST /memory/write``.

    Runs the full write pipeline (W1–W8) over ``text`` from ``source_ref`` under
    the given ``write_intent`` (defaulting to ``new_fact``).
    """

    text: str
    source_ref: str
    write_intent: WriteIntent = WriteIntent.new_fact
    extractor_version: Optional[str] = None


class WriteResponse(BaseModel):
    """Result of a write run — the four outcome lists plus the summary (Req 19.2).

    Mirrors :class:`~ocm.memory.write_pipeline.WriteResult` /
    :class:`~ocm.memory.contracts.WriteSummary`. The summary carries exactly
    ``num_candidates``, ``num_accepted``, ``num_quarantined``, ``num_rejected``,
    and ``num_superseded`` (Req 19.2).
    """

    accepted: list[WriteOutcome] = Field(default_factory=list)
    superseded: list[WriteOutcome] = Field(default_factory=list)
    quarantined: list[WriteOutcome] = Field(default_factory=list)
    rejected: list[WriteOutcome] = Field(default_factory=list)
    summary: WriteSummary


# --------------------------------------------------------------------------- #
# 2. POST /memory/query (Req 19.3)
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    """Body for ``POST /memory/query``.

    Runs the retrieval pipeline (R0–R4) over ``query``. ``include_conflicts``
    forces quarantined items into the semantic results even for a non-conflict
    query.
    """

    query: str
    top_k: int = 5
    include_conflicts: bool = False


class QueryResponse(BaseModel):
    """The evidence-package fields returned by retrieval (Req 19.3, 18.1).

    Mirrors :class:`~ocm.retrieval.evidence_packager.EvidencePackage`, adding the
    R0 ``query_type``. ``retrieved_items`` holds the symbolic + semantic results
    merged and reranked into a single ordered set (Req 28.7); ``answer`` is
    optional (Req 18.5).
    """

    query_type: str
    answer: Optional[str] = None
    confidence: float = 0.0
    supporting_assertions: list[SupportingAssertion] = Field(default_factory=list)
    supporting_sources: list[Provenance] = Field(default_factory=list)
    conflicts: list[ConflictItem] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    retrieved_items: list[RankedItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 3. POST /memory/validate (Req 19.4)
# --------------------------------------------------------------------------- #
class ValidateRequest(BaseModel):
    """Body for ``POST /memory/validate``.

    Carries a fully-formed :class:`~ocm.memory.contracts.CandidateAssertion` to
    run W5→W6→W7 against, **without committing** anything (Req 19.4).
    """

    candidate: CandidateAssertion


class ValidateResponse(BaseModel):
    """Validation verdict for a candidate, with no side effects (Req 19.4).

    Reports whether the candidate is structurally/constraint ``valid``, the
    routing ``decision`` the commit manager would take, and the ``reason`` /
    ``severity`` behind it — without writing to graph/repo/vector/quarantine.
    """

    valid: bool
    decision: Literal["accept", "supersede", "quarantine", "reject"]
    reason: Optional[str] = None
    severity: Optional[Severity] = None
    failed_check: Optional[str] = None
    conflicting_ids: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 4. GET /memory/entity/{id} (Req 19.5)
# --------------------------------------------------------------------------- #
class EntityResponse(BaseModel):
    """An entity plus the assertions it participates in (Req 19.5).

    ``entity`` is the typed entity payload, ``entity_type`` its ontology type,
    and ``assertions`` every accepted assertion where the entity is the subject
    or the object.
    """

    entity: dict
    entity_type: str
    assertions: list[Assertion] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 5. GET /memory/conflicts (Req 19.6)
# --------------------------------------------------------------------------- #
class ConflictsResponse(BaseModel):
    """Unresolved conflicts and quarantined candidates (Req 19.6).

    ``unresolved_conflicts`` is the curated conflict view; ``quarantined_candidates``
    are the raw persisted :class:`~ocm.ontology.models.QuarantineRecord` rows from
    the Quarantine_Store.
    """

    unresolved_conflicts: list[ConflictItem] = Field(default_factory=list)
    quarantined_candidates: list[QuarantineRecord] = Field(default_factory=list)

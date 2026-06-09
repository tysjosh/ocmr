"""The five memory endpoints (task 15.2).

This :class:`~fastapi.APIRouter` exposes the ``API_Service`` surface described in
the design's "API Design" section. The routers are intentionally thin: each one
validates its request model, resolves the wired
:class:`~ocm.core.container.CoreContainer` from the request's application state,
calls into the appropriate pipeline/validator, and serializes the result into a
response model from :mod:`ocm.app.api.schemas`.

Endpoints:

1. ``POST /memory/write``        — run the write pipeline (W1–W8) (Req 19.2, 28.1).
2. ``POST /memory/query``        — run the retrieval pipeline (R0–R4) (Req 19.3, 28.2, 28.7).
3. ``POST /memory/validate``     — validate a candidate without committing (Req 19.4).
4. ``GET  /memory/entity/{id}``  — fetch an entity and its assertions (Req 19.5).
5. ``GET  /memory/conflicts``    — list unresolved conflicts / quarantines (Req 19.6).

The container is stored on ``app.state.container`` by the application factory
(:func:`ocm.app.main.create_app`); :func:`get_container` is the FastAPI
dependency that reads it back, keeping endpoints decoupled from construction.

Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 28.1, 28.2, 28.7.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ocm.app.api.schemas import (
    ConflictsResponse,
    EntityResponse,
    QueryRequest,
    QueryResponse,
    ValidateRequest,
    ValidateResponse,
    WriteRequest,
    WriteResponse,
)
from ocm.core.container import CoreContainer
from ocm.ontology.enums import QuarantineStatus
from ocm.retrieval.evidence_packager import ConflictItem

router = APIRouter()


# --------------------------------------------------------------------------- #
# Dependency: resolve the wired container from application state
# --------------------------------------------------------------------------- #
def get_container(request: Request) -> CoreContainer:
    """Return the :class:`CoreContainer` wired onto the application state.

    The application factory (:func:`ocm.app.main.create_app`) constructs a
    single container per process (or per test) and stores it on
    ``app.state.container``; this dependency hands it to each endpoint so the
    routers never construct components themselves (design "Dependency Wiring").
    """
    container = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - defensive; factory always sets it
        raise HTTPException(status_code=500, detail="CoreContainer is not configured")
    return container


# --------------------------------------------------------------------------- #
# 1. POST /memory/write (Req 19.2, 28.1)
# --------------------------------------------------------------------------- #
@router.post("/memory/write", response_model=WriteResponse)
def write_memory(
    req: WriteRequest, c: CoreContainer = Depends(get_container)
) -> WriteResponse:
    """Run the full write pipeline (W1–W8) and return the outcome lists + summary.

    The four mutually-exclusive outcome buckets (``accepted`` / ``superseded`` /
    ``quarantined`` / ``rejected``) and the rolled-up :class:`WriteSummary`
    mirror :class:`~ocm.memory.write_pipeline.WriteResult` (Req 19.2).
    """
    result = c.write_pipeline.run(
        req.text,
        req.source_ref,
        req.write_intent,
        req.extractor_version,
    )
    return WriteResponse(
        accepted=result.accepted,
        superseded=result.superseded,
        quarantined=result.quarantined,
        rejected=result.rejected,
        summary=result.summary,
    )


# --------------------------------------------------------------------------- #
# 2. POST /memory/query (Req 19.3, 28.2, 28.7)
# --------------------------------------------------------------------------- #
@router.post("/memory/query", response_model=QueryResponse)
def query_memory(
    req: QueryRequest, c: CoreContainer = Depends(get_container)
) -> QueryResponse:
    """Run the retrieval pipeline (R0–R4) and return the evidence-package fields.

    ``query_type`` comes from the R0 classifier; the symbolic and semantic
    results are merged and reranked into a single ordered ``retrieved_items``
    list (Req 28.7). The pipeline classifies internally, but we classify once
    here for the response field so the API exposes the R0 verdict without
    reaching into the package internals.
    """
    classification = c.query_classifier.classify(req.query)
    package = c.retrieval_pipeline.query(
        req.query,
        top_k=req.top_k,
        include_conflicts=req.include_conflicts,
    )
    return QueryResponse(
        query_type=classification.query_type,
        answer=package.answer,
        confidence=package.confidence,
        supporting_assertions=package.supporting_assertions,
        supporting_sources=package.supporting_sources,
        conflicts=package.conflicts,
        missing_information=package.missing_information,
        retrieved_items=package.retrieved_items,
    )


# --------------------------------------------------------------------------- #
# 3. POST /memory/validate (Req 19.4)
# --------------------------------------------------------------------------- #
@router.post("/memory/validate", response_model=ValidateResponse)
def validate_candidate(
    req: ValidateRequest, c: CoreContainer = Depends(get_container)
) -> ValidateResponse:
    """Validate a candidate (W5→W6→W7) **without committing** anything (Req 19.4).

    Runs the structural Schema Validator first; only if it passes does the
    graph-level Constraint Validator (which binds the contradiction gate, W7)
    run. No writes touch the graph/repo/vector/quarantine stores. The routing
    ``decision`` is derived from the validation result's ``recommended_action``
    (defaulting to ``accept`` when valid, otherwise ``reject``).
    """
    result = c.schema_validator.validate(req.candidate, c.graph)
    if result.valid:
        result = c.constraint_validator.validate(
            req.candidate, c.graph, settings=c.settings
        )

    decision = "accept" if result.valid else (result.recommended_action or "reject")
    return ValidateResponse(
        valid=result.valid,
        decision=decision,
        reason=result.reason,
        severity=result.severity,
        failed_check=result.failed_check,
        conflicting_ids=list(result.conflicting_ids or []),
    )


# --------------------------------------------------------------------------- #
# 4. GET /memory/entity/{entity_id} (Req 19.5)
# --------------------------------------------------------------------------- #
@router.get("/memory/entity/{entity_id}", response_model=EntityResponse)
def get_entity(
    entity_id: str, c: CoreContainer = Depends(get_container)
) -> EntityResponse:
    """Return an entity and every accepted assertion it participates in (Req 19.5).

    The entity payload and type are read from the in-memory Graph_Store (the
    accepted-only projection of durable storage); a missing entity yields 404.
    Associated assertions are gathered from the graph edges where the entity is
    the subject or the object, then resolved to full :class:`Assertion` rows via
    the repository.
    """
    payload = c.graph.get_entity_payload(entity_id)
    entity_type = c.graph.get_entity_type(entity_id)
    if payload is None or entity_type is None:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id!r} not found")

    assertions = _gather_assertions(c, entity_id)
    return EntityResponse(
        entity=payload,
        entity_type=entity_type,
        assertions=assertions,
    )


# --------------------------------------------------------------------------- #
# 5. GET /memory/conflicts (Req 19.6)
# --------------------------------------------------------------------------- #
@router.get("/memory/conflicts", response_model=ConflictsResponse)
def get_conflicts(c: CoreContainer = Depends(get_container)) -> ConflictsResponse:
    """Return unresolved conflicts and quarantined candidates (Req 19.6).

    The raw, persisted :class:`QuarantineRecord` rows for unresolved candidates
    are returned in ``quarantined_candidates`` straight from the
    Quarantine_Store; a curated :class:`ConflictItem` view of the same records
    is returned in ``unresolved_conflicts``.
    """
    records = c.quarantine_store.list(QuarantineStatus.unresolved)
    unresolved_conflicts = [
        ConflictItem(
            memory_id=record.id,
            memory_type="quarantine",
            status=record.status.value if hasattr(record.status, "value") else record.status,
            reason=record.reason,
            conflicting_ids=list(record.conflicting_ids or []),
            severity=record.severity.value if hasattr(record.severity, "value") else record.severity,
        )
        for record in records
    ]
    return ConflictsResponse(
        unresolved_conflicts=unresolved_conflicts,
        quarantined_candidates=list(records),
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _gather_assertions(c: CoreContainer, entity_id: str):
    """Collect accepted :class:`Assertion` rows touching ``entity_id``.

    Walks the Graph_Store's outgoing and incoming edges for the entity (the
    accepted-only projection), collecting each edge's ``assertion_id`` and
    resolving it to the durable row via the repository. De-duplicates while
    preserving discovery order so an entity that is both subject and object of
    the same assertion is reported once.
    """
    seen: set[str] = set()
    assertions = []
    edges = c.graph.out_edges(entity_id) + c.graph.in_edges(entity_id)
    for _s, _o, _k, data in edges:
        assertion_id = data.get("assertion_id")
        if assertion_id is None or assertion_id in seen:
            continue
        seen.add(assertion_id)
        assertion = c.repo.get_assertion(assertion_id)
        if assertion is not None:
            assertions.append(assertion)
    return assertions

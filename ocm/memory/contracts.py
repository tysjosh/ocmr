"""Pipeline contract models exchanged between write/resolution stages.

Beyond the persisted ontology models (``ocm/ontology/models.py``), the pipeline
stages exchange these typed Pydantic v2 contracts. They make the
``candidate -> result -> outcome`` flow explicit and validated at every boundary.

- :class:`ExtractionResult` — raw typed dicts emitted by an extractor (W1).
- :class:`ResolutionOutcome` — entity resolution result (W3).
- :class:`CandidateAssertion` — the proposed write unit built by W4.
- :class:`ValidationResult` — schema/constraint validation verdict (W5, W6).
- :class:`ContradictionResult` — contradiction-checker verdict (W7).
- :class:`WriteOutcome` / :class:`WriteSummary` — commit-manager results (W8).

Confidence uses ``confloat(ge=0.0, le=1.0)`` so the [0, 1] bound is enforced
structurally (Req 8.7, supports constraint C6).

Requirements: 6.1, 6.2, 8.1, 9.7, 10.1, 19.2.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, confloat

from ocm.ontology.enums import ResolutionStatus, Severity, WriteIntent


class ExtractionResult(BaseModel):
    """Raw, typed output of an extractor (W1).

    Items are kept as plain dicts here and validated into their concrete entity
    models downstream during normalization/resolution.
    """

    entities: list[dict]
    events: list[dict]
    claims: list[dict]
    documents: list[dict]
    decisions: list[dict]
    relations: list[dict]  # {subject, predicate, object, confidence, write_intent?}
    extractor_version: str


class ResolutionOutcome(BaseModel):
    """Entity resolution result for a single extracted mention (W3)."""

    resolution_status: ResolutionStatus
    entity_id: str | None
    candidate_matches: list[str] = []


class CandidateAssertion(BaseModel):
    """A proposed assertion to be validated and committed (built by W4)."""

    operation: Literal["upsert_assertion"] = "upsert_assertion"
    subject_id: str
    predicate: str
    object_id: str
    confidence: confloat(ge=0.0, le=1.0)
    source_ref: str
    write_intent: WriteIntent = WriteIntent.new_fact
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    extractor_version: str | None = None


class ValidationResult(BaseModel):
    """Schema/constraint validation verdict for a candidate (W5, W6)."""

    valid: bool
    failed_check: str | None = None  # which schema/constraint failed
    reason: str | None = None
    severity: Severity | None = None
    conflicting_ids: list[str] = []
    recommended_action: Literal["accept", "quarantine", "reject", "supersede"] | None = None


class ContradictionResult(BaseModel):
    """Contradiction-checker verdict for a candidate (W7)."""

    has_conflict: bool
    severity: Severity | None = None
    reason: str | None = None
    conflicting_assertion_ids: list[str] = []
    kind: Literal["hard", "soft", "temporal"] | None = None
    recommended_action: Literal["accept", "quarantine", "supersede"] | None = None


class WriteOutcome(BaseModel):
    """The committed result for a single candidate assertion (W8)."""

    candidate: CandidateAssertion
    decision: Literal["accepted", "superseded", "quarantined", "rejected"]
    assertion_id: str | None = None
    quarantine_id: str | None = None
    superseded_assertion_id: str | None = None
    reason: str | None = None


class WriteSummary(BaseModel):
    """Aggregate counts for a batch write run (W8)."""

    num_candidates: int
    num_accepted: int
    num_quarantined: int
    num_rejected: int
    num_superseded: int

"""Pydantic v2 ontology entity and assertion models.

This module is the single source of truth for memory item structure (Req 1.12).
Confidence fields use ``confloat(ge=0.0, le=1.0)`` so the [0, 1] bound is enforced
structurally (Req 1.6, 1.9; supports constraint C6). Status defaulting and the
``status_defaulted`` metadata flag are provided by :class:`StatusDefaultMixin`
(Req 1.13, 1.15). ``Event`` and ``Document`` deliberately omit a status field and
do not use the mixin (Req 1.14).
"""

from datetime import datetime

from pydantic import BaseModel, Field, confloat, model_validator

from ocm.ontology.enums import (
    AssertionStatus,
    ClaimStatus,
    DecisionStatus,
    OrgStatus,
    PersonStatus,
    Priority,
    ProjectStatus,
    QuarantineStatus,
    Severity,
    TaskStatus,
    WriteIntent,
)


class StatusDefaultMixin(BaseModel):
    """Default a missing/None ``status`` to ``unknown`` and record that it was defaulted.

    Implements default-to-``unknown`` (Req 1.13) and records the defaulting in
    ``status_defaulted`` as the "WHERE metadata is available" record (Req 1.15).
    """

    status_defaulted: bool = Field(default=False, exclude=False)

    @model_validator(mode="before")
    @classmethod
    def _default_status(cls, data):
        if isinstance(data, dict) and ("status" not in data or data.get("status") is None):
            data = {**data, "status": "unknown", "status_defaulted": True}
        return data


class Person(StatusDefaultMixin):
    id: str
    name: str
    roles: list[str] = []
    status: PersonStatus = PersonStatus.unknown
    aliases: list[str] = []


class Organization(StatusDefaultMixin):
    id: str
    name: str
    type: str
    status: OrgStatus = OrgStatus.unknown


class Project(StatusDefaultMixin):
    id: str
    name: str
    goal: str | None = None
    status: ProjectStatus = ProjectStatus.unknown
    owner_id: str | None = None


class Task(StatusDefaultMixin):
    id: str
    title: str
    status: TaskStatus = TaskStatus.unknown
    priority: Priority = Priority.unknown
    project_id: str | None = None
    assignee_id: str | None = None
    due_at: datetime | None = None


class Event(BaseModel):  # no status field (Req 1.14)
    id: str
    type: str
    timestamp_start: datetime
    timestamp_end: datetime | None = None
    description: str


class Claim(StatusDefaultMixin):
    id: str
    text: str
    source_ref: str
    confidence: confloat(ge=0.0, le=1.0)
    status: ClaimStatus = ClaimStatus.unknown
    created_at: datetime


class Document(BaseModel):  # no status field (Req 1.14)
    id: str
    title: str
    path_or_url: str
    created_at: datetime
    tags: list[str] = []


class Decision(StatusDefaultMixin):
    id: str
    summary: str
    timestamp: datetime
    made_by: str | None = None
    rationale: str | None = None
    status: DecisionStatus = DecisionStatus.unknown


class StatusValue(BaseModel):  # no status field — it *is* a status value node
    """A first-class status value node (e.g. ``done``), the object of HAS_STATUS.

    Promoting a Task's status to a ``HAS_STATUS(task -> StatusValue)`` assertion
    makes a status flip an assertion-to-assertion contradiction: the quarantined
    flip can point its ``conflicting_ids`` at the *accepted* status assertion, so
    a plain status query surfaces the paired ``{accepted, quarantined}`` conflict
    instead of silently collapsing it. ``id`` is the canonical ``status:<value>``
    node id shared across entities; ``name`` mirrors ``value`` for label lookups.
    """

    id: str
    value: str
    name: str = ""


class Assertion(StatusDefaultMixin):
    id: str
    subject_id: str
    predicate: str
    object_id: str
    confidence: confloat(ge=0.0, le=1.0)
    status: AssertionStatus = AssertionStatus.unknown
    source_ref: str
    created_at: datetime
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    extractor_version: str | None = None
    write_intent: WriteIntent = WriteIntent.new_fact


class QuarantineRecord(BaseModel):
    id: str
    candidate_payload: dict  # serialized candidate assertion/entity
    reason: str
    severity: Severity
    conflicting_ids: list[str] = []
    created_at: datetime
    status: QuarantineStatus = QuarantineStatus.unresolved


class Provenance(BaseModel):
    id: str
    subject_id: str  # id of the assertion/claim/document/quarantine record
    source_ref: str
    created_at: datetime
    extractor_version: str | None = None
    supporting_evidence_ids: list[str] = []

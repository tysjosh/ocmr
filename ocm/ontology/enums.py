"""Ontology enumerations.

All status/priority/severity/intent/resolution fields are ``str`` enums so that
out-of-enum values raise a ``ValidationError`` (Req 1.11). Every status enum
includes ``unknown`` to support default-to-unknown behavior (Req 1.13).
"""

from enum import Enum


class PersonStatus(str, Enum):
    active = "active"
    inactive = "inactive"
    unknown = "unknown"


class OrgStatus(str, Enum):
    active = "active"
    inactive = "inactive"
    unknown = "unknown"


class ProjectStatus(str, Enum):
    active = "active"
    inactive = "inactive"
    completed = "completed"
    cancelled = "cancelled"
    unknown = "unknown"


class TaskStatus(str, Enum):
    todo = "todo"
    in_progress = "in_progress"
    blocked = "blocked"
    done = "done"
    cancelled = "cancelled"
    unknown = "unknown"


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"
    unknown = "unknown"


class ClaimStatus(str, Enum):
    accepted = "accepted"
    rejected = "rejected"
    quarantined = "quarantined"
    superseded = "superseded"
    unknown = "unknown"


class DecisionStatus(str, Enum):
    draft = "draft"
    final = "final"
    superseded = "superseded"
    rejected = "rejected"
    unknown = "unknown"


class AssertionStatus(str, Enum):
    accepted = "accepted"
    rejected = "rejected"
    quarantined = "quarantined"
    superseded = "superseded"
    unknown = "unknown"


class WriteIntent(str, Enum):
    new_fact = "new_fact"
    update = "update"
    correction = "correction"
    deletion = "deletion"
    hypothesis = "hypothesis"


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class QuarantineStatus(str, Enum):
    unresolved = "unresolved"
    resolved = "resolved"
    dismissed = "dismissed"


class ResolutionStatus(str, Enum):
    resolved_existing = "resolved_existing"
    created_new = "created_new"
    possible_match = "possible_match"
    unresolved = "unresolved"

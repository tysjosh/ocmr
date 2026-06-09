"""W2 — Normalizer.

The Normalizer canonicalizes the raw, typed dicts emitted by an extractor (W1)
into consistent representations before entity resolution (W3) sees them. It
performs **value-level** normalization only and is deliberately *conservative*:
it never collapses two distinct entities just because their normalized forms are
close — merging is exclusively the Entity Resolver's job (Req 4.7).

Responsibilities (Req 4.1-4.7):

- Names / aliases -> canonical form: trim, collapse internal whitespace, and
  apply consistent casing, **without** merging distinct entities (Req 4.1, 4.7).
- Timestamps -> ISO-8601 UTC strings (Req 4.2).
- Status synonyms -> canonical enum value, including ``"completed" -> "done"``
  for tasks (Req 4.3).
- Priority synonyms -> canonical enum value, including ``"high priority" ->
  "high"`` (Req 4.4).
- Relation names -> canonical predicate identifiers, e.g. ``"assigned to" ->
  "ASSIGNED_TO"`` (Req 4.5).
- Confidence -> float in ``[0, 1]``, parsing textual/percentage confidences and
  clamping out-of-range numerics (Req 4.6).

The Normalizer returns a brand-new :class:`ExtractionResult`; it does not mutate
its input.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone

from ocm.memory.contracts import ExtractionResult
from ocm.ontology.relations import RELATION_SIGNATURES

# --------------------------------------------------------------------------- #
# Synonym maps
# --------------------------------------------------------------------------- #

# Status synonyms keyed by entity type. Keys are lowercased/whitespace-collapsed
# synonyms; values are the canonical enum *values* declared in
# ``ocm.ontology.enums``. Type-aware mapping is required because "completed"
# canonicalizes to "done" for a Task but stays "completed" for a Project
# (Req 4.3).
_STATUS_SYNONYMS: dict[str, dict[str, str]] = {
    "Task": {
        "todo": "todo",
        "to do": "todo",
        "to-do": "todo",
        "not started": "todo",
        "backlog": "todo",
        "pending": "todo",
        "open": "todo",
        "new": "todo",
        "in progress": "in_progress",
        "in-progress": "in_progress",
        "inprogress": "in_progress",
        "ongoing": "in_progress",
        "started": "in_progress",
        "active": "in_progress",
        "wip": "in_progress",
        "doing": "in_progress",
        "blocked": "blocked",
        "stuck": "blocked",
        "on hold": "blocked",
        "waiting": "blocked",
        "done": "done",
        "complete": "done",
        "completed": "done",
        "finished": "done",
        "closed": "done",
        "resolved": "done",
        "cancelled": "cancelled",
        "canceled": "cancelled",
        "abandoned": "cancelled",
        "dropped": "cancelled",
        "unknown": "unknown",
    },
    "Project": {
        "active": "active",
        "in progress": "active",
        "ongoing": "active",
        "started": "active",
        "inactive": "inactive",
        "paused": "inactive",
        "on hold": "inactive",
        "completed": "completed",
        "complete": "completed",
        "done": "completed",
        "finished": "completed",
        "cancelled": "cancelled",
        "canceled": "cancelled",
        "abandoned": "cancelled",
        "unknown": "unknown",
    },
    "Person": {
        "active": "active",
        "current": "active",
        "employed": "active",
        "inactive": "inactive",
        "departed": "inactive",
        "left": "inactive",
        "former": "inactive",
        "terminated": "inactive",
        "unknown": "unknown",
    },
    "Organization": {
        "active": "active",
        "operating": "active",
        "inactive": "inactive",
        "defunct": "inactive",
        "dissolved": "inactive",
        "closed": "inactive",
        "unknown": "unknown",
    },
    "Decision": {
        "draft": "draft",
        "proposed": "draft",
        "pending": "draft",
        "final": "final",
        "finalized": "final",
        "approved": "final",
        "accepted": "final",
        "confirmed": "final",
        "superseded": "superseded",
        "replaced": "superseded",
        "rejected": "rejected",
        "declined": "rejected",
        "unknown": "unknown",
    },
}

# Fallback status synonyms used when the entity type is unknown. Uses the
# Task-oriented mapping so the headline "completed" -> "done" rule (Req 4.3)
# still applies.
_STATUS_SYNONYMS_DEFAULT: dict[str, str] = _STATUS_SYNONYMS["Task"]

# Priority synonyms -> canonical Priority enum value (Req 4.4).
_PRIORITY_SYNONYMS: dict[str, str] = {
    "low": "low",
    "low priority": "low",
    "minor": "low",
    "p3": "low",
    "p4": "low",
    "medium": "medium",
    "med": "medium",
    "medium priority": "medium",
    "normal": "medium",
    "moderate": "medium",
    "default": "medium",
    "p2": "medium",
    "high": "high",
    "high priority": "high",
    "important": "high",
    "major": "high",
    "p1": "high",
    "urgent": "urgent",
    "urgent priority": "urgent",
    "critical": "urgent",
    "highest": "urgent",
    "blocker": "urgent",
    "p0": "urgent",
    "unknown": "unknown",
}

# Relation-name synonyms -> canonical predicate identifier (Req 4.5). Keys are
# lowercased/whitespace-collapsed. Canonical predicates themselves (already
# uppercase) are recognized directly in :func:`_normalize_predicate`.
_PREDICATE_SYNONYMS: dict[str, str] = {
    "owns": "OWNS",
    "own": "OWNS",
    "owner of": "OWNS",
    "is owner of": "OWNS",
    "assigned to": "ASSIGNED_TO",
    "assigned": "ASSIGNED_TO",
    "assignee": "ASSIGNED_TO",
    "is assigned to": "ASSIGNED_TO",
    "results in": "RESULTS_IN",
    "result in": "RESULTS_IN",
    "leads to": "RESULTS_IN",
    "causes": "RESULTS_IN",
    "produces": "RESULTS_IN",
    "precedes": "PRECEDES",
    "comes before": "PRECEDES",
    "before": "PRECEDES",
    "happens before": "PRECEDES",
    "contradicts": "CONTRADICTS",
    "conflicts with": "CONTRADICTS",
    "contradict": "CONTRADICTS",
    "participates in": "PARTICIPATES_IN",
    "participated in": "PARTICIPATES_IN",
    "takes part in": "PARTICIPATES_IN",
    "member of": "MEMBER_OF",
    "belongs to": "MEMBER_OF",
    "is member of": "MEMBER_OF",
    "contains": "CONTAINS",
    "has task": "CONTAINS",
    "includes": "CONTAINS",
    "supports": "SUPPORTS",
    "support": "SUPPORTS",
    "backs": "SUPPORTS",
    "evidence for": "EVIDENCE_FOR",
    "is evidence for": "EVIDENCE_FOR",
    "evidences": "EVIDENCE_FOR",
    "about": "ABOUT",
    "regarding": "ABOUT",
    "concerns": "ABOUT",
    "relates to": "ABOUT",
    "possibly same as": "POSSIBLY_SAME_AS",
    "same as": "POSSIBLY_SAME_AS",
    "maybe same as": "POSSIBLY_SAME_AS",
    "supersedes": "SUPERSEDES",
    "replaces": "SUPERSEDES",
    "overrides": "SUPERSEDES",
}

# Textual confidence terms -> numeric value in [0, 1] (Req 4.6).
_CONFIDENCE_TERMS: dict[str, float] = {
    "certain": 1.0,
    "definite": 1.0,
    "very high": 0.95,
    "high": 0.9,
    "likely": 0.8,
    "probable": 0.75,
    "medium": 0.6,
    "moderate": 0.6,
    "possible": 0.5,
    "unknown": 0.5,
    "low": 0.3,
    "unlikely": 0.2,
    "very low": 0.1,
    "none": 0.0,
}


# --------------------------------------------------------------------------- #
# Value-level helpers
# --------------------------------------------------------------------------- #

def _normalize_key(value: str) -> str:
    """Lowercase, trim, and collapse internal whitespace for synonym lookups."""
    return " ".join(str(value).split()).lower()


def _canon_token(token: str) -> str:
    """Apply consistent casing to a single name token.

    Preserves likely acronyms (all-uppercase, length > 1) and mixed-case tokens
    (camelCase, e.g. ``iPhone``); title-cases otherwise.
    """
    if len(token) > 1 and token.isupper():
        return token
    if any(c.isupper() for c in token[1:]):
        return token
    return token[:1].upper() + token[1:].lower()


def _canonical_name(value: str) -> str:
    """Canonicalize a name/alias: trim, collapse whitespace, consistent casing.

    This is a pure string transform applied independently to each value; it
    never compares or merges entities (Req 4.1, 4.7).
    """
    collapsed = " ".join(str(value).split())
    if not collapsed:
        return collapsed
    return " ".join(_canon_token(tok) for tok in collapsed.split(" "))


def _normalize_timestamp(value) -> str | None:
    """Normalize a timestamp to an ISO-8601 UTC string (Req 4.2).

    Accepts ``datetime`` objects or strings. Naive datetimes/strings are assumed
    to already be UTC. Unparseable values are returned unchanged so downstream
    validation can reject them rather than silently dropping data.
    """
    if value is None:
        return None
    dt: datetime | None = None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return value
        candidate = raw
        # Python's fromisoformat handles a trailing 'Z' from 3.11+, but accept
        # it defensively for older parsers too.
        if candidate.endswith(("Z", "z")):
            candidate = candidate[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            return value
    else:
        return value

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat()


def _normalize_status(value, entity_type: str | None) -> str:
    """Map a status synonym to its canonical enum value (Req 4.3)."""
    if value is None:
        return value
    key = _normalize_key(value)
    table = _STATUS_SYNONYMS.get(entity_type or "", _STATUS_SYNONYMS_DEFAULT)
    if key in table:
        return table[key]
    # Fall back to the default (Task-oriented) table so the headline
    # "completed" -> "done" rule still applies when the type is unrecognized.
    return _STATUS_SYNONYMS_DEFAULT.get(key, value)


def _normalize_priority(value) -> str:
    """Map a priority synonym to its canonical enum value (Req 4.4)."""
    if value is None:
        return value
    key = _normalize_key(value)
    if key in _PRIORITY_SYNONYMS:
        return _PRIORITY_SYNONYMS[key]
    # Tolerate a trailing "priority" qualifier, e.g. "high priority" already
    # handled above; this catches forms like "high  priority".
    if key.endswith(" priority"):
        stripped = key[: -len(" priority")].strip()
        if stripped in _PRIORITY_SYNONYMS:
            return _PRIORITY_SYNONYMS[stripped]
    return value


def _normalize_predicate(value) -> str:
    """Normalize a relation name to its canonical predicate identifier (Req 4.5)."""
    if value is None:
        return value
    raw = str(value).strip()
    if not raw:
        return value
    # Already a registered canonical predicate?
    upper = raw.upper().replace(" ", "_").replace("-", "_")
    if upper in RELATION_SIGNATURES:
        return upper
    key = _normalize_key(raw)
    if key in _PREDICATE_SYNONYMS:
        return _PREDICATE_SYNONYMS[key]
    # Best-effort canonical form; downstream schema validation (W5) rejects it
    # if it is not a registered predicate.
    return upper


def _normalize_confidence(value) -> float | None:
    """Normalize a confidence value to a float in [0, 1] (Req 4.6).

    Parses numerics, numeric strings, percentage strings ("80%"), and textual
    terms ("high"), then clamps the result into ``[0, 1]``.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # avoid treating True/False as 1/0 numerics
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return _clamp_confidence(float(value))
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        key = raw.lower()
        if key in _CONFIDENCE_TERMS:
            return _CONFIDENCE_TERMS[key]
        if raw.endswith("%"):
            try:
                return _clamp_confidence(float(raw[:-1].strip()) / 100.0)
            except ValueError:
                return None
        try:
            return _clamp_confidence(float(raw))
        except ValueError:
            return None
    return None


def _clamp_confidence(value: float) -> float:
    """Clamp a numeric confidence into [0, 1] (Req 4.6).

    Values above 1.0 clamp to 1.0 and negatives to 0.0. Percentage handling is
    intentionally limited to explicit ``"%"`` strings (see
    :func:`_normalize_confidence`) so a bare ``1.5`` clamps to ``1.0`` rather
    than being silently reinterpreted as ``1.5%``.
    """
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


# --------------------------------------------------------------------------- #
# Normalizer
# --------------------------------------------------------------------------- #

class Normalizer:
    """W2 — normalizes extractor output into canonical, consistent values.

    Stateless and side-effect free: :meth:`normalize` returns a new
    :class:`ExtractionResult` and never mutates the input.
    """

    def normalize(self, extraction: ExtractionResult) -> ExtractionResult:
        """Return a normalized copy of ``extraction`` (Req 4.1-4.7)."""
        entities = [self._normalize_entity(e) for e in extraction.entities]
        events = [self._normalize_event(e) for e in extraction.events]
        claims = [self._normalize_claim(c) for c in extraction.claims]
        documents = [self._normalize_document(d) for d in extraction.documents]
        decisions = [self._normalize_decision(d) for d in extraction.decisions]
        relations = [self._normalize_relation(r) for r in extraction.relations]

        return ExtractionResult(
            entities=entities,
            events=events,
            claims=claims,
            documents=documents,
            decisions=decisions,
            relations=relations,
            extractor_version=extraction.extractor_version,
        )

    # -- per-item normalizers --------------------------------------------- #

    def _normalize_entity(self, entity: dict) -> dict:
        """Normalize a single entity dict, preserving its distinct identity (Req 4.7)."""
        out = copy.deepcopy(entity)
        entity_type = out.get("type")

        # Names / titles and aliases (Req 4.1).
        for name_key in ("name", "title"):
            if isinstance(out.get(name_key), str):
                out[name_key] = _canonical_name(out[name_key])
        if isinstance(out.get("aliases"), list):
            out["aliases"] = [
                _canonical_name(a) if isinstance(a, str) else a for a in out["aliases"]
            ]

        # Status / priority may live at the top level or in a nested "fields"
        # dict (the LLM extractor schema nests them under "fields").
        self._normalize_status_priority_inplace(out, entity_type)
        if isinstance(out.get("fields"), dict):
            self._normalize_status_priority_inplace(out["fields"], entity_type)

        # Timestamp-ish fields commonly attached to entities.
        for ts_key in ("due_at", "created_at"):
            if ts_key in out:
                out[ts_key] = _normalize_timestamp(out[ts_key])
        if isinstance(out.get("fields"), dict):
            for ts_key in ("due_at", "created_at"):
                if ts_key in out["fields"]:
                    out["fields"][ts_key] = _normalize_timestamp(out["fields"][ts_key])
        return out

    @staticmethod
    def _normalize_status_priority_inplace(d: dict, entity_type: str | None) -> None:
        if "status" in d and d["status"] is not None:
            d["status"] = _normalize_status(d["status"], entity_type)
        if "priority" in d and d["priority"] is not None:
            d["priority"] = _normalize_priority(d["priority"])

    def _normalize_event(self, event: dict) -> dict:
        out = copy.deepcopy(event)
        for ts_key in ("timestamp_start", "timestamp_end"):
            if ts_key in out:
                out[ts_key] = _normalize_timestamp(out[ts_key])
        return out

    def _normalize_claim(self, claim: dict) -> dict:
        out = copy.deepcopy(claim)
        if "confidence" in out:
            out["confidence"] = _normalize_confidence(out["confidence"])
        if "created_at" in out:
            out["created_at"] = _normalize_timestamp(out["created_at"])
        if "status" in out and out["status"] is not None:
            out["status"] = _normalize_status(out["status"], "Claim")
        return out

    def _normalize_document(self, document: dict) -> dict:
        out = copy.deepcopy(document)
        if isinstance(out.get("title"), str):
            out["title"] = _canonical_name(out["title"])
        if "created_at" in out:
            out["created_at"] = _normalize_timestamp(out["created_at"])
        return out

    def _normalize_decision(self, decision: dict) -> dict:
        out = copy.deepcopy(decision)
        if "timestamp" in out:
            out["timestamp"] = _normalize_timestamp(out["timestamp"])
        if "status" in out and out["status"] is not None:
            out["status"] = _normalize_status(out["status"], "Decision")
        return out

    def _normalize_relation(self, relation: dict) -> dict:
        out = copy.deepcopy(relation)
        if "predicate" in out:
            out["predicate"] = _normalize_predicate(out["predicate"])
        if "confidence" in out:
            out["confidence"] = _normalize_confidence(out["confidence"])
        for ts_key in ("valid_from", "valid_to"):
            if ts_key in out:
                out[ts_key] = _normalize_timestamp(out[ts_key])
        return out

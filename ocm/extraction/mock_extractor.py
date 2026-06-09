"""Deterministic, offline Mock_Extractor (default W1 extractor).

The :class:`MockExtractor` is the default extractor used whenever no extractor
configuration is supplied (Req 3.4). It runs entirely offline with **no API key
and no network access** (Req 3.7) by applying a fixed set of seeded
regex/keyword rules to the input text. Because the rules are pure functions of
the input and carry no clock, randomness, or external state, identical input
produces **byte-identical** output (Req 3.5): every output list is sorted by a
deterministic key and timestamps are derived from a fixed base plus a
discovery counter.

The assembled candidate payload is validated into an
:class:`~ocm.memory.contracts.ExtractionResult` (Req 3.1, 3.2); if validation
fails an :class:`~ocm.extraction.base.ExtractionError` is raised so the write
pipeline can reject the input and record a validation failure (Req 3.3).

Recognized rules (case-insensitive keywords; entity names preserve case):

* ``X owns [Project] Y``           -> Person ``X``, Project ``Y``, ``OWNS``.
* ``X (is) assigned to [Task] Y``  -> Person ``X``, Task ``Y``, ``ASSIGNED_TO``
  (emitted in registry direction Task -> Person).
* ``X completed/finished [Task] Y``-> Task ``Y`` status ``done`` + a completion
  Event, ``RESULTS_IN`` (Event -> Task), and ``PARTICIPATES_IN`` (Person ->
  Event). Satisfies constraint C4 (a done task needs a completion event).
* ``[Task] Y is <status phrase>``  -> Task ``Y`` with a normalized status
  (``not started``/``todo`` -> todo, ``in progress``/``started`` ->
  in_progress, ``blocked`` -> blocked, ``done``/``completed`` -> done,
  ``cancelled`` -> cancelled). A ``done`` status also creates the completion
  event so C4 holds.
* ``(we) decided ...``             -> a Decision.
* an ``http(s)://`` URL            -> a Document.
* every non-empty sentence         -> a Claim (text + default confidence) so
  semantic retrieval has content.

The word ``actually``/``correction``/``instead``/``in fact`` in a sentence
sets ``write_intent="correction"`` for relations from that sentence; otherwise
``new_fact`` is used (Req 6.3 default is applied downstream too).

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.7.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from pydantic import ValidationError

from ocm.extraction.base import ExtractionError
from ocm.memory.contracts import ExtractionResult

# --- Determinism knobs ------------------------------------------------------
#: Default confidence assigned to extracted claims/relations. High (> 0.8) so
#: the contradiction gate (C7) treats mock-extracted facts as strong.
DEFAULT_CONFIDENCE: float = 0.95

#: Confidence for a relation asserted as a *correction*. Deliberately higher than
#: ``DEFAULT_CONFIDENCE`` so a correction dominates the fact it revises by a
#: margin (Algorithm 1's ``c(a) - c(a_old) > delta``), letting the governance
#: gate route it to ``supersede`` rather than ``quarantine``.
CORRECTION_CONFIDENCE: float = 0.97

#: Fixed base timestamp; event/decision timestamps derive from this plus a
#: discovery counter so output never depends on the wall clock (Req 3.5).
_BASE_TS = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _ts(offset: int) -> str:
    """Return a deterministic ISO-8601 UTC timestamp for the given offset."""
    return (_BASE_TS + timedelta(seconds=offset)).isoformat()


# --- Compiled rule patterns (case-insensitive; names captured verbatim) -----
_CORRECTION_RE = re.compile(r"\b(actually|correction|instead|in fact|to correct)\b", re.IGNORECASE)
_URL_RE = re.compile(r"(https?://\S+)", re.IGNORECASE)

_OWNS_RE = re.compile(
    r"([A-Za-z]\w*)\s+owns\s+(?:the\s+)?(?:project\s+)?([A-Za-z]\w*)",
    re.IGNORECASE,
)
_ASSIGNED_RE = re.compile(
    r"([A-Za-z]\w*)\s+(?:is\s+|was\s+|been\s+|gets\s+)?assigned\s+to\s+(?:the\s+)?(?:task\s+)?([A-Za-z]\w*)",
    re.IGNORECASE,
)
_COMPLETED_RE = re.compile(
    r"([A-Za-z]\w*)\s+(?:has\s+)?(?:completed|finished)\s+(?:the\s+)?(?:task\s+)?([A-Za-z]\w*)",
    re.IGNORECASE,
)
# "Task T1 is not started", "task T1 has been blocked", ...
_TASK_STATUS_RE = re.compile(
    r"\btask\s+([A-Za-z]\w*)\s+(?:is\s+|has\s+been\s+|was\s+|are\s+)?(.+)$",
    re.IGNORECASE,
)
# "Project Orion is active", "project Orion was cancelled", ...
_PROJECT_STATUS_RE = re.compile(
    r"\bproject\s+([A-Za-z]\w*)\s+(?:is\s+|has\s+been\s+|was\s+|are\s+)?(.+)$",
    re.IGNORECASE,
)
# "Organization Acme is inactive", "company Acme has been active", ... — the
# org keyword disambiguates from a bare Person status ("Acme is inactive").
_ORG_STATUS_RE = re.compile(
    r"\b(?:organization|organisation|org|company)\s+([A-Za-z]\w*)\s+"
    r"(?:is\s+|has\s+been\s+|was\s+|are\s+)?(.+)$",
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"\b(?:we\s+|the\s+team\s+)?(?:have\s+)?(?:decided|finali[sz]ed)\b"
    r"\s*(?:the\s+decision\s+)?(?:to\s+|that\s+)?(.+)$",
    re.IGNORECASE,
)
# Marks a decision as final/ratified (drives constraint C8: a final decision
# needs supporting evidence). Without one of these cues a decision is a draft.
_DECISION_FINAL_RE = re.compile(
    r"\b(final|finali[sz]ed|approved|ratified|confirmed)\b", re.IGNORECASE
)
# "Mallory is inactive", "Alice is active", "Bob is no longer active" -> Person
# status (drives constraint C5: an inactive assignee is quarantined).
_INACTIVE_RE = re.compile(
    r"\b([A-Za-z]\w*)\s+is\s+(?:now\s+)?(?:no\s+longer\s+active|inactive|deactivated|departed|former)\b",
    re.IGNORECASE,
)
_ACTIVE_RE = re.compile(
    r"\b([A-Za-z]\w*)\s+is\s+(?:now\s+)?active\b",
    re.IGNORECASE,
)
# Marks a sentence as being about a typed entity's status (Project/Organization)
# so the bare Person status rules are suppressed for it (e.g. "Project Orion is
# active" must not also mint a Person "Orion").
_TYPED_STATUS_RE = re.compile(
    r"\b(project|organization|organisation|org|company)\b", re.IGNORECASE
)
# "Event Kickoff precedes Event Review", "X precedes Y" -> Event/PRECEDES/Event
# (drives constraint C3: a PRECEDES cycle is rejected).
_PRECEDES_RE = re.compile(
    r"\b(?:event\s+)?([A-Za-z]\w*)\s+precedes\s+(?:event\s+)?([A-Za-z]\w*)",
    re.IGNORECASE,
)

# Ordered status detection: "not started" must be checked before "started".
_STATUS_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("cancelled", re.compile(r"\b(cancelled|canceled|abandoned)\b", re.IGNORECASE)),
    ("blocked", re.compile(r"\b(blocked|stuck|on hold)\b", re.IGNORECASE)),
    ("todo", re.compile(r"\b(not started|not yet started|to ?do|pending)\b", re.IGNORECASE)),
    ("done", re.compile(r"\b(completed|complete|done|finished|resolved)\b", re.IGNORECASE)),
    ("in_progress", re.compile(r"\b(in[\s-]?progress|started|working on|ongoing|underway)\b", re.IGNORECASE)),
]


def _detect_status(phrase: str) -> str | None:
    """Map a free-text status phrase to a canonical Task status, if any."""
    for status, pattern in _STATUS_RULES:
        if pattern.search(phrase):
            return status
    return None


# Project status detection (ProjectStatus: active/inactive/completed/cancelled).
# Ordered so "completed" wins over the generic "active"/"on hold" cues.
_PROJECT_STATUS_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("cancelled", re.compile(r"\b(cancelled|canceled|abandoned|scrapped)\b", re.IGNORECASE)),
    ("completed", re.compile(r"\b(completed|complete|finished|done|delivered|shipped)\b", re.IGNORECASE)),
    ("inactive", re.compile(r"\b(inactive|on hold|paused|suspended|dormant)\b", re.IGNORECASE)),
    ("active", re.compile(r"\b(active|ongoing|underway|in progress|live|running)\b", re.IGNORECASE)),
]


def _detect_project_status(phrase: str) -> str | None:
    """Map a free-text status phrase to a canonical Project status, if any."""
    for status, pattern in _PROJECT_STATUS_RULES:
        if pattern.search(phrase):
            return status
    return None


# Organization status detection (OrgStatus: active/inactive).
_ORG_STATUS_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("inactive", re.compile(r"\b(inactive|defunct|dissolved|closed|shut down|wound down)\b", re.IGNORECASE)),
    ("active", re.compile(r"\b(active|operating|operational|running|live)\b", re.IGNORECASE)),
]


def _detect_org_status(phrase: str) -> str | None:
    """Map a free-text status phrase to a canonical Organization status, if any."""
    for status, pattern in _ORG_STATUS_RULES:
        if pattern.search(phrase):
            return status
    return None


def _split_sentences(text: str) -> list[str]:
    """Split input into trimmed, non-empty sentences (deterministic)."""
    parts = re.split(r"[.\n;!?]+", text)
    return [p.strip() for p in parts if p.strip()]


class MockExtractor:
    """Deterministic, offline rule-based extractor (default).

    Implements the :class:`~ocm.extraction.base.Extractor` protocol. Holds no
    mutable state between calls, so it is safe to share a single instance.
    """

    version: str = "mock-1"

    def extract(self, text: str, source_ref: str) -> ExtractionResult:
        """Extract candidate memory items from ``text`` (Req 3.1, 3.2)."""
        entities: dict[tuple[str, str], dict] = {}
        events: dict[str, dict] = {}
        documents: dict[str, dict] = {}
        decisions: list[dict] = []
        claims: list[dict] = []
        relations: dict[tuple[str, str, str], dict] = {}
        counter = {"n": 0}

        def next_offset() -> int:
            counter["n"] += 1
            return counter["n"]

        def add_entity(etype: str, name: str, fields: dict | None = None) -> str:
            key = (etype, name)
            if key not in entities:
                entities[key] = {"type": etype, "name": name, "fields": dict(fields or {})}
            elif fields:
                entities[key]["fields"].update(fields)
            return name

        def add_event(name: str, etype: str, description: str) -> str:
            if name not in events:
                events[name] = {
                    "name": name,
                    "type": etype,
                    "timestamp_start": _ts(next_offset()),
                    "timestamp_end": None,
                    "description": description,
                }
            return name

        def add_relation(subject: str, predicate: str, object_: str, write_intent: str) -> None:
            key = (subject, predicate, object_)
            if key in relations:
                return
            confidence = (
                CORRECTION_CONFIDENCE if write_intent == "correction" else DEFAULT_CONFIDENCE
            )
            relations[key] = {
                "subject": subject,
                "predicate": predicate,
                "object": object_,
                "confidence": confidence,
                "write_intent": write_intent,
            }

        def mark_task_done(task: str, write_intent: str) -> None:
            add_entity("Task", task, {"status": "done"})
            ev = add_event(f"Completion of {task}", "completion", f"Task {task} was completed")
            add_relation(ev, "RESULTS_IN", task, write_intent)

        for sentence in _split_sentences(text):
            intent = "correction" if _CORRECTION_RE.search(sentence) else "new_fact"

            # Every sentence becomes a Claim so semantic retrieval has content.
            claims.append({"text": sentence, "confidence": DEFAULT_CONFIDENCE})

            # OWNS: Person owns [Project] Project
            m = _OWNS_RE.search(sentence)
            if m:
                person, project = m.group(1), m.group(2)
                add_entity("Person", person)
                add_entity("Project", project)
                add_relation(person, "OWNS", project, intent)

            # ASSIGNED_TO: Person assigned to [Task] Task  (emit Task -> Person)
            m = _ASSIGNED_RE.search(sentence)
            if m:
                person, task = m.group(1), m.group(2)
                add_entity("Person", person)
                add_entity("Task", task)
                add_relation(task, "ASSIGNED_TO", person, intent)

            # Completion: Person completed [Task] Task
            m = _COMPLETED_RE.search(sentence)
            if m:
                person, task = m.group(1), m.group(2)
                add_entity("Person", person)
                add_entity("Task", task)
                mark_task_done(task, intent)
                add_relation(person, "PARTICIPATES_IN", f"Completion of {task}", intent)

            # Standalone task status: "Task T1 is not started"
            m = _TASK_STATUS_RE.search(sentence)
            if m:
                task, phrase = m.group(1), m.group(2)
                status = _detect_status(phrase)
                if status is not None:
                    if status == "done":
                        mark_task_done(task, intent)
                    else:
                        add_entity("Task", task, {"status": status})

            # Standalone project status: "Project Orion is active/cancelled".
            # Skipped for "owns Project X" (handled by the OWNS rule, which has
            # no trailing status phrase).
            m = _PROJECT_STATUS_RE.search(sentence)
            if m and "owns" not in sentence.lower():
                project, phrase = m.group(1), m.group(2)
                proj_status = _detect_project_status(phrase)
                if proj_status is not None:
                    add_entity("Project", project, {"status": proj_status})

            # Standalone organization status: "Organization Acme is inactive".
            m = _ORG_STATUS_RE.search(sentence)
            if m:
                org, phrase = m.group(1), m.group(2)
                org_status = _detect_org_status(phrase)
                if org_status is not None:
                    add_entity("Organization", org, {"status": org_status})

            # Person status: "Mallory is inactive" / "Alice is active".
            # Checked inactive-first so "no longer active" maps to inactive, and
            # skipped when the sentence is really about a typed entity (a
            # Project/Organization status) so "Project Orion is active" does not
            # also mint a spurious Person.
            if not _TYPED_STATUS_RE.search(sentence):
                m = _INACTIVE_RE.search(sentence)
                if m:
                    add_entity("Person", m.group(1), {"status": "inactive"})
                else:
                    m = _ACTIVE_RE.search(sentence)
                    if m:
                        add_entity("Person", m.group(1), {"status": "active"})

            # Temporal order: "Event X precedes Event Y" -> Event/PRECEDES/Event.
            m = _PRECEDES_RE.search(sentence)
            if m:
                before, after = m.group(1), m.group(2)
                add_event(before, "event", before)
                add_event(after, "event", after)
                add_relation(before, "PRECEDES", after, intent)

            # Decision: "(we) decided ..." / "we finalized the decision ...".
            m = _DECISION_RE.search(sentence)
            if m:
                topic = m.group(1).strip()
                decision_status = "final" if _DECISION_FINAL_RE.search(sentence) else "draft"
                decisions.append(
                    {
                        "summary": sentence,
                        "topic": topic,
                        "timestamp": _ts(next_offset()),
                        "made_by": None,
                        "status": decision_status,
                    }
                )

            # Document: any URL becomes a Document.
            for url_match in _URL_RE.finditer(sentence):
                url = url_match.group(1)
                if url not in documents:
                    documents[url] = {
                        "title": sentence,
                        "path_or_url": url,
                        "tags": [],
                    }

        payload = {
            "entities": sorted(entities.values(), key=lambda e: (e["type"], e["name"])),
            "events": sorted(events.values(), key=lambda e: (e["timestamp_start"], e["name"])),
            "claims": sorted(claims, key=lambda c: c["text"]),
            "documents": sorted(documents.values(), key=lambda d: d["path_or_url"]),
            "decisions": sorted(decisions, key=lambda d: (d["timestamp"], d["summary"])),
            "relations": sorted(
                relations.values(),
                key=lambda r: (r["predicate"], r["subject"], r["object"]),
            ),
            "extractor_version": self.version,
        }

        try:
            return ExtractionResult.model_validate(payload)
        except ValidationError as exc:  # pragma: no cover - defensive (Req 3.3)
            raise ExtractionError(
                f"MockExtractor produced output that failed validation: {exc}"
            ) from exc

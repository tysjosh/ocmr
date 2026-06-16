"""R4 — Evidence Packager (Req 18.1, 18.2, 18.3, 18.4, 18.5).

The :class:`EvidencePackager` is the final stage of the retrieval pipeline
(R0→R1→R2→R3→**R4**). It turns the reranked candidate list (R3) into a single,
structured :class:`EvidencePackage` — the object the API (`POST /memory/query`,
task 15.2) serializes and the agent (`Answer_Policy`, task 16.x) consumes.

What the package carries (Req 18.1)
-----------------------------------
``answer``, ``confidence``, ``supporting_assertions``, ``supporting_sources``,
``conflicts``, ``missing_information``, and ``retrieved_items``.

How each field is assembled
---------------------------
- **answer (optional, Req 18.5).** Retrieval is *not required* to produce a
  natural-language answer. For a structural ``direct_fact`` query whose top
  supporting item is an exact symbolic match, a concise answer is derived from
  the graph (the owner for an ``OWNS`` hit, the assignee for an ``ASSIGNED_TO``
  hit); otherwise ``answer`` is left ``None`` and the caller works from
  ``retrieved_items`` / ``supporting_assertions``.
- **supporting_assertions (Req 18.2).** The accepted, non-contradicted ranked
  items (highest score first), each carrying its ``id`` and ``confidence``.
- **supporting_sources (Req 18.3, 12.2).** Provenance for every supporting
  assertion, fetched via ``Provenance_Tracker.for_subject`` and de-duplicated.
- **conflicts (Req 18.4).** Contradicted / quarantined items that surfaced for
  the query, plus — for a conflict query — any unresolved quarantine records
  whose ``conflicting_ids`` intersect the retrieved set.
- **missing_information (Req 18.5).** Plain-language notes when nothing relevant
  was found, when only conflicting/quarantined items matched, when confidence is
  low, or when provenance is absent.
- **confidence.** Derived from the top supporting assertion (its ``confidence``,
  falling back to its rerank ``score``); ``0.0`` when nothing is supported.

Requirements: 18.1, 18.2, 18.3, 18.4, 18.5 (and 12.2 for provenance).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from ocm.ontology.models import Provenance
from ocm.retrieval.reranker import RankedItem

# Structural predicates we can phrase a direct answer for (Req 18.5 / 15.x).
OWNS = "OWNS"
ASSIGNED_TO = "ASSIGNED_TO"
PRECEDES = "PRECEDES"

#: Status string for an accepted memory item.
STATUS_ACCEPTED = "accepted"
#: Status string for a quarantined memory item.
STATUS_QUARANTINED = "quarantined"

#: The conflict-query signal (R0 ``query_type``) — see Query Classifier.
CONFLICT_QUERY_TYPE = "contradiction_check"

#: Below this confidence the packager flags low confidence in missing_information.
LOW_CONFIDENCE_THRESHOLD = 0.5

#: Node payload fields that can act as a human-facing label, in priority order.
_LABEL_FIELDS: tuple[str, ...] = ("name", "title", "summary", "description", "text")


class SupportingAssertion(BaseModel):
    """A supporting assertion: its id and confidence (Req 18.2)."""

    id: str
    confidence: float


class ConflictItem(BaseModel):
    """An unresolved conflict relevant to the query (Req 18.4).

    Surfaces either a contradicted / quarantined retrieved item or an
    unresolved ``Quarantine_Store`` record. ``conflicting_ids`` records the
    accepted items the conflict involves where known.
    """

    memory_id: Optional[str] = None
    memory_type: Optional[str] = None
    status: Optional[str] = None
    reason: Optional[str] = None
    conflicting_ids: list[str] = Field(default_factory=list)
    severity: Optional[str] = None
    text: Optional[str] = None
    score: Optional[float] = None
    #: Human-readable accepted side of a paired status conflict (the accepted
    #: HAS_STATUS assertion the quarantined flip contradicts), e.g. "T1: done".
    accepted: Optional[str] = None
    #: Human-readable quarantined side of a paired status conflict, e.g. "T1: todo".
    quarantined: Optional[str] = None


class EvidencePackage(BaseModel):
    """Structured retrieval result returned by R4 (Req 18.1).

    ``answer`` is optional (Req 18.5); the package is the contract the API and
    agent consume — ``confidence`` plus ``supporting_assertions`` (ids +
    confidence), ``supporting_sources`` (provenance), ``conflicts``,
    ``missing_information``, and the full ranked ``retrieved_items``.
    """

    answer: Optional[str] = None
    confidence: float = 0.0
    supporting_assertions: list[SupportingAssertion] = Field(default_factory=list)
    supporting_sources: list[Provenance] = Field(default_factory=list)
    conflicts: list[ConflictItem] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    retrieved_items: list[RankedItem] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def _resolve_name(graph: Any | None, entity_id: str | None) -> Optional[str]:
    """Resolve an entity id to its display name via the graph payload, else the id."""
    if entity_id is None:
        return None
    if graph is None:
        return entity_id
    try:
        payload = graph.get_entity_payload(entity_id)
    except Exception:  # pragma: no cover - defensive: graph without this method
        payload = None
    if not payload:
        return entity_id
    for key in _LABEL_FIELDS:
        value = payload.get(key)
        if value:
            return str(value)
    return entity_id


def _norm(value: str) -> str:
    """Case/whitespace-insensitive normal form for name matching."""
    return " ".join(str(value).split()).casefold()


# Leading words denoting an entity's type (so "Project Orion" also matches the
# node stored under the bare name "Orion").
_GENERIC_TYPE_WORDS: frozenset[str] = frozenset(
    {"project", "task", "event", "document", "decision", "organization", "org", "person"}
)


def _name_variants(raw: str) -> set[str]:
    """Normalized match targets for a name: the full form plus, when it opens
    with a generic type word, the form with that word stripped."""
    variants: set[str] = set()
    normalized = _norm(raw)
    if not normalized:
        return variants
    variants.add(normalized)
    tokens = normalized.split(" ")
    if len(tokens) > 1 and tokens[0] in _GENERIC_TYPE_WORDS:
        variants.add(" ".join(tokens[1:]))
    return variants


def _resolve_entity_ids(graph: Any | None, names: list[str]) -> set[str]:
    """Resolve query entity *names* to graph node ids (Req 18.4 relevance).

    A name matches a node when it equals the node id or any of the node's
    label-ish payload fields (``name`` / ``title`` / ``summary`` /
    ``description``), compared case-insensitively; a mention that opens with a
    generic type word (``"Project Orion"``) also matches the bare-name node
    (``"Orion"``). Used to judge whether a quarantined conflict is about an
    entity the query asked about.
    """
    if graph is None or not names:
        return set()
    try:
        node_ids = list(graph.node_ids())
    except Exception:  # pragma: no cover - defensive: graph without node_ids
        return set()
    targets: set[str] = set()
    for name in names:
        targets |= _name_variants(name)
    if not targets:
        return set()
    resolved: set[str] = set()
    for node_id in node_ids:
        if _norm(node_id) in targets:
            resolved.add(node_id)
            continue
        payload = graph.get_entity_payload(node_id) or {}
        for key in _LABEL_FIELDS:
            value = payload.get(key)
            if value and _norm(value) in targets:
                resolved.add(node_id)
                break
    return resolved


class EvidencePackager:
    """Assembles an :class:`EvidencePackage` from reranked candidates (R4)."""

    def package(
        self,
        query: str,
        classification: Any,
        ranked: list[RankedItem],
        graph: Any | None = None,
        provenance_tracker: Any | None = None,
        quarantine_store: Any | None = None,
        *,
        max_supporting: int | None = None,
    ) -> EvidencePackage:
        """Build the evidence package for a reranked candidate list.

        Args:
            query: The original natural-language query (used for context only).
            classification: The R0 :class:`QueryClassification` (``query_type``
                drives answer derivation and conflict-query handling).
            ranked: The reranked candidates from R3 (highest score first).
            graph: Optional ``Graph_Store`` used to resolve entity ids to names
                when deriving an ``answer``.
            provenance_tracker: Optional ``Provenance_Tracker``; when supplied,
                its ``for_subject`` populates ``supporting_sources`` (Req 18.3).
            quarantine_store: Optional ``Quarantine_Store``; for a conflict query
                its unresolved records augment ``conflicts`` (Req 18.4).
            max_supporting: Optional cap on the number of supporting assertions.

        Returns:
            A populated :class:`EvidencePackage` (Req 18.1).
        """
        ranked = list(ranked or [])

        # --- supporting assertions: accepted, non-contradicted (Req 18.2) ---
        accepted = [
            item
            for item in ranked
            if item.status == STATUS_ACCEPTED and not item.contradicted
        ]
        if max_supporting is not None:
            accepted = accepted[:max_supporting]

        supporting_assertions = [
            SupportingAssertion(
                id=item.memory_id,
                confidence=self._item_confidence(item),
            )
            for item in accepted
        ]

        # --- confidence from the top supporting item ------------------------
        confidence = supporting_assertions[0].confidence if supporting_assertions else 0.0

        # --- supporting sources via provenance (Req 18.3, 12.2) -------------
        supporting_sources = self._collect_provenance(accepted, provenance_tracker)

        # --- conflicts relevant to the query (Req 18.4) ---------------------
        conflicts = self._collect_conflicts(
            classification, ranked, accepted, graph, quarantine_store
        )

        # --- optional answer (Req 18.5) -------------------------------------
        answer = self._derive_answer(query, classification, ranked, accepted, graph)

        # --- missing information (Req 18.5) ---------------------------------
        missing_information = self._missing_information(
            ranked, supporting_assertions, supporting_sources, confidence
        )

        return EvidencePackage(
            answer=answer,
            confidence=confidence,
            supporting_assertions=supporting_assertions,
            supporting_sources=supporting_sources,
            conflicts=conflicts,
            missing_information=missing_information,
            retrieved_items=ranked,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _item_confidence(item: RankedItem) -> float:
        """Confidence for a supporting item: its ``confidence`` else its score."""
        if item.confidence is not None:
            return _clamp(float(item.confidence))
        return _clamp(float(item.score))

    @staticmethod
    def _collect_provenance(
        accepted: list[RankedItem], provenance_tracker: Any | None
    ) -> list[Provenance]:
        """Gather de-duplicated provenance for the supporting assertions (Req 18.3)."""
        if provenance_tracker is None:
            return []
        sources: list[Provenance] = []
        seen: set[str] = set()
        for item in accepted:
            try:
                records = provenance_tracker.for_subject(item.memory_id)
            except Exception:  # pragma: no cover - defensive
                records = []
            for record in records or []:
                key = getattr(record, "id", None) or id(record)
                if key in seen:
                    continue
                seen.add(key)
                sources.append(record)
        return sources

    def _collect_conflicts(
        self,
        classification: Any,
        ranked: list[RankedItem],
        accepted: list[RankedItem],
        graph: Any | None,
        quarantine_store: Any | None,
    ) -> list[ConflictItem]:
        """Surface conflicts relevant to the query (Req 18.4).

        Two sources are merged:

        1. Contradicted / quarantined items already in the reranked set.
        2. Unresolved ``Quarantine_Store`` records whose ``conflicting_ids``
           are *relevant* to this query — i.e. they overlap the retrieved memory
           (item ids and their subject/object ids) **or** the entities named in
           the query (resolved to graph ids). This is what lets a plain status
           query about ``task_t1`` surface the quarantined "not started"
           contradiction whose ``conflicting_ids`` reference ``task_t1``
           (Req 18.4) — not just explicit ``contradiction_check`` queries — while
           the relevance gate keeps unrelated quarantines out (precision).
        """
        conflicts: list[ConflictItem] = []
        seen: set[str] = set()

        for item in ranked:
            if not (item.contradicted or item.status == STATUS_QUARANTINED):
                continue
            if item.memory_id in seen:
                continue
            seen.add(item.memory_id)
            conflicts.append(
                ConflictItem(
                    memory_id=item.memory_id,
                    memory_type=item.memory_type,
                    status=item.status,
                    reason="quarantined" if item.status == STATUS_QUARANTINED else "contradicted",
                    text=item.text,
                    score=item.score,
                )
            )

        # Augment with unresolved quarantine records that are relevant to what
        # the query is about, for ANY query type (Req 18.4). Relevance = the
        # record's conflicting_ids overlap the retrieved set or the query's
        # entities; this surfaces conflicts on the exact entity being queried
        # without flooding unrelated ones.
        if quarantine_store is not None:
            relevant_ids = self._relevant_ids(classification, ranked, graph)
            try:
                records = quarantine_store.list("unresolved")
            except Exception:  # pragma: no cover - defensive
                records = []
            for record in records or []:
                rid = getattr(record, "id", None)
                if rid is None or rid in seen:
                    continue
                conflicting_ids = list(getattr(record, "conflicting_ids", []) or [])
                if not self._conflict_relevant(conflicting_ids, relevant_ids):
                    continue
                seen.add(rid)
                conflicts.append(
                    ConflictItem(
                        memory_id=rid,
                        memory_type="quarantine",
                        status=_enum_value(getattr(record, "status", None)),
                        reason=getattr(record, "reason", None),
                        conflicting_ids=conflicting_ids,
                        severity=_enum_value(getattr(record, "severity", None)),
                        accepted=self._render_accepted_status(graph, conflicting_ids),
                        quarantined=self._render_quarantined_status(graph, record),
                    )
                )
        return conflicts

    @staticmethod
    def _render_accepted_status(graph: Any | None, conflicting_ids: list[str]) -> Optional[str]:
        """Render the accepted side of a paired status conflict (Req 18.4).

        Looks for a ``HAS_STATUS`` assertion among ``conflicting_ids`` (the
        accepted status the quarantined flip contradicts) and renders it as
        ``"<label>: <status>"`` (e.g. ``"T1: done"``).
        """
        if graph is None:
            return None
        find = getattr(graph, "find_assertion", None)
        if find is None:
            return None
        for cid in conflicting_ids:
            try:
                found = find(cid)
            except Exception:  # pragma: no cover - defensive
                found = None
            if not found:
                continue
            subject_id, object_id, predicate, _data = found
            if predicate != "HAS_STATUS":
                continue
            return _render_status(graph, subject_id, object_id)
        return None

    @staticmethod
    def _render_quarantined_status(graph: Any | None, record: Any) -> Optional[str]:
        """Render the quarantined side of a paired status conflict from its payload."""
        payload = getattr(record, "candidate_payload", None) or {}
        if not isinstance(payload, dict):
            return None
        if payload.get("predicate") != "HAS_STATUS":
            return None
        subject_id = payload.get("subject_id")
        object_id = payload.get("object_id")
        if not subject_id or not object_id:
            return None
        return _render_status(graph, subject_id, object_id)

    @staticmethod
    def _conflict_relevant(conflicting_ids: list[str], relevant_ids: set[str]) -> bool:
        """Whether a quarantine record is relevant to this query (Req 18.4).

        Relevant iff any of its ``conflicting_ids`` overlaps the query-relevant
        id set (retrieved item / subject / object ids plus resolved query entity
        ids). A record with no ``conflicting_ids`` is never auto-surfaced — it is
        not tied to anything the query is about.
        """
        return bool(set(conflicting_ids) & relevant_ids)

    def _relevant_ids(
        self, classification: Any, ranked: list[RankedItem], graph: Any | None
    ) -> set[str]:
        """The set of ids this query is "about" (Req 18.4 relevance).

        Combines the reranked items' ids and their subject/object endpoints with
        the graph ids of the entities named in the query, so conflict relevance
        is judged against both what was retrieved and what was asked.
        """
        ids: set[str] = set()
        for item in ranked:
            if item.memory_id:
                ids.add(item.memory_id)
            if item.subject_id:
                ids.add(item.subject_id)
            if item.object_id:
                ids.add(item.object_id)
        ids |= _resolve_entity_ids(graph, getattr(classification, "entities", []) or [])
        return ids

    def _derive_answer(
        self,
        query: str,
        classification: Any,
        ranked: list[RankedItem],
        accepted: list[RankedItem],
        graph: Any | None,
    ) -> Optional[str]:
        """Derive a concise, deterministic answer for the common query types.

        Retrieval is still evidence-first (Req 18.5) — the agent's Answer_Policy
        produces the user-facing text — but for the structural query types the
        benchmark exercises we return a short factual string so end-to-end QA
        can be scored without a generator. Handled (in intent order):

        * **status** (``"... status of Task T1?"``) -> the entity's current
          accepted status, read from the graph (``"<label>: <status>"``).
        * **owner** (``OWNS`` exact match) -> the owning subject's name.
        * **assignee** (``ASSIGNED_TO`` exact match) -> the assigned object's name.
        * **temporal** (``PRECEDES`` exact matches) -> the preceding event(s).
        * **decision** -> a matching Decision's summary (+ status).
        * **contradiction_check** -> a short conflict presence/absence note.

        Anything else returns ``None`` (the caller works from ``retrieved_items``).
        """
        q = (query or "").lower()
        qtype = getattr(classification, "query_type", None)
        entity_ids = _resolve_entity_ids(graph, getattr(classification, "entities", []) or [])

        # 0) Slot value intent: a "value of <slot>" query addressed by the slot's
        #    (qualified) key returns the slot's current accepted HAS_VALUE — the
        #    MultiWOZ dialogue-state recall probe.
        if "value of" in q or "slot" in q:
            slot_answer = self._slot_value_answer(graph, query)
            if slot_answer is not None:
                return slot_answer

        # 1) Status intent takes precedence: a status question about a Task whose
        #    accepted status is "done" must answer "done", not its assignee.
        if "status" in q or "state of" in q:
            status_answer = self._status_answer(graph, entity_ids)
            if status_answer is not None:
                return status_answer

        # 2) Owner / assignee from an exact symbolic match.
        for item in accepted:
            if not item.exact_match:
                continue
            if item.predicate == OWNS and item.subject_id:
                return _resolve_name(graph, item.subject_id)
            if item.predicate == ASSIGNED_TO and item.object_id:
                return _resolve_name(graph, item.object_id)

        # 3) Temporal ordering: the event(s) preceding the queried event.
        preceding = [i for i in accepted if i.exact_match and i.predicate == PRECEDES]
        if preceding:
            names = list(
                dict.fromkeys(_resolve_name(graph, i.subject_id) for i in preceding)
            )
            return "Preceding events: " + ", ".join(names)

        # 4) Decision queries: a Decision whose summary mentions a query entity.
        decision_answer = self._decision_answer(graph, classification)
        if decision_answer is not None:
            return decision_answer

        # 5) Conflict-check queries: a short presence/absence note.
        if qtype == CONFLICT_QUERY_TYPE:
            return None  # the conflicts field carries the detail; avoid a vague string

        # 6) Fallback status answer for non-status-worded planning/temporal asks.
        if qtype in ("planning", "temporal", "direct_fact"):
            status_answer = self._status_answer(graph, entity_ids)
            if status_answer is not None:
                return status_answer
        return None

    @staticmethod
    def _slot_value_answer(graph: Any | None, query: str) -> Optional[str]:
        """Answer a slot-value query from the slot's current accepted HAS_VALUE.

        The query addresses a slot by its qualified key inside a ``[[key]]``
        marker (e.g. ``[[mwz-0001:hotel-area]]``); the key is matched **exactly**
        (normalized) against ``Slot`` nodes, so a slot key that is a substring of
        another never mis-resolves at full dataset scale. When no marker is
        present we fall back to a substring scan (legacy/robustness).

        When more than one HAS_VALUE is accepted for the slot — which only
        happens on an ungoverned arm that failed to supersede — the most recent
        value is returned, so recall is measured against the *current* gold value
        regardless of arm; the difference between arms surfaces in
        constraint_violations, not here.
        """
        if graph is None:
            return None
        qn = _norm(query)
        if not qn:
            return None
        marker = re.search(r"\[\[(.+?)\]\]", query)
        key_norm = _norm(marker.group(1)) if marker else None

        def _current_value(node_id: str) -> Optional[str]:
            values: list[tuple[Any, str]] = []
            for _s, obj, _k, data in graph.out_edges(node_id, "HAS_VALUE"):
                vp = graph.get_entity_payload(obj) or {}
                value = vp.get("value") or vp.get("name") or obj
                values.append((data.get("created_at"), str(value)))
            if not values:
                return None
            values.sort(key=lambda cv: (cv[0] is not None, str(cv[0])))
            return values[-1][1]

        for node_id in graph.node_ids():
            if graph.get_entity_type(node_id) != "Slot":
                continue
            payload = graph.get_entity_payload(node_id) or {}
            slot_name = payload.get("name") or ""
            if not slot_name:
                continue
            name_norm = _norm(slot_name)
            matched = (name_norm == key_norm) if key_norm else (name_norm in qn)
            if not matched:
                continue
            value = _current_value(node_id)
            if value is not None:
                return f"{slot_name}: {value}"
        return None

    @staticmethod
    def _status_answer(graph: Any | None, entity_ids: set[str]) -> Optional[str]:
        """Answer a status query from a resolved status-bearing entity's status."""
        if graph is None:
            return None
        for eid in entity_ids:
            etype = graph.get_entity_type(eid)
            if etype not in ("Task", "Project", "Person", "Organization"):
                continue
            payload = graph.get_entity_payload(eid) or {}
            status = payload.get("status")
            if status and status != "unknown":
                label = payload.get("title") or payload.get("name") or eid
                return f"{label}: {status}"
        return None

    @staticmethod
    def _decision_answer(graph: Any | None, classification: Any) -> Optional[str]:
        """Answer a decision query from a Decision whose summary names a query entity."""
        if graph is None:
            return None
        names = [_norm(n) for n in (getattr(classification, "entities", []) or []) if n]
        if not names:
            return None
        for nid in graph.node_ids():
            if graph.get_entity_type(nid) != "Decision":
                continue
            payload = graph.get_entity_payload(nid) or {}
            summary = payload.get("summary") or ""
            summary_norm = _norm(summary)
            if any(name in summary_norm for name in names):
                status = payload.get("status")
                return f"Decision: {summary}" + (f" ({status})" if status else "")
        return None

    @staticmethod
    def _missing_information(
        ranked: list[RankedItem],
        supporting_assertions: list[SupportingAssertion],
        supporting_sources: list[Provenance],
        confidence: float,
    ) -> list[str]:
        """State what evidence is absent or weak (Req 18.5)."""
        missing: list[str] = []
        if not ranked:
            missing.append("No memory items matched the query.")
            return missing
        if not supporting_assertions:
            missing.append(
                "No accepted supporting assertions were found; only unresolved or "
                "quarantined items matched the query."
            )
        elif confidence < LOW_CONFIDENCE_THRESHOLD:
            missing.append(
                f"Best-supported result has low confidence ({confidence:.2f})."
            )
        if supporting_assertions and not supporting_sources:
            missing.append(
                "No provenance records were found for the supporting assertions."
            )
        return missing


def _enum_value(value: Any) -> Optional[str]:
    """Return ``value.value`` for an enum, the string form otherwise, or ``None``."""
    if value is None:
        return None
    return str(getattr(value, "value", value))


#: Canonical id prefix for a shared StatusValue node (``status:<value>``).
_STATUS_VALUE_PREFIX = "status:"


def _status_value_label(graph: Any | None, status_value_id: str) -> str:
    """Resolve a StatusValue node id to its ``value`` label (e.g. ``done``)."""
    if graph is not None:
        try:
            payload = graph.get_entity_payload(status_value_id)
        except Exception:  # pragma: no cover - defensive
            payload = None
        if payload:
            value = payload.get("value") or payload.get("name")
            if value:
                return str(value)
    if status_value_id.startswith(_STATUS_VALUE_PREFIX):
        return status_value_id[len(_STATUS_VALUE_PREFIX):]
    return status_value_id


def _render_status(graph: Any | None, subject_id: str, status_value_id: str) -> str:
    """Render a status assertion as ``"<subject label>: <status>"``."""
    label = _resolve_name(graph, subject_id) or subject_id
    return f"{label}: {_status_value_label(graph, status_value_id)}"

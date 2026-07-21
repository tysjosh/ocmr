"""Write Pipeline (W1–W8) — the full ordered write path.

``WritePipeline`` wires the eight write stages into the strict ordered sequence
described in the design's "Write Pipeline Design (W1–W8)" section and exposed as
``POST /memory/write`` → ``WritePipeline.run(text, source_ref, write_intent,
extractor_version)``:

* **W1 Extractor** — turn ``text`` into a validated
  :class:`~ocm.memory.contracts.ExtractionResult` (Req 3.1). An
  :class:`~ocm.extraction.base.ExtractionError` rejects the whole input and
  records a validation failure (Req 3.3) instead of aborting the process.
* **W2 Normalizer** — canonicalize values.
* **W3 Entity Resolver** — resolve every extracted entity (and event) to an
  ``entity_id`` and **persist** the newly created/resolved nodes through the
  repository + graph so later stages and relations can reference them.
* **W4 Assertion Builder** — build a :class:`~ocm.memory.contracts.CandidateAssertion`
  per relation, mapping the relation's subject/object *names* to resolved ids.
* **W5 Schema Validator → W6 Constraint Validator (→ W7 Contradiction Checker)
  → W8 Commit Manager** — run **per candidate, independently**: a failure on one
  candidate routes that candidate to reject/quarantine without aborting the
  batch (Req 10.6, 10.7).

The run aggregates every committed outcome into ``accepted`` / ``superseded`` /
``quarantined`` / ``rejected`` lists plus a :class:`~ocm.memory.contracts.WriteSummary`
(Req 19.2), embeds accepted claims / documents / events via an injectable
vector-index hook (Req 16.6; accepted assertions are embedded by the Commit
Manager's own hook, Req 13.5), and records one per-write research log with the
run counts and latency (Req 25.1).

**Entity status reconciliation.** A bare extracted status (e.g. "Task T1 is not
started") carries no relation, so it is not a candidate assertion. Such status
changes are reconciled against accepted memory *after* relations are committed:
a Task set to ``done`` is accepted only when a completion Event exists (C4); any
other transition is checked against the permitted task-status map (C10). An
illegal transition (e.g. ``done`` → ``todo`` under ``new_fact``) is quarantined
as a status contradiction rather than silently overwriting accepted memory
(Req 10.6).

Requirements: 3.1, 10.1, 10.6, 10.7, 13.5, 16.6, 19.2, 25.1.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel

from ocm.core.config import Settings
from ocm.core.ids import IdGenerator
from ocm.core.logging import ResearchLogger
from ocm.extraction.base import ExtractionError
from ocm.memory.assertion_builder import AssertionBuilder
from ocm.memory.commit_manager import CommitManager
from ocm.memory.contracts import (
    CandidateAssertion,
    ExtractionResult,
    ResolutionOutcome,
    ValidationResult,
    WriteOutcome,
    WriteSummary,
)
from ocm.memory.graph_store import GraphStore
from ocm.memory.provenance_tracker import ProvenanceTracker
from ocm.memory.quarantine_store import QuarantineStore
from ocm.memory.repository import StorageRepository
from ocm.ontology.enums import DecisionStatus, ResolutionStatus, Severity, TaskStatus, WriteIntent
from ocm.ontology.models import (
    Claim,
    Decision,
    Document,
    Event,
    Organization,
    Person,
    Project,
    Slot,
    SlotValue,
    StatusValue,
    Task,
)
from ocm.resolution.entity_resolver import EntityResolver, normalize_name
from ocm.resolution.normalizer import Normalizer
from ocm.validation.constraints import (
    ConstraintValidator,
    c4_done_task_completion_event,
    c8_decision_evidence_floor,
    c10_task_status_transition,
)
from ocm.validation.schema_validator import SchemaValidator

#: Hook invoked with ``(memory_type, model)`` for each accepted non-assertion
#: memory item (claim / document / event) so it can be added to the
#: Vector_Index (Req 16.6). Optional and side-effecting.
MemoryEmbedHook = Callable[[str, BaseModel], None]

#: String tokens an LLM may emit for an absent value. JSON ``null`` decodes to
#: Python ``None``, but a model often emits these as literal *strings* inside a
#: JSON string field; they must be treated as "no value" before Pydantic tries
#: to parse them as a datetime.
_NULLISH_STRINGS = frozenset({"", "null", "none", "n/a", "na", "nil", "undefined"})


def _coerce_optional_datetime(value: Any) -> Any:
    """Return ``None`` for nullish/empty LLM placeholders, else the value as-is.

    LLM extractors sometimes fill an unknown timestamp with the literal string
    ``"null"`` (or ``""``/``"none"``) rather than JSON ``null``. Such strings are
    truthy and would reach the ``Event`` model and fail datetime parsing, so we
    normalize them to ``None`` here; genuine datetime/ISO-string values pass
    through untouched for Pydantic to validate.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in _NULLISH_STRINGS:
        return None
    return value

#: Predicate for the first-class status assertion ``Task -[HAS_STATUS]-> StatusValue``.
#: Promoting status to an assertion lets a status flip become an
#: assertion-to-assertion contradiction whose quarantine points at the accepted
#: status assertion (Req 8.11, 10.6).
HAS_STATUS = "HAS_STATUS"

#: Confidence assigned to a derived status assertion (high but < 1.0 so a
#: ``correction`` can still cleanly supersede it).
STATUS_CONFIDENCE = 0.95

#: Canonical id prefix for a shared :class:`StatusValue` node (``status:<value>``).
STATUS_VALUE_PREFIX = "status:"

#: Entity types whose ``status`` is promoted to a first-class HAS_STATUS
#: assertion. Task carries the richer constraint-driven rules (C4/C10); the
#: others use the write-intent rule (update/correction supersede; a conflicting
#: ``new_fact`` is quarantined). Decision adds a C8 ``final`` gate via its own
#: reconcile path. Organization/Project/Person follow the plain write-intent rule.
STATUS_BEARING_TYPES: frozenset[str] = frozenset(
    {"Task", "Project", "Person", "Organization"}
)

logger = logging.getLogger(__name__)


class WriteResult(BaseModel):
    """Aggregate result of a :meth:`WritePipeline.run` call (Req 19.2).

    Carries the four mutually-exclusive outcome lists and the rolled-up
    :class:`WriteSummary`. Entity status conflicts are surfaced in
    ``quarantined`` as a :class:`WriteOutcome` with a synthetic status candidate
    so callers see a uniform shape; the durable :class:`QuarantineRecord` lives
    in the Quarantine_Store.
    """

    accepted: list[WriteOutcome] = []
    superseded: list[WriteOutcome] = []
    quarantined: list[WriteOutcome] = []
    rejected: list[WriteOutcome] = []
    summary: WriteSummary


# Map an extraction entity ``type`` to the concrete ontology model + the field
# its display name lives under (Task uses ``title`` rather than ``name``).
_ENTITY_MODELS: dict[str, tuple[type[BaseModel], str]] = {
    "Person": (Person, "name"),
    "Organization": (Organization, "name"),
    "Project": (Project, "name"),
    "Task": (Task, "title"),
    "Decision": (Decision, "summary"),
    "Slot": (Slot, "name"),
    "SlotValue": (SlotValue, "name"),
}


class WritePipeline:
    """Orchestrates W1→W8 for a single write request."""

    def __init__(
        self,
        extractor: Any,
        normalizer: Normalizer,
        resolver: EntityResolver,
        assertion_builder: AssertionBuilder,
        schema_validator: SchemaValidator,
        constraint_validator: ConstraintValidator,
        commit_manager: CommitManager,
        repo: StorageRepository,
        graph: GraphStore,
        ids: IdGenerator,
        provenance_tracker: ProvenanceTracker,
        quarantine_store: QuarantineStore | None = None,
        embed_hook: Callable[[Any], None] | None = None,
        memory_embed_hook: MemoryEmbedHook | None = None,
        research_logger: ResearchLogger | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Wire the pipeline to its eight stages and the stores they touch.

        Args:
            extractor: W1 extractor (``extract(text, source_ref) -> ExtractionResult``).
            normalizer: W2 :class:`Normalizer`.
            resolver: W3 :class:`EntityResolver`.
            assertion_builder: W4 :class:`AssertionBuilder`.
            schema_validator: W5 :class:`SchemaValidator`.
            constraint_validator: W6 :class:`ConstraintValidator` (binds W7).
            commit_manager: W8 :class:`CommitManager`.
            repo: Durable :class:`StorageRepository`.
            graph: In-memory accepted-only :class:`GraphStore`.
            ids: :class:`IdGenerator` for entity / claim / document ids.
            provenance_tracker: Records provenance for persisted memory.
            quarantine_store: Store for entity status-conflict quarantines.
                Defaults to the commit manager's store when omitted.
            embed_hook: Optional assertion embed hook (forwarded behaviour lives
                on the commit manager; kept here for reference/wiring parity).
            memory_embed_hook: Optional ``(memory_type, model)`` hook to embed
                accepted claims / documents / events (Req 16.6).
            research_logger: Optional :class:`ResearchLogger` for per-write logs.
            settings: Optional :class:`Settings` (thresholds, determinism).
        """
        self.extractor = extractor
        self.normalizer = normalizer
        self.resolver = resolver
        self.assertion_builder = assertion_builder
        self.schema_validator = schema_validator
        self.constraint_validator = constraint_validator
        self.commit_manager = commit_manager
        self.repo = repo
        self.graph = graph
        self.ids = ids
        self.provenance_tracker = provenance_tracker
        self.quarantine_store = quarantine_store or commit_manager.quarantine_store
        self.embed_hook = embed_hook
        self.memory_embed_hook = memory_embed_hook
        self.research_logger = research_logger
        self.settings = settings

    # ====================================================================
    # public API
    # ====================================================================
    def run(
        self,
        text: str,
        source_ref: str,
        write_intent: str | WriteIntent | None = None,
        extractor_version: str | None = None,
        *,
        created_at: datetime | None = None,
    ) -> WriteResult:
        """Run the full W1–W8 pipeline over ``text`` and return a :class:`WriteResult`.

        On an extractor failure (W1) the whole input is rejected and a validation
        failure is recorded (Req 3.3); an empty :class:`WriteResult` is returned.
        Otherwise every relation is processed independently through W4–W8 so one
        bad candidate never aborts the batch (Req 10.6, 10.7).
        """
        started = time.perf_counter()
        now = created_at or datetime.now(timezone.utc)
        run_intent = self._coerce_intent(write_intent)

        # --- W1: extract -------------------------------------------------
        try:
            extraction = self.extractor.extract(text, source_ref)
        except ExtractionError as exc:
            logger.warning("W1 extraction failed for %s: %s", source_ref, exc)
            return self._record_failed_extraction(source_ref, started)

        # --- W2: normalize ----------------------------------------------
        extraction = self.normalizer.normalize(extraction)
        version = extractor_version or extraction.extractor_version

        # --- W3: resolve + persist entities, events, and other memory ----
        ref_to_id: dict[str, str] = {}
        status_reconciles: list[dict[str, Any]] = []
        self._persist_entities(extraction, source_ref, now, ref_to_id, status_reconciles)
        self._persist_events(extraction, source_ref, now, ref_to_id)
        self._persist_claims(extraction, source_ref, now)
        self._persist_documents(extraction, source_ref, now, ref_to_id)
        self._persist_decisions(extraction, source_ref, now, ref_to_id, status_reconciles)

        accepted: list[WriteOutcome] = []
        superseded: list[WriteOutcome] = []
        quarantined: list[WriteOutcome] = []
        rejected: list[WriteOutcome] = []
        validation_failures = 0
        contradiction_failures = 0

        # --- W4–W8: per-candidate (independent failure routing) ----------
        for relation in extraction.relations:
            outcome, vfail, cfail = self._process_relation(
                relation, ref_to_id, source_ref, version, run_intent, now
            )
            if outcome is None:
                continue
            validation_failures += vfail
            contradiction_failures += cfail
            self._bucket(outcome, accepted, superseded, quarantined, rejected)

        # --- Entity status reconciliation (after relations commit) -------
        # Includes Task / Project / Person statuses and Decision draft->final
        # (the latter gated by C8, which needs this write's EVIDENCE_FOR edges).
        for record in status_reconciles:
            outcome = self._reconcile_entity_status(record, run_intent, now)
            if outcome is not None:
                self._bucket(outcome, accepted, superseded, quarantined, rejected)

        all_outcomes = accepted + superseded + quarantined + rejected
        summary = CommitManager.summarize(all_outcomes)

        latency_ms = (time.perf_counter() - started) * 1000.0
        self._log_write(
            source_ref=source_ref,
            summary=summary,
            validation_failures=validation_failures,
            contradiction_failures=contradiction_failures,
            latency_ms=latency_ms,
            used_llm=bool(getattr(self.settings, "extractor", "mock") == "llm"),
        )

        return WriteResult(
            accepted=accepted,
            superseded=superseded,
            quarantined=quarantined,
            rejected=rejected,
            summary=summary,
        )

    # ====================================================================
    # W3 — persistence of entities / events / claims / documents / decisions
    # ====================================================================
    def _persist_entities(
        self,
        extraction: ExtractionResult,
        source_ref: str,
        now: datetime,
        ref_to_id: dict[str, str],
        status_reconciles: list[dict[str, Any]],
    ) -> None:
        """Resolve every extracted entity to an id and persist new nodes (W3)."""
        for ent in extraction.entities:
            etype = ent.get("type")
            spec = _ENTITY_MODELS.get(etype)
            if spec is None:
                logger.debug("Skipping entity of unknown type %r", etype)
                continue
            name = ent.get("name") or ent.get("title") or ""
            entity_ref = {"type": etype, "name": name, **(ent.get("fields") or {})}
            outcome = self.resolver.resolve(entity_ref, self.graph, self.ids, source_ref)
            if outcome.entity_id is None:
                logger.debug("Entity %r (%s) unresolved; skipping", name, etype)
                continue

            entity_id = outcome.entity_id
            ref_to_id[normalize_name(name)] = entity_id

            model = self._build_entity_model(etype, entity_id, ent)
            desired_status = self._desired_status(etype, ent)
            is_new = not self.graph.has_entity(entity_id)

            if is_new:
                node_model = model
                # Defer a brand-new Task's ``done`` status (set the node to
                # ``unknown``) so a ``done`` status does not trip C4 before its
                # completion Event + RESULTS_IN edge are committed; the status
                # reconciliation below mints the HAS_STATUS assertion and syncs
                # the node attribute. Other status-bearing types keep their
                # stated status on the node immediately (e.g. so C5 sees an
                # inactive Person during relation processing).
                if etype == "Task" and desired_status == TaskStatus.done.value:
                    node_model = model.model_copy(update={"status": TaskStatus.unknown})
                self._persist_node(etype, node_model)
                self.provenance_tracker.record(
                    subject_id=entity_id,
                    source_ref=source_ref,
                    created_at=now,
                    extractor_version=extraction.extractor_version,
                )

            # Record a status reconciliation when a real status was stated on a
            # status-bearing entity (Task / Project / Person).
            if etype in STATUS_BEARING_TYPES and desired_status not in (
                None,
                TaskStatus.unknown.value,
            ):
                status_reconciles.append(
                    {
                        "entity_id": entity_id,
                        "etype": etype,
                        "desired_status": desired_status,
                        "source_ref": source_ref,
                    }
                )

    def _persist_events(
        self,
        extraction: ExtractionResult,
        source_ref: str,
        now: datetime,
        ref_to_id: dict[str, str],
    ) -> None:
        """Resolve events to Event entities and persist them (W3)."""
        for ev in extraction.events:
            name = ev.get("name") or ev.get("description") or ""
            entity_ref = {"type": "Event", "name": name}
            outcome = self.resolver.resolve(entity_ref, self.graph, self.ids, source_ref)
            if outcome.entity_id is None:
                continue
            event_id = outcome.entity_id
            ref_to_id[normalize_name(name)] = event_id
            if outcome.resolution_status == ResolutionStatus.resolved_existing:
                continue
            model = Event(
                id=event_id,
                type=ev.get("type") or "event",
                timestamp_start=_coerce_optional_datetime(ev.get("timestamp_start")) or now,
                timestamp_end=_coerce_optional_datetime(ev.get("timestamp_end")),
                description=ev.get("description") or name or "",
            )
            self._persist_node("Event", model, name=name)
            self.provenance_tracker.record(
                subject_id=event_id,
                source_ref=source_ref,
                created_at=now,
                extractor_version=extraction.extractor_version,
            )
            self._embed_memory("Event", model)

    def _persist_claims(
        self, extraction: ExtractionResult, source_ref: str, now: datetime
    ) -> None:
        """Persist every extracted claim and embed it (Req 16.6)."""
        for claim in extraction.claims:
            text = claim.get("text", "")
            if not text:
                continue
            claim_id = self.ids.generic_id("claim", text, source_ref)
            model = Claim(
                id=claim_id,
                text=text,
                source_ref=source_ref,
                confidence=claim.get("confidence", 1.0),
                created_at=_coerce_optional_datetime(claim.get("created_at")) or now,
            )
            self.repo.upsert_claim(model)
            self.provenance_tracker.record(
                subject_id=claim_id,
                source_ref=source_ref,
                created_at=now,
                extractor_version=extraction.extractor_version,
            )
            self._embed_memory("Claim", model)

    def _persist_documents(
        self,
        extraction: ExtractionResult,
        source_ref: str,
        now: datetime,
        ref_to_id: dict[str, str],
    ) -> None:
        """Persist every extracted document and embed it (Req 16.6)."""
        for doc in extraction.documents:
            path = doc.get("path_or_url") or ""
            title = doc.get("title") or path or ""
            if not (path or title):
                # Nothing identifiable to store (LLM emitted an empty/null doc).
                continue
            doc_id = self.ids.generic_id("doc", path or title, source_ref)
            model = Document(
                id=doc_id,
                title=title,
                path_or_url=path,
                created_at=_coerce_optional_datetime(doc.get("created_at")) or now,
                tags=list(doc.get("tags") or []),
            )
            self.repo.upsert_document(model)
            # A document can be a relation endpoint (ABOUT / EVIDENCE_FOR), so
            # mirror it as a graph node too.
            self._persist_node("Document", model, name=title, persist_repo=False)
            ref_to_id[normalize_name(title)] = doc_id
            self.provenance_tracker.record(
                subject_id=doc_id,
                source_ref=source_ref,
                created_at=now,
                extractor_version=extraction.extractor_version,
            )
            self._embed_memory("Document", model)

    def _persist_decisions(
        self,
        extraction: ExtractionResult,
        source_ref: str,
        now: datetime,
        ref_to_id: dict[str, str],
        status_reconciles: list[dict[str, Any]],
    ) -> None:
        """Persist every extracted decision as a graph entity (W3).

        A Decision is identified by its **topic** (the decision content, e.g.
        "launch Project Orion"), not the full summary sentence or the source, so
        the same decision mentioned across sessions resolves to the *same*
        Decision entity — letting a draft -> final status change (or a
        contradictory restatement) reconcile as a HAS_STATUS assertion.

        Draft decisions are persisted durably immediately. A **final** decision
        is only mirrored into the graph (durable persistence deferred) so any
        EVIDENCE_FOR edges in this write can attach; the status reconciliation
        then gates ``final`` on C8 (evidence floor) — a final decision lacking
        evidence is quarantined and its mirrored node retracted.
        """
        for dec in extraction.decisions:
            summary = dec.get("summary") or ""
            topic = dec.get("topic") or summary
            if not topic:
                # No decision content to identify/store (empty/null decision).
                continue
            normalized_topic = " ".join(str(topic).lower().split())
            dec_id = self.ids.stable_id("dec", normalized_topic)
            status = dec.get("status") or DecisionStatus.draft.value
            is_new = not self.graph.has_entity(dec_id)
            model = Decision(
                id=dec_id,
                summary=summary,
                timestamp=_coerce_optional_datetime(dec.get("timestamp")) or now,
                made_by=dec.get("made_by"),
                status=status,
            )
            ref_to_id[normalize_name(summary)] = dec_id
            ref_to_id[normalize_name(topic)] = dec_id

            if status == DecisionStatus.final.value:
                # A brand-new final decision is mirrored into the graph (durable
                # persistence deferred) so EVIDENCE_FOR edges can attach and C8
                # can be checked; an existing decision (e.g. a prior draft) is
                # left untouched so a failed ``final`` upgrade cannot overwrite
                # the accepted draft before reconciliation runs.
                if is_new:
                    self._persist_node("Decision", model, name=summary, persist_repo=False)
            elif is_new:
                self._persist_node("Decision", model, name=summary)
                self.provenance_tracker.record(
                    subject_id=dec_id,
                    source_ref=source_ref,
                    created_at=now,
                    extractor_version=extraction.extractor_version,
                )

            # Reconcile the decision's status as a HAS_STATUS assertion.
            status_reconciles.append(
                {
                    "entity_id": dec_id,
                    "etype": "Decision",
                    "desired_status": status,
                    "source_ref": source_ref,
                }
            )

    # ====================================================================
    # W4–W8 — per-candidate processing
    # ====================================================================
    def _process_relation(
        self,
        relation: dict,
        ref_to_id: dict[str, str],
        source_ref: str,
        version: str,
        run_intent: WriteIntent,
        now: datetime,
    ) -> tuple[WriteOutcome | None, int, int]:
        """Build, validate, and commit one relation candidate (W4→W8).

        Returns ``(outcome, validation_failures, contradiction_failures)``.
        ``outcome`` is ``None`` when the relation's endpoints did not resolve
        (the relation is dropped — its endpoints are surfaced elsewhere).
        """
        subject_ref = relation.get("subject", "")
        object_ref = relation.get("object", "")
        subject_id = ref_to_id.get(normalize_name(subject_ref))
        object_id = ref_to_id.get(normalize_name(object_ref))

        if subject_id is None or object_id is None:
            logger.debug(
                "Dropping relation %s -[%s]-> %s: unresolved endpoint(s)",
                subject_ref,
                relation.get("predicate"),
                object_ref,
            )
            return None, 0, 0

        resolved = {
            subject_ref: ResolutionOutcome(
                resolution_status=ResolutionStatus.resolved_existing, entity_id=subject_id
            ),
            object_ref: ResolutionOutcome(
                resolution_status=ResolutionStatus.resolved_existing, entity_id=object_id
            ),
        }
        rel = dict(relation)
        rel.setdefault("write_intent", run_intent.value)
        rel.setdefault("extractor_version", version)

        candidate = self.assertion_builder.build(rel, resolved, source_ref)

        # W5 — structural schema validation (typed-schema ablation skips it).
        if getattr(self.settings, "enable_schema_validation", True):
            vr = self.schema_validator.validate(candidate, self.graph)
            if not vr.valid:
                outcome = self.commit_manager.commit(candidate, vr, created_at=now)
                return outcome, 1, 0

        # Reviewer ablation: latest-value supersession for Slot HAS_VALUE only.
        # This bypasses the broader W6/W7 governance stack and asks whether the
        # observed gains are explained by a plain "keep the newest slot value"
        # policy.
        if getattr(self.settings, "supersession_only_has_value", False):
            vr = self._supersession_only_verdict(candidate)
            outcome = self.commit_manager.commit(candidate, vr, created_at=now)
            return outcome, 0 if vr.valid else 1, 0

        # Reviewer baseline: MemGPT-style LLM-managed memory. The (LLM-decided)
        # write intent alone governs the store op — no schema/contradiction/
        # quarantine governance. Unlike Bsup (always overwrite), this supersedes
        # only when the model chose to *update*; an *insert* decision on a slot
        # that already has a value leaves both active (a violation), which is the
        # MemGPT failure mode OCMR's gate prevents.
        if getattr(self.settings, "memgpt_intent_supersede", False):
            vr = self._memgpt_intent_verdict(candidate)
            outcome = self.commit_manager.commit(candidate, vr, created_at=now)
            return outcome, 0 if vr.valid else 1, 0

        # W6 (+W7) — graph-level constraints incl. the contradiction gate. The
        # contradiction-gate ablation disables C7 by passing no checker, so
        # contradictions are no longer blocked/quarantined at write time.
        if not getattr(self.settings, "enable_constraint_validation", True):
            vr = ValidationResult(valid=True, recommended_action="accept")
        elif getattr(self.settings, "enable_contradiction_gate", True):
            vr = self.constraint_validator.validate(
                candidate, self.graph, settings=self.settings
            )
        else:
            vr = self.constraint_validator.validate(
                candidate, self.graph, settings=self.settings, contradiction_checker=None
            )
        outcome = self.commit_manager.commit(candidate, vr, created_at=now)

        vfail = 0 if vr.valid else 1
        cfail = 1 if (not vr.valid and vr.failed_check == "C7") else 0
        return outcome, vfail, cfail

    def _supersession_only_verdict(
        self, candidate: CandidateAssertion
    ) -> ValidationResult:
        """Latest-value routing for the Bsup ablation.

        Only ``Slot -[HAS_VALUE]-> SlotValue`` participates. If the slot already
        has one or more active ``HAS_VALUE`` assertions, they are superseded and
        the candidate becomes current. No schema, temporal, provenance, evidence,
        or contradiction quarantine checks are consulted here; the baseline is
        intentionally just a same-slot overwrite policy.
        """
        if candidate.predicate != "HAS_VALUE":
            return ValidationResult(valid=True, recommended_action="accept")

        if self.graph.get_entity_type(candidate.subject_id) != "Slot":
            return ValidationResult(valid=True, recommended_action="accept")

        current_ids: list[str] = []
        new_id = self.ids.assertion_id(
            candidate.subject_id,
            candidate.predicate,
            candidate.object_id,
            candidate.source_ref,
        )
        for _s, _o, _k, data in self.graph.out_edges(
            candidate.subject_id, candidate.predicate
        ):
            old_id = data.get("assertion_id")
            if old_id and old_id != new_id:
                current_ids.append(str(old_id))

        if not current_ids:
            return ValidationResult(valid=True, recommended_action="accept")

        return ValidationResult(
            valid=True,
            reason=(
                "Bsup latest-value supersession for Slot HAS_VALUE "
                f"replaces {current_ids}"
            ),
            conflicting_ids=current_ids,
            recommended_action="supersede",
        )

    def _memgpt_intent_verdict(
        self, candidate: CandidateAssertion
    ) -> ValidationResult:
        """Intent-conditional routing for the MemGPT-style baseline.

        Only ``Slot -[HAS_VALUE]-> SlotValue`` participates. The (LLM-decided)
        ``write_intent`` on the candidate is the sole gate: an ``update`` intent
        supersedes any active HAS_VALUE assertion(s) for the same Slot (the model
        chose to overwrite); any other intent is accepted as an additional value
        (the model chose to insert). No schema/contradiction/quarantine checks
        run. Consequently an ``insert`` decision on a slot that already holds a
        value leaves both active — the single-valued violation OCMR's gate would
        have prevented.
        """
        if candidate.predicate != "HAS_VALUE":
            return ValidationResult(valid=True, recommended_action="accept")
        if self.graph.get_entity_type(candidate.subject_id) != "Slot":
            return ValidationResult(valid=True, recommended_action="accept")
        if candidate.write_intent != WriteIntent.update:
            # insert / new_fact: add without superseding (may leave a conflict).
            return ValidationResult(valid=True, recommended_action="accept")

        new_id = self.ids.assertion_id(
            candidate.subject_id,
            candidate.predicate,
            candidate.object_id,
            candidate.source_ref,
        )
        current_ids: list[str] = []
        for _s, _o, _k, data in self.graph.out_edges(
            candidate.subject_id, candidate.predicate
        ):
            old_id = data.get("assertion_id")
            if old_id and old_id != new_id:
                current_ids.append(str(old_id))
        if not current_ids:
            return ValidationResult(valid=True, recommended_action="accept")
        return ValidationResult(
            valid=True,
            reason=f"MemGPT-style overwrite (LLM update) replaces {current_ids}",
            conflicting_ids=current_ids,
            recommended_action="supersede",
        )

    # ====================================================================
    # Entity status reconciliation (HAS_STATUS assertions)
    # ====================================================================
    def _reconcile_entity_status(
        self, record: dict[str, Any], run_intent: WriteIntent, now: datetime
    ) -> WriteOutcome | None:
        """Reconcile an entity's stated status as a first-class HAS_STATUS assertion.

        Builds a ``<entity> -[HAS_STATUS]-> StatusValue`` candidate and routes it
        through the Commit Manager exactly like any other assertion, so the
        accepted status becomes durable, retrievable memory and a status flip
        becomes an assertion-to-assertion contradiction (Req 8.11, 10.6):

        * **accept** — the first status for the entity (no prior accepted status).
        * **supersede** — a legal change: the prior HAS_STATUS assertion is
          retired and the new one accepted. For a Task this is a permitted
          transition (C10) or a ``correction``; for other status-bearing
          entities (Project / Person) it is an ``update`` or ``correction``.
        * **quarantine** — a conflicting change that is not permitted (e.g. a
          Task ``done`` -> ``todo`` flip, or a Project ``active`` -> ``cancelled``
          stated as a bare ``new_fact``): quarantined as a *status contradiction*
          whose ``conflicting_ids`` point at the **accepted** HAS_STATUS assertion
          (and the entity), never overwriting accepted memory.

        The denormalized entity ``status`` attribute is kept in sync on accept
        and supersede for backward compatibility. Returns the
        :class:`WriteOutcome` for the status assertion (or ``None`` on a no-op).
        """
        entity_id = record["entity_id"]
        etype = record.get("etype", "Task")
        desired = record["desired_status"]
        source_ref = record["source_ref"]

        current, current_aid = self._current_status(entity_id)
        if desired == current:
            return None  # idempotent no-op

        if getattr(self.settings, "supersession_only_has_value", False):
            # Bsup has no status/evidence/temporal governance. It accepts status
            # assertions as ordinary memory and intentionally does not supersede
            # them; only Slot HAS_VALUE gets latest-value supersession.
            status_value_id = self._ensure_status_value(desired, now)
            candidate = CandidateAssertion(
                subject_id=entity_id,
                predicate=HAS_STATUS,
                object_id=status_value_id,
                confidence=STATUS_CONFIDENCE,
                source_ref=source_ref,
                write_intent=run_intent,
                extractor_version=None,
            )
            vr = ValidationResult(valid=True, recommended_action="accept")
            outcome = self.commit_manager.commit(candidate, vr, created_at=now)
            self._sync_entity_status(entity_id, desired)
            return outcome

        action, reason = self._classify_status_change(
            etype, entity_id, current, current_aid, desired, run_intent
        )

        status_value_id = self._ensure_status_value(desired, now)
        candidate = CandidateAssertion(
            subject_id=entity_id,
            predicate=HAS_STATUS,
            object_id=status_value_id,
            confidence=STATUS_CONFIDENCE,
            source_ref=source_ref,
            write_intent=run_intent,
            extractor_version=None,
        )

        if action == "quarantine":
            # Point the contradiction at the accepted status assertion (so a
            # status query can pair {accepted, quarantined}) and the entity
            # (backward-compatible entity-linked surfacing).
            conflicting_ids = [cid for cid in (current_aid, entity_id) if cid]
            full_reason = (
                f"status contradiction: {etype} {entity_id!r} is {current!r} and "
                f"cannot change to {desired!r} ({reason or 'not a permitted change'})"
            )
            vr = ValidationResult(
                valid=False,
                failed_check="HAS_STATUS",
                reason=full_reason,
                severity=Severity.medium,
                conflicting_ids=conflicting_ids,
                recommended_action="quarantine",
            )
            outcome = self.commit_manager.commit(candidate, vr, created_at=now)
            # A brand-new final Decision that fails C8 was only mirrored into the
            # graph (durable persistence deferred); retract it so no accepted
            # Decision node lingers (Req 8.9, 10.3). A decision that already had
            # an accepted status (e.g. a prior draft) is left intact — only the
            # failed upgrade is quarantined.
            if etype == "Decision" and current_aid is None:
                self.graph.remove_entity(entity_id)
            return outcome

        if action == "supersede" and current_aid is not None:
            vr = ValidationResult(
                valid=True,
                reason=reason,
                conflicting_ids=[current_aid],
                recommended_action="supersede",
            )
            outcome = self.commit_manager.commit(candidate, vr, created_at=now)
        else:  # accept (first status)
            vr = ValidationResult(valid=True, recommended_action="accept")
            outcome = self.commit_manager.commit(candidate, vr, created_at=now)

        # Keep the denormalized node attribute in sync with accepted memory.
        self._sync_entity_status(entity_id, desired)
        return outcome

    def _classify_status_change(
        self,
        etype: str,
        entity_id: str,
        current: str,
        current_aid: str | None,
        desired: str,
        run_intent: WriteIntent,
    ) -> tuple[str, str | None]:
        """Decide accept / supersede / quarantine for an entity status change.

        With no prior accepted status the change is always accepted (first
        status). Otherwise the rule depends on the entity type:

        **Task** (constraint-driven):
        * ``done`` is gated by C4 (a completion Event must exist); failing C4
          quarantines the change.
        * a ``correction`` always supersedes;
        * a transition permitted by ``TASK_STATUS_TRANSITIONS`` (C10) supersedes,
          anything else is quarantined.

        **Other status-bearing entities** (Project / Person — write-intent rule):
        * ``update`` or ``correction`` supersedes the prior status;
        * a conflicting ``new_fact`` (or any other intent) is quarantined as a
          status contradiction, so the change surfaces instead of silently
          overwriting accepted memory.
        """
        # First status for the entity: always accept (except a Task ``done`` must
        # still clear C4 below, and a Decision ``final`` must clear C8).
        first_status = current in (None, TaskStatus.unknown.value) or current_aid is None

        if etype == "Task":
            if desired == TaskStatus.done.value:
                c4 = c4_done_task_completion_event(entity_id, TaskStatus.done, self.graph)
                if not c4.valid:
                    return "quarantine", c4.reason
                return ("supersede" if current_aid is not None else "accept", None)
            if first_status:
                return "accept", None
            if run_intent == WriteIntent.correction:
                return "supersede", "status correction"
            c10 = c10_task_status_transition(current, desired, WriteIntent.new_fact)
            if c10.valid:
                return "supersede", None
            return "quarantine", c10.reason

        if etype == "Decision":
            # A ``final`` Decision is gated by C8 (an EVIDENCE_FOR floor), exactly
            # as a Task ``done`` is gated by C4: lacking evidence it is quarantined
            # rather than accepted.
            if desired == DecisionStatus.final.value:
                c8 = c8_decision_evidence_floor(
                    entity_id, DecisionStatus.final, self.graph, self.settings
                )
                if not c8.valid:
                    return "quarantine", c8.reason
                return ("supersede" if current_aid is not None else "accept", None)
            # draft (or other) decision statuses follow the write-intent rule.

        # Project / Person / Decision-draft and other status-bearing entities:
        # write-intent rule.
        if first_status:
            return "accept", None
        if run_intent in (WriteIntent.update, WriteIntent.correction):
            label = "status correction" if run_intent == WriteIntent.correction else "status update"
            return "supersede", label
        return "quarantine", f"{run_intent.value} cannot overwrite an accepted status"

    def _current_status(self, entity_id: str) -> tuple[str, str | None]:
        """Return ``(status_value, has_status_assertion_id)`` for an entity.

        The accepted ``HAS_STATUS`` out-edge is the source of truth for the
        status *assertion*: when present its StatusValue and assertion id are
        returned. When absent the entity has no accepted status assertion yet, so
        ``(unknown, None)`` is returned — the denormalized node attribute is
        deliberately ignored here so the *first* stated status always mints a
        HAS_STATUS assertion (the attribute is kept in sync separately).
        """
        for _s, obj, _k, data in self.graph.out_edges(entity_id, HAS_STATUS):
            value = self._status_value_of(obj)
            return value, data.get("assertion_id")
        return TaskStatus.unknown.value, None

    def _status_value_of(self, status_value_id: str) -> str:
        """Resolve a StatusValue node id to its ``value`` (label) string."""
        payload = self.graph.get_entity_payload(status_value_id) or {}
        value = payload.get("value") or payload.get("name")
        if value:
            return str(value)
        if status_value_id.startswith(STATUS_VALUE_PREFIX):
            return status_value_id[len(STATUS_VALUE_PREFIX):]
        return status_value_id

    def _ensure_status_value(self, value: str, now: datetime) -> str:
        """Create (idempotently) and persist the shared StatusValue node.

        The node id is ``status:<value>`` and is shared across every entity that
        holds that status (HAS_STATUS is m:1). Returns the node id so it can be
        used as the assertion's object.
        """
        status_value_id = f"{STATUS_VALUE_PREFIX}{value}"
        if not self.graph.has_entity(status_value_id):
            model = StatusValue(id=status_value_id, value=value, name=value)
            self.repo.upsert_entity("StatusValue", model)
            self.graph.add_entity("StatusValue", model.model_dump(mode="json"))
        return status_value_id

    def _sync_entity_status(self, entity_id: str, desired: str) -> None:
        """Mirror the accepted status onto the denormalized node attribute.

        Keeps the graph node's ``status`` (and the durable row) consistent with
        the accepted HAS_STATUS assertion so existing readers that inspect the
        entity attribute keep working (backward compatibility).
        """
        payload = self.graph.get_entity_payload(entity_id)
        if payload is None:
            return
        etype = self.graph.get_entity_type(entity_id) or "Task"
        updated = dict(payload)
        updated["status"] = desired
        updated["status_defaulted"] = False
        self.graph.add_entity(etype, updated)
        # Persist the attribute change durably via the concrete entity model.
        spec = _ENTITY_MODELS.get(etype)
        if spec is not None:
            model_cls = spec[0]
            try:
                self.repo.upsert_entity(etype, model_cls(**updated))
            except Exception:  # pragma: no cover - defensive: payload mismatch
                logger.debug("Could not persist synced status for %s", entity_id)

    # ====================================================================
    # helpers
    # ====================================================================
    def _build_entity_model(
        self, etype: str, entity_id: str, ent: dict
    ) -> BaseModel:
        """Build the concrete ontology model for an extracted entity dict."""
        fields = dict(ent.get("fields") or {})
        # Normalize LLM nullish placeholders ("null"/"none"/"") in datetime
        # fields to real None so model defaults apply instead of failing parse.
        for _dt_key in ("due_at", "timestamp"):
            if _dt_key in fields:
                fields[_dt_key] = _coerce_optional_datetime(fields[_dt_key])
        name = ent.get("name") or ent.get("title") or ""

        def present(*keys: str) -> dict:
            """Collect only the non-None field values so model defaults apply."""
            out: dict[str, Any] = {}
            for key in keys:
                value = fields.get(key)
                if value is not None:
                    out[key] = value
            return out

        if etype == "Person":
            return Person(
                id=entity_id,
                name=name,
                roles=list(fields.get("roles") or []),
                aliases=list(fields.get("aliases") or ent.get("aliases") or []),
                **present("status"),
            )
        if etype == "Organization":
            return Organization(
                id=entity_id,
                name=name,
                type=fields.get("type") or "organization",
                **present("status"),
            )
        if etype == "Project":
            return Project(
                id=entity_id,
                name=name,
                **present("goal", "status", "owner_id"),
            )
        if etype == "Task":
            return Task(
                id=entity_id,
                title=name,
                **present("status", "priority", "project_id", "assignee_id", "due_at"),
            )
        if etype == "Decision":
            return Decision(
                id=entity_id,
                summary=name or fields.get("summary", ""),
                timestamp=fields.get("timestamp") or datetime.now(timezone.utc),
                **present("made_by", "status"),
            )
        if etype == "Slot":
            return Slot(id=entity_id, name=name)
        if etype == "SlotValue":
            value = fields.get("value") or name
            return SlotValue(id=entity_id, value=value, name=name or value)
        raise ValueError(f"Unsupported entity type {etype!r}")

    @staticmethod
    def _desired_status(etype: str, ent: dict) -> str | None:
        """Return the stated status for an entity (Task-relevant)."""
        fields = ent.get("fields") or {}
        status = fields.get("status") or ent.get("status")
        return status

    def _persist_node(
        self,
        entity_type: str,
        model: BaseModel,
        *,
        name: str | None = None,
        persist_repo: bool = True,
    ) -> None:
        """Persist an entity model to the repo and mirror it in the graph.

        The graph payload always carries a ``name`` key (falling back to the
        model's ``title`` / ``summary``) so the Entity Resolver — which matches
        on ``payload['name']`` — resolves entities whose model names live under
        a different field (e.g. ``Task.title``).
        """
        if persist_repo:
            self.repo.upsert_entity(entity_type, model)
        payload = model.model_dump(mode="json")
        if "name" not in payload or not payload.get("name"):
            payload["name"] = name or self._model_name(model)
        self.graph.add_entity(entity_type, payload)

    @staticmethod
    def _model_name(model: BaseModel) -> str:
        for attr in ("name", "title", "summary", "description"):
            value = getattr(model, attr, None)
            if value:
                return str(value)
        return getattr(model, "id", "")

    def _embed_memory(self, memory_type: str, model: BaseModel) -> None:
        """Embed an accepted claim / document / event via the hook (Req 16.6)."""
        if self.memory_embed_hook is not None:
            self.memory_embed_hook(memory_type, model)

    @staticmethod
    def _coerce_intent(value: str | WriteIntent | None) -> WriteIntent:
        if value is None:
            return WriteIntent.new_fact
        if isinstance(value, WriteIntent):
            return value
        return WriteIntent(value)

    @staticmethod
    def _bucket(
        outcome: WriteOutcome,
        accepted: list[WriteOutcome],
        superseded: list[WriteOutcome],
        quarantined: list[WriteOutcome],
        rejected: list[WriteOutcome],
    ) -> None:
        """Route a :class:`WriteOutcome` into its decision bucket."""
        if outcome.decision == "accepted":
            accepted.append(outcome)
        elif outcome.decision == "superseded":
            superseded.append(outcome)
        elif outcome.decision == "quarantined":
            quarantined.append(outcome)
        else:
            rejected.append(outcome)

    # -- logging / failure handling --------------------------------------
    def _record_failed_extraction(self, source_ref: str, started: float) -> WriteResult:
        """Build the empty result for a failed extraction and log it (Req 3.3)."""
        summary = WriteSummary(
            num_candidates=0,
            num_accepted=0,
            num_quarantined=0,
            num_rejected=0,
            num_superseded=0,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._log_write(
            source_ref=source_ref,
            summary=summary,
            validation_failures=1,
            contradiction_failures=0,
            latency_ms=latency_ms,
            used_llm=bool(getattr(self.settings, "extractor", "mock") == "llm"),
        )
        return WriteResult(summary=summary)

    def _log_write(
        self,
        *,
        source_ref: str,
        summary: WriteSummary,
        validation_failures: int,
        contradiction_failures: int,
        latency_ms: float,
        used_llm: bool,
    ) -> None:
        """Record one per-write research log (Req 25.1)."""
        if self.research_logger is None:
            return
        self.research_logger.log_write(
            input_id=source_ref,
            source_ref=source_ref,
            number_of_candidates=summary.num_candidates,
            number_accepted=summary.num_accepted,
            number_quarantined=summary.num_quarantined,
            number_rejected=summary.num_rejected,
            validation_failures=validation_failures,
            contradiction_failures=contradiction_failures,
            latency_ms=latency_ms,
            token_count_if_llm_used=None,
        )

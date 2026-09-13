"""W8 — Commit Manager (``Commit_Manager``).

The Commit Manager is the final stage of the write pipeline. It takes a
:class:`~ocm.memory.contracts.CandidateAssertion` together with the
:class:`~ocm.memory.contracts.ValidationResult` produced by W5/W6 (which folds in
the W7 contradiction verdict) and routes the candidate to exactly one of four
mutually-exclusive outcomes:

* **accept** — the candidate passed schema + all constraints + no blocking
  contradiction. A new ``accepted`` :class:`~ocm.ontology.models.Assertion` is
  minted and written through to the durable ``Storage_Repository``, mirrored as
  an edge in the ``Graph_Store``, embedded into the ``Vector_Index`` (via an
  injectable hook), and its provenance is recorded.
* **supersede** — a ``correction`` that replaces one or more existing accepted
  assertions. Each old assertion is flipped to ``superseded`` (row + graph edge
  removed), the new assertion is accepted, and a ``SUPERSEDES`` edge links the
  new assertion to each old one. Provenance is preserved for **both** sides
* **quarantine** — the candidate is structurally valid but reviewable or
  conflicting. A :class:`~ocm.ontology.models.QuarantineRecord` is written to the
  ``Quarantine_Store`` and the candidate is **excluded from accepted memory**
  (never added to the graph).
* **reject** — the candidate is malformed or ontology-illegal. The rejection is
  logged and the candidate is **never** written to the graph or default
  retrieval.

Invariants enforced here: quarantined and rejected candidates are never written
to the ``Graph_Store`` as accepted memory; every validation failure is
excluded from accepted memory and reported back on the
:class:`~ocm.memory.contracts.WriteOutcome`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Iterable

from ocm.core.ids import IdGenerator
from ocm.memory.contracts import CandidateAssertion, ValidationResult, WriteOutcome, WriteSummary
from ocm.memory.graph_store import GraphStore
from ocm.memory.provenance_tracker import ProvenanceTracker
from ocm.memory.quarantine_store import QuarantineStore
from ocm.memory.repository import StorageRepository
from ocm.ontology.enums import AssertionStatus, Severity
from ocm.ontology.models import Assertion

#: Predicate used for the assertion-to-assertion supersession link.
SUPERSEDES = "SUPERSEDES"

#: Hook invoked with an accepted :class:`Assertion` so its embedding can be added
#: to the Vector_Index. Optional and side-effecting.
EmbedHook = Callable[[Assertion], None]

#: Hook invoked with ``(memory_id, new_status)`` when an embedded item's status
#: changes (e.g. an assertion is superseded) so the Vector_Index metadata stays
#: consistent with durable storage. Optional and side-effecting.
StatusHook = Callable[[str, str], None]


class CommitManager:
    """Routes a validated candidate to accept / supersede / quarantine / reject (W8)."""

    def __init__(
        self,
        repo: StorageRepository,
        graph: GraphStore,
        ids: IdGenerator,
        quarantine_store: QuarantineStore,
        provenance_tracker: ProvenanceTracker,
        embed_hook: EmbedHook | None = None,
        status_hook: StatusHook | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Wire the commit manager to the stores it writes through.

        Args:
            repo: Durable :class:`StorageRepository` (source of truth on disk).
            graph: In-memory :class:`GraphStore` projection of accepted memory.
            ids: :class:`IdGenerator` used to mint assertion ids.
            quarantine_store: :class:`QuarantineStore` for reviewable/conflicting
                candidates.
            provenance_tracker: :class:`ProvenanceTracker` recording origin of
                every committed/quarantined item.
            embed_hook: Optional callable invoked with each accepted
                :class:`Assertion` to add it to the Vector_Index. Skipped when
                ``None`` (e.g. when embeddings are disabled in tests).
            status_hook: Optional ``(memory_id, new_status)`` callable invoked
                when an embedded assertion's status changes (a supersession), so
                the Vector_Index metadata is re-tagged and the superseded item
                drops out of accepted-only retrieval. Skipped
                when ``None``.
            logger: Optional logger for rejections; defaults to a module logger.
        """
        self.repo = repo
        self.graph = graph
        self.ids = ids
        self.quarantine_store = quarantine_store
        self.provenance_tracker = provenance_tracker
        self.embed_hook = embed_hook
        self.status_hook = status_hook
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    # -- public API --------------------------------------------------------
    def commit(
        self,
        candidate: CandidateAssertion,
        vr: ValidationResult,
        *,
        created_at: datetime | None = None,
    ) -> WriteOutcome:
        """Commit a single candidate per its validation verdict.

        Args:
            candidate: The proposed assertion from W4.
            vr: The W5/W6 (+W7) verdict. Its ``recommended_action`` drives the
                routing; when absent, ``valid`` decides accept vs reject.
            created_at: Optional fixed creation timestamp (defaults to now UTC).

        Returns:
            A :class:`WriteOutcome` describing the single outcome, carrying any
            ids minted and the reported reason for non-accept outcomes
        """
        now = created_at or datetime.now(timezone.utc)
        action = self._resolve_action(vr)

        if action == "accept":
            return self._accept(candidate, now)
        if action == "supersede":
            return self._supersede(candidate, vr, now)
        if action == "quarantine":
            return self._quarantine(candidate, vr, now)
        return self._reject(candidate, vr)

    @staticmethod
    def summarize(outcomes: Iterable[WriteOutcome]) -> WriteSummary:
        """Aggregate a batch of outcomes into a :class:`WriteSummary`."""
        outcomes = list(outcomes)
        counts = {"accepted": 0, "superseded": 0, "quarantined": 0, "rejected": 0}
        for outcome in outcomes:
            counts[outcome.decision] += 1
        return WriteSummary(
            num_candidates=len(outcomes),
            num_accepted=counts["accepted"],
            num_superseded=counts["superseded"],
            num_quarantined=counts["quarantined"],
            num_rejected=counts["rejected"],
        )

    # -- routing -----------------------------------------------------------
    @staticmethod
    def _resolve_action(vr: ValidationResult) -> str:
        """Resolve the routing action from a validation verdict.

        ``recommended_action`` is authoritative when present. Otherwise a valid
        result is accepted and an invalid one is rejected (the conservative
        default for a verdict that did not classify itself).
        """
        if vr.recommended_action is not None:
            return vr.recommended_action
        return "accept" if vr.valid else "reject"

    # -- accept -------------------------------------------------
    def _accept(self, candidate: CandidateAssertion, now: datetime) -> WriteOutcome:
        """Mint, persist, graph, embed, and record provenance for an accept."""
        existing_id = self._existing_accepted_triple_id(candidate)
        if existing_id is not None:
            self.provenance_tracker.record(
                subject_id=existing_id,
                source_ref=candidate.source_ref,
                created_at=now,
                extractor_version=candidate.extractor_version,
            )
            return WriteOutcome(
                candidate=candidate,
                decision="accepted",
                assertion_id=existing_id,
            )

        assertion = self._build_accepted_assertion(candidate, now)
        self._persist_accepted(assertion)
        return WriteOutcome(
            candidate=candidate,
            decision="accepted",
            assertion_id=assertion.id,
        )

    def _existing_accepted_triple_id(self, candidate: CandidateAssertion) -> str | None:
        """Return the accepted assertion id for an already-active identical triple.

        The accepted graph stores one edge per ``(subject, predicate, object)``
        triple. Treating a repeated identical assertion as a fresh durable row
        lets the graph edge metadata hide the earlier row; a later supersession
        can then retire only the graph-visible duplicate and leave the older row
        accepted in SQLite. Re-asserting the same triple is therefore an
        idempotent accept with additional provenance on the existing assertion.
        """
        edge = self.graph.get_assertion_edge(
            candidate.subject_id, candidate.object_id, candidate.predicate
        )
        if edge is None:
            return None
        assertion_id = edge.get("assertion_id")
        if not assertion_id:
            return None
        existing = self.repo.get_assertion(str(assertion_id))
        if (
            existing is None
            or existing.status is not AssertionStatus.accepted
            or existing.subject_id != candidate.subject_id
            or existing.object_id != candidate.object_id
            or existing.predicate != candidate.predicate
        ):
            return None
        return existing.id

    # -- supersede ----------------------------------
    def _supersede(
        self, candidate: CandidateAssertion, vr: ValidationResult, now: datetime
    ) -> WriteOutcome:
        """Supersede the conflicting accepted assertion(s) with the correction.

        Falls back to a quarantine when no target assertion is identified — a
        supersession needs an existing accepted assertion to replace, so a
        ``supersede`` recommendation with no ``conflicting_ids`` is treated as
        reviewable rather than silently accepted.
        """
        if not vr.conflicting_ids:
            return self._quarantine(candidate, vr, now)

        # 1) Retire the superseded assertion(s) first. This is done before the
        #    new edge is added so that when the correction shares the old
        #    triple (same subject/predicate/object) the removal does not drop
        #    the freshly-accepted edge.
        new_assertion = self._build_accepted_assertion(candidate, now)
        for old_id in vr.conflicting_ids:
            self._mark_superseded(old_id, now)

        # 2) Accept the new (correcting) assertion and link new -> old.
        self._persist_accepted(new_assertion)
        for old_id in vr.conflicting_ids:
            self._add_supersedes_edge(new_assertion, old_id, candidate, now)

        return WriteOutcome(
            candidate=candidate,
            decision="superseded",
            assertion_id=new_assertion.id,
            superseded_assertion_id=vr.conflicting_ids[0],
            reason=vr.reason,
        )

    def _mark_superseded(self, old_id: str, now: datetime) -> None:
        """Flip an old assertion to ``superseded`` and drop its accepted edge.

        Provenance for the old assertion is preserved: existing rows are never
        deleted, and we ensure at least one provenance row exists for it so the
        "both sides retain provenance" guarantee holds.
        """
        old = self.repo.get_assertion(old_id)
        # Persist the status change (row stays, status flips to superseded).
        self.repo.set_assertion_status(old_id, AssertionStatus.superseded.value)
        if old is not None:
            # Remove the now-superseded edge from the accepted-only graph.
            self.graph.remove_assertion(old.subject_id, old.object_id, old.predicate)
            # Re-tag the superseded assertion in the Vector_Index so it no longer
            # surfaces in accepted-only semantic retrieval.
            if self.status_hook is not None:
                self.status_hook(old_id, AssertionStatus.superseded.value)
            # Preserve provenance for the old assertion.
            if not self.provenance_tracker.for_subject(old_id):
                self.provenance_tracker.record(
                    subject_id=old_id,
                    source_ref=old.source_ref,
                    created_at=old.created_at,
                    extractor_version=old.extractor_version,
                )

    def _add_supersedes_edge(
        self,
        new_assertion: Assertion,
        old_id: str,
        candidate: CandidateAssertion,
        now: datetime,
    ) -> None:
        """Create an accepted ``SUPERSEDES`` assertion linking new -> old."""
        link = Assertion(
            id=self.ids.assertion_id(new_assertion.id, SUPERSEDES, old_id, candidate.source_ref),
            subject_id=new_assertion.id,
            predicate=SUPERSEDES,
            object_id=old_id,
            confidence=candidate.confidence,
            status=AssertionStatus.accepted,
            source_ref=candidate.source_ref,
            created_at=now,
            extractor_version=candidate.extractor_version,
            write_intent=candidate.write_intent,
        )
        self.repo.upsert_assertion(link)
        self.graph.add_assertion(link)

    # -- quarantine ---------------------------------------
    def _quarantine(
        self, candidate: CandidateAssertion, vr: ValidationResult, now: datetime
    ) -> WriteOutcome:
        """Persist a QuarantineRecord; never add the candidate to the graph."""
        reason = vr.reason or vr.failed_check or "quarantined for review"
        severity = vr.severity or Severity.medium
        record = self.quarantine_store.add(
            candidate_payload=candidate.model_dump(mode="json"),
            reason=reason,
            severity=severity,
            conflicting_ids=list(vr.conflicting_ids),
            created_at=now,
        )
        # Record provenance for the quarantined candidate.
        self.provenance_tracker.record(
            subject_id=record.id,
            source_ref=candidate.source_ref,
            created_at=now,
            extractor_version=candidate.extractor_version,
        )
        return WriteOutcome(
            candidate=candidate,
            decision="quarantined",
            quarantine_id=record.id,
            reason=reason,
        )

    # -- reject -------------------------------------------
    def _reject(self, candidate: CandidateAssertion, vr: ValidationResult) -> WriteOutcome:
        """Log the rejection; never touch the graph or default retrieval."""
        reason = vr.reason or vr.failed_check or "rejected: malformed or ontology-illegal"
        self.logger.warning(
            "Rejected candidate %s -[%s]-> %s (check=%s): %s",
            candidate.subject_id,
            candidate.predicate,
            candidate.object_id,
            vr.failed_check,
            reason,
        )
        return WriteOutcome(
            candidate=candidate,
            decision="rejected",
            reason=reason,
        )

    # -- shared helpers ----------------------------------------------------
    def _build_accepted_assertion(
        self, candidate: CandidateAssertion, now: datetime
    ) -> Assertion:
        """Build an ``accepted`` :class:`Assertion` from a candidate."""
        assertion_id = self.ids.assertion_id(
            candidate.subject_id,
            candidate.predicate,
            candidate.object_id,
            candidate.source_ref,
        )
        return Assertion(
            id=assertion_id,
            subject_id=candidate.subject_id,
            predicate=candidate.predicate,
            object_id=candidate.object_id,
            confidence=candidate.confidence,
            status=AssertionStatus.accepted,
            source_ref=candidate.source_ref,
            created_at=now,
            valid_from=candidate.valid_from,
            valid_to=candidate.valid_to,
            extractor_version=candidate.extractor_version,
            write_intent=candidate.write_intent,
        )

    def _persist_accepted(self, assertion: Assertion) -> None:
        """Write an accepted assertion through to repo + graph + vector + provenance.

        Order: durable row first (source of truth), then the in-memory graph
        edge (write-through), then the optional embedding, then
        provenance keyed by the assertion id.
        """
        self.repo.upsert_assertion(assertion)
        self.graph.add_assertion(assertion)
        if self.embed_hook is not None:
            self.embed_hook(assertion)
        self.provenance_tracker.record(
            subject_id=assertion.id,
            source_ref=assertion.source_ref,
            created_at=assertion.created_at,
            extractor_version=assertion.extractor_version,
        )

"""Provenance Tracker (``Provenance_Tracker``).

``ProvenanceTracker`` writes to the ``provenance`` table through the
:class:`~ocm.memory.repository.StorageRepository` interface (Req 12.4). When any
assertion, claim, document, or quarantine record is created, it records the
item's ``source_ref``, ``created_at``, ``extractor_version`` where available, and
``supporting_evidence_ids`` where available (Req 12.1).

Provenance is keyed by ``subject_id`` (the id of the assertion/claim/document/
quarantine record it describes). A single subject can accrue multiple provenance
records — for example, supersession preserves provenance for both the old
superseded assertion and the new accepted one (Req 12.3) — so :meth:`record`
always inserts a new row and :meth:`for_subject` returns every row for a subject.

At retrieval, the Evidence Packager calls :meth:`for_subject` to populate
``supporting_sources`` (Req 12.2).

Requirements: 12.1, 12.4.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ocm.core.ids import IdGenerator
from ocm.memory.repository import StorageRepository
from ocm.ontology.models import Provenance


class ProvenanceTracker:
    """Records and retrieves provenance, backed by the ``provenance`` table."""

    def __init__(self, repo: StorageRepository, ids: IdGenerator | None = None) -> None:
        """Create a provenance tracker.

        Args:
            repo: The durable :class:`StorageRepository` whose ``provenance``
                table holds the records (Req 12.4).
            ids: Optional :class:`IdGenerator`. When omitted a non-deterministic
                generator is used so each provenance record gets a unique id.
        """
        self.repo = repo
        self.ids = ids if ids is not None else IdGenerator(deterministic=False)

    def record(
        self,
        subject_id: str,
        source_ref: str,
        created_at: datetime | None = None,
        extractor_version: str | None = None,
        supporting_evidence_ids: list[str] | None = None,
    ) -> Provenance:
        """Build and persist a :class:`Provenance` record, returning it.

        Records ``source_ref`` and ``created_at`` always, and
        ``extractor_version`` / ``supporting_evidence_ids`` where available
        (Req 12.1). The record is written to the ``provenance`` table (Req 12.4).

        Args:
            subject_id: Id of the assertion/claim/document/quarantine record
                this provenance describes.
            source_ref: Origin reference of the memory item.
            created_at: Creation timestamp; defaults to ``now`` (UTC).
            extractor_version: Extractor version where available.
            supporting_evidence_ids: Supporting evidence ids where available.

        Returns:
            The persisted :class:`Provenance` record.
        """
        provenance = Provenance(
            id=self.ids.generic_id("prov", subject_id, source_ref),
            subject_id=subject_id,
            source_ref=source_ref,
            created_at=created_at or datetime.now(timezone.utc),
            extractor_version=extractor_version,
            supporting_evidence_ids=list(supporting_evidence_ids or []),
        )
        self.repo.upsert_provenance(provenance)
        return provenance

    def for_subject(self, subject_id: str) -> list[Provenance]:
        """Return every provenance record for ``subject_id``.

        Delegates to ``repo.get_provenance_for`` (Req 12.4). Returning all rows
        preserves provenance for both sides of a supersession (Req 12.3).
        """
        return list(self.repo.get_provenance_for(subject_id))

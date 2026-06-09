"""Quarantine Store (``Quarantine_Store``).

``QuarantineStore`` is a thin facade over the ``quarantine_records`` table,
accessed exclusively through the :class:`~ocm.memory.repository.StorageRepository`
interface. It persists :class:`~ocm.ontology.models.QuarantineRecord` items so
they survive process restarts (Req 11.7) and powers ``GET /memory/conflicts``.

Because every record is written straight to durable storage via
``repo.upsert_quarantine`` and read back via ``repo.list_quarantine``, the store
holds no in-memory state of its own: a fresh ``QuarantineStore`` constructed
against the same repository (e.g. after a restart) sees exactly the records that
were persisted before (Req 11.7).

The store exposes three operations matching the design contract
(``Quarantine_Store.add/list/set_status``):

* :meth:`add` — build and persist a new :class:`QuarantineRecord`.
* :meth:`list` — list records, optionally filtered by status.
* :meth:`set_status` — transition an existing record's status.

The repository surface intentionally offers only ``upsert_quarantine`` and
``list_quarantine`` (no targeted get/update), so :meth:`set_status` loads the
record via :meth:`list`, mutates its status, and re-upserts it.

Requirements: 11.7.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ocm.core.ids import IdGenerator
from ocm.memory.repository import StorageRepository
from ocm.ontology.enums import QuarantineStatus, Severity
from ocm.ontology.models import QuarantineRecord


class QuarantineStore:
    """Durable store of quarantined candidates, backed by the repository."""

    def __init__(self, repo: StorageRepository, ids: IdGenerator | None = None) -> None:
        """Create a quarantine store.

        Args:
            repo: The durable :class:`StorageRepository` (source of truth on
                disk). All reads and writes go through it so records persist
                across restarts (Req 11.7).
            ids: Optional :class:`IdGenerator`. When omitted a non-deterministic
                generator is used so each record gets a unique id.
        """
        self.repo = repo
        self.ids = ids if ids is not None else IdGenerator(deterministic=False)

    def add(
        self,
        candidate_payload: dict,
        reason: str,
        severity: Severity,
        conflicting_ids: list[str] | None = None,
        created_at: datetime | None = None,
        status: QuarantineStatus = QuarantineStatus.unresolved,
        quarantine_id: str | None = None,
    ) -> QuarantineRecord:
        """Build and persist a :class:`QuarantineRecord`, returning it.

        Args:
            candidate_payload: Serialized candidate assertion/entity that was
                quarantined.
            reason: Human-readable reason the candidate was quarantined.
            severity: Severity of the issue (low/medium/high).
            conflicting_ids: Ids of accepted items this candidate conflicts with.
            created_at: Creation timestamp; defaults to ``now`` (UTC).
            status: Initial status; defaults to ``unresolved``.
            quarantine_id: Explicit id; generated when omitted.

        Returns:
            The persisted :class:`QuarantineRecord`.
        """
        record = QuarantineRecord(
            id=quarantine_id or self.ids.generic_id("qua", reason),
            candidate_payload=candidate_payload,
            reason=reason,
            severity=severity,
            conflicting_ids=list(conflicting_ids or []),
            created_at=created_at or datetime.now(timezone.utc),
            status=status,
        )
        self.repo.upsert_quarantine(record)
        return record

    def list(
        self, status: QuarantineStatus | str | None = None
    ) -> list[QuarantineRecord]:
        """List persisted quarantine records, optionally filtered by status.

        Delegates to ``repo.list_quarantine`` so the result reflects durable
        storage and therefore persists across restarts (Req 11.7).

        Args:
            status: Optional :class:`QuarantineStatus` (or its string value) to
                filter on. When ``None`` all records are returned.

        Returns:
            A list of :class:`QuarantineRecord` items.
        """
        status_value = status.value if isinstance(status, QuarantineStatus) else status
        return list(self.repo.list_quarantine(status=status_value))

    def set_status(
        self, quarantine_id: str, status: QuarantineStatus | str
    ) -> QuarantineRecord:
        """Transition a quarantine record's status and persist the change.

        The repository exposes only ``upsert_quarantine`` and
        ``list_quarantine``, so the record is loaded via :meth:`list`, its
        status is mutated, and it is re-upserted.

        Args:
            quarantine_id: Id of the record to update.
            status: New :class:`QuarantineStatus` (or its string value).

        Returns:
            The updated :class:`QuarantineRecord`.

        Raises:
            KeyError: If no record with ``quarantine_id`` exists.
        """
        new_status = (
            status if isinstance(status, QuarantineStatus) else QuarantineStatus(status)
        )
        for record in self.repo.list_quarantine():
            if record.id == quarantine_id:
                updated = record.model_copy(update={"status": new_status})
                self.repo.upsert_quarantine(updated)
                return updated
        raise KeyError(f"No quarantine record with id {quarantine_id!r}")

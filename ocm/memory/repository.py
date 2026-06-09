"""Storage repository interface (``Storage_Repository``).

``StorageRepository`` is an abstract base class that abstracts **all** durable
persistence away from callers (Req 11.3). The default :class:`SQLiteRepository`
(in ``ocm/memory/sqlite_repository.py``) implements it against SQLite (Req 11.2);
a future ``PostgresRepository`` is a drop-in replacement because callers depend
only on this interface, never on a concrete backend (Req 11.4).

The interface covers the seven required tables (Req 11.1): entities, assertions,
claims, documents, quarantine_records, provenance, and embeddings. Ontology
models are serialized with Pydantic v2 (``model_dump_json`` / ``model_validate``)
so the round-trip is lossless and backend-independent.

Requirements: 11.1, 11.2, 11.3, 11.4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from pydantic import BaseModel

from ocm.ontology.models import (
    Assertion,
    Claim,
    Document,
    Provenance,
    QuarantineRecord,
)


class StorageRepository(ABC):
    """Durable persistence interface shared by every caller.

    Implementations provision the seven tables and persist/load the ontology
    models. No SQLite (or any backend) specifics leak through this surface, so
    the backend can move from SQLite to Postgres without changing callers.
    """

    # --- entities ---------------------------------------------------------
    @abstractmethod
    def upsert_entity(self, entity_type: str, entity: BaseModel) -> None:
        """Insert or replace an entity, recording its ``entity_type``.

        The entity's full typed model is stored as a JSON payload so any of the
        entity kinds (Person/Organization/Project/Task/Event/Decision/...) round
        trips through a single table. ``entity_type`` is persisted so the graph
        rebuild and the C9 domain/range checks can recover the type.
        """

    @abstractmethod
    def get_entity(self, entity_id: str) -> tuple[str, dict] | None:
        """Return ``(entity_type, payload)`` for ``entity_id`` or ``None``."""

    @abstractmethod
    def list_entities(self) -> Iterable[tuple[str, dict]]:
        """Yield ``(entity_type, payload)`` for every stored entity."""

    # --- assertions -------------------------------------------------------
    @abstractmethod
    def upsert_assertion(self, a: Assertion) -> None:
        """Insert or replace an assertion (any status is persisted)."""

    @abstractmethod
    def get_assertion(self, assertion_id: str) -> Assertion | None:
        """Return the :class:`Assertion` for ``assertion_id`` or ``None``."""

    @abstractmethod
    def list_assertions(self, status: str | None = None) -> Iterable[Assertion]:
        """Yield assertions, optionally filtered to a single ``status``."""

    @abstractmethod
    def set_assertion_status(self, assertion_id: str, status: str) -> None:
        """Update only the ``status`` of an existing assertion."""

    # --- claims / documents ----------------------------------------------
    @abstractmethod
    def upsert_claim(self, c: Claim) -> None:
        """Insert or replace a claim."""

    @abstractmethod
    def upsert_document(self, d: Document) -> None:
        """Insert or replace a document."""

    @abstractmethod
    def get_claim(self, claim_id: str) -> Claim | None:
        """Return the :class:`Claim` for ``claim_id`` or ``None``."""

    @abstractmethod
    def get_document(self, document_id: str) -> Document | None:
        """Return the :class:`Document` for ``document_id`` or ``None``."""

    # --- quarantine -------------------------------------------------------
    @abstractmethod
    def upsert_quarantine(self, q: QuarantineRecord) -> None:
        """Insert or replace a quarantine record (persists across restarts)."""

    @abstractmethod
    def list_quarantine(self, status: str | None = None) -> Iterable[QuarantineRecord]:
        """Yield quarantine records, optionally filtered to a single ``status``."""

    # --- provenance -------------------------------------------------------
    @abstractmethod
    def upsert_provenance(self, p: Provenance) -> None:
        """Insert or replace a provenance record."""

    @abstractmethod
    def get_provenance_for(self, subject_id: str) -> list[Provenance]:
        """Return all provenance records whose ``subject_id`` matches."""

    # --- embeddings metadata ---------------------------------------------
    @abstractmethod
    def upsert_embedding_meta(
        self, memory_id: str, memory_type: str, status: str, dim: int
    ) -> None:
        """Insert or replace the metadata mirror of a Chroma vector."""

    @abstractmethod
    def list_embedding_meta(self) -> Iterable[dict]:
        """Yield embedding-metadata rows as dicts."""

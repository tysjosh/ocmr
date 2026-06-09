"""SQLite implementation of :class:`StorageRepository`.

``SQLiteRepository`` provisions the seven required tables on construction
(Req 11.1) and implements every persistence operation behind the
:class:`~ocm.memory.repository.StorageRepository` interface so a Postgres
adapter is a drop-in (Req 11.2, 11.3, 11.4). It supports ``:memory:`` mode for
hermetic tests (the ``in_memory_repository`` fixture uses
``SQLiteRepository(":memory:")``); because an in-memory SQLite database lives
only as long as its connection, a single long-lived connection is held for the
repository's lifetime.

Serialization uses Pydantic v2 (``model_dump_json`` / ``model_validate``):
entities are stored as a ``(entity_type, JSON payload)`` pair in one table so
every entity kind round-trips and its type is recoverable for the graph rebuild
and the C9 domain/range checks. Assertions, claims, documents, quarantine
records, and provenance are stored with promoted typed columns and rebuilt into
their models on read.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable

from pydantic import BaseModel

from ocm.memory.repository import StorageRepository
from ocm.ontology.models import (
    Assertion,
    Claim,
    Document,
    Provenance,
    QuarantineRecord,
)

# --- DDL: the seven required tables (Req 11.1) ----------------------------
_SCHEMA = """
-- 1. entities: one row per resolved entity, payload holds the full typed model
CREATE TABLE IF NOT EXISTS entities (
    id              TEXT PRIMARY KEY,
    entity_type     TEXT NOT NULL,
    normalized_name TEXT,
    status          TEXT,
    payload         TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entities_type_name ON entities(entity_type, normalized_name);

-- 2. assertions: typed graph statements (all statuses persisted)
CREATE TABLE IF NOT EXISTS assertions (
    id                TEXT PRIMARY KEY,
    subject_id        TEXT NOT NULL,
    predicate         TEXT NOT NULL,
    object_id         TEXT NOT NULL,
    confidence        REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    status            TEXT NOT NULL,
    source_ref        TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    valid_from        TEXT,
    valid_to          TEXT,
    extractor_version TEXT,
    write_intent      TEXT NOT NULL,
    supersedes_id     TEXT,
    status_defaulted  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_assertions_status       ON assertions(status);
CREATE INDEX IF NOT EXISTS idx_assertions_subject_pred ON assertions(subject_id, predicate);

-- 3. claims
CREATE TABLE IF NOT EXISTS claims (
    id          TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    source_ref  TEXT NOT NULL,
    confidence  REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    status_defaulted INTEGER NOT NULL DEFAULT 0
);

-- 4. documents
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    path_or_url TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    tags        TEXT NOT NULL DEFAULT '[]'
);

-- 5. quarantine_records (persist across restarts, Req 11.7)
CREATE TABLE IF NOT EXISTS quarantine_records (
    id                TEXT PRIMARY KEY,
    candidate_payload TEXT NOT NULL,
    reason            TEXT NOT NULL,
    severity          TEXT NOT NULL,
    conflicting_ids   TEXT NOT NULL DEFAULT '[]',
    created_at        TEXT NOT NULL,
    status            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_quarantine_status ON quarantine_records(status);

-- 6. provenance (Req 12.4)
CREATE TABLE IF NOT EXISTS provenance (
    id                      TEXT PRIMARY KEY,
    subject_id              TEXT NOT NULL,
    source_ref              TEXT NOT NULL,
    created_at              TEXT NOT NULL,
    extractor_version       TEXT,
    supporting_evidence_ids TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_provenance_subject ON provenance(subject_id);

-- 7. embeddings (metadata mirror of the Chroma vectors)
CREATE TABLE IF NOT EXISTS embeddings (
    memory_id   TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL,
    status      TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteRepository(StorageRepository):
    """SQLite-backed :class:`StorageRepository`.

    :param path: filesystem path to the database file, or ``":memory:"`` for an
        ephemeral in-memory database (used by tests).
    """

    def __init__(self, path: str = "ocm.db") -> None:
        self.path = path
        # A single long-lived connection: required for ``:memory:`` (the DB is
        # discarded when the last connection closes) and fine for file mode.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    # -- lifecycle ---------------------------------------------------------
    def _create_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _entity_field(entity: BaseModel, *names: str) -> Any:
        for name in names:
            if hasattr(entity, name):
                value = getattr(entity, name)
                if value is not None:
                    return value.value if hasattr(value, "value") else value
        return None

    # --- entities ---------------------------------------------------------
    def upsert_entity(self, entity_type: str, entity: BaseModel) -> None:
        payload = entity.model_dump_json()
        normalized_name = self._entity_field(entity, "name", "title")
        status = self._entity_field(entity, "status")
        entity_id = getattr(entity, "id")
        self._conn.execute(
            """
            INSERT INTO entities (id, entity_type, normalized_name, status, payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                entity_type=excluded.entity_type,
                normalized_name=excluded.normalized_name,
                status=excluded.status,
                payload=excluded.payload
            """,
            (entity_id, entity_type, normalized_name, status, payload, _now_iso()),
        )
        self._conn.commit()

    def get_entity(self, entity_id: str) -> tuple[str, dict] | None:
        row = self._conn.execute(
            "SELECT entity_type, payload FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        if row is None:
            return None
        return row["entity_type"], json.loads(row["payload"])

    def list_entities(self) -> Iterable[tuple[str, dict]]:
        rows = self._conn.execute(
            "SELECT entity_type, payload FROM entities"
        ).fetchall()
        return [(r["entity_type"], json.loads(r["payload"])) for r in rows]

    # --- assertions -------------------------------------------------------
    def upsert_assertion(self, a: Assertion) -> None:
        self._conn.execute(
            """
            INSERT INTO assertions (
                id, subject_id, predicate, object_id, confidence, status,
                source_ref, created_at, valid_from, valid_to, extractor_version,
                write_intent, supersedes_id, status_defaulted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                subject_id=excluded.subject_id,
                predicate=excluded.predicate,
                object_id=excluded.object_id,
                confidence=excluded.confidence,
                status=excluded.status,
                source_ref=excluded.source_ref,
                created_at=excluded.created_at,
                valid_from=excluded.valid_from,
                valid_to=excluded.valid_to,
                extractor_version=excluded.extractor_version,
                write_intent=excluded.write_intent,
                status_defaulted=excluded.status_defaulted
            """,
            (
                a.id,
                a.subject_id,
                a.predicate,
                a.object_id,
                float(a.confidence),
                a.status.value,
                a.source_ref,
                a.created_at.isoformat(),
                a.valid_from.isoformat() if a.valid_from else None,
                a.valid_to.isoformat() if a.valid_to else None,
                a.extractor_version,
                a.write_intent.value,
                None,
                1 if a.status_defaulted else 0,
            ),
        )
        self._conn.commit()

    @staticmethod
    def _row_to_assertion(row: sqlite3.Row) -> Assertion:
        return Assertion.model_validate(
            {
                "id": row["id"],
                "subject_id": row["subject_id"],
                "predicate": row["predicate"],
                "object_id": row["object_id"],
                "confidence": row["confidence"],
                "status": row["status"],
                "source_ref": row["source_ref"],
                "created_at": row["created_at"],
                "valid_from": row["valid_from"],
                "valid_to": row["valid_to"],
                "extractor_version": row["extractor_version"],
                "write_intent": row["write_intent"],
                "status_defaulted": bool(row["status_defaulted"]),
            }
        )

    def get_assertion(self, assertion_id: str) -> Assertion | None:
        row = self._conn.execute(
            "SELECT * FROM assertions WHERE id = ?", (assertion_id,)
        ).fetchone()
        return self._row_to_assertion(row) if row else None

    def list_assertions(self, status: str | None = None) -> Iterable[Assertion]:
        if status is None:
            rows = self._conn.execute("SELECT * FROM assertions").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM assertions WHERE status = ?", (status,)
            ).fetchall()
        return [self._row_to_assertion(r) for r in rows]

    def set_assertion_status(self, assertion_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE assertions SET status = ? WHERE id = ?", (status, assertion_id)
        )
        self._conn.commit()

    # --- claims / documents ----------------------------------------------
    def upsert_claim(self, c: Claim) -> None:
        self._conn.execute(
            """
            INSERT INTO claims (id, text, source_ref, confidence, status, created_at, status_defaulted)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                text=excluded.text,
                source_ref=excluded.source_ref,
                confidence=excluded.confidence,
                status=excluded.status,
                created_at=excluded.created_at,
                status_defaulted=excluded.status_defaulted
            """,
            (
                c.id,
                c.text,
                c.source_ref,
                float(c.confidence),
                c.status.value,
                c.created_at.isoformat(),
                1 if c.status_defaulted else 0,
            ),
        )
        self._conn.commit()

    def get_claim(self, claim_id: str) -> Claim | None:
        row = self._conn.execute(
            "SELECT * FROM claims WHERE id = ?", (claim_id,)
        ).fetchone()
        if row is None:
            return None
        return Claim.model_validate(
            {
                "id": row["id"],
                "text": row["text"],
                "source_ref": row["source_ref"],
                "confidence": row["confidence"],
                "status": row["status"],
                "created_at": row["created_at"],
                "status_defaulted": bool(row["status_defaulted"]),
            }
        )

    def upsert_document(self, d: Document) -> None:
        self._conn.execute(
            """
            INSERT INTO documents (id, title, path_or_url, created_at, tags)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                path_or_url=excluded.path_or_url,
                created_at=excluded.created_at,
                tags=excluded.tags
            """,
            (
                d.id,
                d.title,
                d.path_or_url,
                d.created_at.isoformat(),
                json.dumps(d.tags),
            ),
        )
        self._conn.commit()

    def get_document(self, document_id: str) -> Document | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        if row is None:
            return None
        return Document.model_validate(
            {
                "id": row["id"],
                "title": row["title"],
                "path_or_url": row["path_or_url"],
                "created_at": row["created_at"],
                "tags": json.loads(row["tags"]),
            }
        )

    # --- quarantine -------------------------------------------------------
    def upsert_quarantine(self, q: QuarantineRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO quarantine_records
                (id, candidate_payload, reason, severity, conflicting_ids, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                candidate_payload=excluded.candidate_payload,
                reason=excluded.reason,
                severity=excluded.severity,
                conflicting_ids=excluded.conflicting_ids,
                created_at=excluded.created_at,
                status=excluded.status
            """,
            (
                q.id,
                json.dumps(q.candidate_payload),
                q.reason,
                q.severity.value,
                json.dumps(q.conflicting_ids),
                q.created_at.isoformat(),
                q.status.value,
            ),
        )
        self._conn.commit()

    def list_quarantine(self, status: str | None = None) -> Iterable[QuarantineRecord]:
        if status is None:
            rows = self._conn.execute("SELECT * FROM quarantine_records").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM quarantine_records WHERE status = ?", (status,)
            ).fetchall()
        return [
            QuarantineRecord.model_validate(
                {
                    "id": r["id"],
                    "candidate_payload": json.loads(r["candidate_payload"]),
                    "reason": r["reason"],
                    "severity": r["severity"],
                    "conflicting_ids": json.loads(r["conflicting_ids"]),
                    "created_at": r["created_at"],
                    "status": r["status"],
                }
            )
            for r in rows
        ]

    # --- provenance -------------------------------------------------------
    def upsert_provenance(self, p: Provenance) -> None:
        self._conn.execute(
            """
            INSERT INTO provenance
                (id, subject_id, source_ref, created_at, extractor_version, supporting_evidence_ids)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                subject_id=excluded.subject_id,
                source_ref=excluded.source_ref,
                created_at=excluded.created_at,
                extractor_version=excluded.extractor_version,
                supporting_evidence_ids=excluded.supporting_evidence_ids
            """,
            (
                p.id,
                p.subject_id,
                p.source_ref,
                p.created_at.isoformat(),
                p.extractor_version,
                json.dumps(p.supporting_evidence_ids),
            ),
        )
        self._conn.commit()

    def get_provenance_for(self, subject_id: str) -> list[Provenance]:
        rows = self._conn.execute(
            "SELECT * FROM provenance WHERE subject_id = ?", (subject_id,)
        ).fetchall()
        return [
            Provenance.model_validate(
                {
                    "id": r["id"],
                    "subject_id": r["subject_id"],
                    "source_ref": r["source_ref"],
                    "created_at": r["created_at"],
                    "extractor_version": r["extractor_version"],
                    "supporting_evidence_ids": json.loads(r["supporting_evidence_ids"]),
                }
            )
            for r in rows
        ]

    # --- embeddings metadata ---------------------------------------------
    def upsert_embedding_meta(
        self, memory_id: str, memory_type: str, status: str, dim: int
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO embeddings (memory_id, memory_type, status, dim, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                memory_type=excluded.memory_type,
                status=excluded.status,
                dim=excluded.dim
            """,
            (memory_id, memory_type, status, dim, _now_iso()),
        )
        self._conn.commit()

    def list_embedding_meta(self) -> Iterable[dict]:
        rows = self._conn.execute("SELECT * FROM embeddings").fetchall()
        return [dict(r) for r in rows]

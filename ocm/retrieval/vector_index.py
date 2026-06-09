"""Chroma-backed semantic ``Vector_Index`` (Req 13.4, 13.5, 13.6, 16.6).

The :class:`VectorIndex` is the semantic half of retrieval: it stores a dense
embedding per memory item together with the metadata the retrieval pipeline
filters on (``{memory_id, memory_type, status}``) and answers nearest-neighbour
queries ranked by cosine similarity.

Storage backends
----------------
- **persistent** (default) — a local on-disk Chroma collection that survives
  process restarts (Req 13.4, 13.6): ``chromadb.PersistentClient(path=...)``.
- **memory** — an ephemeral, in-process Chroma collection for hermetic tests
  (``settings.chroma_mode == "memory"``, Req 13.6):
  ``chromadb.EphemeralClient()``.

Offline / hermetic fallback
---------------------------
``chromadb`` is the intended backend, but it is a heavy dependency that may not
be installed in every environment. ``chromadb`` is therefore imported lazily in
the constructor; if it is unavailable a dependency-free, pure-Python in-memory
index (:class:`_FallbackCollection`) implementing the same ``add`` / ``upsert``
/ ``query`` surface is used instead, so the vector-index and retrieval tests run
hermetically without Chroma installed. The fallback uses the same cosine space
and the same metadata filtering semantics as Chroma.

What gets embedded (Req 13.5, 16.6)
-----------------------------------
When a Claim, Document, accepted Assertion, or Event is accepted, the write path
embeds it via one of the connect hooks:

- :meth:`embed_assertion` — the ``Commit_Manager`` ``embed_hook`` for accepted
  assertions (Req 13.5).
- :meth:`embed_memory` — the ``WritePipeline`` ``memory_embed_hook`` for accepted
  claims / documents / events (Req 16.6).

The embedding text is a compact natural-language rendering of the item (see the
``build_*_text`` helpers). For an assertion the subject/object **names** are
resolved from the ``Graph_Store`` (when wired) so the embedded text is
semantically meaningful — ``Assertion(subject=per_x, OWNS, object=prj_y)``
embeds as ``"Alice OWNS Project Orion"`` rather than raw ids.

Requirements: 13.4, 13.5, 13.6, 16.6.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from pydantic import BaseModel

from ocm.ontology.models import Assertion, Claim, Document, Event
from ocm.retrieval.embeddings import EmbeddingProvider

#: Default Chroma collection name (matches the design snippet).
DEFAULT_COLLECTION = "ocm_memory"

#: Memory-type tags carried in each vector's metadata (Req 13.5).
MEMORY_TYPE_ASSERTION = "assertion"
MEMORY_TYPE_CLAIM = "claim"
MEMORY_TYPE_DOCUMENT = "document"
MEMORY_TYPE_EVENT = "event"

#: Default status for an accepted memory item embedded by the write path.
STATUS_ACCEPTED = "accepted"
STATUS_QUARANTINED = "quarantined"


class VectorHit(BaseModel):
    """A single semantic match returned by :meth:`VectorIndex.query`.

    ``similarity`` is a cosine-derived score in ``[0, 1]`` (``1.0`` = identical
    direction) used directly by the reranker (R3).
    """

    memory_id: str
    memory_type: str
    status: str
    similarity: float
    text: Optional[str] = None


# ---------------------------------------------------------------------------
# Embedding-text construction helpers (Req 13.5, 16.6)
# ---------------------------------------------------------------------------
def build_assertion_text(assertion: Assertion, graph: Any | None = None) -> str:
    """Render an assertion as ``"<subject_name> <PREDICATE> <object_name>"``.

    When a ``Graph_Store`` is supplied the subject/object **names** are resolved
    from their node payloads so the embedded text is semantically meaningful;
    otherwise the raw ids are used as a fallback.
    """
    subject = _resolve_entity_name(graph, assertion.subject_id)
    obj = _resolve_entity_name(graph, assertion.object_id)
    return f"{subject} {assertion.predicate} {obj}"


def build_claim_text(claim: Claim) -> str:
    """Render a claim as its verbatim ``text``."""
    return claim.text


def build_document_text(document: Document) -> str:
    """Render a document as ``"<title> :: <tags joined>"`` (title only if no tags)."""
    if document.tags:
        return f"{document.title} :: {', '.join(document.tags)}"
    return document.title


def build_event_text(event: Event) -> str:
    """Render an event as ``"<type>: <description> @ <timestamp_start>"``."""
    return f"{event.type}: {event.description} @ {event.timestamp_start.isoformat()}"


def _resolve_entity_name(graph: Any | None, entity_id: str) -> str:
    """Resolve an entity id to its display name via the graph, else the id.

    Looks at the node payload's ``name`` / ``title`` / ``summary`` /
    ``description`` (in that order), matching how the write path stores names.
    """
    if graph is None:
        return entity_id
    payload = None
    try:
        payload = graph.get_entity_payload(entity_id)
    except Exception:  # pragma: no cover - defensive: graph without this method
        payload = None
    if not payload:
        return entity_id
    for key in ("name", "title", "summary", "description"):
        value = payload.get(key)
        if value:
            return str(value)
    return entity_id


# ---------------------------------------------------------------------------
# VectorIndex
# ---------------------------------------------------------------------------
class VectorIndex:
    """A Chroma collection (or pure-Python fallback) over embedded memory items."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        chroma_mode: str = "memory",
        chroma_path: str = ".chroma",
        collection_name: str = DEFAULT_COLLECTION,
        graph: Any | None = None,
    ) -> None:
        """Open (or create) the backing collection.

        Args:
            provider: The swappable :class:`EmbeddingProvider` used to embed
                documents and queries (Req 13.1).
            chroma_mode: ``"persistent"`` for an on-disk collection that survives
                restarts (Req 13.4, 13.6) or ``"memory"`` for an ephemeral,
                in-process collection for hermetic tests (Req 13.6).
            chroma_path: On-disk directory for ``persistent`` mode.
            collection_name: Chroma collection name.
            graph: Optional ``Graph_Store`` used to resolve assertion
                subject/object ids to names for embedding text. May be set later
                via :meth:`set_graph`.

        Raises:
            ValueError: If ``chroma_mode`` is not ``"persistent"`` or ``"memory"``.
        """
        if chroma_mode not in ("persistent", "memory"):
            raise ValueError(
                f"chroma_mode must be 'persistent' or 'memory', got {chroma_mode!r}"
            )
        self.provider = provider
        self.chroma_mode = chroma_mode
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.graph = graph
        self._using_fallback = False
        self.col = self._open_collection()

    def set_graph(self, graph: Any) -> None:
        """Wire (or replace) the ``Graph_Store`` used to resolve assertion names."""
        self.graph = graph

    @property
    def using_fallback(self) -> bool:
        """Whether the pure-Python fallback index is in use (no ``chromadb``)."""
        return self._using_fallback

    # -- backend selection -------------------------------------------------
    def _open_collection(self) -> Any:
        """Open a Chroma collection, falling back to a pure-Python index.

        ``chromadb`` is imported lazily so importing this module never requires
        the dependency. If it is missing, an in-memory fallback with the same
        ``add`` / ``upsert`` / ``query`` interface is used so tests stay
        hermetic.
        """
        try:
            import chromadb  # type: ignore
        except ImportError:
            self._using_fallback = True
            return _FallbackCollection()

        if self.chroma_mode == "persistent":
            client = chromadb.PersistentClient(path=self.chroma_path)
        else:
            client = chromadb.EphemeralClient()
        return client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # -- write surface (Req 13.5) -----------------------------------------
    def add(
        self,
        memory_id: str,
        text: str,
        memory_type: str,
        status: str = STATUS_ACCEPTED,
    ) -> None:
        """Embed ``text`` and upsert it with ``{memory_id, memory_type, status}``.

        Uses ``upsert`` so re-embedding the same ``memory_id`` (e.g. after a
        status change) replaces the existing vector rather than duplicating it.
        """
        embedding = self.provider.embed_one(text)
        metadata = {
            "memory_id": memory_id,
            "memory_type": memory_type,
            "status": status,
        }
        upsert = getattr(self.col, "upsert", None)
        if callable(upsert):
            upsert(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata],
            )
        else:  # pragma: no cover - older chroma without upsert
            self.col.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata],
            )

    def set_status(self, memory_id: str, status: str) -> None:
        """Update the ``status`` metadata of an embedded item in place.

        Used by the Commit_Manager when an assertion is superseded so the
        vector index stays consistent with durable storage: the superseded
        assertion is re-tagged ``status="superseded"`` and therefore drops out
        of the default accepted-only semantic results (Req 10.3, 10.5, 16.2),
        while remaining retrievable for conflict/provenance queries.

        A no-op when ``memory_id`` was never embedded (e.g. embeddings disabled).
        The embedding itself is preserved — only metadata changes.
        """
        existing = self._get_metadata(memory_id)
        if existing is None:
            return
        meta = dict(existing)
        meta["status"] = status
        updater = getattr(self.col, "update", None)
        if callable(updater):
            updater(ids=[memory_id], metadatas=[meta])
        else:  # pragma: no cover - backend without update: re-add via stored doc
            doc = self._get_document(memory_id)
            if doc is not None:
                self.add(memory_id, doc, meta.get("memory_type", ""), status)

    def delete(self, memory_id: str) -> None:
        """Remove an embedded item from the index (no-op when absent)."""
        deleter = getattr(self.col, "delete", None)
        if callable(deleter):
            deleter(ids=[memory_id])

    def _get_metadata(self, memory_id: str) -> dict | None:
        """Return the stored metadata dict for ``memory_id`` or ``None``."""
        getter = getattr(self.col, "get", None)
        if not callable(getter):  # pragma: no cover - backend without get
            return None
        res = getter(ids=[memory_id])
        metadatas = (res or {}).get("metadatas") or []
        return dict(metadatas[0]) if metadatas else None

    def _get_document(self, memory_id: str) -> str | None:
        """Return the stored embedding text for ``memory_id`` or ``None``."""
        getter = getattr(self.col, "get", None)
        if not callable(getter):  # pragma: no cover - backend without get
            return None
        res = getter(ids=[memory_id])
        documents = (res or {}).get("documents") or []
        return documents[0] if documents else None

    # -- read surface (Req 16.6) ------------------------------------------
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[VectorHit]:
        """Embed ``query_text`` and return the ``top_k`` nearest hits.

        ``where`` is a Chroma-style metadata filter (e.g. ``{"status":
        "accepted"}`` or ``{"memory_type": "claim"}``) used to constrain results
        — the Semantic Retriever uses it to keep accepted-only memory by default
        and to include quarantined items for conflict queries.
        """
        if top_k <= 0:
            return []
        embedding = self.provider.embed_one(query_text)
        res = self.col.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where or None,
        )
        return self._hits_from_result(res)

    @staticmethod
    def _hits_from_result(res: dict) -> list[VectorHit]:
        """Convert a Chroma query result into ranked :class:`VectorHit` objects.

        Cosine distance is converted to ``similarity = 1.0 - distance`` (the
        design's canonical conversion) and clamped to ``[0, 1]`` so the score is
        a well-behaved reranker input even when the cosine angle is obtuse.
        """
        metadatas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        documents = (res.get("documents") or [[]])[0] if res.get("documents") else []

        hits: list[VectorHit] = []
        for idx, (meta, dist) in enumerate(zip(metadatas, distances)):
            similarity = 1.0 - float(dist)
            if similarity < 0.0:
                similarity = 0.0
            elif similarity > 1.0:
                similarity = 1.0
            text = documents[idx] if idx < len(documents) else None
            hits.append(
                VectorHit(
                    memory_id=meta["memory_id"],
                    memory_type=meta["memory_type"],
                    status=meta["status"],
                    similarity=similarity,
                    text=text,
                )
            )
        return hits

    # -- Commit_Manager / WritePipeline hooks (Req 13.5, 16.6) ------------
    def embed_assertion(self, assertion: Assertion) -> None:
        """``Commit_Manager`` embed hook for an accepted assertion (Req 13.5).

        Embeds ``build_assertion_text`` with ``memory_type="assertion"`` and the
        assertion's own status (accepted assertions are the only ones the commit
        manager passes here).

        ``HAS_STATUS`` assertions are intentionally *not* embedded: a status is a
        structural fact answered through the symbolic retriever (an exact graph
        lookup), and its synthetic text ("T1 HAS_STATUS done") adds no useful
        semantic signal while risking false matches for unrelated queries. Status
        conflicts surface through the graph + Quarantine_Store, not the vector
        index.
        """
        if assertion.predicate == "HAS_STATUS":
            return
        text = build_assertion_text(assertion, self.graph)
        status = getattr(assertion.status, "value", assertion.status)
        self.add(assertion.id, text, MEMORY_TYPE_ASSERTION, str(status))

    def embed_memory(self, memory_type: str, model: BaseModel) -> None:
        """``WritePipeline`` memory embed hook for claim / document / event (Req 16.6).

        ``memory_type`` is the extraction-side label ("Claim" / "Document" /
        "Event", case-insensitive); the stored metadata tag is normalized to the
        lower-case canonical form.
        """
        key = memory_type.strip().lower()
        if key == MEMORY_TYPE_CLAIM and isinstance(model, Claim):
            text = build_claim_text(model)
            status = getattr(model.status, "value", STATUS_ACCEPTED)
            self.add(model.id, text, MEMORY_TYPE_CLAIM, str(status))
        elif key == MEMORY_TYPE_DOCUMENT and isinstance(model, Document):
            self.add(model.id, build_document_text(model), MEMORY_TYPE_DOCUMENT, STATUS_ACCEPTED)
        elif key == MEMORY_TYPE_EVENT and isinstance(model, Event):
            self.add(model.id, build_event_text(model), MEMORY_TYPE_EVENT, STATUS_ACCEPTED)
        else:  # pragma: no cover - defensive: unknown memory type / mismatch
            raise ValueError(
                f"Cannot embed memory_type={memory_type!r} with model "
                f"{type(model).__name__}"
            )


# ---------------------------------------------------------------------------
# Pure-Python in-memory fallback collection (no chromadb required)
# ---------------------------------------------------------------------------
class _FallbackCollection:
    """A minimal in-memory stand-in for a Chroma cosine collection.

    Implements just the ``add`` / ``upsert`` / ``query`` surface the
    :class:`VectorIndex` uses, with the same cosine space and ``where`` metadata
    filtering semantics. Vectors are L2-normalized on insert so cosine distance
    is ``1 - dot(query_unit, stored_unit)`` — identical to Chroma's cosine
    space. Used only when ``chromadb`` is not installed, keeping the
    vector-index and retrieval tests hermetic.
    """

    def __init__(self) -> None:
        # memory_id -> (unit_vector, metadata, document)
        self._items: dict[str, tuple[list[float], dict, str]] = {}

    def upsert(self, ids, embeddings, documents=None, metadatas=None) -> None:
        documents = documents or [None] * len(ids)
        metadatas = metadatas or [{} for _ in ids]
        for _id, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            self._items[_id] = (_normalize(list(emb)), dict(meta), doc)

    # add behaves as upsert here (replace by id) so re-adds never duplicate.
    add = upsert

    def get(self, ids=None, **kwargs) -> dict:
        """Return stored metadata/documents for ``ids`` (or all when ``None``).

        Mirrors the subset of Chroma's ``get`` the :class:`VectorIndex` uses for
        metadata-only status updates.
        """
        if ids is None:
            selected = list(self._items.items())
        else:
            selected = [(i, self._items[i]) for i in ids if i in self._items]
        return {
            "ids": [i for i, _ in selected],
            "metadatas": [dict(meta) for _, (_vec, meta, _doc) in selected],
            "documents": [doc for _, (_vec, _meta, doc) in selected],
        }

    def update(self, ids, metadatas=None, embeddings=None, documents=None) -> None:
        """Update metadata/embedding/document for existing ids (missing ids skipped)."""
        for idx, _id in enumerate(ids):
            if _id not in self._items:
                continue
            vec, meta, doc = self._items[_id]
            if metadatas is not None:
                meta = dict(metadatas[idx])
            if embeddings is not None:
                vec = _normalize(list(embeddings[idx]))
            if documents is not None:
                doc = documents[idx]
            self._items[_id] = (vec, meta, doc)

    def delete(self, ids) -> None:
        """Remove the given ids from the index (missing ids ignored)."""
        for _id in ids:
            self._items.pop(_id, None)

    def query(self, query_embeddings, n_results=10, where=None) -> dict:
        query_vec = _normalize(list(query_embeddings[0]))
        scored: list[tuple[float, str, dict, str]] = []
        for _id, (vec, meta, doc) in self._items.items():
            if not _matches_where(meta, where):
                continue
            distance = 1.0 - _dot(query_vec, vec)
            scored.append((distance, _id, meta, doc))
        # Ascending distance == descending similarity (nearest first).
        scored.sort(key=lambda row: row[0])
        top = scored[: max(0, n_results)]
        return {
            "ids": [[row[1] for row in top]],
            "metadatas": [[row[2] for row in top]],
            "distances": [[row[0] for row in top]],
            "documents": [[row[3] for row in top]],
        }


def _normalize(vec: list[float]) -> list[float]:
    """Return the L2-normalized vector (unit length); zero vector unchanged."""
    norm = math.sqrt(sum(component * component for component in vec))
    if norm == 0.0:
        return vec
    return [component / norm for component in vec]


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product of two equal-length vectors."""
    return sum(x * y for x, y in zip(a, b))


def _matches_where(meta: dict, where: dict | None) -> bool:
    """Evaluate a Chroma-style ``where`` filter against a metadata dict.

    Supports the subset OCM uses: top-level field equality (``{"status":
    "accepted"}``), explicit operators (``{"field": {"$eq": v}}`` /
    ``{"field": {"$ne": v}}`` / ``{"field": {"$in": [...]}}`` /
    ``{"field": {"$nin": [...]}}``) and the boolean combinators ``$and`` /
    ``$or``.
    """
    if not where:
        return True
    for key, condition in where.items():
        if key == "$and":
            if not all(_matches_where(meta, sub) for sub in condition):
                return False
        elif key == "$or":
            if not any(_matches_where(meta, sub) for sub in condition):
                return False
        elif isinstance(condition, dict):
            if not _matches_operator(meta.get(key), condition):
                return False
        else:
            if meta.get(key) != condition:
                return False
    return True


def _matches_operator(value: Any, condition: dict) -> bool:
    """Evaluate a single ``{$op: operand}`` metadata condition."""
    for op, operand in condition.items():
        if op == "$eq" and value != operand:
            return False
        if op == "$ne" and value == operand:
            return False
        if op == "$in" and value not in operand:
            return False
        if op == "$nin" and value in operand:
            return False
    return True

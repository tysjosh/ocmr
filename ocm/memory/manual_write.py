"""Minimal manual write path (pre-validator end-to-end storage exercise).

This module wires the persistence layer together *before* the validation and
commit pipeline (W5–W8) exists, so the storage end-to-end path can be exercised
in isolation. Given a set of pre-resolved entities and a single
:class:`~ocm.memory.contracts.CandidateAssertion`, :func:`manual_write`:

1. persists each entity through the :class:`~ocm.memory.repository.StorageRepository`
   and mirrors it as a node in the :class:`~ocm.memory.graph_store.GraphStore`
   (Req 11.6 — write-through), and
2. promotes the candidate to an **accepted** :class:`~ocm.ontology.models.Assertion`
   (operation ``upsert_assertion``, Req 6.1), persists it, and reflects it as an
   edge in the graph (Req 11.6) — keeping the standing invariant that graph
   edges equal the ``accepted`` assertion rows (Req 11.5).

It deliberately performs **no validation, contradiction checking, or
supersession** — those land with the real Commit Manager. It mirrors only the
*accept* leg of the design's write-through-on-commit contract.

Requirements: 6.1, 11.5, 11.6.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Union

from pydantic import BaseModel

from ocm.core.ids import IdGenerator
from ocm.memory.contracts import CandidateAssertion
from ocm.memory.graph_store import GraphStore
from ocm.memory.repository import StorageRepository
from ocm.ontology.enums import AssertionStatus
from ocm.ontology.models import Assertion


@dataclass(frozen=True)
class ResolvedEntity:
    """A pre-resolved entity paired with its ontology ``entity_type``.

    ``entity_type`` (e.g. ``"Person"``, ``"Project"``) is carried alongside the
    typed model because both the repository and the graph need it to recover the
    entity kind for later domain/range checks and rebuilds.
    """

    entity_type: str
    entity: BaseModel


# An entity may be supplied as a ``ResolvedEntity`` or a plain
# ``(entity_type, model)`` tuple for convenience.
EntityInput = Union[ResolvedEntity, tuple[str, BaseModel]]


def _coerce_entity(item: EntityInput) -> ResolvedEntity:
    """Normalize an entity input into a :class:`ResolvedEntity`."""
    if isinstance(item, ResolvedEntity):
        return item
    try:
        entity_type, entity = item
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Each entity must be a ResolvedEntity or a (entity_type, model) "
            f"tuple; got {item!r}"
        ) from exc
    return ResolvedEntity(entity_type=entity_type, entity=entity)


def assertion_from_candidate(
    candidate: CandidateAssertion,
    ids: IdGenerator,
    *,
    created_at: datetime | None = None,
) -> Assertion:
    """Build an **accepted** :class:`Assertion` from a candidate.

    A generated id (deterministic or random per ``ids``) and a ``created_at``
    timestamp (defaulting to ``now(UTC)``) are stamped on; status is forced to
    ``accepted`` (this minimal path has no validators to gate it).
    """
    assertion_id = ids.assertion_id(
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
        created_at=created_at or datetime.now(timezone.utc),
        valid_from=candidate.valid_from,
        valid_to=candidate.valid_to,
        extractor_version=candidate.extractor_version,
        write_intent=candidate.write_intent,
    )


def manual_write(
    entities: Iterable[EntityInput],
    candidate: CandidateAssertion,
    repo: StorageRepository,
    graph: GraphStore,
    ids: IdGenerator,
    *,
    created_at: datetime | None = None,
) -> Assertion:
    """Persist pre-resolved entities and one accepted assertion (write-through).

    Args:
        entities: Pre-resolved entities as :class:`ResolvedEntity` items or
            ``(entity_type, model)`` tuples. Each is upserted to the repository
            and mirrored as a graph node.
        candidate: The proposed assertion to accept and persist.
        repo: Durable storage backend (source of truth).
        graph: In-memory accepted-only projection kept in lock-step (Req 11.6).
        ids: Generator used to mint the assertion id.
        created_at: Optional fixed creation timestamp (defaults to ``now(UTC)``).

    Returns:
        The accepted :class:`Assertion` that was persisted and added to the
        graph.
    """
    # 1) Persist entities to durable storage and mirror them in the graph.
    for item in entities:
        resolved = _coerce_entity(item)
        repo.upsert_entity(resolved.entity_type, resolved.entity)
        graph.add_entity(resolved.entity_type, resolved.entity)

    # 2) Promote the candidate to an accepted assertion and write it through to
    #    both stores (repository row + graph edge). The graph rejects any
    #    non-accepted status, enforcing the accepted-only edge invariant.
    assertion = assertion_from_candidate(candidate, ids, created_at=created_at)
    repo.upsert_assertion(assertion)
    graph.add_assertion(assertion)
    return assertion

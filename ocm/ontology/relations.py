"""Relation signature registry and task status transition map.

This module is the single source of truth for the directed relation
signatures (Req 2.1-2.14) and the task status transition map (Req 8.11)
that the OCM ontology layer exposes.

- ``RELATION_SIGNATURES`` declares every one of the 13 relations with its
  allowed source types, target types, and cardinality.
- ``get_relation_signature`` is the registry lookup API (Req 2.14); it raises
  ``UnknownPredicateError`` for unregistered predicates.
- ``TASK_STATUS_TRANSITIONS`` declares the permitted task status transitions
  that drive constraint C10 (Req 8.11).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

try:  # Prefer the canonical enum once the ontology enums module exists.
    from ocm.ontology.enums import TaskStatus
except ImportError:  # pragma: no cover - fallback until enums.py lands.
    class TaskStatus(str, Enum):
        """Fallback TaskStatus mirroring the ontology enum (Req 1.4).

        Replaced automatically by ``ocm.ontology.enums.TaskStatus`` once that
        module is available. Members must stay in sync with the design.
        """

        todo = "todo"
        in_progress = "in_progress"
        blocked = "blocked"
        done = "done"
        cancelled = "cancelled"
        unknown = "unknown"


class UnknownPredicateError(KeyError):
    """Raised when a predicate is not present in ``RELATION_SIGNATURES``."""

    def __init__(self, predicate: str) -> None:
        self.predicate = predicate
        super().__init__(f"Unknown relation predicate: {predicate!r}")


class Cardinality(str, Enum):
    """Directed relation cardinality."""

    ONE_TO_ONE = "1:1"
    ONE_TO_N = "1:n"
    M_TO_ONE = "m:1"
    M_TO_N = "m:n"


@dataclass(frozen=True)
class RelationSignature:
    """Directed signature for a relation predicate.

    ``source_types`` and ``target_types`` are sets of entity-type names;
    ``cardinality`` constrains how many targets a subject may have.
    """

    predicate: str
    source_types: frozenset[str]
    target_types: frozenset[str]
    cardinality: Cardinality


def _sig(
    predicate: str,
    source_types: set[str],
    target_types: set[str],
    cardinality: Cardinality,
) -> RelationSignature:
    return RelationSignature(
        predicate=predicate,
        source_types=frozenset(source_types),
        target_types=frozenset(target_types),
        cardinality=cardinality,
    )


# Frozen registry of all 15 relations: the 13 specified in Req 2.1-2.13 (incl.
# SUPERSEDES 2.13), plus two derived ones — HAS_STATUS (Req 8.11, promotes an
# entity's status to a first-class assertion) and HAS_VALUE (added with the
# MultiWOZ adapter, maps an external single-valued field onto governed memory).
#
# Three are single-valued (m:1): ASSIGNED_TO, HAS_STATUS, HAS_VALUE. Those are the
# only relations on which a second distinct object is a contradiction, and so the
# only ones the durable-violation metric can measure.
#
# Rendered as a diagram by ``python -m ocm.scripts.render_ontology_graph``; re-run
# it after changing this registry (``--check`` fails if docs/ontology_graph.md is
# stale) so the documentation cannot drift from the declaration again.
RELATION_SIGNATURES: dict[str, RelationSignature] = {
    "PARTICIPATES_IN": _sig("PARTICIPATES_IN", {"Person"}, {"Event"}, Cardinality.M_TO_N),
    "MEMBER_OF": _sig("MEMBER_OF", {"Person"}, {"Organization"}, Cardinality.M_TO_N),
    "OWNS": _sig("OWNS", {"Person", "Organization"}, {"Project"}, Cardinality.M_TO_N),
    "CONTAINS": _sig("CONTAINS", {"Project"}, {"Task"}, Cardinality.ONE_TO_N),
    "ASSIGNED_TO": _sig("ASSIGNED_TO", {"Task"}, {"Person"}, Cardinality.M_TO_ONE),
    "PRECEDES": _sig("PRECEDES", {"Event"}, {"Event"}, Cardinality.M_TO_N),
    "SUPPORTS": _sig("SUPPORTS", {"Claim"}, {"Claim", "Decision"}, Cardinality.M_TO_N),
    "CONTRADICTS": _sig(
        "CONTRADICTS",
        {"Claim", "Assertion"},
        {"Claim", "Assertion"},
        Cardinality.M_TO_N,
    ),
    "EVIDENCE_FOR": _sig(
        "EVIDENCE_FOR",
        {"Document", "Event"},
        {"Claim", "Decision", "Assertion"},
        Cardinality.M_TO_N,
    ),
    "RESULTS_IN": _sig(
        "RESULTS_IN",
        {"Event", "Decision"},
        {"Event", "Task", "Project"},
        Cardinality.M_TO_N,
    ),
    "ABOUT": _sig(
        "ABOUT",
        {"Document", "Claim"},
        {"Person", "Project", "Task", "Event", "Decision"},
        Cardinality.M_TO_N,
    ),
    "POSSIBLY_SAME_AS": _sig(
        "POSSIBLY_SAME_AS",
        {"Person", "Organization", "Project", "Task", "Event"},
        {"Person", "Organization", "Project", "Task", "Event"},
        Cardinality.M_TO_N,
    ),
    "SUPERSEDES": _sig("SUPERSEDES", {"Assertion"}, {"Assertion"}, Cardinality.M_TO_N),
    # HAS_STATUS promotes a status-bearing entity's status to a first-class
    # assertion so a status flip becomes an assertion-to-assertion contradiction
    # (Req 8.11). m:1 — a subject has at most one accepted status at a time.
    # Sources cover every status-bearing entity; the write pipeline currently
    # reconciles Task / Project / Person (Decision keeps its C8-governed
    # draft->final lifecycle).
    "HAS_STATUS": _sig(
        "HAS_STATUS",
        {"Person", "Organization", "Project", "Task", "Claim", "Decision"},
        {"StatusValue"},
        Cardinality.M_TO_ONE,
    ),
    # HAS_VALUE assigns a single current value to a Slot (a typed key, e.g. a
    # dialogue-state field such as ``hotel-area``). m:1 — each slot has at most
    # one accepted value (single-valued on the *subject*), but a value may be
    # shared across many slots (``centre`` can be both a restaurant area and a
    # hotel area), so it is NOT a bijection. A new conflicting value for the same
    # slot is a single-valued contradiction (quarantine under ``new_fact``,
    # supersede under ``update`` with the authoritative-update policy). Used by
    # the MultiWOZ real-data adapter.
    "HAS_VALUE": _sig("HAS_VALUE", {"Slot"}, {"SlotValue"}, Cardinality.M_TO_ONE),
}


def get_relation_signature(predicate: str) -> RelationSignature:
    """Return the declared signature for ``predicate`` (Req 2.14).

    Raises ``UnknownPredicateError`` if the predicate is not registered.
    """
    try:
        return RELATION_SIGNATURES[predicate]
    except KeyError:
        raise UnknownPredicateError(predicate) from None


# Permitted task status transitions driving constraint C10 (Req 8.11).
# A ``correction`` write_intent bypasses this map (handled in C10).
TASK_STATUS_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.todo: {TaskStatus.in_progress, TaskStatus.blocked, TaskStatus.cancelled},
    TaskStatus.in_progress: {TaskStatus.blocked, TaskStatus.done, TaskStatus.cancelled},
    TaskStatus.blocked: {TaskStatus.in_progress, TaskStatus.cancelled},
    TaskStatus.done: set(),
    TaskStatus.cancelled: set(),
}

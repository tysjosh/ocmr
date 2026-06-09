"""Unit tests for the relation signature registry and task transition map.

Covers task 2.6:

* ``get_relation_signature`` returns the declared source/target types and
  cardinality for representative predicates (Req 2.14).
* ``get_relation_signature`` raises ``UnknownPredicateError`` for an
  unregistered predicate (Req 2.14).
* ``TASK_STATUS_TRANSITIONS`` matches the design map exactly (Req 8.11).
* All 13 relations are registered (Req 2.1-2.13).
"""

from __future__ import annotations

import pytest

from ocm.ontology.enums import TaskStatus
from ocm.ontology.relations import (
    RELATION_SIGNATURES,
    Cardinality,
    RelationSignature,
    TASK_STATUS_TRANSITIONS,
    UnknownPredicateError,
    get_relation_signature,
)


# ---------------------------------------------------------------------------
# Registry completeness (Req 2.1-2.13).
# ---------------------------------------------------------------------------
EXPECTED_PREDICATES = {
    "PARTICIPATES_IN",
    "MEMBER_OF",
    "OWNS",
    "CONTAINS",
    "ASSIGNED_TO",
    "PRECEDES",
    "SUPPORTS",
    "CONTRADICTS",
    "EVIDENCE_FOR",
    "RESULTS_IN",
    "ABOUT",
    "POSSIBLY_SAME_AS",
    "SUPERSEDES",
    "HAS_STATUS",
}


def test_all_thirteen_relations_registered() -> None:
    """The registry declares exactly the 14 design relations (Req 2.1-2.13 + HAS_STATUS)."""
    assert len(RELATION_SIGNATURES) == 14
    assert set(RELATION_SIGNATURES) == EXPECTED_PREDICATES


def test_every_signature_predicate_matches_its_key() -> None:
    """Each signature's ``predicate`` field matches its registry key."""
    for key, sig in RELATION_SIGNATURES.items():
        assert isinstance(sig, RelationSignature)
        assert sig.predicate == key


# ---------------------------------------------------------------------------
# Lookup returns declared source/target types and cardinality (Req 2.14).
# ---------------------------------------------------------------------------
# (predicate, expected source_types, expected target_types, expected cardinality)
SIGNATURE_CASES = [
    ("ASSIGNED_TO", {"Task"}, {"Person"}, Cardinality.M_TO_ONE),
    ("OWNS", {"Person", "Organization"}, {"Project"}, Cardinality.M_TO_N),
    ("SUPERSEDES", {"Assertion"}, {"Assertion"}, Cardinality.M_TO_N),
    ("CONTAINS", {"Project"}, {"Task"}, Cardinality.ONE_TO_N),
    ("PARTICIPATES_IN", {"Person"}, {"Event"}, Cardinality.M_TO_N),
    ("MEMBER_OF", {"Person"}, {"Organization"}, Cardinality.M_TO_N),
    ("PRECEDES", {"Event"}, {"Event"}, Cardinality.M_TO_N),
    ("SUPPORTS", {"Claim"}, {"Claim", "Decision"}, Cardinality.M_TO_N),
    (
        "CONTRADICTS",
        {"Claim", "Assertion"},
        {"Claim", "Assertion"},
        Cardinality.M_TO_N,
    ),
    (
        "EVIDENCE_FOR",
        {"Document", "Event"},
        {"Claim", "Decision", "Assertion"},
        Cardinality.M_TO_N,
    ),
    (
        "RESULTS_IN",
        {"Event", "Decision"},
        {"Event", "Task", "Project"},
        Cardinality.M_TO_N,
    ),
    (
        "ABOUT",
        {"Document", "Claim"},
        {"Person", "Project", "Task", "Event", "Decision"},
        Cardinality.M_TO_N,
    ),
    (
        "POSSIBLY_SAME_AS",
        {"Person", "Organization", "Project", "Task", "Event"},
        {"Person", "Organization", "Project", "Task", "Event"},
        Cardinality.M_TO_N,
    ),
    ("HAS_STATUS",
     {"Person", "Organization", "Project", "Task", "Claim", "Decision"},
     {"StatusValue"},
     Cardinality.M_TO_ONE),
]


@pytest.mark.parametrize(
    "predicate,source_types,target_types,cardinality", SIGNATURE_CASES
)
def test_lookup_returns_declared_signature(
    predicate: str,
    source_types: set[str],
    target_types: set[str],
    cardinality: Cardinality,
) -> None:
    """Lookup returns the declared source/target types and cardinality (Req 2.14)."""
    sig = get_relation_signature(predicate)
    assert sig.predicate == predicate
    assert set(sig.source_types) == source_types
    assert set(sig.target_types) == target_types
    assert sig.cardinality is cardinality


def test_lookup_covers_every_registered_predicate() -> None:
    """Every registered predicate is retrievable and returns its own signature."""
    for predicate in RELATION_SIGNATURES:
        assert get_relation_signature(predicate) is RELATION_SIGNATURES[predicate]


def test_signature_collections_are_frozen() -> None:
    """Declared type collections are immutable frozensets."""
    sig = get_relation_signature("OWNS")
    assert isinstance(sig.source_types, frozenset)
    assert isinstance(sig.target_types, frozenset)


# ---------------------------------------------------------------------------
# Unknown predicate raises (Req 2.14).
# ---------------------------------------------------------------------------
def test_lookup_unknown_predicate_raises() -> None:
    """An unregistered predicate raises ``UnknownPredicateError`` (Req 2.14)."""
    with pytest.raises(UnknownPredicateError) as exc_info:
        get_relation_signature("NOT_A_REAL_PREDICATE")
    assert exc_info.value.predicate == "NOT_A_REAL_PREDICATE"


def test_unknown_predicate_error_is_keyerror_subclass() -> None:
    """``UnknownPredicateError`` is a ``KeyError`` so dict-style callers still work."""
    assert issubclass(UnknownPredicateError, KeyError)
    with pytest.raises(KeyError):
        get_relation_signature("")


# ---------------------------------------------------------------------------
# TASK_STATUS_TRANSITIONS matches the design map exactly (Req 8.11).
# ---------------------------------------------------------------------------
EXPECTED_TRANSITIONS = {
    TaskStatus.todo: {
        TaskStatus.in_progress,
        TaskStatus.blocked,
        TaskStatus.cancelled,
    },
    TaskStatus.in_progress: {
        TaskStatus.blocked,
        TaskStatus.done,
        TaskStatus.cancelled,
    },
    TaskStatus.blocked: {TaskStatus.in_progress, TaskStatus.cancelled},
    TaskStatus.done: set(),
    TaskStatus.cancelled: set(),
}


def test_task_status_transitions_match_design_map() -> None:
    """The transition map equals the design map exactly (Req 8.11)."""
    assert TASK_STATUS_TRANSITIONS == EXPECTED_TRANSITIONS


@pytest.mark.parametrize(
    "current,allowed", list(EXPECTED_TRANSITIONS.items())
)
def test_each_transition_entry(
    current: TaskStatus, allowed: set[TaskStatus]
) -> None:
    """Each source status maps to exactly its allowed target statuses (Req 8.11)."""
    assert TASK_STATUS_TRANSITIONS[current] == allowed


def test_terminal_states_have_no_outgoing_transitions() -> None:
    """``done`` and ``cancelled`` are terminal: no permitted transitions (Req 8.11)."""
    assert TASK_STATUS_TRANSITIONS[TaskStatus.done] == set()
    assert TASK_STATUS_TRANSITIONS[TaskStatus.cancelled] == set()

"""Durable-state correctness: is the value the store *kept* the right one?

The existing durable-state metric,
:func:`~ocm.evaluation.experiment.durable_constraint_violations`, measures
**consistency**: it counts single-valued keys left holding two or more accepted
objects. It cannot measure **correctness**, because it never looks at *which*
object survived. Every write policy that retires the loser of a contradiction —
last-writer-wins, evidence-weighted merge, per-rule policy, and OCMR's gate
alike — leaves exactly one value and therefore scores 0.

That blind spot is exactly where competing write policies differ, so this module
adds the orthogonal measure. Given a gold *current* value per single-valued key,
each key falls into exactly one bucket:

===============  ==================================================  ==========================
Bucket           Condition                                           Meaning
===============  ==================================================  ==========================
``correct``      accepted objects == {gold}                          one value, and it is right
``stale``        exactly one accepted object, != gold, **unflagged**  silently wrong
``split``        two or more accepted objects                        cardinality breach
``abstained``    wrong or absent, but the key was flagged            declined, and said so
``missing``      no accepted object and nothing flagged              never extracted
===============  ==================================================  ==========================

Two definitions carry the weight.

**``stale`` requires that nothing was flagged for the key.** When OCMR quarantines
an incoming value the incumbent stays accepted, so the store holds a non-gold
value *by design*. Scoring that as silently stale would penalise the governed arm
for behaving correctly. The flag check separates "wrong and silent" from "wrong
and declared", which is the distinction the whole comparison rests on.

**``missing`` is its own bucket** so a shared extraction bottleneck does not
contaminate the governance comparison — the ``*_gov`` rates exclude it. On
LongMemEval Arm B raw accuracy is extraction-bound rather than governance-bound,
so a metric that folds extraction misses into governance failures would compare
extractors, not policies.

``split`` reproduces ``durable_constraint_violations`` and is disjoint from
``stale``, so the two metrics never double-count.

Scope is defined by the caller's ``gold`` map, not by the ontology: only keys with
a known gold current value are scored. Keys the store holds but gold does not
mention are ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from ocm.ontology.enums import QuarantineStatus

#: A durable key: ``(subject_id, predicate)``.
Key = Tuple[str, str]

#: Value-node id prefix for ``StatusValue`` objects (mirrors ``write_pipeline``).
_STATUS_VALUE_PREFIX = "status:"

#: Quarantine states that count as "the system declined and said so". A
#: ``resolved`` record was released into the store after review, so the value it
#: produced is judged on its merits rather than shielded by the original flag.
_HOLDING_STATUSES = frozenset(
    {QuarantineStatus.unresolved.value, QuarantineStatus.dismissed.value}
)

#: Bucket labels, in report order.
CORRECT = "correct"
STALE = "stale"
SPLIT = "split"
ABSTAINED = "abstained"
MISSING = "missing"

BUCKETS: tuple[str, ...] = (CORRECT, STALE, SPLIT, ABSTAINED, MISSING)

_WS = re.compile(r"\s+")


def normalize_value(value: Any) -> str:
    """Case/whitespace-insensitive form used to compare a stored value with gold."""
    return _WS.sub(" ", str(value).strip()).casefold()


@dataclass
class DurableStateReport:
    """Per-bucket counts and derived rates over the gold-labelled keys.

    ``outcomes`` retains the per-key bucket so a reviewer can audit any single
    decision rather than trusting the aggregate, and ``observed`` records the
    normalized value the store actually held (empty when it held none).
    """

    correct: int = 0
    stale: int = 0
    split: int = 0
    abstained: int = 0
    missing: int = 0
    outcomes: Dict[Key, str] = field(default_factory=dict)
    observed: Dict[Key, str] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """Number of gold-labelled keys scored."""
        return self.correct + self.stale + self.split + self.abstained + self.missing

    @property
    def scored(self) -> int:
        """Keys attributable to governance (``total`` minus never-extracted)."""
        return self.total - self.missing

    # -- rates over every gold key ------------------------------------------
    @property
    def dsc(self) -> float:
        """Durable State Correctness: share of keys holding exactly the gold value."""
        return self._rate(self.correct, self.total)

    @property
    def ssr(self) -> float:
        """Silent Stale Rate: share holding one wrong, unflagged value (minimize)."""
        return self._rate(self.stale, self.total)

    @property
    def split_rate(self) -> float:
        """Share with two or more accepted values (the legacy violation measure)."""
        return self._rate(self.split, self.total)

    @property
    def abstention_rate(self) -> float:
        """Share declined and flagged — the cost side, reported not optimized."""
        return self._rate(self.abstained, self.total)

    @property
    def missing_rate(self) -> float:
        """Share never extracted — the extraction floor, a diagnostic."""
        return self._rate(self.missing, self.total)

    # -- governance-scoped rates (exclude never-extracted) ------------------
    @property
    def dsc_gov(self) -> float:
        """:attr:`dsc` over keys the extractor actually produced."""
        return self._rate(self.correct, self.scored)

    @property
    def ssr_gov(self) -> float:
        """:attr:`ssr` over keys the extractor actually produced."""
        return self._rate(self.stale, self.scored)

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        return (100.0 * numerator / denominator) if denominator else 0.0

    def as_dict(self) -> dict[str, float]:
        """Flat summary suitable for a results table or a checkpoint payload."""
        return {
            "correct": float(self.correct),
            "stale": float(self.stale),
            "split": float(self.split),
            "abstained": float(self.abstained),
            "missing": float(self.missing),
            "total": float(self.total),
            "dsc": self.dsc,
            "ssr": self.ssr,
            "split_rate": self.split_rate,
            "abstention_rate": self.abstention_rate,
            "missing_rate": self.missing_rate,
            "dsc_gov": self.dsc_gov,
            "ssr_gov": self.ssr_gov,
        }


def _resolve_value(graph: Any, object_id: str) -> str:
    """Resolve an assertion's object id to the value string it denotes.

    ``SlotValue`` and ``StatusValue`` nodes carry the value in their payload; for
    an ordinary entity object the id itself is the value (e.g. ``ASSIGNED_TO``
    pointing at a Person). Falls back through payload ``value`` -> payload
    ``name`` -> ``status:`` prefix strip -> the raw id.
    """
    payload = None
    if graph is not None:
        try:
            payload = graph.get_entity_payload(object_id)
        except Exception:  # pragma: no cover - defensive
            payload = None
    if payload:
        for key in ("value", "name"):
            candidate = payload.get(key)
            if candidate:
                return str(candidate)
    if isinstance(object_id, str) and object_id.startswith(_STATUS_VALUE_PREFIX):
        return object_id[len(_STATUS_VALUE_PREFIX):]
    return str(object_id)


def _flagged_keys(container: Any) -> set[Key]:
    """Keys with a quarantine record still holding (unresolved or dismissed).

    The record's ``candidate_payload`` is a serialized
    :class:`~ocm.memory.contracts.CandidateAssertion`, so it carries the
    ``(subject_id, predicate)`` the hold applies to.
    """
    store = getattr(container, "quarantine_store", None)
    if store is None:
        return set()
    try:
        records = list(store.list())
    except Exception:  # pragma: no cover - defensive
        return set()

    flagged: set[Key] = set()
    for record in records:
        status = getattr(record, "status", None)
        status_value = getattr(status, "value", status)
        if str(status_value) not in _HOLDING_STATUSES:
            continue
        payload = getattr(record, "candidate_payload", None) or {}
        subject = payload.get("subject_id")
        predicate = payload.get("predicate")
        if subject and predicate:
            flagged.add((str(subject), str(predicate)))
    return flagged


def _accepted_objects_by_key(container: Any) -> Dict[Key, set[str]]:
    """Map each ``(subject_id, predicate)`` to its accepted object ids."""
    try:
        accepted = list(container.repo.list_assertions("accepted"))
    except Exception:  # pragma: no cover - defensive
        return {}
    by_key: Dict[Key, set[str]] = {}
    for assertion in accepted:
        by_key.setdefault(
            (assertion.subject_id, assertion.predicate), set()
        ).add(assertion.object_id)
    return by_key


def durable_state_outcomes(
    container: Any,
    gold: Mapping[Key, str],
) -> DurableStateReport:
    """Classify each gold-labelled key by what the durable store ended up holding.

    Args:
        container: A :class:`~ocm.core.container.CoreContainer` after a replay.
        gold: ``(subject_id, predicate) -> gold current value``. Defines the
            scope: only these keys are scored.

    Returns:
        A :class:`DurableStateReport`. Never mutates state.
    """
    graph = getattr(container, "graph", None)
    by_key = _accepted_objects_by_key(container)
    flagged = _flagged_keys(container)

    report = DurableStateReport()
    for key, gold_value in gold.items():
        objects = by_key.get(key, set())
        target = normalize_value(gold_value)

        if len(objects) >= 2:
            bucket = SPLIT
            observed = " | ".join(
                sorted(normalize_value(_resolve_value(graph, o)) for o in objects)
            )
        elif len(objects) == 1:
            observed = normalize_value(_resolve_value(graph, next(iter(objects))))
            if observed == target:
                bucket = CORRECT
            elif key in flagged:
                bucket = ABSTAINED
            else:
                bucket = STALE
        else:
            observed = ""
            bucket = ABSTAINED if key in flagged else MISSING

        setattr(report, bucket, getattr(report, bucket) + 1)
        report.outcomes[key] = bucket
        report.observed[key] = observed

    return report


def gold_from_slot_values(
    entries: Iterable[tuple[str, str]], predicate: str = "HAS_VALUE"
) -> Dict[Key, str]:
    """Build a gold map from ``(slot_id, current_value)`` pairs.

    Convenience for the adapters, whose gold is naturally a slot-to-value map on
    a single predicate.
    """
    return {(slot_id, predicate): value for slot_id, value in entries}


def resolve_gold_keys(
    container: Any,
    gold_by_name: Mapping[Key, str],
    *,
    entity_type: str = "Slot",
) -> Dict[Key, str]:
    """Rekey a gold map from entity *name* to the entity *id* the store uses.

    Adapters know their subjects by qualified name (e.g. ``"<qid>:residence"``),
    but entity ids are minted with a per-run counter and cannot be recomputed, so
    the mapping has to be read back from the store after a replay.

    A name that never materialized as an entity is **kept** in the returned map
    under its original name. That id matches nothing in the store, so the key
    lands in the ``missing`` bucket rather than silently vanishing from the
    denominator — an unextracted fact is a measured outcome, not an excluded one.
    """
    name_to_id: Dict[str, str] = {}
    try:
        entities = list(container.repo.list_entities())
    except Exception:  # pragma: no cover - defensive
        entities = []
    for etype, payload in entities:
        if etype != entity_type or not payload:
            continue
        name = payload.get("name")
        entity_id = payload.get("id")
        if name and entity_id:
            name_to_id[normalize_value(name)] = str(entity_id)

    resolved: Dict[Key, str] = {}
    for (name, predicate), value in gold_by_name.items():
        entity_id = name_to_id.get(normalize_value(name), name)
        resolved[(entity_id, predicate)] = value
    return resolved


def summarize_reports(
    reports: Mapping[str, DurableStateReport],
    metric: str = "ssr",
) -> Dict[str, float]:
    """Pull one rate out of a ``arm -> report`` mapping, for table assembly."""
    out: Dict[str, float] = {}
    for arm, report in reports.items():
        value: Optional[float] = getattr(report, metric, None)
        if value is not None:
            out[arm] = float(value)
    return out

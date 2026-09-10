"""Retrieval-free durable-state scoring (``exact_state_accuracy``).

Why this exists
---------------
The harness's headline recall metric, ``task_success``, is answer-token recall
over :func:`~ocm.evaluation.runner.BaselineRunner._haystack` — the rendered
answer *plus the text of every retrieved item*. That makes it **monotone
non-decreasing in the size of the retrieved set**: surfacing more memory can
only raise it.

The volume difference between arms is created at **write** time, not read time.
``status_filter`` widens only to ``{"status": {"$in": ["accepted",
"quarantined"]}}`` -- ``superseded`` is never retrievable, under any toggle or
query type. So an arm does not gain recall by *retrieving* retired state; it
gains recall by *never retiring it*. An arm that skips supersession leaves every
stale value at ``status='accepted'``, where it stays a first-class retrieval
candidate, while a governed arm's retired values drop out of the index entirely.

The consequence is that an arm is *rewarded* for retaining exactly the stale
state ``constraint_violations`` penalises. The two metrics are therefore
anti-correlated by construction -- one phenomenon measured twice with opposite
signs -- and raw ``task_success`` cannot be read as a memory-quality comparison
across arms with different write policies.

(Retrieval *composition* was tested directly and found not to explain the gap: a
variant carrying B3's write settings with B2's retrieval toggles scored
identically to B3 on raw LongMemEval, because ``include_conflicts`` only admits
quarantined items and that arm quarantines none.)

On MultiWOZ the effect is stark enough to saturate the metric:
``EvidencePackager._slot_value_answer`` deliberately returns the *most recent*
accepted ``HAS_VALUE`` when several are active — "which only happens on an
ungoverned arm that failed to supersede" — so an ungoverned arm is handed the
current gold value despite carrying single-valued violations, and every arm
scores ~99.96.

This module scores the **durable store** instead, ignoring retrieval entirely.
A probe is *correct* only when the governed state is unambiguous **and** right:
exactly one accepted object for ``(subject, predicate)``, equal to the gold
value. An arm that failed to supersede has two accepted values and is scored
``ambiguous``, not correct — which is the distinction ``task_success`` erases.

Reading the graph is sound here because ``GraphStore`` maintains an
accepted-only invariant (edges == ``assertions`` rows with
``status == 'accepted'``); supersession removes the retired edge via
``CommitManager._supersede`` -> ``GraphStore.remove_assertion``.

The scorer is surface-agnostic: it takes ``(subject_name, predicate,
expected_object)`` probes, so MultiWOZ (slot key in the question marker),
cached-oracle LongMemEval (gold value trajectories), and the synthetic
benchmark (typed relations) can all reuse it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

#: Node payload fields consulted for a human-facing label, in priority order.
_NAME_FIELDS: tuple[str, ...] = ("name", "title", "summary", "description")

#: Node payload fields consulted for a *value* (SlotValue / StatusValue nodes).
_VALUE_FIELDS: tuple[str, ...] = ("value", "name", "title")


def _norm(text: Any) -> str:
    """Casefold + collapse whitespace, so ``"San Francisco "`` == ``"san francisco"``."""
    return " ".join(str(text or "").split()).casefold()


@dataclass(frozen=True)
class StateProbe:
    """One durable-state question: what should ``(subject, predicate)`` hold?

    ``subject_name`` is matched against entity payload names (not ids), because
    every caller knows the business key -- a MultiWOZ slot key such as
    ``"MUL0001:hotel-area"``, or a synthetic entity name such as
    ``"Project Orion"`` -- rather than the hashed entity id.
    """

    subject_name: str
    predicate: str
    expected_object: str
    #: Optional label carried through to the per-probe records (e.g. question id).
    probe_id: Optional[str] = None


@dataclass
class ExactStateReport:
    """Outcome counts for a set of :class:`StateProbe` s over one arm's store."""

    total: int = 0
    #: Exactly one accepted object, and it matches the gold value.
    correct: int = 0
    #: Exactly one accepted object, but it is the wrong value.
    wrong_value: int = 0
    #: Two or more *distinct* accepted objects -- unresolved single-valued state.
    ambiguous: int = 0
    #: Subject absent, or no accepted object for the predicate.
    missing: int = 0
    #: Per-probe detail, for audit and error analysis.
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def exact_state_accuracy(self) -> float:
        """Percent of probes whose durable state is unambiguous and correct."""
        return (100.0 * self.correct / self.total) if self.total else 0.0

    @property
    def ambiguous_rate(self) -> float:
        """Percent of probes left with two or more accepted values."""
        return (100.0 * self.ambiguous / self.total) if self.total else 0.0

    @property
    def lenient_accuracy(self) -> float:
        """Percent correct when a *containment* match is allowed on the value.

        Reported alongside the strict figure so a formatting mismatch (``"4"``
        vs ``"four"``) is visible as a scoring artifact rather than silently
        counted as a governance failure.
        """
        hits = sum(1 for r in self.records if r["lenient_correct"])
        return (100.0 * hits / self.total) if self.total else 0.0

    def to_dict(self, *, include_records: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "exact_state_accuracy": self.exact_state_accuracy,
            "lenient_accuracy": self.lenient_accuracy,
            "ambiguous_rate": self.ambiguous_rate,
            "counts": {
                "total": self.total,
                "correct": self.correct,
                "wrong_value": self.wrong_value,
                "ambiguous": self.ambiguous,
                "missing": self.missing,
            },
        }
        if include_records:
            out["records"] = self.records
        return out


def _index_names(graph: Any) -> dict[str, str]:
    """Map normalized entity name -> entity id.

    On a name collision the first id wins and the duplicate is ignored; callers
    scope their subject names (``"<dialogue>:<slot>"``, ``"<qid>:<attr>"``) so
    collisions do not occur in practice.
    """
    index: dict[str, str] = {}
    for node_id in graph.node_ids():
        payload = graph.get_entity_payload(node_id) or {}
        for key in _NAME_FIELDS:
            value = payload.get(key)
            if value:
                index.setdefault(_norm(value), node_id)
                break
    return index


def _object_label(graph: Any, object_id: str) -> str:
    """Render an object node as its value/name, falling back to its id."""
    payload = graph.get_entity_payload(object_id) or {}
    for key in _VALUE_FIELDS:
        value = payload.get(key)
        if value:
            return str(value)
    return object_id


def exact_state_report(
    container: Any,
    probes: Iterable[StateProbe],
) -> ExactStateReport:
    """Score the durable store of one arm against ``probes`` (retrieval-free).

    Args:
        container: A :class:`~ocm.core.container.CoreContainer` whose graph holds
            the arm's accepted state after ingestion.
        probes: The ``(subject_name, predicate, expected_object)`` triples to check.

    Returns:
        An :class:`ExactStateReport`. ``correct`` requires **exactly one**
        accepted object equal to the gold value, so an arm that left a stale
        value active is counted ``ambiguous`` rather than correct.
    """
    graph = container.graph
    names = _index_names(graph)
    report = ExactStateReport()

    for probe in probes:
        report.total += 1
        subject_id = names.get(_norm(probe.subject_name))
        expected = _norm(probe.expected_object)

        if subject_id is None:
            outcome, actual = "missing", []
        else:
            actual = [
                _object_label(graph, object_id)
                for _s, object_id, _k, _d in graph.out_edges(subject_id, probe.predicate)
            ]
            distinct = {_norm(a) for a in actual}
            if not distinct:
                outcome = "missing"
            elif len(distinct) > 1:
                outcome = "ambiguous"
            elif expected in distinct:
                outcome = "correct"
            else:
                outcome = "wrong_value"

        setattr(report, outcome, getattr(report, outcome) + 1)

        # Lenient variant: tolerate containment either way, so "four" matches
        # "four times" and "$400,000" matches "400,000". Only meaningful when the
        # state is unambiguous; an ambiguous store is never leniently correct.
        lenient = False
        if outcome in ("correct", "wrong_value"):
            only = _norm(actual[0])
            lenient = expected == only or expected in only or only in expected

        report.records.append(
            {
                "probe_id": probe.probe_id,
                "subject_name": probe.subject_name,
                "predicate": probe.predicate,
                "expected": probe.expected_object,
                "actual": actual,
                "outcome": outcome,
                "lenient_correct": bool(lenient),
            }
        )

    return report


def probes_from_examples(
    examples: Iterable[Any],
    *,
    predicate: str = "HAS_VALUE",
    subject_from_marker: bool = True,
) -> list[StateProbe]:
    """Derive probes from benchmark examples whose queries carry a ``[[key]]`` marker.

    This covers the slot-addressed surfaces (MultiWOZ, and the oracle LongMemEval
    arm) where the question already names the slot it asks about and
    ``expected_answer_contains`` carries the gold value. Questions without a
    marker are skipped when ``subject_from_marker`` is set, because guessing the
    subject from free text would silently mis-score.
    """
    import re

    probes: list[StateProbe] = []
    for example in examples:
        for index, question in enumerate(getattr(example, "questions", []) or []):
            expected = list(getattr(question, "expected_answer_contains", []) or [])
            if not expected:
                continue
            query = str(getattr(question, "query", ""))
            marker = re.search(r"\[\[(.+?)\]\]", query)
            if marker is None:
                if subject_from_marker:
                    continue
                raise ValueError(f"no [[slot]] marker in query: {query!r}")
            probes.append(
                StateProbe(
                    subject_name=marker.group(1),
                    predicate=predicate,
                    expected_object=expected[0],
                    probe_id=f"{getattr(example, 'id', '?')}:q{index}",
                )
            )
    return probes

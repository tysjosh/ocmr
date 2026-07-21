"""Schema/Provenance Stress Workload → OCM data models + oracle extractor.

This module is the data layer of the **Schema/Provenance Stress Workload**, a
targeted diagnostic (IEEE ICTAI 2026 paper) that constructs synthetic durable
writes which are **not** single-valued contradictions (so the contradiction gate
C7 has nothing to fire on) yet **do** violate the schema/domain-range (C9),
temporal (C2), decision-evidence (C8), and task-status (C4/C10) checks. Fed
identical inputs, a gate-only configuration leaves those violations accepted in
durable memory while a schema+constraint configuration removes them.

This file implements the additive **data models** and the **oracle extractor**
only (the ``generate_stress_workload`` generator is a separate task and lives
alongside these types once implemented):

* :class:`WriteClass` — the five write-class labels (four poison classes plus a
  benign valid class) stored on each example so classes are distinguishable in
  the workload output (Req 5.5).
* :class:`SessionWrites` — the oracle's gold writes for one session, reusing the
  LongMemEval oracle's shape extended with events/documents/decisions. This is
  the value keyed by ``source_ref`` in the oracle's ``writes_by_ref`` map.
* :class:`StressCase` — one synthetic durable write: its class label, its
  ``source_ref`` key, its :class:`SessionWrites` payload, and the ground-truth
  expectation used by tests.
* :class:`StressOracleExtractor` — a W1 extractor with the **same contract** as
  :class:`~ocm.evaluation.datasets.longmemeval_adapter.LongMemEvalOracleExtractor`:
  ``version="stress-oracle-1"``, ``extract(text, source_ref)`` keyed on
  ``writes_by_ref``, returning an empty :class:`ExtractionResult` for an
  unknown/empty ``source_ref``. It is stateless, offline, and deterministic — it
  carries no clock, RNG, or network access (Req 6.1, 6.3).

The extractor is wired into the harness exactly as the LongMemEval oracle is::

    CoreContainer(settings, extractor=oracle)

Requirements: 6.1, 6.3, 5.5.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

from ocm.evaluation.benchmark import BenchmarkExample, Question, Session
from ocm.memory.contracts import ExtractionResult


# --------------------------------------------------------------------------- #
# Write-class labels (Req 5.5)
# --------------------------------------------------------------------------- #
class WriteClass(str, Enum):
    """The write class of a synthetic stress case.

    Each poison class maps to exactly one already-existing check so the mapping
    between a poison write and its intended detector is explicit and auditable;
    ``VALID`` violates nothing and is used to measure precision. The value is
    stored on ``BenchmarkExample.category`` so Valid_Writes and each Poison_Write
    class are distinguishable in the workload output and groupable in the metric.
    """

    SCHEMA = "schema_stress"      # C9 (+ W5) — discriminating on the relation path
    TEMPORAL = "temporal_stress"  # C2        — discriminating on the relation path
    EVIDENCE = "evidence_stress"  # C8        — discriminating via the reconcile-path guard
    STATUS = "status_stress"      # C10 + C4  — discriminating via the reconcile-path guard
    VALID = "valid_write"         # violates nothing


# --------------------------------------------------------------------------- #
# Oracle payload (reuses the LongMemEval oracle's shape + events/decisions)
# --------------------------------------------------------------------------- #
@dataclass
class SessionWrites:
    """The oracle's gold writes for one session (the stress-case payload).

    Mirrors the LongMemEval oracle's ``_SessionWrites`` shape, extended with the
    ``events``, ``documents``, and ``decisions`` lists the stress classes need.
    Every list holds plain dicts in the same shape the offline ``MockExtractor``
    emits, so the downstream normalization/resolution path validates them
    identically:

    * ``entities``  — ``{"type", "name", "fields"?}``
    * ``events``    — ``{"name", "type", "timestamp_start", "timestamp_end", "description"}``
    * ``relations`` — ``{"subject", "predicate", "object", "confidence", "write_intent"}``
    * ``documents`` — ``{"title", "path_or_url", "tags"?}``
    * ``decisions`` — ``{"summary", "topic", "timestamp"?, "status", "made_by"}``
    """

    entities: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    documents: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Stress case (one synthetic durable write)
# --------------------------------------------------------------------------- #
@dataclass
class StressCase:
    """One synthetic durable write: its class label and the oracle payload.

    ``source_ref`` is the sole key linking this case's injected
    entities/events/relations/decisions to the oracle's ``writes_by_ref`` map,
    exactly as the LongMemEval oracle keys per-session writes. ``write_class``
    labels the case (Req 5.5) and ``expects_active_when_ungoverned`` records the
    ground-truth expectation used by the metric tests (it is not consumed by the
    runner): ``True`` for poison writes (which are left as Invalid_Active_State
    under the ungoverned/gate-only arms) and ``True`` for valid writes (which are
    admitted under every arm), so it is ``False`` only where a write should never
    reach the accepted store.
    """

    case_id: str
    write_class: WriteClass
    source_ref: str
    writes: SessionWrites
    expects_active_when_ungoverned: bool = True


# --------------------------------------------------------------------------- #
# Oracle extractor (same contract as LongMemEvalOracleExtractor)
# --------------------------------------------------------------------------- #
class StressOracleExtractor:
    """A W1 extractor returning stress-workload gold writes keyed by source_ref.

    Implements the extractor protocol (``extract(text, source_ref)``) with the
    same contract as
    :class:`~ocm.evaluation.datasets.longmemeval_adapter.LongMemEvalOracleExtractor`:
    it is **stateless** over a run (it only reads its fixed ``writes_by_ref``
    map), **offline**, and **deterministic** — it holds no clock, RNG, or network
    access. A ``source_ref`` it does not know (including an empty/unknown ref)
    returns an empty :class:`ExtractionResult` carrying only the extractor
    version (Req 6.1, 6.3).
    """

    version: str = "stress-oracle-1"

    def __init__(self, writes_by_ref: dict[str, SessionWrites]) -> None:
        self._writes = writes_by_ref

    def extract(self, text: str, source_ref: str) -> ExtractionResult:
        """Replay the gold writes for ``source_ref`` as an ``ExtractionResult``.

        ``text`` is ignored — the oracle is keyed solely on ``source_ref``, which
        is the offline/deterministic contract. An unknown or empty ``source_ref``
        yields an empty extraction (all six item lists default to empty).
        """
        w = self._writes.get(source_ref)
        if w is None:
            return ExtractionResult(extractor_version=self.version)
        return ExtractionResult(
            entities=list(w.entities),
            events=list(w.events),
            documents=list(w.documents),
            decisions=list(w.decisions),
            relations=list(w.relations),
            extractor_version=self.version,
        )


# --------------------------------------------------------------------------- #
# Generator (Req 1-6): the four poison classes + benign valid writes
# --------------------------------------------------------------------------- #
#: Confidence assigned to every synthetic relation. Above the contradiction
#: high-confidence threshold (mirrors the MockExtractor's ``DEFAULT_CONFIDENCE``)
#: so nothing is silently dropped for a low-confidence reason — the poison writes
#: must be caught by their *intended* structural/constraint check, not by C6.
_CONFIDENCE: float = 0.85

#: Fixed base timestamp for synthetic Events (no wall clock → deterministic).
_BASE_TS = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

#: Deterministic flavor pools driven by the seeded RNG so generation is a pure
#: function of the seed (Req 6.2) while still exercising ``random.Random(seed)``.
_TASK_WORDS = ["Migrate", "Refactor", "Deploy", "Audit", "Design", "Ship", "Patch", "Draft"]
_PROJECT_WORDS = ["Orion", "Helios", "Atlas", "Vega", "Nimbus", "Titan", "Lyra", "Nova"]
_PERSON_WORDS = ["Ada", "Bran", "Cleo", "Dev", "Esme", "Finn", "Gita", "Hugo"]
_TOPIC_WORDS = ["vendor", "budget", "rollout", "hiring", "pricing", "roadmap", "launch", "merge"]


def _ts(offset_seconds: int) -> str:
    """Return a deterministic ISO-8601 UTC timestamp at ``offset_seconds``."""
    return (_BASE_TS + timedelta(seconds=offset_seconds)).isoformat()


def generate_stress_workload(
    seed: int = 1337,
    *,
    n_schema: int = 4,
    n_temporal: int = 4,
    n_evidence: int = 2,
    n_status: int = 2,
    n_valid: int = 6,
) -> tuple[list[BenchmarkExample], StressOracleExtractor, list[StressCase]]:
    """Build the Schema/Provenance Stress Workload (Req 1-6).

    Drives every choice from a single ``random.Random(seed)`` and returns
    ``(examples, oracle, cases)``:

    * ``examples`` — one :class:`BenchmarkExample` per :class:`StressCase`, each
      ``category = write_class.value`` and ``id = f"stress-{write_class.value}-{i:03d}"``.
      Every session's ``source_ref`` (``f"{example.id}:{session_id}"``) keys the
      oracle, and ``input`` is empty because the oracle is keyed solely on
      ``source_ref`` (the offline/deterministic contract, Req 6.1).
    * ``oracle`` — a :class:`StressOracleExtractor` replaying each case's exact
      entities/events/relations/documents/decisions by ``source_ref``.
    * ``cases`` — the :class:`StressCase` list (labels + payloads) used by the
      metric ground-truth tests; not consumed by the runner.

    The four poison classes are constructed so **none** is a single-valued
    contradiction (so the contradiction gate C7 has nothing to fire on) yet each
    violates exactly one already-existing check:

    * **SCHEMA (C9 + W5)** — a ``Task -[ASSIGNED_TO]-> Document`` range violation
      (object is a Document, not a Person) and a ``Task -[HAS_STATUS]-> Person``
      range violation (object is a Person, not a StatusValue). Both are C9
      domain/range violations on a resolvable, non-contradictory write. (A bare
      out-of-vocabulary *status string* is not independently catchable by W5/C9 —
      a ``StatusValue`` node accepts any string and cannot be minted as a relation
      endpoint through extraction — so the Schema class is represented by the
      robust, verified HAS_STATUS **range** violation instead; Req 1.1-1.4.)
    * **TEMPORAL (C2)** — an ``Event`` whose ``timestamp_end`` precedes its
      ``timestamp_start`` referenced by a ``PRECEDES`` relation, plus an
      expired-interval variant using the same end-before-start condition (the
      only sanity C2 checks); Req 3.1-3.3.
    * **EVIDENCE (C8)** — a ``final`` Decision with no ``EVIDENCE_FOR`` edge and no
      supporting evidence; Req 2.1, 2.2.
    * **STATUS (C10 + C4)** — a ``done`` Task with no completion Event / ``RESULTS_IN``
      (C4), and a two-session illegal ``done`` -> ``todo`` transition (C10);
      Req 4.1-4.3.

    Benign **VALID** writes (``Person -[OWNS]-> Project``, ``Task -[ASSIGNED_TO]->
    Person``, a well-formed ``Event -[PRECEDES]-> Event``, and a ``final`` Decision
    *with* a ``Document -[EVIDENCE_FOR]-> Decision`` edge) violate nothing and are
    admitted under the Full arm (Req 5.1-5.3). The workload always contains
    ``> 0`` valid and ``> 0`` poison writes and at least one case of each poison
    class, and every example carries its :class:`WriteClass` (Req 5.2, 5.5).
    """
    rng = random.Random(seed)
    examples: list[BenchmarkExample] = []
    cases: list[StressCase] = []
    writes_by_ref: dict[str, SessionWrites] = {}

    def _register(
        write_class: WriteClass,
        idx: int,
        sessions_writes: list[tuple[str, SessionWrites]],
    ) -> None:
        """Turn one logical case into a BenchmarkExample + StressCase + oracle keys.

        ``sessions_writes`` is an ordered list of ``(session_id, SessionWrites)``;
        the **last** session is the poison/representative session recorded on the
        :class:`StressCase` (the earlier sessions, if any, set up prerequisite
        accepted state — e.g. the C10 case's legitimately-completed ``done``).
        """
        example_id = f"stress-{write_class.value}-{idx:03d}"
        sessions: list[Session] = []
        primary_ref = ""
        primary_writes = SessionWrites()
        for session_id, sw in sessions_writes:
            source_ref = f"{example_id}:{session_id}"
            writes_by_ref[source_ref] = sw
            sessions.append(Session(session_id=session_id, input=""))
            primary_ref = source_ref
            primary_writes = sw
        question = Question(
            query=f"stress probe {example_id}",
            expected_answer_contains=[],
            expected_conflict=False,
        )
        examples.append(
            BenchmarkExample(
                id=example_id,
                category=write_class.value,
                sessions=sessions,
                questions=[question],
            )
        )
        cases.append(
            StressCase(
                case_id=example_id,
                write_class=write_class,
                source_ref=primary_ref,
                writes=primary_writes,
            )
        )

    # --- SCHEMA (C9 + W5) ------------------------------------------------- #
    for i in range(n_schema):
        task = f"Task{_rng_word(rng, _TASK_WORDS)}Schema{i}"
        if i % 2 == 0:
            # Range violation: ASSIGNED_TO object must be a Person, not a Document.
            doc = f"Spec{_rng_word(rng, _PROJECT_WORDS)}Schema{i}"
            sw = SessionWrites(
                entities=[{"type": "Task", "name": task}],
                documents=[
                    {
                        "title": doc,
                        "path_or_url": f"https://docs.example.com/{doc.lower()}",
                        "tags": [],
                    }
                ],
                relations=[_rel(task, "ASSIGNED_TO", doc)],
            )
        else:
            # Range violation: HAS_STATUS object must be a StatusValue, not a Person.
            person = f"{_rng_word(rng, _PERSON_WORDS)}Schema{i}"
            sw = SessionWrites(
                entities=[
                    {"type": "Task", "name": task},
                    {"type": "Person", "name": person},
                ],
                relations=[_rel(task, "HAS_STATUS", person)],
            )
        _register(WriteClass.SCHEMA, i, [("s1", sw)])

    # --- TEMPORAL (C2) ---------------------------------------------------- #
    for i in range(n_temporal):
        bad = f"Evt{_rng_word(rng, _PROJECT_WORDS)}Bad{i}"
        other = f"Evt{_rng_word(rng, _PROJECT_WORDS)}Ref{i}"
        # timestamp_end precedes timestamp_start — the C2 condition. The second
        # variant frames the same end-before-start as an expired interval whose
        # end lies before a later reference event with no renewal between them.
        expired = i % 2 == 1
        bad_event = {
            "name": bad,
            "type": "interval" if expired else "event",
            "timestamp_start": _ts(3600 + i),
            "timestamp_end": _ts(60 + i),  # end < start
            "description": (
                f"expired interval {bad} (lapsed before {other}, unrenewed)"
                if expired
                else f"impossible interval {bad} (ends before it starts)"
            ),
        }
        other_event = {
            "name": other,
            "type": "event",
            "timestamp_start": _ts(7200 + i),
            "timestamp_end": None,
            "description": f"reference event {other}",
        }
        sw = SessionWrites(
            events=[bad_event, other_event],
            relations=[_rel(bad, "PRECEDES", other)],
        )
        _register(WriteClass.TEMPORAL, i, [("s1", sw)])

    # --- EVIDENCE (C8) ---------------------------------------------------- #
    for i in range(n_evidence):
        topic = f"{_rng_word(rng, _TOPIC_WORDS)}-evidence-{i}"
        sw = SessionWrites(
            decisions=[
                {
                    "summary": f"We finalized the {topic} decision.",
                    "topic": topic,
                    "status": "final",  # final + zero EVIDENCE_FOR → C8 floor
                    "made_by": None,
                }
            ],
        )
        _register(WriteClass.EVIDENCE, i, [("s1", sw)])

    # --- STATUS (C10 + C4) ------------------------------------------------ #
    for i in range(n_status):
        task = f"Task{_rng_word(rng, _TASK_WORDS)}Status{i}"
        if i % 2 == 0:
            # C4: a done Task with no completion Event / RESULTS_IN edge.
            sw = SessionWrites(
                entities=[{"type": "Task", "name": task, "fields": {"status": "done"}}]
            )
            _register(WriteClass.STATUS, i, [("s1", sw)])
        else:
            # C10: a two-session illegal done -> todo transition. Session s1
            # legitimately completes the Task (done + completion Event +
            # RESULTS_IN, satisfying C4 so done is accepted); session s2 states
            # todo, an illegal transition out of the terminal done status.
            completion = f"Complete{task}"
            setup = SessionWrites(
                entities=[{"type": "Task", "name": task, "fields": {"status": "done"}}],
                events=[
                    {
                        "name": completion,
                        "type": "completion",
                        "timestamp_start": _ts(100 + i),
                        "timestamp_end": None,
                        "description": f"completion of {task}",
                    }
                ],
                relations=[_rel(completion, "RESULTS_IN", task)],
            )
            poison = SessionWrites(
                entities=[{"type": "Task", "name": task, "fields": {"status": "todo"}}]
            )
            _register(WriteClass.STATUS, i, [("s1", setup), ("s2", poison)])

    # --- VALID (violates nothing) ----------------------------------------- #
    for i in range(n_valid):
        variant = i % 4
        if variant == 0:
            person = f"{_rng_word(rng, _PERSON_WORDS)}Valid{i}"
            project = f"{_rng_word(rng, _PROJECT_WORDS)}Valid{i}"
            sw = SessionWrites(
                entities=[
                    {"type": "Person", "name": person},
                    {"type": "Project", "name": project},
                ],
                relations=[_rel(person, "OWNS", project)],
            )
        elif variant == 1:
            task = f"Task{_rng_word(rng, _TASK_WORDS)}Valid{i}"
            person = f"{_rng_word(rng, _PERSON_WORDS)}Valid{i}"
            sw = SessionWrites(
                entities=[
                    {"type": "Task", "name": task},
                    {"type": "Person", "name": person},
                ],
                relations=[_rel(task, "ASSIGNED_TO", person)],
            )
        elif variant == 2:
            first = f"Evt{_rng_word(rng, _PROJECT_WORDS)}First{i}"
            second = f"Evt{_rng_word(rng, _PROJECT_WORDS)}Second{i}"
            sw = SessionWrites(
                events=[
                    {
                        "name": first,
                        "type": "event",
                        "timestamp_start": _ts(3600 + i),
                        "timestamp_end": _ts(7200 + i),  # end >= start (well-formed)
                        "description": f"well-formed event {first}",
                    },
                    {
                        "name": second,
                        "type": "event",
                        "timestamp_start": _ts(10800 + i),
                        "timestamp_end": None,
                        "description": f"well-formed event {second}",
                    },
                ],
                relations=[_rel(first, "PRECEDES", second)],
            )
        else:
            # A valid final Decision must gain its EVIDENCE_FOR support *before* it
            # is marked final: C8 counts only already-accepted EVIDENCE_FOR edges, so
            # a single-session "final + EVIDENCE_FOR" write would quarantine the edge
            # itself (0 supports at the moment the edge — whose endpoint is already a
            # final Decision — is validated). Split into two sessions: s1 records the
            # draft Decision plus its Document -[EVIDENCE_FOR]-> Decision edge (both
            # accepted, because a draft Decision does not trip the C8 floor), and s2
            # promotes the Decision to final — now backed by one accepted EVIDENCE_FOR
            # edge, so C8 passes and the final status is accepted (Req 5.1, 5.3).
            topic = f"{_rng_word(rng, _TOPIC_WORDS)}-valid-{i}"
            doc = f"Rationale{_rng_word(rng, _PROJECT_WORDS)}Valid{i}"
            setup = SessionWrites(
                documents=[
                    {
                        "title": doc,
                        "path_or_url": f"https://docs.example.com/{doc.lower()}",
                        "tags": [],
                    }
                ],
                decisions=[
                    {
                        "summary": f"We finalized the {topic} decision.",
                        "topic": topic,
                        "status": "draft",
                        "made_by": None,
                    }
                ],
                relations=[_rel(doc, "EVIDENCE_FOR", topic)],
            )
            finalize = SessionWrites(
                decisions=[
                    {
                        "summary": f"We finalized the {topic} decision.",
                        "topic": topic,
                        "status": "final",
                        "made_by": None,
                    }
                ],
            )
            _register(WriteClass.VALID, i, [("s1", setup), ("s2", finalize)])
            continue
        _register(WriteClass.VALID, i, [("s1", sw)])

    return examples, StressOracleExtractor(writes_by_ref), cases


def _rng_word(rng: random.Random, pool: list[str]) -> str:
    """Deterministically pick a flavor word (drives generation by the seed)."""
    return rng.choice(pool)


def _rel(subject: str, predicate: str, object_: str) -> dict:
    """Build a relation dict in the MockExtractor's shape (new_fact intent)."""
    return {
        "subject": subject,
        "predicate": predicate,
        "object": object_,
        "confidence": _CONFIDENCE,
        "write_intent": "new_fact",
    }

"""MultiWOZ → OCM adapter: dialogue-state slots as governed single-valued memory.

Maps MultiWOZ task-oriented dialogues onto the OCM write pipeline so the
governance mechanism can be evaluated on **real human dialogue** rather than only
the synthetic benchmark. The mapping is faithful and minimal:

* Each dialogue-state slot (e.g. ``hotel-area``) becomes a :class:`Slot` node,
  qualified per dialogue (``<dialogue_id>:hotel-area``).
* Each slot value becomes a :class:`SlotValue` node.
* The current value is a ``Slot -[HAS_VALUE]-> SlotValue`` assertion, which is
  **1:1** (one accepted value per slot), so the governance gate treats a changed
  value as a single-valued contradiction: under ``correction`` intent it
  **supersedes** the prior value (the natural belief-state update), and under
  ``new_fact`` it would **quarantine** the conflict.

Why an *oracle* extractor (`MultiWOZOracleExtractor`)
----------------------------------------------------
Rather than ask the LLM to re-extract slots from utterances (which would conflate
dialogue-state-tracking error with governance behaviour), the adapter keys an
oracle extractor off MultiWOZ's **gold per-turn belief state**: for each turn it
emits exactly the slots that are *new* or *changed* at that turn (the delta),
with ``new_fact`` / ``correction`` intent respectively. This isolates the
governance evaluation — the experiment measures what the governed write path does
*given* correct slots — and lets the entire existing pipeline, baselines, and
metrics run unchanged (the oracle plugs in via the container's ``extractor``
slot, exactly like the LLM extractor). A second condition using the real LLM
extractor can be run for end-to-end-from-text numbers.

What you get per OCM metric
---------------------------
* **constraint_violations** ✅ — single-valued cardinality on real slots: a
  governed arm supersedes a changed slot (one accepted value, zero violations);
  an ungoverned arm keeps both values (violation).
* **task_success** ✅ — recall of the *current* slot value.
* **contradiction_rate** ◑ — partial: ``expected_conflict`` marks slots that
  changed during the dialogue (a stale value must not be surfaced); this is a
  stale-value-surfacing signal, not a logical contradiction.

This module is data-source agnostic: :func:`build_from_dialogues` takes a
normalized dialogue structure (``dialogue_id`` + per-turn ``utterance`` and
cumulative ``state``), so a MultiWOZ-2.2 loader or a hand-built fixture feed it
identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ocm.evaluation.benchmark import BenchmarkExample, Question, Session
from ocm.memory.contracts import ExtractionResult

#: Confidence for a brand-new slot value (a first assertion). Mirrors the mock
#: extractor's default so the contradiction gate treats it as a strong fact.
NEW_FACT_CONFIDENCE: float = 0.85

#: Confidence for a *changed* slot value, emitted as a ``correction`` so it
#: dominates the prior value by the Algorithm-1 supersede margin (0.97 - 0.85 =
#: 0.12 > 0.1) and routes to supersede rather than quarantine.
CORRECTION_CONFIDENCE: float = 0.97

#: Benchmark category label for MultiWOZ-derived examples.
CATEGORY: str = "dialogue_state_slots"


@dataclass
class _TurnWrites:
    """The oracle's gold writes for one turn (the belief-state delta)."""

    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)


class MultiWOZOracleExtractor:
    """A W1 extractor that returns MultiWOZ gold slot deltas keyed by source_ref.

    Implements the :class:`~ocm.extraction.base.Extractor` protocol. Holds no
    mutable state, so a single instance is safe to share across a run. For a
    ``source_ref`` it does not know it returns an empty extraction (the turn
    carried no belief-state change).
    """

    version: str = "multiwoz-oracle-1"

    def __init__(self, writes_by_ref: dict[str, _TurnWrites]) -> None:
        self._writes = writes_by_ref

    def extract(self, text: str, source_ref: str) -> ExtractionResult:
        w = self._writes.get(source_ref)
        if w is None:
            return ExtractionResult(extractor_version=self.version)
        return ExtractionResult(
            entities=list(w.entities),
            relations=list(w.relations),
            extractor_version=self.version,
        )


def _slot_key(dialogue_id: str, slot: str) -> str:
    """Qualified Slot name: scope a slot to its dialogue (one store, many dialogues)."""
    return f"{dialogue_id}:{slot}"


def build_from_dialogues(
    dialogues: Iterable[dict[str, Any]],
) -> tuple[list[BenchmarkExample], MultiWOZOracleExtractor]:
    """Build benchmark examples + an oracle extractor from normalized dialogues.

    Each input dialogue is ``{"dialogue_id": str, "turns": [...]}`` where each
    turn is ``{"utterance": str, "state": {slot: value, ...}}`` and ``state`` is
    the **cumulative** gold belief state after that turn (MultiWOZ 2.2 form).
    For every turn we emit the *delta* versus the previous turn: a brand-new slot
    as ``new_fact`` and a changed slot as ``correction``. ``source_ref`` is
    ``"<dialogue_id>:t<i>"`` so the oracle and the harness agree.

    Returns ``(examples, oracle_extractor)``; pass the extractor into the
    container/harness as ``extractor=...``.
    """
    examples: list[BenchmarkExample] = []
    writes_by_ref: dict[str, _TurnWrites] = {}

    for dialogue in dialogues:
        did = str(dialogue["dialogue_id"])
        prev_state: dict[str, str] = {}
        changed_slots: set[str] = set()
        sessions: list[Session] = []

        for i, turn in enumerate(dialogue.get("turns", [])):
            utterance = str(turn.get("utterance", ""))
            state = {str(k): str(v) for k, v in (turn.get("state") or {}).items()}
            tw = _TurnWrites()
            for slot, value in sorted(state.items()):
                if prev_state.get(slot) == value:
                    continue  # unchanged this turn — no write
                is_change = slot in prev_state
                if is_change:
                    changed_slots.add(slot)
                intent = "correction" if is_change else "new_fact"
                conf = CORRECTION_CONFIDENCE if is_change else NEW_FACT_CONFIDENCE
                slot_name = _slot_key(did, slot)
                tw.entities.append({"type": "Slot", "name": slot_name})
                tw.entities.append(
                    {"type": "SlotValue", "name": value, "fields": {"value": value}}
                )
                tw.relations.append(
                    {
                        "subject": slot_name,
                        "predicate": "HAS_VALUE",
                        "object": value,
                        "confidence": conf,
                        "write_intent": intent,
                    }
                )
            source_ref = f"{did}:t{i}"
            writes_by_ref[source_ref] = tw
            sessions.append(Session(session_id=f"t{i}", input=utterance))
            prev_state = dict(state)

        # Questions: recall the *final* value of each slot. A slot that changed
        # during the dialogue is flagged expected_conflict (a stale value exists
        # that a governed reader must not surface).
        questions = [
            Question(
                query=f"What is the value of {slot}?",
                expected_answer_contains=[value],
                expected_conflict=(slot in changed_slots),
            )
            for slot, value in sorted(prev_state.items())
        ]
        examples.append(
            BenchmarkExample(
                id=did, category=CATEGORY, sessions=sessions, questions=questions
            )
        )

    return examples, MultiWOZOracleExtractor(writes_by_ref)


# --------------------------------------------------------------------------- #
# Tiny hand-built fixture (MultiWOZ-shaped) for smoke tests / offline demos.
# --------------------------------------------------------------------------- #
def sample_dialogues() -> list[dict[str, Any]]:
    """A small MultiWOZ-shaped fixture exercising new / changed / stable slots.

    * ``mwz-0001`` changes ``hotel-area`` (centre -> south) — a supersession.
    * ``mwz-0002`` sets two slots once — clean accepts, no conflict.
    * ``mwz-0003`` changes ``restaurant-pricerange`` and keeps ``restaurant-food``.
    """
    return [
        {
            "dialogue_id": "mwz-0001",
            "turns": [
                {"utterance": "I want a hotel in the centre.",
                 "state": {"hotel-area": "centre"}},
                {"utterance": "Actually, make it the south instead.",
                 "state": {"hotel-area": "south"}},
            ],
        },
        {
            "dialogue_id": "mwz-0002",
            "turns": [
                {"utterance": "Find an expensive italian restaurant.",
                 "state": {"restaurant-pricerange": "expensive", "restaurant-food": "italian"}},
            ],
        },
        {
            "dialogue_id": "mwz-0003",
            "turns": [
                {"utterance": "A cheap chinese place please.",
                 "state": {"restaurant-pricerange": "cheap", "restaurant-food": "chinese"}},
                {"utterance": "On second thought, something expensive.",
                 "state": {"restaurant-pricerange": "expensive", "restaurant-food": "chinese"}},
            ],
        },
    ]

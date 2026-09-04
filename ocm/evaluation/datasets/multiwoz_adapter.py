"""MultiWOZ → OCM adapter: dialogue-state slots as governed single-valued memory.

Maps MultiWOZ task-oriented dialogues onto the OCM write pipeline so the
governance mechanism can be evaluated on **real human dialogue** rather than only
the synthetic benchmark. The mapping is faithful and minimal:

* Each dialogue-state slot (e.g. ``hotel-area``) becomes a :class:`Slot` node,
  qualified per dialogue (``<dialogue_id>:hotel-area``).
* Each slot value becomes a :class:`SlotValue` node.
* The current value is a ``Slot -[HAS_VALUE]-> SlotValue`` assertion, which is
  **m:1** — single-valued on the *slot* (at most one accepted value per slot), but
  **not** a bijection: a value may be shared across slots, since ``centre`` is
  legitimately both a restaurant area and a hotel area. Declaring it ``1:1`` made
  every shared value look like a contradiction and produced the quarantine flood
  fixed in ``7383cbe``; do not reintroduce that reading.
  The gate therefore treats a *changed* value as a single-valued contradiction:
  under ``update`` intent it **supersedes** the prior value (the natural
  belief-state update, via ``authoritative_update_supersede``), and under
  ``new_fact`` it would **quarantine** the conflict.

Why an *oracle* extractor (`MultiWOZOracleExtractor`)
----------------------------------------------------
Rather than ask the LLM to re-extract slots from utterances (which would conflate
dialogue-state-tracking error with governance behaviour), the adapter keys an
oracle extractor off MultiWOZ's **gold per-turn belief state**: for each turn it
emits exactly the slots that are *new* or *changed* at that turn (the delta),
with ``new_fact`` / ``update`` intent respectively. This isolates the
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

#: Confidence for a *changed* slot value. Kept above the gate threshold so the
#: single-valued conflict is detected; the authoritative-``update`` supersede
#: path does not depend on a confidence margin (the latest value wins).
UPDATE_CONFIDENCE: float = 0.85

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
    as ``new_fact`` and a changed slot as ``update``. The distinction matters: an
    ``update`` takes the ``authoritative_update_supersede`` path in C7 (the latest
    value replaces the incumbent unconditionally, no confidence margin), whereas a
    ``correction`` must dominate the incumbent by ``supersede_margin`` and carry
    ``supersede_evidence_min`` evidence. MultiWOZ slots are authoritative state the
    user just changed, so ``update`` is the correct label. ``source_ref`` is
    ``"<dialogue_id>:t<i>"`` so the oracle and the harness agree.

    Returns ``(examples, oracle_extractor)``; pass the extractor into the
    container/harness as ``extractor=...``.
    """
    examples: list[BenchmarkExample] = []
    writes_by_ref: dict[str, _TurnWrites] = {}

    for dialogue in dialogues:
        did = str(dialogue["dialogue_id"])
        # Running cumulative belief across the whole dialogue. MultiWOZ is
        # multi-domain: a service's slots appear only in turns where it is
        # active, so a per-turn state DROPS slots when the user switches topic.
        # Comparing against the previous *turn* would then misread a reappearing
        # slot as a new_fact (and quarantine its conflict). We instead accumulate
        # a belief dict that never drops slots, so a changed slot is always an
        # ``update``.
        belief: dict[str, str] = {}
        sessions: list[Session] = []

        for i, turn in enumerate(dialogue.get("turns", [])):
            utterance = str(turn.get("utterance", ""))
            state = {str(k): str(v) for k, v in (turn.get("state") or {}).items()}
            tw = _TurnWrites()
            for slot, value in sorted(state.items()):
                if belief.get(slot) == value:
                    continue  # unchanged vs cumulative belief — no write
                is_change = slot in belief
                intent = "update" if is_change else "new_fact"
                conf = UPDATE_CONFIDENCE if is_change else NEW_FACT_CONFIDENCE
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
                belief[slot] = value  # fold into the cumulative belief
            source_ref = f"{did}:t{i}"
            writes_by_ref[source_ref] = tw
            sessions.append(Session(session_id=f"t{i}", input=utterance))

        # Questions: recall the final cumulative value of every slot, addressed
        # by the slot's qualified key so it resolves in a shared multi-dialogue
        # store. expected_conflict=False: a changed slot is a resolved
        # supersession, not an unresolved contradiction (contradiction-surfacing
        # is N/A for MultiWOZ).
        questions = [
            Question(
                query=f"What is the current value of slot [[{_slot_key(did, slot)}]]?",
                expected_answer_contains=[value],
                expected_conflict=False,
            )
            for slot, value in sorted(belief.items())
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


# --------------------------------------------------------------------------- #
# MultiWOZ 2.2 (HuggingFace ``multi_woz_v22``) loader / normalizer
# --------------------------------------------------------------------------- #
#: HF speaker code for a user turn (system turns carry no belief state).
_HF_USER_SPEAKER = 0


def _collect_slot_values(state_entries: Any) -> dict[str, str]:
    """Pull ``{slot: value}`` from a turn's ``frames[].state`` (HF 2.2 shape).

    MultiWOZ 2.2 stores per-service state with ``slots_values`` =
    ``{"slots_values_name": [...], "slots_values_list": [[...], ...]}``. Slot
    names are already domain-qualified (e.g. ``hotel-area``); we take the first
    listed value (the canonical surface form). Defensive to both list-of-dicts
    and the columnar dict-of-lists nesting HF sometimes uses.
    """
    out: dict[str, str] = {}

    def _ingest_slots_values(sv: Any) -> None:
        if not isinstance(sv, dict):
            return
        names = sv.get("slots_values_name") or []
        values = sv.get("slots_values_list") or []
        for name, vlist in zip(names, values):
            if vlist:
                first = vlist[0] if isinstance(vlist, (list, tuple)) else vlist
                if first not in (None, ""):
                    out[str(name)] = str(first)

    # Shape A: list of state dicts (one per service).
    if isinstance(state_entries, list):
        for st in state_entries:
            if isinstance(st, dict):
                _ingest_slots_values(st.get("slots_values"))
    # Shape B: columnar dict with a parallel "slots_values" list.
    elif isinstance(state_entries, dict):
        svs = state_entries.get("slots_values")
        if isinstance(svs, list):
            for sv in svs:
                _ingest_slots_values(sv)
        else:
            _ingest_slots_values(svs)
    return out


def normalize_hf_multiwoz(hf_dialogue: dict[str, Any]) -> dict[str, Any]:
    """Normalize one HF ``multi_woz_v22`` dialogue to ``{dialogue_id, turns}``.

    Keeps only **user** turns (they carry the belief state); each normalized turn
    is ``{"utterance": str, "state": {slot: value}}`` with the *cumulative* gold
    state at that turn, ready for :func:`build_from_dialogues`.
    """
    turns = hf_dialogue.get("turns", {}) or {}
    speakers = turns.get("speaker", []) or []
    utterances = turns.get("utterance", []) or []
    frames = turns.get("frames", []) or []

    norm_turns: list[dict[str, Any]] = []
    for i, speaker in enumerate(speakers):
        if int(speaker) != _HF_USER_SPEAKER:
            continue
        frame = frames[i] if i < len(frames) else {}
        # frame is typically {"service": [...], "state": [...], "slots": [...]}.
        state_entries = frame.get("state") if isinstance(frame, dict) else None
        state = _collect_slot_values(state_entries)
        utterance = utterances[i] if i < len(utterances) else ""
        norm_turns.append({"utterance": str(utterance), "state": state})

    return {"dialogue_id": str(hf_dialogue.get("dialogue_id", "")), "turns": norm_turns}


def normalize_raw_multiwoz(raw_dialogue: dict[str, Any]) -> dict[str, Any]:
    """Normalize one **raw MultiWOZ 2.2 JSON** dialogue to ``{dialogue_id, turns}``.

    The official JSON shape differs from the HF columnar form: each turn has
    ``speaker`` in ``{"USER","SYSTEM"}`` and ``frames[].state.slot_values`` =
    ``{slot: [values]}``. We keep user turns and take the first listed value as
    the cumulative gold state.
    """
    norm_turns: list[dict[str, Any]] = []
    for turn in raw_dialogue.get("turns", []) or []:
        if str(turn.get("speaker", "")).upper() != "USER":
            continue
        state: dict[str, str] = {}
        for frame in turn.get("frames", []) or []:
            slot_values = (frame.get("state") or {}).get("slot_values") or {}
            for slot, vals in slot_values.items():
                if vals:
                    state[str(slot)] = str(vals[0])
        norm_turns.append({"utterance": str(turn.get("utterance", "")), "state": state})
    return {"dialogue_id": str(raw_dialogue.get("dialogue_id", "")), "turns": norm_turns}


def load_multiwoz(
    split: str = "validation", limit: Optional[int] = None
) -> list[dict[str, Any]]:
    """Load + normalize MultiWOZ 2.2 from the official GitHub JSON shards.

    HuggingFace's ``multi_woz_v22`` is a *script-based* dataset, which recent
    ``datasets`` versions refuse to load. To stay version-independent we read the
    raw 2.2 dialogue JSON directly from the upstream repo
    (``budzianowski/multiwoz``), probing ``dialogues_001.json``,
    ``dialogues_002.json``, … until a shard is missing. ``split`` accepts
    ``"validation"`` (= MultiWOZ ``dev``), ``"train"``, or ``"test"``. Returns
    normalized dialogues ready for :func:`build_from_dialogues`. Requires network
    access (Colab has it); confirm the dataset license before use.
    """
    import json as _json
    import urllib.error
    import urllib.request

    split_dir = {"validation": "dev", "train": "train", "test": "test"}.get(split, split)
    base = (
        "https://raw.githubusercontent.com/budzianowski/multiwoz/master/"
        f"data/MultiWOZ_2.2/{split_dir}"
    )
    out: list[dict[str, Any]] = []
    shard = 1
    while True:
        url = f"{base}/dialogues_{shard:03d}.json"
        try:
            with urllib.request.urlopen(url) as resp:  # noqa: S310 (trusted host)
                payload = _json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and shard > 1:
                break  # no more shards
            raise FileNotFoundError(
                f"Could not fetch MultiWOZ 2.2 shard {url!r} (HTTP {exc.code}). "
                "Check the split name and network access."
            ) from exc
        for raw in payload:
            out.append(normalize_raw_multiwoz(raw))
            if limit is not None and len(out) >= limit:
                return out
        shard += 1
    return out


def load_multiwoz_hf(
    split: str = "validation", limit: Optional[int] = None
) -> list[dict[str, Any]]:
    """Legacy HuggingFace loader (kept for reference).

    Only works on ``datasets`` versions that still permit script-based datasets;
    newer versions reject ``multi_woz_v22``. Prefer :func:`load_multiwoz`.
    """
    from datasets import load_dataset  # lazy import

    ds = load_dataset("multi_woz_v22", split=split, trust_remote_code=True)
    out: list[dict[str, Any]] = []
    for i, d in enumerate(ds):
        if limit is not None and i >= limit:
            break
        out.append(normalize_hf_multiwoz(d))
    return out


# --------------------------------------------------------------------------- #
# Suite runner (reuses the multi-seed harness + stats)
# --------------------------------------------------------------------------- #
def run_multiwoz_suite(
    dialogues: Iterable[dict[str, Any]],
    *,
    baselines: Iterable[str] = ("B0", "B2", "B3"),
    seeds: Iterable[int] = (1337,),
    settings_factory: Any = None,
    embeddings: object | None = None,
    checkpoint_dir: Optional[str] = None,
    run_fingerprint: Optional[str] = None,
) -> dict[str, Any]:
    """Run the governed vs ungoverned comparison on MultiWOZ and aggregate.

    Builds examples + the oracle extractor from ``dialogues``, runs the selected
    baselines through the multi-seed harness (the oracle is the W1 extractor, so
    governance is evaluated given gold slots), and returns decisive metrics
    (mean ± 95% CI) plus write outcomes. The headline is ``constraint_violations``:
    governed arms (B3) supersede a changed slot to zero; ungoverned arms
    (B0/B2, contradiction gate off) accumulate single-valued violations.

    Note: ``task_success`` here is a recall proxy over retrieved slot text; a
    dedicated HAS_VALUE answer-derivation rule is not yet wired, so read the
    governance metrics (violations, write outcomes) as the primary result.
    """
    from ocm.core.config import Settings
    from ocm.evaluation.experiment import aggregate_methods, run_multiseed
    from ocm.evaluation.run_identity import fingerprint_suffix

    if settings_factory is None:
        # MultiWOZ slots are authoritative single-valued state: a changed slot
        # should supersede the prior value (latest wins), not quarantine. Enable
        # the authoritative-update policy by default for this dataset.
        def settings_factory() -> Settings:  # type: ignore[misc]
            return Settings(
                deterministic_test_mode=True,
                chroma_mode="memory",
                extractor="mock",
                authoritative_update_supersede=True,
            )

    examples, oracle = build_from_dialogues(dialogues)
    methods = list(baselines)
    ms = run_multiseed(
        methods,
        seeds=seeds,
        settings_factory=settings_factory,
        extractor=oracle,
        embeddings=embeddings,
        checkpoint_dir=checkpoint_dir,
        # belief-tracking fix changes writes for all arms; the fingerprint fragment
        # additionally separates checkpoints produced by different embedding stacks.
        key_suffix="__multiwoz_v2" + fingerprint_suffix(run_fingerprint),
        provided_examples=examples,
    )
    agg = aggregate_methods(ms)
    return {
        "dataset": "multiwoz",
        "methods": methods,
        "seeds": list(seeds),
        "n_examples": len(examples),
        "decisive_metrics": {
            m: {metric: agg[m][metric].__dict__ for metric in agg[m]} for m in agg
        },
        "write_outcomes": ms.write_outcomes,
    }

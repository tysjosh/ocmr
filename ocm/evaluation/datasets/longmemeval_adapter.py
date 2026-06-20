"""LongMemEval → OCM adapter: knowledge-update tracking as governed memory.

Maps the **LongMemEval** benchmark (Wu et al., ICLR 2025; 500 questions over
multi-session chat, five memory abilities) onto the OCM governed write pipeline,
to validate the governance mechanism on a *recognized agent-memory benchmark* in
the **open-domain** setting — complementing the task-oriented MultiWOZ result.

Scope (deliberately narrow, mirroring the MultiWOZ scoping):

* **`knowledge-update` questions** — a user fact is stated, then *changed* across
  sessions; the correct answer is the **latest** value. This is exactly OCMR's
  single-valued ``Slot -[HAS_VALUE]-> SlotValue`` governance (``m:1``): the
  governed arm **supersedes** the prior value (current value retrievable, zero
  durable violations); an ungoverned arm keeps both (stale-surfacing + a
  single-valued violation). Same claim as MultiWOZ, new domain + benchmark.
* **`abstention` questions** (``question_id`` ends in ``_abs``) — handled by a
  later increment; governed quarantine / insufficient evidence → *abstain*.

Two evaluation arms share this module:

* **Arm A (oracle).** Governance is isolated from extraction by replaying a
  *gold* per-session value trajectory (analogous to MultiWOZ's gold belief
  state). LongMemEval ships no structured per-turn state, so the trajectory is
  produced **once, offline** (see :mod:`ocm.evaluation.datasets.longmemeval_annotate`)
  and cached/committed; this module consumes it deterministically. Built here.
* **Arm B (end-to-end).** The real LLM extractor runs over the full haystack and
  governance acts on noisy candidates; entity resolution becomes load-bearing.
  Wired in a later increment (reuses the same examples + scoring).

Dataset format (``longmemeval_{oracle,s,m}.json``; HF ``xiaowu0162/longmemeval-cleaned``):
each of 500 instances has ``question_id``, ``question_type``, ``question``,
``answer``, ``question_date``, ``haystack_session_ids``, ``haystack_dates``,
``haystack_sessions`` (list of sessions; each a list of ``{"role", "content",
has_answer?}`` turns), and ``answer_session_ids``. Confirm the dataset license
before use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ocm.evaluation.benchmark import BenchmarkExample, Question, Session
from ocm.memory.contracts import ExtractionResult

#: Confidence for a brand-new fact value (first assertion). Mirrors the MultiWOZ
#: adapter / mock extractor so the contradiction gate treats it as a strong fact.
NEW_FACT_CONFIDENCE: float = 0.85

#: Confidence for a *changed* value. Above the gate threshold so the single-valued
#: conflict is detected; the authoritative-``update`` supersede path does not
#: depend on a confidence margin (latest value wins).
UPDATE_CONFIDENCE: float = 0.85

#: Benchmark category label for LongMemEval knowledge-update examples.
CATEGORY: str = "knowledge_update"

#: LongMemEval question_type for the knowledge-update ability.
KNOWLEDGE_UPDATE_TYPE: str = "knowledge-update"

#: Abstention questions carry this question_id suffix (per the official schema).
_ABSTENTION_SUFFIX: str = "_abs"


# --------------------------------------------------------------------------- #
# Oracle extractor (generic; identical contract to the MultiWOZ oracle)
# --------------------------------------------------------------------------- #
@dataclass
class _SessionWrites:
    """The oracle's gold writes for one session (the fact-trajectory delta)."""

    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)


class LongMemEvalOracleExtractor:
    """A W1 extractor returning LongMemEval gold fact deltas keyed by source_ref.

    Implements the extractor protocol (``extract(text, source_ref)``). Stateless
    over a run; a session it does not know returns an empty extraction.
    """

    version: str = "longmemeval-oracle-1"

    def __init__(self, writes_by_ref: dict[str, _SessionWrites]) -> None:
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


# --------------------------------------------------------------------------- #
# Loader / subset filtering
# --------------------------------------------------------------------------- #
def is_abstention(instance: dict[str, Any]) -> bool:
    """True when an instance is an abstention question (``question_id`` ``_abs``)."""
    return str(instance.get("question_id", "")).endswith(_ABSTENTION_SUFFIX)


def load_longmemeval(
    path: str,
    *,
    question_type: Optional[str] = KNOWLEDGE_UPDATE_TYPE,
    abstention: Optional[bool] = False,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Load + filter LongMemEval instances from a local JSON file.

    The dataset is distributed as a single JSON array of 500 instances (download
    one of ``longmemeval_oracle.json`` / ``longmemeval_s.json`` /
    ``longmemeval_m.json`` from the HF repo first). Filtering:

    * ``question_type`` — keep only this ability (default ``knowledge-update``);
      pass ``None`` to keep all types.
    * ``abstention`` — ``False`` drops ``_abs`` questions (default), ``True`` keeps
      *only* them, ``None`` is indifferent.
    * ``limit`` — cap the number returned (after filtering).
    """
    import json as _json

    with open(path, "r", encoding="utf-8") as fh:
        data = _json.load(fh)

    out: list[dict[str, Any]] = []
    for inst in data:
        if question_type is not None and str(inst.get("question_type")) != question_type:
            continue
        if abstention is True and not is_abstention(inst):
            continue
        if abstention is False and is_abstention(inst):
            continue
        out.append(inst)
        if limit is not None and len(out) >= limit:
            break
    return out


def _slot_key(question_id: str, attribute: str) -> str:
    """Qualified Slot name: scope an attribute to its question (shared store)."""
    return f"{question_id}:{attribute}"


def _session_text(session: Any) -> str:
    """Flatten a haystack session (list of ``{role, content}`` turns) to text."""
    if not isinstance(session, list):
        return str(session)
    parts = []
    for turn in session:
        if isinstance(turn, dict):
            parts.append(f"{turn.get('role', 'user')}: {turn.get('content', '')}")
        else:
            parts.append(str(turn))
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Arm A — oracle adapter (governance isolated from extraction)
# --------------------------------------------------------------------------- #
def build_from_kupdate_oracle(
    instances: Iterable[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
) -> tuple[list[BenchmarkExample], LongMemEvalOracleExtractor]:
    """Build benchmark examples + an oracle extractor for the knowledge-update arm.

    ``annotations`` maps ``question_id`` → a **gold value trajectory** (produced
    once, offline; see :mod:`longmemeval_annotate`)::

        {
          "attribute": "residence",          # the single fact the question asks about
          "trajectory": [                      # ordered oldest → newest
            {"session_id": "<haystack id>", "value": "New York"},
            {"session_id": "<haystack id>", "value": "San Francisco"}
          ],
          "current_value": "San Francisco"     # == benchmark answer (sanity check)
        }

    For each instance we emit, per haystack session: a ``new_fact`` for the first
    value of the attribute and an ``update`` for each later change, keyed to that
    session's ``source_ref`` (``"<question_id>:s<idx>"``). The recall question
    addresses the attribute by its qualified ``[[key]]`` marker so the
    ``HAS_VALUE`` answer-derivation rule resolves the current accepted value.

    Returns ``(examples, oracle_extractor)``; pass the extractor into the
    container/harness as ``extractor=...``.
    """
    examples: list[BenchmarkExample] = []
    writes_by_ref: dict[str, _SessionWrites] = {}

    for inst in instances:
        qid = str(inst["question_id"])
        ann = annotations.get(qid)
        if ann is None:
            continue  # no gold trajectory → skip (Arm A is annotation-scoped)
        attribute = str(ann["attribute"])
        slot_name = _slot_key(qid, attribute)

        sessions_raw = inst.get("haystack_sessions", []) or []
        session_ids = inst.get("haystack_session_ids", []) or []
        # Map a haystack session id → its enumerated index so the trajectory
        # (keyed by session id) aligns with the per-session source_ref.
        id_to_idx = {str(sid): i for i, sid in enumerate(session_ids)}

        # Bucket trajectory values by the session index they are stated in,
        # preserving oldest→newest order for stable new_fact/update assignment.
        per_session_values: dict[int, list[str]] = {}
        ordered = list(ann.get("trajectory", []))
        for entry in ordered:
            sid = str(entry.get("session_id", ""))
            idx = id_to_idx.get(sid)
            if idx is None:
                continue  # trajectory references a session not in this haystack
            per_session_values.setdefault(idx, []).append(str(entry["value"]))

        sessions: list[Session] = []
        belief_set = False
        for idx, session in enumerate(sessions_raw):
            sw = _SessionWrites()
            for value in per_session_values.get(idx, []):
                intent = "update" if belief_set else "new_fact"
                conf = UPDATE_CONFIDENCE if belief_set else NEW_FACT_CONFIDENCE
                sw.entities.append({"type": "Slot", "name": slot_name})
                sw.entities.append(
                    {"type": "SlotValue", "name": value, "fields": {"value": value}}
                )
                sw.relations.append(
                    {
                        "subject": slot_name,
                        "predicate": "HAS_VALUE",
                        "object": value,
                        "confidence": conf,
                        "write_intent": intent,
                    }
                )
                belief_set = True
            source_ref = f"{qid}:s{idx}"
            writes_by_ref[source_ref] = sw
            sessions.append(Session(session_id=f"s{idx}", input=_session_text(session)))

        current = str(ann.get("current_value", ordered[-1]["value"] if ordered else ""))
        question = Question(
            query=f"{inst.get('question', '')} [[{slot_name}]]",
            expected_answer_contains=[current],
            expected_conflict=False,
        )
        examples.append(
            BenchmarkExample(
                id=qid, category=CATEGORY, sessions=sessions, questions=[question]
            )
        )

    return examples, LongMemEvalOracleExtractor(writes_by_ref)


# --------------------------------------------------------------------------- #
# Tiny hand-built fixture (LongMemEval-shaped) for offline tests.
# --------------------------------------------------------------------------- #
def sample_instances() -> list[dict[str, Any]]:
    """A small LongMemEval-shaped knowledge-update fixture (no download)."""
    return [
        {
            "question_id": "ku_0001",
            "question_type": KNOWLEDGE_UPDATE_TYPE,
            "question": "Where does the user currently live?",
            "answer": "San Francisco",
            "haystack_session_ids": ["sess_a", "sess_b", "sess_c"],
            "haystack_sessions": [
                [{"role": "user", "content": "I just moved to New York."},
                 {"role": "assistant", "content": "Nice!"}],
                [{"role": "user", "content": "The weather is cold today."}],
                [{"role": "user", "content": "Actually I relocated to San Francisco."}],
            ],
            "answer_session_ids": ["sess_a", "sess_c"],
        },
    ]


def sample_annotations() -> dict[str, dict[str, Any]]:
    """Gold trajectory matching :func:`sample_instances` (stands in for the cache)."""
    return {
        "ku_0001": {
            "attribute": "residence",
            "trajectory": [
                {"session_id": "sess_a", "value": "New York"},
                {"session_id": "sess_c", "value": "San Francisco"},
            ],
            "current_value": "San Francisco",
        }
    }

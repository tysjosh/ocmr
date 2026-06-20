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


# --------------------------------------------------------------------------- #
# Abstention arm — governed refusal instead of fabrication
# --------------------------------------------------------------------------- #
#: Sentinel attribute for an abstention probe: a fact the question asks about
#: that was never grounded in memory, so a faithful system must abstain.
_ABSTAIN_ATTR: str = "__ungrounded__"


def build_abstention_examples(
    instances: Iterable[dict[str, Any]],
) -> tuple[list[BenchmarkExample], LongMemEvalOracleExtractor]:
    """Build abstention probes: a question whose answer is **not** grounded.

    For each ``_abs`` instance we ingest the haystack sessions as *context* but
    emit **no** gold ``HAS_VALUE`` write for the queried fact (the question refers
    to something never established). A faithful system must therefore **abstain**
    — derive no current value — rather than fabricate one. The recall query
    addresses an ungrounded slot key via the ``[[key]]`` marker so the
    answer-derivation rule returns ``None`` when memory holds no accepted value.

    Honest scope: in the **oracle** setting this validates the *governed-abstain
    plumbing* (no grounded value → no answer). It is **not** discriminative across
    baselines here, because no arm has a fabricated value to surface. The
    discriminating result — a governed arm abstaining where an ungoverned arm
    surfaces a distractor-derived false answer — requires the **end-to-end arm**
    (real extraction + retrieval over the full haystack), wired in a later
    increment. ``evaluate_abstention`` reports the plumbing metric for now.
    """
    examples: list[BenchmarkExample] = []
    writes_by_ref: dict[str, _SessionWrites] = {}

    for inst in instances:
        qid = str(inst["question_id"])
        slot_name = _slot_key(qid, _ABSTAIN_ATTR)
        sessions: list[Session] = []
        for idx, session in enumerate(inst.get("haystack_sessions", []) or []):
            # Context only — no grounded write for the queried fact.
            writes_by_ref[f"{qid}:s{idx}"] = _SessionWrites()
            sessions.append(Session(session_id=f"s{idx}", input=_session_text(session)))
        # expected_answer_contains is empty: correctness is "abstained", scored by
        # evaluate_abstention (the standard token-recall metric does not apply).
        question = Question(
            query=f"{inst.get('question', '')} value of slot [[{slot_name}]]",
            expected_answer_contains=[],
            expected_conflict=False,
        )
        examples.append(
            BenchmarkExample(
                id=qid, category="abstention", sessions=sessions, questions=[question]
            )
        )
    return examples, LongMemEvalOracleExtractor(writes_by_ref)


def _answer_is_abstention(answer: Optional[str]) -> bool:
    """True when an answer counts as an abstention (empty / explicit refusal)."""
    if answer is None:
        return True
    text = str(answer).strip().lower()
    if not text:
        return True
    refusals = ("i don't", "i do not", "no information", "not sure", "cannot",
                "can't", "unknown", "no record", "not available")
    return any(r in text for r in refusals)


def evaluate_abstention(
    instances: Iterable[dict[str, Any]],
    *,
    baselines: Iterable[str] = ("B0", "B2", "B3"),
    settings_factory: Any = None,
    embeddings: object | None = None,
) -> dict[str, Any]:
    """Score governed abstention on ``_abs`` probes (oracle plumbing metric).

    For each baseline: ingest the haystack context, query the ungrounded fact,
    and count the response as correct when the system **abstains**
    (:func:`_answer_is_abstention`). Returns per-baseline abstention accuracy.

    See :func:`build_abstention_examples` for the scope caveat: this is the
    oracle plumbing check; cross-baseline divergence is an end-to-end result.
    """
    from ocm.core.config import Settings
    from ocm.core.container import CoreContainer
    from ocm.evaluation.baselines import build_baseline

    if settings_factory is None:
        def settings_factory() -> Settings:  # type: ignore[misc]
            return Settings(
                deterministic_test_mode=True, chroma_mode="memory", extractor="mock",
                authoritative_update_supersede=True,
            )

    examples, oracle = build_abstention_examples(instances)
    out: dict[str, Any] = {}
    for method in baselines:
        container = CoreContainer(settings_factory(), extractor=oracle, embeddings=embeddings)
        for ex in examples:
            for s in ex.sessions:
                container.write_pipeline.run(s.input, f"{ex.id}:{s.session_id}")
        baseline = build_baseline(method, container)
        correct = 0
        for ex in examples:
            pkg = baseline.query(ex.questions[0].query, top_k=10)
            if _answer_is_abstention(getattr(pkg, "answer", None)):
                correct += 1
        n = len(examples)
        out[method] = {
            "abstention_accuracy": (100.0 * correct / n) if n else 0.0,
            "n": n,
        }
    return out


# --------------------------------------------------------------------------- #
# Suite runner (knowledge-update arm; reuses the multi-seed harness + stats)
# --------------------------------------------------------------------------- #
def run_longmemeval_suite(
    instances: Iterable[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    *,
    baselines: Iterable[str] = ("B0", "B2", "B3"),
    seeds: Iterable[int] = (1337,),
    settings_factory: Any = None,
    embeddings: object | None = None,
    checkpoint_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Run governed vs ungoverned on LongMemEval knowledge-update (Arm A / oracle).

    Builds examples + the oracle extractor from ``instances`` + gold
    ``annotations`` and runs the selected baselines through the multi-seed
    harness (the oracle is the W1 extractor, so governance is evaluated *given*
    gold facts). Returns decisive metrics (mean ± 95% CI) + write outcomes. The
    headline mirrors MultiWOZ: B3 supersedes a changed fact (one accepted value,
    zero durable violations); ungoverned arms accumulate single-valued
    violations. Recall is preserved via the ``HAS_VALUE`` answer-derivation rule.

    Use the dataset-specific ``key_suffix`` ``__lme_kupdate`` so checkpoints stay
    separate from the synthetic and MultiWOZ runs.
    """
    from ocm.core.config import Settings
    from ocm.evaluation.experiment import aggregate_methods, run_multiseed

    if settings_factory is None:
        # Knowledge updates are authoritative single-valued state (latest value
        # wins), exactly like MultiWOZ slots — enable the policy by default.
        def settings_factory() -> Settings:  # type: ignore[misc]
            return Settings(
                deterministic_test_mode=True, chroma_mode="memory", extractor="mock",
                authoritative_update_supersede=True,
            )

    examples, oracle = build_from_kupdate_oracle(instances, annotations)
    methods = list(baselines)
    ms = run_multiseed(
        methods,
        seeds=seeds,
        settings_factory=settings_factory,
        extractor=oracle,
        embeddings=embeddings,
        checkpoint_dir=checkpoint_dir,
        key_suffix="__lme_kupdate",
        provided_examples=examples,
    )
    agg = aggregate_methods(ms)
    return {
        "dataset": "longmemeval",
        "subset": "knowledge-update",
        "methods": methods,
        "seeds": list(seeds),
        "n_examples": len(examples),
        "decisive_metrics": {
            m: {metric: agg[m][metric].__dict__ for metric in agg[m]} for m in agg
        },
        "write_outcomes": ms.write_outcomes,
    }


# --------------------------------------------------------------------------- #
# Arm B — end-to-end from text (real extraction, belief-tracked, cached)
# --------------------------------------------------------------------------- #
# Design: extract user facts ONCE with the real LLM, tracking a per-question
# belief so a changed fact is emitted as an authoritative ``update`` (and a
# repeated value is skipped); cache the writes per session ``source_ref`` and
# replay them through the *stateless* oracle replayer. This (a) is genuinely
# end-to-end from raw text, (b) keeps the LLM off the per-method/seed hot path
# (extraction is not repeated per arm), and (c) reuses all of Arm A's machinery.
# The entity-resolution stress lives in attribute normalization: the extractor
# must map a fact's mentions across sessions to ONE stable slot key, or the
# update is never detected as a single-valued conflict.

#: Prompt a model to extract durable user facts from one message/session.
FACT_EXTRACTION_PROMPT = """\
Extract durable USER facts (stable attributes the user states about themselves,
e.g. where they live, their job, preferences) from the message below.

Message:
{text}

Respond with ONLY a JSON array, no prose. Each item:
{{"attribute": "<short snake_case name, stable across messages>", "value": "<short value>"}}
Use the SAME attribute name whenever the user refers to the same fact. If the
message states no durable user fact, respond with [].
"""


def normalize_attribute(attribute: str) -> str:
    """Normalize a free-text attribute to a stable snake_case slot key fragment."""
    import re

    s = str(attribute).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def parse_facts_json(text: str) -> list[dict[str, Any]]:
    """Extract the first JSON array of objects from a model response (tolerant).

    Returns a list of dicts (possibly empty). Accepts a bare array, or an object
    with a ``"facts"`` list. Malformed responses yield ``[]``.
    """
    import json as _json

    if not text:
        return []
    start = text.find("[")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        arr = _json.loads(text[start : i + 1])
                        if isinstance(arr, list):
                            return [x for x in arr if isinstance(x, dict)]
                    except _json.JSONDecodeError:
                        break
        start = text.find("[", start + 1)
    # Fallback: an object with a "facts" array.
    obj_start = text.find("{")
    if obj_start != -1:
        try:
            obj = _json.loads(text[obj_start : text.rfind("}") + 1])
            if isinstance(obj, dict) and isinstance(obj.get("facts"), list):
                return [x for x in obj["facts"] if isinstance(x, dict)]
        except _json.JSONDecodeError:
            pass
    return []


#: A fact extractor maps one session's text → a list of ``{attribute, value}``.
def build_fact_extract_fn(chat_fn) -> Any:
    """Build a per-session fact extractor from a chat callable (``prompt -> text``)."""

    def _extract(text: str) -> list[dict[str, Any]]:
        return parse_facts_json(chat_fn(FACT_EXTRACTION_PROMPT.format(text=text)))

    return _extract


def build_e2e_from_extraction(
    instances: Iterable[dict[str, Any]],
    fact_extract_fn: Any,
    *,
    intent_mode: str = "auto",
) -> tuple[list[BenchmarkExample], LongMemEvalOracleExtractor]:
    """Build end-to-end examples by extracting facts from the haystack with an LLM.

    ``fact_extract_fn`` maps one session's text → ``[{attribute, value}, ...]``
    (see :func:`build_fact_extract_fn`). Per question we walk the sessions in
    order, normalize each attribute to a stable slot key, and track a belief so a
    changed value is emitted as an ``update`` (``intent_mode="auto"``) — or always
    as ``new_fact`` (``intent_mode="new_fact"``, the conservative case where the
    gate quarantines conflicts). Writes are cached per ``source_ref`` and replayed
    via the stateless oracle, so the LLM runs once regardless of arms/seeds.

    The recall question is the **natural** LongMemEval question (no slot marker);
    the answer is scored against the gold ``answer`` via the harness's
    token-containment metric over retrieved evidence. Returns
    ``(examples, oracle_extractor)``.
    """
    if intent_mode not in ("auto", "new_fact"):
        raise ValueError("intent_mode must be 'auto' or 'new_fact'")

    examples: list[BenchmarkExample] = []
    writes_by_ref: dict[str, _SessionWrites] = {}

    for inst in instances:
        qid = str(inst["question_id"])
        belief: dict[str, str] = {}
        sessions: list[Session] = []
        for idx, session in enumerate(inst.get("haystack_sessions", []) or []):
            text = _session_text(session)
            sw = _SessionWrites()
            for fact in fact_extract_fn(text):
                attr = normalize_attribute(fact.get("attribute", ""))
                value = str(fact.get("value", "")).strip()
                if not attr or not value:
                    continue
                prev = belief.get(attr)
                if prev == value:
                    continue  # re-asserted same value — no write
                if prev is None:
                    intent, conf = "new_fact", NEW_FACT_CONFIDENCE
                else:
                    intent = "update" if intent_mode == "auto" else "new_fact"
                    conf = UPDATE_CONFIDENCE
                slot_name = _slot_key(qid, attr)
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
                belief[attr] = value
            writes_by_ref[f"{qid}:s{idx}"] = sw
            sessions.append(Session(session_id=f"s{idx}", input=text))

        question = Question(
            query=str(inst.get("question", "")),
            expected_answer_contains=[str(inst.get("answer", ""))],
            expected_conflict=False,
        )
        examples.append(
            BenchmarkExample(
                id=qid, category="knowledge_update_e2e",
                sessions=sessions, questions=[question],
            )
        )

    return examples, LongMemEvalOracleExtractor(writes_by_ref)


def run_longmemeval_e2e(
    instances: Iterable[dict[str, Any]],
    fact_extract_fn: Any,
    *,
    intent_mode: str = "auto",
    baselines: Iterable[str] = ("B0", "B2", "B3"),
    seeds: Iterable[int] = (1337,),
    settings_factory: Any = None,
    embeddings: object | None = None,
    checkpoint_dir: Optional[str] = None,
) -> dict[str, Any]:
    """End-to-end LongMemEval knowledge-update run (Arm B): real extraction.

    Extracts facts from the haystack with ``fact_extract_fn`` (belief-tracked,
    cached per session), then runs the governed vs ungoverned comparison via the
    multi-seed harness, replaying the cached writes through the stateless oracle.
    Checkpoint suffix ``__lme_e2e`` keeps these separate from the Arm-A oracle
    run. Unlike Arm A, recall is scored on the **natural** question via retrieval,
    so it reflects real extraction + retrieval, not gold facts.
    """
    from ocm.core.config import Settings
    from ocm.evaluation.experiment import aggregate_methods, run_multiseed

    if settings_factory is None:
        def settings_factory() -> Settings:  # type: ignore[misc]
            return Settings(
                deterministic_test_mode=True, chroma_mode="memory", extractor="mock",
                authoritative_update_supersede=True,
            )

    examples, oracle = build_e2e_from_extraction(
        instances, fact_extract_fn, intent_mode=intent_mode
    )
    methods = list(baselines)
    ms = run_multiseed(
        methods,
        seeds=seeds,
        settings_factory=settings_factory,
        extractor=oracle,
        embeddings=embeddings,
        checkpoint_dir=checkpoint_dir,
        key_suffix="__lme_e2e",
        provided_examples=examples,
    )
    agg = aggregate_methods(ms)
    return {
        "dataset": "longmemeval",
        "subset": "knowledge-update",
        "arm": "end_to_end",
        "intent_mode": intent_mode,
        "methods": methods,
        "seeds": list(seeds),
        "n_examples": len(examples),
        "decisive_metrics": {
            m: {metric: agg[m][metric].__dict__ for metric in agg[m]} for m in agg
        },
        "write_outcomes": ms.write_outcomes,
    }

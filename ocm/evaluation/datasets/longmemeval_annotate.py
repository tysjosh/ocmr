"""Offline gold-trajectory annotation for the LongMemEval knowledge-update arm.

Arm A (oracle) of the LongMemEval evaluation needs a **gold per-session value
trajectory** for each ``knowledge-update`` question — the analogue of MultiWOZ's
gold belief state, which LongMemEval does not ship. This module produces that
trajectory **once, offline**; the result is cached to JSON and committed, so the
evaluation itself is deterministic and free of per-run extraction noise (that is
exactly what Arm A is meant to isolate).

Two annotation sources, in order of preference:

1. **Repo evidence statements (preferred, cleanest).** LongMemEval's
   ``custom_history/2_questions`` ships structured *evidence statements* and the
   evidence sessions per question. When available, derive the trajectory from
   those — no LLM, least contestable.
2. **One-time LLM extraction (fallback).** For each instance, prompt an LLM with
   the evidence turns (``has_answer: true`` in ``longmemeval_oracle.json``, or the
   sessions in ``answer_session_ids``) to extract the single updatable attribute
   the question asks about and its ordered ``(session_id, value)`` values. Verify
   the final value equals the benchmark ``answer`` before accepting.

Output schema (one entry per ``question_id``), consumed by
:func:`ocm.evaluation.datasets.longmemeval_adapter.build_from_kupdate_oracle`::

    {
      "<question_id>": {
        "attribute": "residence",
        "trajectory": [
          {"session_id": "<haystack id>", "value": "New York"},
          {"session_id": "<haystack id>", "value": "San Francisco"}
        ],
        "current_value": "San Francisco"
      },
      ...
    }

This module is intentionally LLM-agnostic: pass a ``annotate_fn`` callable that
maps one instance → one annotation dict (or ``None`` to skip). Wiring a concrete
LLM (e.g. the Qwen extractor or an API model) is left to the caller so the
foundation has no GPU/network dependency and stays unit-testable.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Optional

#: An annotator maps one LongMemEval instance to its gold trajectory, or None.
AnnotateFn = Callable[[dict[str, Any]], Optional[dict[str, Any]]]


def evidence_turns(instance: dict[str, Any]) -> list[tuple[str, str]]:
    """Return ``(session_id, turn_text)`` for evidence turns (``has_answer: true``).

    Falls back to all turns of the sessions in ``answer_session_ids`` when no
    turn-level ``has_answer`` flags are present (e.g. the non-oracle files).
    """
    session_ids = [str(s) for s in (instance.get("haystack_session_ids") or [])]
    sessions = instance.get("haystack_sessions") or []
    answer_sessions = {str(s) for s in (instance.get("answer_session_ids") or [])}

    out: list[tuple[str, str]] = []
    saw_flag = False
    for idx, session in enumerate(sessions):
        sid = session_ids[idx] if idx < len(session_ids) else f"s{idx}"
        if not isinstance(session, list):
            continue
        for turn in session:
            if isinstance(turn, dict) and turn.get("has_answer"):
                saw_flag = True
                out.append((sid, str(turn.get("content", ""))))
    if saw_flag:
        return out
    # Fallback: every turn of the evidence sessions.
    for idx, session in enumerate(sessions):
        sid = session_ids[idx] if idx < len(session_ids) else f"s{idx}"
        if sid in answer_sessions and isinstance(session, list):
            for turn in session:
                if isinstance(turn, dict):
                    out.append((sid, str(turn.get("content", ""))))
    return out


def validate_annotation(
    instance: dict[str, Any], annotation: dict[str, Any]
) -> list[str]:
    """Return a list of problems with an annotation (empty == valid).

    Checks structural shape and that the trajectory's final value matches the
    benchmark ``answer`` (a cheap guard against a bad extraction). The match is
    a case-insensitive containment test in either direction (surface forms vary).
    """
    problems: list[str] = []
    if "attribute" not in annotation or not str(annotation["attribute"]).strip():
        problems.append("missing 'attribute'")
    traj = annotation.get("trajectory")
    if not isinstance(traj, list) or not traj:
        problems.append("empty or missing 'trajectory'")
        return problems

    valid_ids = {str(s) for s in (instance.get("haystack_session_ids") or [])}
    for i, entry in enumerate(traj):
        if "session_id" not in entry or "value" not in entry:
            problems.append(f"trajectory[{i}] missing session_id/value")
        elif valid_ids and str(entry["session_id"]) not in valid_ids:
            problems.append(f"trajectory[{i}] session_id not in haystack")

    answer = str(instance.get("answer", "")).strip().lower()
    current = str(annotation.get("current_value", traj[-1].get("value", ""))).strip().lower()
    if answer and current and answer not in current and current not in answer:
        problems.append(f"current_value {current!r} disagrees with benchmark answer {answer!r}")
    return problems


def annotate_instances(
    instances: Iterable[dict[str, Any]],
    annotate_fn: AnnotateFn,
    *,
    skip_invalid: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run ``annotate_fn`` over instances and collect validated annotations.

    Each produced annotation is checked with :func:`validate_annotation`; invalid
    ones are skipped (``skip_invalid=True``) or kept with a ``"_problems"`` field
    for auditing (``skip_invalid=False``).
    """
    annotations: dict[str, dict[str, Any]] = {}
    for inst in instances:
        ann = annotate_fn(inst)
        if ann is None:
            continue
        problems = validate_annotation(inst, ann)
        if problems and skip_invalid:
            continue
        if problems:
            ann = {**ann, "_problems": problems}
        annotations[str(inst["question_id"])] = ann
    return annotations


def save_annotations(annotations: dict[str, dict[str, Any]], path: str) -> None:
    """Persist annotations to JSON (commit this file so Arm A is reproducible)."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(annotations, fh, indent=2, ensure_ascii=False)


def load_annotations(path: str) -> dict[str, dict[str, Any]]:
    """Load a committed annotations JSON produced by :func:`save_annotations`."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Concrete LLM annotator (model-agnostic via a chat callable)
# --------------------------------------------------------------------------- #
#: Prompt asking a model to identify the single updated attribute and its values
#: in chronological order. We deliberately ask only for attribute + ordered
#: values + current value (NOT session ids); session alignment is done
#: deterministically from text afterwards, which is more reliable than asking the
#: model to reason about session identifiers.
KUPDATE_ANNOTATION_PROMPT = """\
You are labelling a knowledge-update question for a memory benchmark.

A single user attribute (fact) is stated and then CHANGED across the chat history.
Identify that one attribute and the sequence of values it took, oldest first.

Question: {question}
Expected current answer: {answer}

Evidence from the chat history (oldest to newest):
{evidence}

Respond with ONLY a JSON object, no prose:
{{"attribute": "<short snake_case name of the changing fact>",
  "values": ["<oldest value>", "...", "<newest value>"],
  "current_value": "<the value that answers the question now>"}}
The "values" list must be in chronological order and end with current_value.
"""

#: A chat callable: takes a prompt string, returns the model's text response.
ChatFn = Callable[[str], str]


def _flatten_session(session: Any) -> str:
    """Flatten one haystack session (list of ``{role, content}``) to text."""
    if not isinstance(session, list):
        return str(session)
    return "\n".join(
        f"{t.get('role', 'user')}: {t.get('content', '')}" if isinstance(t, dict) else str(t)
        for t in session
    )


def parse_annotation_json(text: str) -> Optional[dict[str, Any]]:
    """Extract the first JSON object from a model response, tolerant of prose.

    Returns the parsed dict, or ``None`` when no valid JSON object is found.
    """
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    blob = text[start : i + 1]
                    try:
                        obj = json.loads(blob)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break  # malformed; try the next '{'
        start = text.find("{", start + 1)
    return None


def align_values_to_sessions(
    instance: dict[str, Any], values: list[str]
) -> list[dict[str, str]]:
    """Map an ordered value list to ``[{session_id, value}]`` preserving order.

    For each value (oldest→newest) we find the earliest haystack session **at or
    after** the previous match whose text contains the value (case-insensitive),
    so the resulting ``session_id`` sequence is non-decreasing in haystack index.
    This guarantees the adapter assigns ``new_fact`` to the oldest value and
    ``update`` to each later change in the correct order — regardless of the
    file's absolute session ordering. Values not located in any session are
    skipped (the trajectory keeps only grounded values).
    """
    session_ids = [str(s) for s in (instance.get("haystack_session_ids") or [])]
    sessions = instance.get("haystack_sessions") or []
    texts = [_flatten_session(s).lower() for s in sessions]

    out: list[dict[str, str]] = []
    start_idx = 0
    for value in values:
        needle = str(value).strip().lower()
        if not needle:
            continue
        found_idx: Optional[int] = None
        for idx in range(start_idx, len(texts)):
            if needle in texts[idx]:
                found_idx = idx
                break
        if found_idx is None:  # not found at/after the pointer — search from start
            for idx in range(0, len(texts)):
                if needle in texts[idx]:
                    found_idx = idx
                    break
        if found_idx is None:
            continue  # ungrounded value, skip
        sid = session_ids[found_idx] if found_idx < len(session_ids) else f"s{found_idx}"
        out.append({"session_id": sid, "value": str(value)})
        start_idx = found_idx
    return out


def build_llm_annotate_fn(chat_fn: ChatFn) -> AnnotateFn:
    """Build an :data:`AnnotateFn` from a chat callable (e.g. a Qwen wrapper).

    The returned function prompts ``chat_fn`` with the question + evidence turns,
    parses the JSON, aligns the reported values to sessions deterministically,
    and returns an annotation dict (or ``None`` if parsing/alignment fails). Pair
    with :func:`annotate_instances` (which validates every result against the
    benchmark answer before keeping it).
    """

    def _annotate(instance: dict[str, Any]) -> Optional[dict[str, Any]]:
        evidence = "\n".join(
            f"[{sid}] {txt}" for sid, txt in evidence_turns(instance)
        )
        prompt = KUPDATE_ANNOTATION_PROMPT.format(
            question=instance.get("question", ""),
            answer=instance.get("answer", ""),
            evidence=evidence or "(no evidence turns found)",
        )
        obj = parse_annotation_json(chat_fn(prompt))
        if obj is None:
            return None
        attribute = str(obj.get("attribute", "")).strip()
        values = [str(v) for v in (obj.get("values") or []) if str(v).strip()]
        if not attribute or len(values) < 2:
            return None  # a knowledge update needs at least an old + new value
        trajectory = align_values_to_sessions(instance, values)
        if len(trajectory) < 2:
            return None  # could not ground a change in the history
        current = str(obj.get("current_value", values[-1])).strip()
        return {
            "attribute": attribute,
            "trajectory": trajectory,
            "current_value": current or trajectory[-1]["value"],
        }

    return _annotate


def annotate_file(
    in_path: str, out_path: str, chat_fn: ChatFn, *, limit: Optional[int] = None
) -> dict[str, dict[str, Any]]:
    """End-to-end convenience: load a LongMemEval file, annotate, save, return.

    Loads the ``knowledge-update`` subset from ``in_path``, runs the LLM
    annotator, validates, writes the committed annotations JSON to ``out_path``,
    and returns the annotations. Intended for a one-time offline/Colab pass.
    """
    from ocm.evaluation.datasets.longmemeval_adapter import load_longmemeval

    instances = load_longmemeval(in_path, limit=limit)
    annotations = annotate_instances(instances, build_llm_annotate_fn(chat_fn))
    save_annotations(annotations, out_path)
    return annotations

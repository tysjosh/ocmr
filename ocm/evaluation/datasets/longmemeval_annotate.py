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

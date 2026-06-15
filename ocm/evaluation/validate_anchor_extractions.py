"""Read-only validation scan for the six hand-authored anchors (Req 23.4).

This is a **diagnostic, not a fix**. It inspects a persisted ``CachingExtractor``
cache (the JSON produced during a governed run) and reports, per anchor, whether
the *precondition* the anchor's target constraint needs is actually present in
the extraction. It writes nothing and never calls the LLM — it only re-derives
the cache keys for the anchor sessions (exactly as the harness does) and reads
the stored :class:`~ocm.memory.contracts.ExtractionResult`.

Motivation: some anchors only exercise their constraint if the extractor lands a
specific field. Two known-fragile cases:

* **anchor-inactive-assignee (C5)** needs the Person node to carry
  ``fields.status == "inactive"``. If the model emits "Mallory is inactive" as a
  free-text *claim* with empty ``fields``, the inactive flag never reaches the
  node and C5 cannot fire.
* **anchor-temporal-cycle** needs both ``PRECEDES`` edges to reference the *same*
  two event nodes after :func:`normalize_name`. If s1 says ``Kickoff``/``Review``
  but s2 says ``Event Kickoff``/``Event Review``, the edges land on different
  nodes and the cycle never closes.

Usage (CLI)::

    python -m ocm.evaluation.validate_anchor_extractions --cache /path/to/extraction_cache.json

Usage (notebook)::

    from ocm.evaluation.validate_anchor_extractions import scan_anchor_cache
    report = scan_anchor_cache("/content/ocm_results/extraction_cache.json")

The function returns a structured report and prints a human-readable table; it
exits 0 regardless of findings (it is a report, not a gate).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional

from ocm.evaluation.benchmark import BenchmarkGenerator
from ocm.extraction.caching_extractor import CachingExtractor
from ocm.memory.contracts import ExtractionResult
from ocm.resolution.entity_resolver import normalize_name

#: Anchor id -> the constraint/behaviour the anchor is meant to exercise.
ANCHOR_TARGET: dict[str, str] = {
    "anchor-task-t1-conflict": "C7 contradiction gate (Task status done vs todo)",
    "anchor-joseph-pharaoh": "longitudinal recall (OWNS + ASSIGNED_TO + done)",
    "anchor-project-owner-conflict": "single-owner conflict (two OWNS, same Project)",
    "anchor-inactive-assignee": "C5 (assign Task to an inactive Person)",
    "anchor-final-decision-no-evidence": "C8 (final Decision needs EVIDENCE_FOR)",
    "anchor-temporal-cycle": "temporal cycle (PRECEDES forms a cycle)",
}

_INACTIVE_TOKENS = {"inactive", "deactivated", "disabled", "former", "left"}


def _load_cache(cache_path: str) -> dict[str, ExtractionResult]:
    """Load a persisted CachingExtractor JSON into ``{key: ExtractionResult}``."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Extraction cache not found: {cache_path}")
    with open(cache_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    out: dict[str, ExtractionResult] = {}
    for key, value in payload.items():
        try:
            out[key] = ExtractionResult.model_validate(value)
        except Exception:  # pragma: no cover - tolerate a partial/corrupt entry
            continue
    return out


def _anchor_sessions() -> dict[str, list[tuple[str, str]]]:
    """Return ``{anchor_id: [(source_ref, text), ...]}`` for the six anchors."""
    examples = BenchmarkGenerator().generate()
    sessions: dict[str, list[tuple[str, str]]] = {}
    for ex in examples:
        if not ex.id.startswith("anchor-"):
            continue
        sessions[ex.id] = [
            (f"{ex.id}:{s.session_id}", s.input) for s in ex.sessions
        ]
    return sessions


def _lookup(
    cache: dict[str, ExtractionResult], source_ref: str, text: str
) -> Optional[ExtractionResult]:
    """Re-derive the harness cache key and return the stored extraction (or None)."""
    return cache.get(CachingExtractor._key(text, source_ref))


# --- per-anchor precondition checks ----------------------------------------
# Each returns (ok, detail) where ``ok`` is True when the constraint's
# precondition is present in the extraction(s), so the anchor can fire live.


def _check_inactive_assignee(sess: dict[str, ExtractionResult]) -> tuple[bool, str]:
    s1 = sess.get("anchor-inactive-assignee:s1")
    if s1 is None:
        return False, "s1 not in cache"
    status = None
    for ent in s1.entities:
        if ent.get("type") == "Person" and (ent.get("name") or "").strip().lower() == "mallory":
            status = (ent.get("fields") or {}).get("status")
            break
    if status and str(status).lower() in _INACTIVE_TOKENS:
        return True, f"Mallory.fields.status = {status!r}"
    has_claim = any("inactive" in (c.get("text", "").lower()) for c in s1.claims)
    where = "claim-only" if has_claim else "absent"
    return False, f"Mallory.fields.status = {status!r} (inactive captured as: {where})"


def _check_temporal_cycle(sess: dict[str, ExtractionResult]) -> tuple[bool, str]:
    s1 = sess.get("anchor-temporal-cycle:s1")
    s2 = sess.get("anchor-temporal-cycle:s2")
    if s1 is None or s2 is None:
        return False, "s1 or s2 not in cache"

    def precedes_pairs(r: ExtractionResult) -> set[tuple[str, str]]:
        return {
            (normalize_name(rel.get("subject", "")), normalize_name(rel.get("object", "")))
            for rel in r.relations
            if rel.get("predicate") == "PRECEDES"
        }

    p1, p2 = precedes_pairs(s1), precedes_pairs(s2)
    if not p1 or not p2:
        return False, f"missing PRECEDES (s1={sorted(p1)}, s2={sorted(p2)})"
    nodes1 = {n for pair in p1 for n in pair}
    nodes2 = {n for pair in p2 for n in pair}
    if nodes1 == nodes2:
        return True, f"both edges over same nodes {sorted(nodes1)}"
    return (
        False,
        f"endpoint names differ: s1={sorted(nodes1)} vs s2={sorted(nodes2)} "
        "(edges resolve to different nodes -> cycle will not close)",
    )


def _check_task_t1_conflict(sess: dict[str, ExtractionResult]) -> tuple[bool, str]:
    s1 = sess.get("anchor-task-t1-conflict:s1")
    s2 = sess.get("anchor-task-t1-conflict:s2")
    if s1 is None or s2 is None:
        return False, "s1 or s2 not in cache"

    def task_status(r: ExtractionResult, name: str) -> Optional[str]:
        for ent in r.entities:
            if ent.get("type") == "Task" and (ent.get("name") or "").strip().lower() == name:
                return (ent.get("fields") or {}).get("status")
        return None

    st1 = (task_status(s1, "t1") or "").lower()
    st2 = (task_status(s2, "t1") or "").lower()
    ok = st1 == "done" and st2 in {"todo", "not_started"}
    return ok, f"T1 s1.status={st1!r}, s2.status={st2!r}"


def _check_owner_conflict(sess: dict[str, ExtractionResult]) -> tuple[bool, str]:
    s1 = sess.get("anchor-project-owner-conflict:s1")
    s2 = sess.get("anchor-project-owner-conflict:s2")
    if s1 is None or s2 is None:
        return False, "s1 or s2 not in cache"

    def owns(r: ExtractionResult) -> Optional[tuple[str, str]]:
        for rel in r.relations:
            if rel.get("predicate") == "OWNS":
                return (
                    normalize_name(rel.get("subject", "")),
                    normalize_name(rel.get("object", "")),
                )
        return None

    o1, o2 = owns(s1), owns(s2)
    if not o1 or not o2:
        return False, f"missing OWNS (s1={o1}, s2={o2})"
    same_project = o1[1] == o2[1]
    diff_owner = o1[0] != o2[0]
    ok = same_project and diff_owner
    return ok, f"s1={o1}, s2={o2} (same project={same_project}, diff owner={diff_owner})"


def _check_final_decision_no_evidence(sess: dict[str, ExtractionResult]) -> tuple[bool, str]:
    s1 = sess.get("anchor-final-decision-no-evidence:s1")
    if s1 is None:
        return False, "s1 not in cache"
    final = any((d.get("status") or "").lower() == "final" for d in s1.decisions)
    has_evidence = any(rel.get("predicate") == "EVIDENCE_FOR" for rel in s1.relations)
    ok = final and not has_evidence
    return ok, f"final decision={final}, EVIDENCE_FOR present={has_evidence}"


def _check_joseph_pharaoh(sess: dict[str, ExtractionResult]) -> tuple[bool, str]:
    s1 = sess.get("anchor-joseph-pharaoh:s1")
    s3 = sess.get("anchor-joseph-pharaoh:s3")
    if s1 is None or s3 is None:
        return False, "s1 or s3 not in cache"
    owns_ok = any(
        rel.get("predicate") == "OWNS"
        and normalize_name(rel.get("subject", "")) == "joseph"
        and "pharaoh" in normalize_name(rel.get("object", ""))
        for rel in s1.relations
    )
    dream_done = any(
        ent.get("type") == "Task"
        and "dream" in (ent.get("name") or "").lower()
        and ((ent.get("fields") or {}).get("status") or "").lower() == "done"
        for ent in s3.entities
    )
    ok = owns_ok and dream_done
    return ok, f"OWNS Joseph->Pharaoh={owns_ok}, Dream done in s3={dream_done}"


_CHECKS = {
    "anchor-task-t1-conflict": _check_task_t1_conflict,
    "anchor-joseph-pharaoh": _check_joseph_pharaoh,
    "anchor-project-owner-conflict": _check_owner_conflict,
    "anchor-inactive-assignee": _check_inactive_assignee,
    "anchor-final-decision-no-evidence": _check_final_decision_no_evidence,
    "anchor-temporal-cycle": _check_temporal_cycle,
}


def scan_anchor_cache(cache_path: str, *, verbose: bool = True) -> dict[str, Any]:
    """Scan ``cache_path`` and report each anchor's precondition status.

    Returns a structured report ``{anchor_id: {target, cached, precondition_ok,
    detail}}`` and (when ``verbose``) prints a readable table. Read-only.
    """
    cache = _load_cache(cache_path)
    sessions = _anchor_sessions()

    report: dict[str, Any] = {}
    for anchor_id, refs in sessions.items():
        resolved = {
            source_ref: _lookup(cache, source_ref, text) for source_ref, text in refs
        }
        cached = all(v is not None for v in resolved.values())
        check = _CHECKS.get(anchor_id)
        if check is None:
            ok, detail = False, "no check defined"
        elif not cached:
            missing = [sr for sr, v in resolved.items() if v is None]
            ok, detail = False, f"not in cache: {missing}"
        else:
            ok, detail = check(resolved)  # type: ignore[arg-type]
        report[anchor_id] = {
            "target": ANCHOR_TARGET.get(anchor_id, "?"),
            "cached": cached,
            "precondition_ok": ok,
            "detail": detail,
        }

    if verbose:
        _print_report(cache_path, cache, report)
    return report


def _print_report(
    cache_path: str, cache: dict[str, ExtractionResult], report: dict[str, Any]
) -> None:
    print(f"Anchor extraction validation — cache: {cache_path}")
    print(f"  cache entries: {len(cache)}")
    print("=" * 78)
    for anchor_id, row in report.items():
        if not row["cached"]:
            verdict = "SKIP (not cached)"
        elif row["precondition_ok"]:
            verdict = "OK  — constraint can fire live"
        else:
            verdict = "GAP — precondition absent; constraint will NOT fire"
        print(f"[{verdict}] {anchor_id}")
        print(f"    target: {row['target']}")
        print(f"    detail: {row['detail']}")
    print("=" * 78)
    gaps = [a for a, r in report.items() if r["cached"] and not r["precondition_ok"]]
    if gaps:
        print(f"GAPS ({len(gaps)}): {', '.join(gaps)}")
        print("These anchors will not demonstrate their constraint live with the "
              "current extraction. Fix the extractor prompt (then clear caches + "
              "re-run) or scope the per-constraint claim in the paper.")
    else:
        print("No gaps: every cached anchor carries its constraint's precondition.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        required=True,
        help="Path to the persisted CachingExtractor JSON (extraction cache).",
    )
    args = parser.parse_args()
    scan_anchor_cache(args.cache)


if __name__ == "__main__":
    main()

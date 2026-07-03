#!/usr/bin/env python3
"""LongMemEval extraction-quality and B3 failure diagnostics.

This is a cache-only diagnostic pass for Cell 7f. It compares cached raw Qwen
fact extractions against the cached oracle knowledge-update annotations, then
replays B3 once to label failed examples. Cache misses are fatal: this script
must not silently start an LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _norm_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _contains_value(haystack: str, needle: str) -> bool:
    n = _norm_text(needle)
    return bool(n) and n in _norm_text(haystack)


def _f1(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "f1": 100.0 * f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


class StrictCachedChat:
    """Prompt-md5 cache reader compatible with run_7f_local.CachedChat."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"Expected dict cache at {path}")
        self.cache = {str(k): str(v) for k, v in raw.items()}
        self.hits = 0
        self.misses: list[str] = []

    def __call__(self, prompt: str) -> str:
        key = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        if key not in self.cache:
            self.misses.append(key)
            raise KeyError(f"missing extraction-cache entry {key}")
        self.hits += 1
        return self.cache[key]


def _slot_attr(slot_name: str, qid: str) -> str:
    prefix = f"{qid}:"
    return slot_name[len(prefix):] if slot_name.startswith(prefix) else slot_name


def _answer_haystack(package: Any) -> str:
    parts: list[str] = []
    answer = getattr(package, "answer", None)
    if answer:
        parts.append(str(answer))
    for item in getattr(package, "retrieved_items", []) or []:
        for attr in ("text", "predicate", "subject_id", "object_id", "memory_id"):
            value = getattr(item, attr, None)
            if value:
                parts.append(str(value))
    for conflict in getattr(package, "conflicts", []) or []:
        value = getattr(conflict, "text", None)
        if value:
            parts.append(str(value))
    return " ".join(parts)


def _entity_label(container: Any, entity_id: str | None) -> str:
    if not entity_id:
        return ""
    try:
        payload = container.graph.get_entity_payload(entity_id) or {}
    except Exception:
        payload = {}
    for key in ("value", "name", "title", "summary", "description", "text"):
        if payload.get(key):
            return str(payload[key])
    return str(entity_id)


def _accepted_values(container: Any, slot_name: str) -> set[str]:
    out: set[str] = set()
    try:
        assertions = list(container.repo.list_assertions("accepted"))
    except Exception:
        return out
    for assertion in assertions:
        if assertion.predicate == "HAS_VALUE" and assertion.subject_id == slot_name:
            out.add(_entity_label(container, assertion.object_id))
    return out


def _nonaccepted_current_status(container: Any, slot_name: str, current: str) -> str | None:
    try:
        assertions = list(container.repo.list_assertions(None))
    except Exception:
        return None
    for assertion in assertions:
        if assertion.predicate != "HAS_VALUE" or assertion.subject_id != slot_name:
            continue
        if _contains_value(_entity_label(container, assertion.object_id), current):
            return str(assertion.status.value if hasattr(assertion.status, "value") else assertion.status)
    return None


def _package_values_for_slot(package: Any, slot_name: str, container: Any) -> list[str]:
    values: list[str] = []
    for item in getattr(package, "retrieved_items", []) or []:
        if getattr(item, "predicate", None) == "HAS_VALUE" and getattr(item, "subject_id", None) == slot_name:
            values.append(_entity_label(container, getattr(item, "object_id", None)))
    return values


def _build_extraction_index(examples: list[Any], oracle: Any) -> dict[str, dict[str, Any]]:
    by_qid: dict[str, dict[str, Any]] = {}
    for ex in examples:
        qid = str(ex.id)
        slots: dict[str, set[str]] = {}
        pairs: set[tuple[str, str]] = set()
        values_any_slot: set[str] = set()
        for session in ex.sessions:
            source_ref = f"{qid}:{session.session_id}"
            result = oracle.extract(session.input, source_ref)
            for rel in result.relations:
                if str(rel.get("predicate")) != "HAS_VALUE":
                    continue
                slot = str(rel.get("subject", ""))
                value = str(rel.get("object", "")).strip()
                if not slot or not value:
                    continue
                slots.setdefault(slot, set()).add(value)
                pairs.add((slot, value))
                values_any_slot.add(value)
        by_qid[qid] = {
            "slots": slots,
            "pairs": pairs,
            "values_any_slot": values_any_slot,
        }
    return by_qid


def extraction_quality(
    instances: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    extraction_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    annotated_qids = [str(inst["question_id"]) for inst in instances if str(inst["question_id"]) in annotations]

    slot_tp = slot_fp = slot_fn = 0
    value_tp = value_fp = value_fn = 0
    value_gold_slot_tp = value_gold_slot_fp = value_gold_slot_fn = 0
    current_hits_any_slot = 0
    current_hits_gold_slot = 0
    link_hits = 0
    link_total = 0
    stale_only = 0
    stale_only_gold_slot = 0
    stale_emitted = 0

    per_example: list[dict[str, Any]] = []
    for qid in annotated_qids:
        ann = annotations[qid]
        gold_slot = f"{qid}:{ann['attribute']}"
        gold_values = {str(x["value"]) for x in ann.get("trajectory", []) if x.get("value") is not None}
        current = str(ann.get("current_value", ""))
        stale_values = {v for v in gold_values if _norm_text(v) != _norm_text(current)}
        pred = extraction_index.get(qid, {"slots": {}, "values_any_slot": set(), "pairs": set()})
        pred_slots = set(pred["slots"])
        pred_values_for_gold_slot = set(pred["slots"].get(gold_slot, set()))
        pred_values_any_slot = set(pred["values_any_slot"])

        slot_hit = gold_slot in pred_slots
        slot_tp += int(slot_hit)
        slot_fn += int(not slot_hit)
        slot_fp += len({s for s in pred_slots if s != gold_slot})

        matched_gold_values_any_slot = {
            gv for gv in gold_values
            if any(_contains_value(pv, gv) or _contains_value(gv, pv) for pv in pred_values_any_slot)
        }
        value_tp += len(matched_gold_values_any_slot)
        value_fn += len(gold_values - matched_gold_values_any_slot)
        value_fp += len([
            pv for pv in pred_values_any_slot
            if not any(_contains_value(pv, gv) or _contains_value(gv, pv) for gv in gold_values)
        ])

        matched_gold_values_on_gold_slot = {
            gv for gv in gold_values
            if any(_contains_value(pv, gv) or _contains_value(gv, pv) for pv in pred_values_for_gold_slot)
        }
        value_gold_slot_tp += len(matched_gold_values_on_gold_slot)
        value_gold_slot_fn += len(gold_values - matched_gold_values_on_gold_slot)
        value_gold_slot_fp += len([
            pv for pv in pred_values_for_gold_slot
            if not any(_contains_value(pv, gv) or _contains_value(gv, pv) for gv in gold_values)
        ])

        current_on_gold_slot = any(
            _contains_value(pv, current) or _contains_value(current, pv)
            for pv in pred_values_for_gold_slot
        )
        current_any_slot = any(
            _contains_value(pv, current) or _contains_value(current, pv)
            for pv in pred_values_any_slot
        )
        stale_any_slot = any(
            _contains_value(pv, sv) or _contains_value(sv, pv)
            for pv in pred_values_any_slot
            for sv in stale_values
        )
        stale_on_gold_slot = any(
            _contains_value(pv, sv) or _contains_value(sv, pv)
            for pv in pred_values_for_gold_slot
            for sv in stale_values
        )
        current_hits_any_slot += int(current_any_slot)
        current_hits_gold_slot += int(current_on_gold_slot)
        if current_any_slot:
            link_total += 1
            link_hits += int(current_on_gold_slot)
        stale_emitted += int(stale_on_gold_slot)
        stale_only += int(stale_any_slot and not current_any_slot)
        stale_only_gold_slot += int(stale_on_gold_slot and not current_on_gold_slot)

        per_example.append({
            "question_id": qid,
            "gold_slot": gold_slot,
            "gold_current_value": current,
            "predicted_slot_count": len(pred_slots),
            "slot_extracted": slot_hit,
            "current_value_extracted_any_slot": current_any_slot,
            "current_value_extracted_on_gold_slot": current_on_gold_slot,
            "stale_value_extracted_on_gold_slot": stale_on_gold_slot,
        })

    n = len(annotated_qids)
    return {
        "n_annotated_examples": n,
        "slot_extraction": _f1(slot_tp, slot_fp, slot_fn),
        "slot_value_extraction": _f1(value_tp, value_fp, value_fn),
        "slot_value_extraction_on_gold_slot": _f1(
            value_gold_slot_tp, value_gold_slot_fp, value_gold_slot_fn
        ),
        "current_value_extraction_recall": {
            "percent": 100.0 * current_hits_any_slot / n if n else 0.0,
            "hits": current_hits_any_slot,
            "n": n,
            "definition": "gold current value emitted under any extracted slot",
        },
        "current_value_linked_recall": {
            "percent": 100.0 * current_hits_gold_slot / n if n else 0.0,
            "hits": current_hits_gold_slot,
            "n": n,
            "definition": "gold current value emitted under the oracle slot",
        },
        "entity_slot_linking_accuracy": {
            "percent": 100.0 * link_hits / link_total if link_total else 0.0,
            "correct": link_hits,
            "n_current_values_extracted_any_slot": link_total,
        },
        "false_stale_value_extraction_rate": {
            "percent": 100.0 * stale_only / n if n else 0.0,
            "stale_only": stale_only,
            "stale_only_on_gold_slot": stale_only_gold_slot,
            "stale_emitted_on_gold_slot": stale_emitted,
            "n": n,
            "definition": "annotated examples where a stale gold value was emitted but the current gold value was not emitted anywhere",
        },
        "per_example": per_example,
    }


def run_b3_failures(
    examples: list[Any],
    oracle: Any,
    instances_by_qid: dict[str, dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    extraction_index: dict[str, dict[str, Any]],
    *,
    embeddings: Any,
) -> dict[str, Any]:
    from ocm.core.config import Settings
    from ocm.core.container import CoreContainer
    from ocm.evaluation.baselines import baseline_settings_overrides, build_baseline
    from ocm.evaluation.runner import BaselineRunner

    def settings_factory() -> Settings:
        return Settings(
            deterministic_test_mode=True,
            chroma_mode="memory",
            extractor="mock",
            authoritative_update_supersede=True,
        )

    settings = settings_factory().model_copy(update=baseline_settings_overrides("B3"))
    container = CoreContainer(settings, extractor=oracle, embeddings=embeddings)
    strategy = build_baseline("B3", container)
    runner = BaselineRunner(settings_factory=settings_factory, top_k=10)

    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    n = 0
    failed = 0
    write_outcomes = {"candidates": 0, "accepted": 0, "superseded": 0, "quarantined": 0, "rejected": 0}

    for ex in examples:
        qid = str(ex.id)
        wc = runner._ingest_sessions(strategy, ex)
        for key in write_outcomes:
            write_outcomes[key] += int(wc.get(key, 0))

        question = ex.questions[0]
        package = strategy.query(question.query, top_k=10)
        haystack = _answer_haystack(package)
        expected = list(question.expected_answer_contains or [])
        answer_correct = all(_contains_value(haystack, token) for token in expected)
        n += 1
        if answer_correct:
            continue

        failed += 1
        inst = instances_by_qid.get(qid, {})
        current = str(expected[0] if expected else inst.get("answer", ""))
        ann = annotations.get(qid)
        gold_slot = f"{qid}:{ann['attribute']}" if ann else None
        stale_values = [
            str(x["value"])
            for x in (ann or {}).get("trajectory", [])
            if _norm_text(x.get("value")) != _norm_text(current)
        ]
        pred = extraction_index.get(qid, {"slots": {}, "values_any_slot": set()})
        current_any_slot = any(
            _contains_value(v, current) or _contains_value(current, v)
            for v in pred["values_any_slot"]
        )
        current_on_gold_slot = bool(gold_slot) and any(
            _contains_value(v, current) or _contains_value(current, v)
            for v in pred["slots"].get(gold_slot, set())
        )

        accepted_current = False
        nonaccepted_status = None
        package_slot_values: list[str] = []
        if gold_slot:
            accepted_current = any(
                _contains_value(v, current) or _contains_value(current, v)
                for v in _accepted_values(container, gold_slot)
            )
            nonaccepted_status = _nonaccepted_current_status(container, gold_slot, current)
            package_slot_values = _package_values_for_slot(package, gold_slot, container)

        stale_ranked = any(
            _contains_value(haystack, stale) for stale in stale_values
        )
        current_ranked = _contains_value(haystack, current)

        if not current_any_slot:
            cause = "missing_current_value"
        elif gold_slot and not current_on_gold_slot:
            cause = "wrong_slot_link"
        elif gold_slot and current_on_gold_slot and nonaccepted_status and nonaccepted_status != "accepted":
            cause = "governance_suppressed_valid"
        elif stale_ranked and not current_ranked:
            cause = "stale_value_ranked"
        elif accepted_current or current_ranked:
            cause = "answer_policy_failure"
        elif not ann and current_any_slot:
            cause = "wrong_slot_link"
        else:
            cause = "answer_policy_failure"

        counts[cause] += 1
        rows.append({
            "question_id": qid,
            "question": str(inst.get("question", question.query)),
            "gold_current_value": current,
            "gold_slot": gold_slot,
            "cause": cause,
            "annotated": bool(ann),
            "current_value_extracted_any_slot": current_any_slot,
            "current_value_extracted_on_gold_slot": current_on_gold_slot if gold_slot else None,
            "current_value_accepted_on_gold_slot": accepted_current if gold_slot else None,
            "current_value_nonaccepted_status": nonaccepted_status,
            "stale_value_ranked": stale_ranked,
            "current_value_ranked": current_ranked,
            "package_values_for_gold_slot": package_slot_values,
        })

    return {
        "n_examples": n,
        "n_failed": failed,
        "b3_task_success": 100.0 * (n - failed) / n if n else 0.0,
        "write_outcomes": write_outcomes,
        "counts": {
            key: {
                "count": int(value),
                "percent_of_failures": 100.0 * value / failed if failed else 0.0,
            }
            for key, value in sorted(counts.items())
        },
        "per_failed_example": rows,
    }


def main() -> int:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=repo_dir)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--intent-mode", choices=("auto", "new_fact"), default="auto")
    parser.add_argument(
        "--extract-prompt",
        choices=("durable", "longmemeval", "generic"),
        default="longmemeval",
        help=(
            "Fact-extraction prompt whose cache to read. MUST match the prompt "
            "used by the 7f run that populated the cache (run_7f_local.py "
            "defaults to 'longmemeval'; 'generic' is the non-benchmark-primed "
            "ablation)."
        ),
    )
    parser.add_argument("--e2e-limit", type=int, default=None)
    args = parser.parse_args()

    repo_dir = args.repo_dir.resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from ocm.evaluation.datasets.longmemeval_adapter import (
        FACT_EXTRACTION_PROMPTS,
        build_e2e_from_extraction,
        load_longmemeval,
        parse_facts_json,
    )
    from ocm.retrieval.embeddings import DeterministicEmbeddingProvider

    extract_prompt = FACT_EXTRACTION_PROMPTS[args.extract_prompt]

    data_dir = (args.data_dir or repo_dir / "data").resolve()
    output_dir = (args.output_dir or repo_dir / "local_results").resolve()
    out_path = (args.out or output_dir / "results_longmemeval_diagnostics.json").resolve()
    ann_path = output_dir / "longmemeval_kupdate_annotations.json"
    # Cache filename must match run_7f_local.py's prompt-specific naming so the
    # diagnostic reads the SAME cached extractions the 7f run produced.
    cache_name = (
        "lme_e2e_extract_cache.json"
        if args.extract_prompt == "durable"
        else f"lme_e2e_extract_cache_{args.extract_prompt}.json"
    )
    cache_path = output_dir / cache_name
    lme_path = data_dir / "longmemeval_s.json"

    annotations = json.loads(ann_path.read_text(encoding="utf-8"))
    instances = load_longmemeval(
        str(lme_path),
        question_type="knowledge-update",
        abstention=False,
        limit=args.e2e_limit,
    )
    instances_by_qid = {str(inst["question_id"]): inst for inst in instances}

    cached_chat = StrictCachedChat(cache_path)

    def fact_extract_fn(text: str) -> list[dict[str, Any]]:
        return parse_facts_json(cached_chat(extract_prompt.format(text=text)))

    examples, oracle = build_e2e_from_extraction(
        instances,
        fact_extract_fn,
        intent_mode=args.intent_mode,
    )
    extraction_index = _build_extraction_index(examples, oracle)

    quality = extraction_quality(instances, annotations, extraction_index)
    failures = run_b3_failures(
        examples,
        oracle,
        instances_by_qid,
        annotations,
        extraction_index,
        embeddings=DeterministicEmbeddingProvider(),
    )

    report = {
        "dataset": "longmemeval",
        "subset": "knowledge-update",
        "arm": "end_to_end",
        "intent_mode": args.intent_mode,
        "extract_prompt": args.extract_prompt,
        "n_examples": len(examples),
        "n_oracle_annotations": len(annotations),
        "extraction_cache": {
            "path": str(cache_path),
            "entries": len(cached_chat.cache),
            "hits": cached_chat.hits,
            "misses": len(cached_chat.misses),
        },
        "extraction_quality": quality,
        "b3_failure_breakdown": failures,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved -> {out_path}")
    print(f"examples: {len(examples)}; oracle annotations: {len(annotations)}")
    print("Extraction quality:")
    for key in ("slot_extraction", "slot_value_extraction", "slot_value_extraction_on_gold_slot"):
        row = quality[key]
        print(f"  {key}: P={row['precision']:.2f} R={row['recall']:.2f} F1={row['f1']:.2f}")
    cvr = quality["current_value_extraction_recall"]
    link = quality["entity_slot_linking_accuracy"]
    stale = quality["false_stale_value_extraction_rate"]
    print(f"  current_value_extraction_recall: {cvr['percent']:.2f} ({cvr['hits']}/{cvr['n']})")
    print(
        "  entity_slot_linking_accuracy: "
        f"{link['percent']:.2f} ({link['correct']}/{link['n_current_values_extracted_any_slot']})"
    )
    print(
        "  false_stale_value_extraction_rate: "
        f"{stale['percent']:.2f} ({stale['stale_only']}/{stale['n']})"
    )
    print("B3 failure breakdown:")
    for cause, row in failures["counts"].items():
        print(f"  {cause}: {row['count']} ({row['percent_of_failures']:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

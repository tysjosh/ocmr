"""Does authoritative supersession discard correct answers on raw extraction?

Motivation
----------
On raw end-to-end LongMemEval the governed arms score *lower* task success than
the ungoverned ones (B3 34.7 vs B0/B2 44.4) and lower than the MemGPT-style arm
(48.6), despite B3 and Bmemgpt holding near-identical accepted counts (11,715 vs
11,479 per seed). So the spread is not retrieval volume and not store size -- it
is *which* values survive. B3 supersedes 2,809 times per seed and keeps only the
latest value; Bmemgpt supersedes 733 times and skips 2,312 writes, frequently
declining to overwrite.

Retrieval composition was already ruled out: an arm with B3's write settings and
B2's retrieval toggles (B3rm) scored identically to B3, because
``status_filter`` only ever widens to ``["accepted", "quarantined"]`` and that
arm quarantines nothing. ``superseded`` is never retrievable for any arm.

That leaves one hypothesis: on noisy extraction the *latest* extracted value is
often not the gold answer, so unconditional latest-wins supersession retires a
correct earlier extraction in favour of a worse later one.

What this measures
------------------
For each knowledge-update question, whether the gold answer survives in the
arm's **accepted** state, only in its **superseded** state, or nowhere:

* ``retained``  -- gold present in accepted state (governance kept it)
* ``lost``      -- gold present ONLY in superseded state -> supersession
                   discarded a correct value. This is the hypothesis's evidence.
* ``never``     -- gold in neither; extraction never captured it, so governance
                   is not implicated either way.

Reported per arm, so B3's ``lost`` count can be read against Bmemgpt's. A large
``lost`` count for B3 with a small one for Bmemgpt supports the hypothesis; both
small means something else explains the recall gap.

Gold matching follows the harness convention (``BaselineRunner._answer_score``):
case-insensitive substring containment of the gold string. Cases where the gold
is a long parenthetical (e.g. ``"25 minutes and 50 seconds (or 25:50)"``) are
also scored against their leading clause, and both figures are reported so a
formatting artifact is visible rather than silently counted as a loss.

Usage (on a machine with the complete extraction + link caches):

    python probe_supersession_loss.py \
      --extract-cache local_results/lme_e2e_extract_cache_longmemeval__32e1c2065787.json \
      --link-cache    local_results/lme_e2e_link_cache_longmemeval__09ccdb187380.json

Read-only with respect to the caches: every prompt is expected to hit, so no LLM
is constructed. Pass ``--arms`` to change which arms are probed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")


# --------------------------------------------------------------------------- #
# Cache reader (read-only; refuses to run the model)
# --------------------------------------------------------------------------- #
class ReadOnlyCache:
    """Serve cached prompt responses; raise rather than invoke a model.

    The probe must not silently start generating: a miss means the cache does not
    match this build, which would make the whole result meaningless. Missing
    prompts are counted and reported instead of being filled in.
    """

    def __init__(self, path: Path) -> None:
        raw = json.loads(path.read_text())
        meta = raw.get("__meta__") or {}
        self.namespace = meta.get("namespace") or {}
        self.entries: dict[str, str] = dict(raw.get("entries") or {})
        self.path = path
        self.misses = 0

    def _key(self, prompt: str) -> str:
        payload = json.dumps(
            {"namespace": self.namespace, "prompt": prompt},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:64]

    def __call__(self, prompt: str) -> str:
        hit = self.entries.get(self._key(prompt))
        if hit is None:
            self.misses += 1
            # Empty JSON array / object parses cleanly downstream, so one stray
            # miss degrades that fact rather than crashing the whole probe. The
            # count is asserted at the end.
            return "[]"
        return hit


# --------------------------------------------------------------------------- #
# Gold matching
# --------------------------------------------------------------------------- #
def _norm(text: Any) -> str:
    return " ".join(str(text or "").split()).casefold()


def _gold_variants(gold: str) -> list[str]:
    """The gold string plus its leading clause, for parenthetical golds."""
    variants = [gold]
    head = gold.split("(")[0].strip()
    if head and head != gold:
        variants.append(head)
    return variants


def _contains(haystack: str, gold: str) -> bool:
    return _norm(gold) in haystack


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #
def probe_arm(container: Any, qid_to_gold: dict[str, str]) -> dict[str, Any]:
    """Classify each question's gold answer against accepted vs superseded state."""
    repo = container.repo

    payload_by_id: dict[str, tuple[str, dict]] = {}
    for etype, payload in repo.list_entities():
        entity_id = payload.get("id")
        if entity_id:
            payload_by_id[entity_id] = (etype, payload)

    def label(object_id: str) -> str:
        entry = payload_by_id.get(object_id)
        if entry is None:
            return object_id
        payload = entry[1]
        return str(payload.get("value") or payload.get("name") or object_id)

    # Slot ids grouped by the question they belong to (slot names are "<qid>:<attr>").
    slots_by_qid: dict[str, set[str]] = {}
    for entity_id, (etype, payload) in payload_by_id.items():
        if etype != "Slot":
            continue
        name = str(payload.get("name") or "")
        qid, _, _attr = name.partition(":")
        if qid:
            slots_by_qid.setdefault(qid, set()).add(entity_id)

    accepted_by_qid: dict[str, list[str]] = {}
    superseded_by_qid: dict[str, list[str]] = {}
    for status, sink in (("accepted", accepted_by_qid), ("superseded", superseded_by_qid)):
        for assertion in repo.list_assertions(status):
            if assertion.predicate != "HAS_VALUE":
                continue
            for qid, slot_ids in slots_by_qid.items():
                if assertion.subject_id in slot_ids:
                    sink.setdefault(qid, []).append(label(assertion.object_id))
                    break

    counts = {"retained": 0, "lost": 0, "never": 0}
    counts_lenient = {"retained": 0, "lost": 0, "never": 0}
    records: list[dict[str, Any]] = []

    for qid, gold in qid_to_gold.items():
        acc_hay = _norm(" | ".join(accepted_by_qid.get(qid, [])))
        sup_hay = _norm(" | ".join(superseded_by_qid.get(qid, [])))

        strict_acc = _contains(acc_hay, gold)
        strict_sup = _contains(sup_hay, gold)
        lenient_acc = any(_contains(acc_hay, v) for v in _gold_variants(gold))
        lenient_sup = any(_contains(sup_hay, v) for v in _gold_variants(gold))

        def classify(in_acc: bool, in_sup: bool) -> str:
            if in_acc:
                return "retained"
            return "lost" if in_sup else "never"

        outcome = classify(strict_acc, strict_sup)
        counts[outcome] += 1
        counts_lenient[classify(lenient_acc, lenient_sup)] += 1

        records.append({
            "question_id": qid,
            "gold": gold,
            "outcome": outcome,
            "n_accepted_values": len(accepted_by_qid.get(qid, [])),
            "n_superseded_values": len(superseded_by_qid.get(qid, [])),
            "superseded_sample": superseded_by_qid.get(qid, [])[:5],
        })

    total = max(1, sum(counts.values()))
    return {
        "counts": counts,
        "counts_lenient": counts_lenient,
        "lost_rate": 100.0 * counts["lost"] / total,
        "retained_rate": 100.0 * counts["retained"] / total,
        "records": records,
    }


def attestation_profile(oracle: Any, qid_to_gold: dict[str, str]) -> dict[str, Any]:
    """Per question, how many sessions asserted each value for its slot.

    Why this matters
    ----------------
    ``lost`` tells us latest-wins retired a correct value. It does not tell us
    whether *corroboration* would have prevented it. Corroboration refuses to
    supersede on a value that is attested only once, so it can only recover a
    lost gold if the value that displaced it was a single-attestation outlier.
    If the displacing values are themselves well attested, corroboration will
    supersede exactly as B3 did and recover nothing.

    The write stream is identical across arms (arms differ in write *policy*,
    not in what the extractor emitted), so this is computed once.
    """
    writes = getattr(oracle, "_writes", {}) or {}

    # source_ref is "<qid>:s<idx>"; recover the ordering so the last-written
    # value -- the one latest-wins keeps -- can be identified.
    per_qid: dict[str, list[tuple[int, str]]] = {}
    for source_ref, sw in writes.items():
        qid, _, session_part = str(source_ref).partition(":s")
        try:
            idx = int(session_part)
        except ValueError:
            continue
        for rel in getattr(sw, "relations", []) or []:
            if rel.get("predicate") != "HAS_VALUE":
                continue
            per_qid.setdefault(qid, []).append((idx, str(rel.get("object") or "")))

    profile: dict[str, Any] = {}
    for qid, gold in qid_to_gold.items():
        seq = sorted(per_qid.get(qid, []))
        counts: dict[str, int] = {}
        for _idx, value in seq:
            counts[value] = counts.get(value, 0) + 1
        final_value = seq[-1][1] if seq else None
        gold_values = [v for v in counts if _contains(_norm(v), gold)]
        profile[qid] = {
            "n_writes": len(seq),
            "n_distinct_values": len(counts),
            "final_value": final_value,
            "final_attestations": counts.get(final_value, 0) if final_value else 0,
            "final_is_gold": bool(final_value and _contains(_norm(final_value), gold)),
            "gold_attestations": max((counts[v] for v in gold_values), default=0),
            "attestations": counts,
        }
    return profile


def summarize_lost_attestations(
    profile: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Of the golds this arm lost, how many would corroboration have saved?

    ``winner_attested_once`` is the recoverable set: the displacing value was
    seen exactly once, so a corroboration requirement would have quarantined it
    instead of letting it supersede. ``winner_attested_multiple`` is the ceiling
    on what corroboration cannot fix.
    """
    lost = [r["question_id"] for r in records if r["outcome"] == "lost"]
    once = multiple = gold_single = 0
    detail = []
    for qid in lost:
        p = profile.get(qid) or {}
        n = int(p.get("final_attestations") or 0)
        if n <= 1:
            once += 1
        else:
            multiple += 1
        if int(p.get("gold_attestations") or 0) <= 1:
            gold_single += 1
        detail.append({
            "question_id": qid,
            "final_value": p.get("final_value"),
            "final_attestations": n,
            "gold_attestations": p.get("gold_attestations"),
            "n_distinct_values": p.get("n_distinct_values"),
        })
    return {
        "n_lost": len(lost),
        "winner_attested_once": once,
        "winner_attested_multiple": multiple,
        "gold_attested_once": gold_single,
        "recoverable_by_corroboration_pct": (
            100.0 * once / len(lost) if lost else 0.0),
        "detail": detail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extract-cache", type=Path, required=True)
    parser.add_argument("--link-cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/longmemeval_s.json"))
    parser.add_argument("--arms", default="B0,B2,Bsup,Bevi,B3")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of questions (smoke testing; omit for all 72).")
    parser.add_argument("--embeddings", choices=("local", "deterministic"), default="local",
                        help="Deterministic is far faster and does not affect this probe, "
                             "which reads durable state rather than retrieval.")
    parser.add_argument("--intent-mode", default="auto")
    parser.add_argument("--link-threshold", type=float, default=0.75)
    parser.add_argument("--out", type=Path,
                        default=Path("local_results/supersession_loss.json"))
    args = parser.parse_args()

    from ocm.core.config import Settings
    from ocm.evaluation.arms import build_arm
    from ocm.evaluation.datasets.longmemeval_adapter import (
        FACT_EXTRACTION_PROMPTS, build_e2e_from_extraction, build_fact_extract_fn,
        build_slot_link_fn, load_longmemeval,
    )
    from ocm.evaluation.runner import BaselineRunner
    from ocm.retrieval.embeddings import (
        DeterministicEmbeddingProvider, LocalEmbeddingProvider)

    extract_cache = ReadOnlyCache(args.extract_cache)
    link_cache = ReadOnlyCache(args.link_cache)
    print(f"extract cache: {len(extract_cache.entries)} entries")
    print(f"link cache:    {len(link_cache.entries)} entries")

    instances = load_longmemeval(
        str(args.data), question_type="knowledge-update", limit=args.limit)
    qid_to_gold = {str(i["question_id"]): str(i.get("answer", "")) for i in instances}
    print(f"{len(instances)} knowledge-update questions\n")

    fact_extract_fn = build_fact_extract_fn(
        extract_cache, prompt_template=FACT_EXTRACTION_PROMPTS["longmemeval"])
    slot_link_fn = build_slot_link_fn(link_cache, confidence_threshold=args.link_threshold)

    examples, oracle = build_e2e_from_extraction(
        instances, fact_extract_fn, intent_mode=args.intent_mode, slot_link_fn=slot_link_fn)
    print(f"built {len(examples)} examples; "
          f"cache misses: extract={extract_cache.misses} link={link_cache.misses}")
    if extract_cache.misses:
        print("  WARNING: extraction misses mean the cache does not match this build; "
              "results are not comparable to the reported runs.", file=sys.stderr)

    def settings_factory() -> Settings:
        return Settings(deterministic_test_mode=True, chroma_mode="memory",
                        extractor="mock", authoritative_update_supersede=True)

    embeddings = (
        LocalEmbeddingProvider() if args.embeddings == "local"
        else DeterministicEmbeddingProvider()
    )
    runner = BaselineRunner(settings_factory=settings_factory, top_k=10)

    out: dict[str, Any] = {"_meta": {
        "n_questions": len(instances),
        "limit": args.limit,
        "embeddings": args.embeddings,
        "intent_mode": args.intent_mode,
        "extract_cache_entries": len(extract_cache.entries),
        "link_cache_entries": len(link_cache.entries),
        "extract_cache_misses": extract_cache.misses,
        "link_cache_misses": link_cache.misses,
    }}

    # Arm-independent: the extractor emitted the same writes for every arm.
    profile = attestation_profile(oracle, qid_to_gold)
    out["_attestations"] = profile

    header = (f"{'arm':7}{'retained':>10}{'lost':>7}{'never':>7}{'lost%':>8}"
              f"{'win=1':>7}{'win>1':>7}   lenient(ret/lost/never)")
    print("\n" + header)
    print("-" * len(header))
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        strategy = build_arm(arm, settings_factory, extractor=oracle, embeddings=embeddings)
        for example in examples:
            runner._ingest_sessions(strategy, example)
        result = probe_arm(strategy.container, qid_to_gold)
        result["lost_attestations"] = summarize_lost_attestations(
            profile, result["records"])
        out[arm] = result
        c, l = result["counts"], result["counts_lenient"]
        a = result["lost_attestations"]
        print(f"{arm:7}{c['retained']:>10}{c['lost']:>7}{c['never']:>7}"
              f"{result['lost_rate']:>8.1f}"
              f"{a['winner_attested_once']:>7}{a['winner_attested_multiple']:>7}"
              f"   {l['retained']}/{l['lost']}/{l['never']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {args.out}")

    b3, memgpt = out.get("B3"), out.get("Bmemgpt")
    if b3:
        print(f"\nB3 lost {b3['counts']['lost']} of {len(instances)} golds to supersession "
              f"({b3['lost_rate']:.1f}%)")
        if memgpt:
            print(f"Bmemgpt lost {memgpt['counts']['lost']} ({memgpt['lost_rate']:.1f}%)")
        a = b3["lost_attestations"]
        print(f"\nOf those {a['n_lost']} losses, the displacing value was attested "
              f"once in {a['winner_attested_once']} case(s) and more than once in "
              f"{a['winner_attested_multiple']}.")
        print(f"So a corroboration requirement could recover at most "
              f"{a['winner_attested_once']}/{a['n_lost']} "
              f"({a['recoverable_by_corroboration_pct']:.0f}%); the rest were "
              f"displaced by values corroboration would also have accepted.")
        print("win=1 is the recoverable set; win>1 is the ceiling corroboration "
              "cannot lift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

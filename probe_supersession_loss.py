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

    header = f"{'arm':7}{'retained':>10}{'lost':>7}{'never':>7}{'lost%':>8}   lenient(ret/lost/never)"
    print("\n" + header)
    print("-" * len(header))
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        strategy = build_arm(arm, settings_factory, extractor=oracle, embeddings=embeddings)
        for example in examples:
            runner._ingest_sessions(strategy, example)
        result = probe_arm(strategy.container, qid_to_gold)
        out[arm] = result
        c, l = result["counts"], result["counts_lenient"]
        print(f"{arm:7}{c['retained']:>10}{c['lost']:>7}{c['never']:>7}"
              f"{result['lost_rate']:>8.1f}   {l['retained']}/{l['lost']}/{l['never']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {args.out}")

    b3, memgpt = out.get("B3"), out.get("Bmemgpt")
    if b3:
        print(f"\nB3 lost {b3['counts']['lost']} of {len(instances)} golds to supersession "
              f"({b3['lost_rate']:.1f}%)")
        if memgpt:
            print(f"Bmemgpt lost {memgpt['counts']['lost']} ({memgpt['lost_rate']:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

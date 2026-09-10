"""MultiWOZ: durable-state scoring per arm (companion to run_lmo_durable_state.py).

Why
---
``task_success`` is answer-token recall over the rendered answer *plus every
retrieved item*, so it is monotone in retrieved volume. On MultiWOZ it saturates
outright: ``EvidencePackager._slot_value_answer`` deliberately returns the *most
recent* accepted ``HAS_VALUE`` when several are active -- "which only happens on an
ungoverned arm that failed to supersede" -- so an arm carrying single-valued
violations is still handed the current gold value. Every arm scores ~99.9 and the
headline number separates nothing.

Measured on the full validation split (1,000 dialogues, 8,104 probed slots):

    arm      task     durable-state
    B0/B2    99.88    93.09   (557 slots left with two or more accepted values)
    Bsup     99.96    99.95
    Bevi     99.96    99.95
    B3       99.96    99.95

0.08 points of separation on recall versus 6.86 on durable state, from the same
run. The 579 violations behind that also reproduce the published 7.14 rate
(579 / 8,104 x 100).

Scored with :mod:`ocm.evaluation.durable_state`, whose buckets keep the comparison
fair to a governed arm:

    correct   -- holds exactly the gold value
    stale     -- one accepted value, wrong, nothing flagged (silently wrong)
    split     -- two or more accepted values (the cardinality breach)
    abstained -- wrong or absent, but the key WAS flagged (declined, and said so)
    missing   -- never extracted (extraction floor, diagnostic)

The stale/abstained split matters for any policy that quarantines rather than
overwrites: the incumbent then stays accepted and the store holds a non-gold value
by design, which must not be scored as silently stale.

Uses the oracle extractor keyed off MultiWOZ's gold per-turn belief state, so
there are no LLM calls and no prompt caches involved -- only the dialogues (fetched
from GitHub) and the embedding model.

Usage:

    python run_multiwoz_durable_state.py                 # full validation split
    python run_multiwoz_durable_state.py --limit 20      # smoke test
    python run_multiwoz_durable_state.py --fixture       # offline, no network
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the dialogue count; omit for the full split "
                             "(validation is 1,000 dialogues).")
    parser.add_argument("--arms", default="B0,B2,Bsup,Bevi,B3")
    parser.add_argument("--embeddings", choices=("local", "deterministic"),
                        default="local",
                        help="Durable-state buckets are retrieval-independent, but "
                             "task_success is not -- keep 'local' to report both.")
    parser.add_argument("--fixture", action="store_true",
                        help="Use the built-in 3-dialogue fixture (no network).")
    parser.add_argument("--out", type=Path,
                        default=Path("local_results/multiwoz_durable_state.json"))
    args = parser.parse_args()

    from ocm.core.config import Settings
    from ocm.evaluation.arms import build_arm
    from ocm.evaluation.datasets.multiwoz_adapter import (
        build_from_dialogues, load_multiwoz, sample_dialogues,
    )
    from ocm.evaluation.durable_state import (
        durable_state_outcomes, gold_from_marked_questions, resolve_gold_keys,
    )
    from ocm.evaluation.experiment import durable_constraint_violations
    from ocm.evaluation.runner import BaselineRunner
    from ocm.retrieval.embeddings import (
        DeterministicEmbeddingProvider, LocalEmbeddingProvider)

    if args.fixture:
        dialogues = sample_dialogues()
        print("using the offline fixture")
    else:
        dialogues = load_multiwoz(split=args.split, limit=args.limit)

    examples, oracle = build_from_dialogues(dialogues)
    gold_by_name = gold_from_marked_questions(examples)
    print(f"dialogues={len(dialogues)} examples={len(examples)} "
          f"gold_keys={len(gold_by_name)}")
    if not gold_by_name:
        print("no slot-addressed questions -- nothing to score")
        return 1

    def settings_factory() -> Settings:
        return Settings(
            deterministic_test_mode=True, chroma_mode="memory", extractor="mock",
            authoritative_update_supersede=True,
        )

    embeddings = (
        LocalEmbeddingProvider() if args.embeddings == "local"
        else DeterministicEmbeddingProvider()
    )
    runner = BaselineRunner(settings_factory=settings_factory, top_k=10)

    header = (f"{'arm':7}{'DSC':>8}{'SSR':>8}{'split':>8}{'abst':>8}{'miss':>8}"
              f"{'task':>8}{'viol':>7}   correct/stale/split/abst/missing")
    print("\n" + header)
    print("-" * len(header))

    out: dict[str, Any] = {"_meta": {
        "split": args.split,
        "limit": args.limit,
        "fixture": bool(args.fixture),
        "n_dialogues": len(dialogues),
        "n_gold_keys": len(gold_by_name),
        "embeddings": args.embeddings,
    }}

    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        strategy = build_arm(arm, settings_factory, extractor=oracle, embeddings=embeddings)
        records = []
        for example in examples:
            counts = runner._ingest_sessions(strategy, example)
            for q_index, question in enumerate(example.questions):
                records.append(runner._run_question(
                    arm, strategy, example, q_index, question,
                    write_quarantined=counts["quarantined"]))

        # Entity ids are minted per run, so the name-keyed gold map has to be
        # rekeyed against the store after the replay.
        gold = resolve_gold_keys(strategy.container, gold_by_name)
        report = durable_state_outcomes(strategy.container, gold)
        violations, _accepted = durable_constraint_violations(strategy.container)
        task = 100.0 * sum(r["score"] for r in records) / max(1, len(records))

        out[arm] = {
            **report.as_dict(),
            "durable_violations": violations,
            "violation_rate_per_100": 100.0 * violations / max(1, len(records)),
            "task_success": task,
        }
        print(f"{arm:7}{report.dsc:8.2f}{report.ssr:8.2f}{report.split_rate:8.2f}"
              f"{report.abstention_rate:8.2f}{report.missing_rate:8.2f}"
              f"{task:8.2f}{violations:7}   "
              f"{report.correct}/{report.stale}/{report.split}/"
              f"{report.abstained}/{report.missing}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved -> {args.out}")
    print("\nDSC=durable state correctness (up)  SSR=silent stale rate (down)")
    print("split=two+ accepted values (down)   abst=declined and flagged")
    print("miss=never extracted (extraction floor, diagnostic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

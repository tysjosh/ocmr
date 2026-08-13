"""Read ``ocmr_arm_results.json`` and print the parts that decide the claim.

The raw file is a few thousand lines: five seeds by four reviewers by five arms,
each with write counts, volume-proof metrics, and a per-category breakdown. Piping
it through ``json.tool`` is not a way to read a result.

Usage::

    python -m ocm.evaluation.rahgm.summarize_ocmr_arm local_results/ocmr_arm/ocmr_arm_results.json

Sections, in the order they should be read:

1. **Gate** — whether this run reproduces OCMR's published Table III at all.
   Nothing below matters if it failed.
2. **Extraction health** — recomputed over distinct corpus inputs, since a warm
   cache makes the in-run figure read far too high.
3. **Arms** — the decisive metrics, with task success set beside the
   memory-induced hallucination rate, because task success is answer-token recall
   and rises with the volume of admitted memory.
4. **Integrity retention** — how much of governance's contradiction gain each
   reviewer keeps. This is the quantity the adjudication claim rests on, and it is
   not visible in the arm table.
5. **Per category** — where the recall cost and the recovery actually land.
6. **Selectivity** — whether the learned policy escalates a different set than
   "every quarantine", and whether the features separate false from genuine
   quarantines at all.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from typing import Any, Sequence

#: Categories in OCMR's benchmark, in reporting order.
CATEGORIES = (
    "longitudinal_factual_qa",
    "multi_step_planning_entity_consistency",
    "contradiction_heavy_update_stream",
    "temporal_reasoning_ordered_events",
    "entity_resolution_ambiguity",
    "evidence_required_decisions",
)

#: Short labels so the per-category table fits a terminal.
SHORT = {
    "longitudinal_factual_qa": "factual",
    "multi_step_planning_entity_consistency": "planning",
    "contradiction_heavy_update_stream": "contradict",
    "temporal_reasoning_ordered_events": "temporal",
    "entity_resolution_ambiguity": "entity-res",
    "evidence_required_decisions": "evidence",
}


def _mean(values: Sequence[float]) -> float:
    finite = [v for v in values if isinstance(v, (int, float)) and v == v]
    return st.mean(finite) if finite else float("nan")


def _header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def _gate(report: dict[str, Any]) -> None:
    _header("1. REPRODUCTION GATE")
    gate = report.get("reproduction_gate")
    if not gate:
        print("absent: this file predates the gate. Compare B0/B3 by hand against "
              "published B0 task 77.20 / contrad 14.49 / viol 50.72 and "
              "B3 60.00 / 1.26 / 0.00.")
        return
    print(f"  {'arm':4s} {'metric':8s} {'published':>10s} {'observed':>9s} {'delta':>8s}  verdict")
    for check in gate["checks"]:
        print(
            f"  {check['arm']:4s} {check['metric']:8s} {check['published']:10.2f} "
            f"{check['observed']:9.2f} {check['delta']:+8.2f}  "
            f"{'pass' if check['pass'] else 'FAIL'}"
        )
    print(f"\n  => {'PASSED' if gate['passed'] else 'FAILED'}")
    if not gate["passed"]:
        print("  Stop here. A failed gate means this run is not configured like the\n"
              "  published one, so no row from it can join Table III.")


def _extraction(report: dict[str, Any]) -> None:
    _header("2. EXTRACTION HEALTH")
    ext = report.get("extraction")
    if not ext:
        print("  absent")
        return
    denom = ext.get("distinct_corpus_inputs") or ext.get("distinct_inputs") or 0
    cache = ext.get("cache") or {}
    if cache.get("distinct_requested"):
        denom = cache["distinct_requested"]
    failed = ext.get("distinct_unparseable_inputs", 0)
    rate = (failed / denom) if denom else 0.0
    print(f"  distinct corpus inputs : {denom}")
    print(f"  unparseable            : {failed}  ({rate:.2%})")
    print(f"  generation calls       : {ext.get('calls')}")
    if rate > 0.05:
        print("\n  WARNING: above 5%. Memory is under-populated relative to the\n"
              "  published run, so read the gate before trusting any row.")
    else:
        print("\n  Within tolerance: extraction is not a confound for these numbers.")
    for example in ext.get("model_failure_examples", []):
        print(f"    {example}")


def _arms(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    _header("3. ARMS (decisive metrics, mean over seeds)")
    agg = report.get("aggregate", {})
    print(
        f"  {'arm':22s} {'task':>14s} {'halluc':>8s} {'contrad':>14s} "
        f"{'viol':>14s} {'rev/100':>8s}"
    )
    for name, values in agg.items():
        print(
            f"  {name:22s} "
            f"{values['task_success_mean']:7.2f}±{values['task_success_sd']:5.2f} "
            f"{values.get('memory_induced_hallucination_rate_mean') or float('nan'):8.3f} "
            f"{values['contradiction_rate_mean']:7.2f}±{values['contradiction_rate_sd']:5.2f} "
            f"{values['constraint_violations_mean']:7.2f}±{values['constraint_violations_sd']:5.2f} "
            f"{values['reviews_per_100_writes_mean']:8.1f}"
        )
    print(
        "\n  Task success is answer-token recall over a haystack of retrieved text,\n"
        "  so it rises with the volume of admitted memory. A gain is only real if\n"
        "  the hallucination rate does not rise with it."
    )
    return agg


def _integrity(agg: dict[str, dict[str, float]]) -> None:
    """How much of governance's contradiction gain does each reviewer keep?"""
    _header("4. INTEGRITY RETENTION (the quantity the claim rests on)")
    if "B0" not in agg or "B3" not in agg:
        print("  need B0 and B3 to compute retention")
        return
    ungoverned = agg["B0"]["contradiction_rate_mean"]
    governed = agg["B3"]["contradiction_rate_mean"]
    gain = ungoverned - governed
    b3_task = agg["B3"]["task_success_mean"]
    recoverable = agg["B0"]["task_success_mean"] - b3_task
    print(f"  governance buys {gain:.2f} contradiction points "
          f"({ungoverned:.2f} ungoverned -> {governed:.2f} governed)")
    print(f"  governance costs {recoverable:.2f} task-success points "
          f"({agg['B0']['task_success_mean']:.2f} -> {b3_task:.2f})\n")
    print(f"  {'arm':22s} {'contrad':>8s} {'kept':>8s} {'recall recovered':>17s} {'rev/100':>8s}")
    for name, values in agg.items():
        if name in ("B0", "B2", "B3"):
            continue
        contrad = values["contradiction_rate_mean"]
        kept = ((ungoverned - contrad) / gain) if gain else float("nan")
        recovered = (
            (values["task_success_mean"] - b3_task) / recoverable
            if recoverable else float("nan")
        )
        print(
            f"  {name:22s} {contrad:8.2f} {kept:7.0%} {recovered:16.0%} "
            f"{values['reviews_per_100_writes_mean']:8.1f}"
        )
    print(
        "\n  'kept' is the fraction of the contradiction gain retained; 'recall\n"
        "  recovered' is the fraction of the task-success cost recovered. A\n"
        "  reviewer that recovers recall by giving back the integrity gain is not\n"
        "  doing adjudication, it is just releasing."
    )


def _per_category(report: dict[str, Any]) -> None:
    _header("5. PER-CATEGORY TASK SUCCESS (mean over seeds)")
    # arm -> category -> [values across seeds and reviewers]
    collected: dict[str, dict[str, list[float]]] = {}
    for seed_entry in report.get("per_seed", []):
        for reviewer, rep in (seed_entry.get("reviewers") or {}).items():
            for arm, arm_report in (rep.get("arms") or {}).items():
                label = arm if arm in ("B0", "B2", "B3") else f"{arm}:{reviewer}"
                per_cat = arm_report.get("per_category_task_success") or {}
                bucket = collected.setdefault(label, {})
                for category, value in per_cat.items():
                    bucket.setdefault(category, []).append(value)
    if not collected:
        print("  no per-category data in this file")
        return

    categories = [c for c in CATEGORIES if any(c in v for v in collected.values())]
    print("  arm                    " + " ".join(f"{SHORT.get(c, c)[:10]:>11s}" for c in categories))
    for label, buckets in collected.items():
        row = " ".join(f"{_mean(buckets.get(c, [])):11.2f}" for c in categories)
        print(f"  {label:22s} {row}")
    print(
        "\n  A category at 0.00 across every governed arm is blocked by a hard gate,\n"
        "  not by a marginal routing decision: review cannot release what was never\n"
        "  quarantined."
    )


def _selectivity(report: dict[str, Any], agg: dict[str, dict[str, float]]) -> None:
    _header("6. SELECTIVITY (is the learned policy doing anything?)")
    # B3R vs B3Q: B3Q reviews every quarantine by construction. If they match,
    # the learned policy escalates exactly that set and adds nothing.
    pairs = [
        (name, name.replace("B3R", "B3Q"))
        for name in agg
        if name.startswith("B3R")
    ]
    print(f"  {'reviewer':14s} {'B3R task':>9s} {'B3Q task':>9s} {'B3R rev':>8s} {'B3Q rev':>8s}  verdict")
    for b3r, b3q in pairs:
        if b3q not in agg:
            continue
        r, q = agg[b3r], agg[b3q]
        same = (
            abs(r["task_success_mean"] - q["task_success_mean"]) < 0.01
            and abs(r["reviews_per_100_writes_mean"] - q["reviews_per_100_writes_mean"]) < 0.01
        )
        print(
            f"  {b3r.split(':', 1)[-1]:14s} {r['task_success_mean']:9.2f} "
            f"{q['task_success_mean']:9.2f} {r['reviews_per_100_writes_mean']:8.1f} "
            f"{q['reviews_per_100_writes_mean']:8.1f}  "
            f"{'DEGENERATE (same set)' if same else 'selective'}"
        )

    print("\n  false-vs-genuine quarantine separability:")
    lifts = []
    for entry in report.get("separability", []):
        lifts.append(entry.get("lift_over_base_rate", float("nan")))
        print(
            f"    seed {entry['seed']}: quarantined={entry['n_quarantined']} "
            f"false={entry['n_false_quarantine']} "
            f"base={entry['base_rate_false']:.3f} "
            f"precision={entry['precision']:.3f} "
            f"lift={entry['lift_over_base_rate']:+.3f}"
        )
    if lifts:
        print(
            f"\n  mean lift {_mean(lifts):+.4f}. A lift at zero means the\n"
            "  constraint-failure features carry no signal for telling a false\n"
            "  quarantine from a genuine one, which is the mechanism behind any\n"
            "  degeneracy above."
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "path",
        nargs="?",
        default="local_results/ocmr_arm/ocmr_arm_results.json",
    )
    args = parser.parse_args(argv)

    try:
        with open(args.path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
    except FileNotFoundError:
        print(f"No such file: {args.path}", file=sys.stderr)
        return 1

    config = report.get("config", {})
    print(
        f"config: extractor={config.get('extractor')} "
        f"model={config.get('model')} "
        f"embeddings={config.get('embeddings')} "
        f"per_category={config.get('per_category')} "
        f"seeds={config.get('seeds')}"
    )
    print(f"elapsed: {report.get('elapsed_seconds')}s")

    _gate(report)
    _extraction(report)
    agg = _arms(report)
    _integrity(agg)
    _per_category(report)
    _selectivity(report, agg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

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


def _cache_size(results_path: str) -> int:
    """Count entries in the extraction cache sitting beside the results file.

    Needed for files written before the rate was computed at the cache boundary.
    In those, ``distinct_inputs`` counts only cache misses, so on a warm cache it
    equals the number of perpetually-failing inputs and the rate reads 100%. The
    cache file holds one entry per successfully extracted input, which is the
    denominator that was missing.
    """
    import os

    path = os.path.join(os.path.dirname(results_path) or ".", "extraction_cache.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return len(json.load(handle))
    except Exception:
        return 0


def _extraction(report: dict[str, Any], results_path: str) -> None:
    _header("2. EXTRACTION HEALTH")
    ext = report.get("extraction")
    if not ext:
        print("  absent")
        return
    failed = ext.get("distinct_unparseable_inputs", 0)
    cache = ext.get("cache") or {}
    denom = (
        cache.get("distinct_requested")
        or ext.get("distinct_corpus_inputs")
        or 0
    )
    note = ""
    if not denom:
        # Pre-fix file: reconstruct the denominator from the cache on disk.
        successes = _cache_size(results_path)
        if successes:
            denom = successes + failed
            note = (
                "  (denominator reconstructed from extraction_cache.json; this "
                "results file\n   predates the cache-boundary fix and its own "
                "figure reads 100%)"
            )
        else:
            denom = ext.get("distinct_inputs") or 0
            note = (
                "  (WARNING: this file predates the cache-boundary fix and no "
                "cache file was\n   found, so the rate below counts only cache "
                "misses and is far too high)"
            )
    rate = (failed / denom) if denom else 0.0
    print(f"  distinct corpus inputs : {denom}")
    print(f"  unparseable            : {failed}  ({rate:.2%})")
    print(f"  generation calls       : {ext.get('calls')} (this run only)")
    if note:
        print(note)
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


def _released_fraction(report: dict[str, Any], label: str) -> float:
    """Mean fraction of escalated writes that a reviewer released."""
    released: list[float] = []
    arm, _, reviewer = label.partition(":")
    for seed_entry in report.get("per_seed", []):
        rep = (seed_entry.get("reviewers") or {}).get(reviewer)
        if not rep:
            continue
        arm_report = (rep.get("arms") or {}).get(arm)
        if not arm_report:
            continue
        escalated = arm_report.get("escalated") or 0
        if escalated:
            released.append(arm_report.get("released", 0) / escalated)
    return _mean(released)


def _frontier(report: dict[str, Any], agg: dict[str, dict[str, float]]) -> None:
    """Does any reviewer beat releasing at random with the same volume?

    Retention and recovery are both monotone in release volume, so an intermediate
    pair of numbers is not by itself evidence of judgment. The random rows trace
    what no discrimination achieves; a reviewer earns the word "adjudication" only
    by retaining more integrity than the random row at its own release rate.
    """
    _header("7. NO-SKILL FRONTIER (does adjudication beat release volume?)")
    if "B0" not in agg or "B3" not in agg:
        print("  need B0 and B3")
        return
    ungoverned = agg["B0"]["contradiction_rate_mean"]
    governed = agg["B3"]["contradiction_rate_mean"]
    gain = ungoverned - governed

    rows = []
    for name, values in agg.items():
        if not name.startswith("B3R"):
            continue
        reviewer = name.split(":", 1)[-1]
        kept = ((ungoverned - values["contradiction_rate_mean"]) / gain) if gain else float("nan")
        # Seed-to-seed spread on the contradiction rate, expressed on the same
        # scale as `kept`, so an excess can be read against its own noise instead
        # of against an arbitrary threshold.
        kept_sd = (values["contradiction_rate_sd"] / gain) if gain else float("nan")
        rows.append(
            {
                "reviewer": reviewer,
                "released_frac": _released_fraction(report, name),
                "kept": kept,
                "kept_sd": kept_sd,
                "task": values["task_success_mean"],
                "random": reviewer.startswith("random") or reviewer in ("release_all", "uphold_all"),
            }
        )
    if not rows:
        print("  no B3R rows")
        return

    controls = sorted(
        [r for r in rows if r["random"]], key=lambda r: r["released_frac"]
    )
    print("  no-skill controls (release without judgment):")
    print(f"    {'reviewer':12s} {'released':>9s} {'kept':>7s} {'±sd':>6s} {'task':>7s}")
    for row in controls:
        print(
            f"    {row['reviewer']:12s} {row['released_frac']:9.1%} "
            f"{row['kept']:7.0%} {row['kept_sd']:6.0%} {row['task']:7.2f}"
        )

    def interpolate(fraction: float) -> tuple[float, float]:
        """No-skill retention at this release fraction, and its uncertainty.

        The frontier is a linear interpolation between the two bracketing controls,
        so its uncertainty inherits theirs. Returned so an excess can be judged
        against noise rather than a fixed cutoff.
        """
        if not controls or fraction != fraction:
            return float("nan"), float("nan")
        pts = [
            (c["released_frac"], c["kept"], c["kept_sd"])
            for c in controls
            if c["released_frac"] == c["released_frac"]
        ]
        if len(pts) < 2:
            return float("nan"), float("nan")
        if fraction <= pts[0][0]:
            return pts[0][1], pts[0][2]
        if fraction >= pts[-1][0]:
            return pts[-1][1], pts[-1][2]
        for (x0, y0, s0), (x1, y1, s1) in zip(pts, pts[1:]):
            if x0 <= fraction <= x1:
                span = (x1 - x0) or 1.0
                w = (fraction - x0) / span
                # Variance of a linear interpolation of two independent estimates.
                sd = (((1 - w) * s0) ** 2 + (w * s1) ** 2) ** 0.5
                return y0 + (y1 - y0) * w, sd
        return float("nan"), float("nan")

    print("\n  judgment-based reviewers vs the frontier at their own release rate:")
    print(
        f"    {'reviewer':12s} {'released':>9s} {'kept':>7s} {'no-skill':>9s} "
        f"{'excess':>8s} {'±':>6s} {'ratio':>6s}  verdict"
    )
    for row in rows:
        if row["random"]:
            continue
        expected, expected_sd = interpolate(row["released_frac"])
        excess = row["kept"] - expected
        # Combined spread of the reviewer and the interpolated frontier. Two of
        # these is the bar for calling an effect real; anything inside it is
        # consistent with no discrimination at all.
        combined = (row["kept_sd"] ** 2 + expected_sd ** 2) ** 0.5
        if not (excess == excess) or not (combined == combined):
            verdict = "indeterminate"
        elif excess > 2 * combined:
            verdict = "DISCRIMINATES"
        elif excess < -2 * combined:
            verdict = "ANTI-SELECTIVE (worse than chance)"
        else:
            verdict = "no evidence of discrimination"
        ratio = (excess / combined) if combined else float("nan")
        print(
            f"    {row['reviewer']:12s} {row['released_frac']:9.1%} "
            f"{row['kept']:7.0%} {expected:9.0%} {excess:+8.0%} {combined:6.0%} "
            f"{ratio:+6.1f}  {verdict}"
        )
    print(
        "\n  'excess' is integrity kept above what releasing the same number of\n"
        "  writes at random would keep. '±' is the combined seed-to-seed spread of\n"
        "  the reviewer and the interpolated frontier, and the last column is the\n"
        "  ratio between them. A verdict is claimed only past 2.0; anything inside\n"
        "  is consistent with the reviewer choosing volume rather than writes.\n"
        "  The frontier's own spread dominates here, so tightening it needs more\n"
        "  seeds rather than a better reviewer."
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
    _extraction(report, args.path)
    agg = _arms(report)
    _integrity(agg)
    _per_category(report)
    _selectivity(report, agg)
    _frontier(report, agg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

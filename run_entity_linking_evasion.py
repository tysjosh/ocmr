#!/usr/bin/env python3
"""Run the entity-linking-evasion attack against the governed write path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ocm.evaluation.entity_linking_evasion import (
    DEFAULT_ATTACK_SEED,
    DEFAULT_AXES,
    DEFAULT_BENIGN_PER_FAMILY,
    PAPER_ATTACK_AXES,
    PAPER_ATTACK_BASELINES,
    PAPER_ATTACK_SEEDS,
    run_benign_linkage_corpus,
    run_entity_linking_evasion_attack,
    run_paper_grade_linkage_evasion_suite,
)


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _metric(row: dict | None, condition: str, metric: str) -> dict:
    if not row:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    return row["conditions"][condition][metric]


def _fmt_ci(ci: dict, *, pct: bool = True) -> str:
    scale = 100.0 if pct else 1.0
    return f"{ci['mean'] * scale:.1f} [{ci['low'] * scale:.1f},{ci['high'] * scale:.1f}]"


def _print_paper_suite(report: dict, output: Path) -> None:
    print("=== Entity-linking-evasion paper-grade suite ===")
    print(f"revision: {report['freeze']['repo_revision']}")
    print(f"seeds: {', '.join(str(s) for s in report['freeze']['seeds'])}")
    print(f"axes: {', '.join(report['freeze']['axes'])}")
    print(f"per axis: {report['freeze']['per_axis']}")
    print()
    print(
        f"{'Baseline':<10}{'Mutation':<10}{'Attack accept':<22}"
        f"{'Detect':<22}{'Quarantine':<22}{'Latency ms':<18}"
    )
    for row in report["baseline_comparison"]["rows"]:
        evasive = row["conditions"]["evasive"]
        print(
            f"{row['baseline']:<10}{row['mutation']:<10}"
            f"{_fmt_ci(evasive['accepted_rate']):<22}"
            f"{_fmt_ci(evasive['detection_rate']):<22}"
            f"{_fmt_ci(evasive['quarantine_rate']):<22}"
            f"{_fmt_ci(evasive['mean_injection_latency_ms'], pct=False):<18}"
        )

    benign = report["benign_utility"]["summary"]
    print()
    print("Benign utility:")
    print(f"  false positives: {_fmt_ci(benign['false_positive_rate'])}")
    print(f"  quarantine burden: {_fmt_ci(benign['quarantine_burden_rate'])}")
    print(f"  utility success: {_fmt_ci(benign['utility_success_rate'])}")

    ablation = report["defense_ablations"]["c7_fail_closed_off_b3"]
    off_accept = _metric(ablation, "evasive", "accepted_rate")
    print()
    print(f"C7 off attack acceptance: {_fmt_ci(off_accept)}")
    print(f"release gate: {report['release_gate']['status']}")
    print(f"Saved -> {output}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the synthetic entity-linking-evasion attack."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_ATTACK_SEED)
    parser.add_argument("--baseline", default="B3")
    parser.add_argument(
        "--axes",
        default=None,
        help=(
            "Comma-separated axes: novel_alias,spelling_variant,"
            "spacing_variant,partial,adaptive_alias,role_description,"
            "unrelated_alias"
        ),
    )
    parser.add_argument("--per-axis", type=int, default=8)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in PAPER_ATTACK_SEEDS),
        help="Comma-separated seeds for --paper-suite.",
    )
    parser.add_argument(
        "--baselines",
        default=",".join(PAPER_ATTACK_BASELINES),
        help="Comma-separated baselines for --paper-suite.",
    )
    parser.add_argument(
        "--benign-per-family",
        type=int,
        default=DEFAULT_BENIGN_PER_FAMILY,
        help="Number of generated benign examples per benign family.",
    )
    parser.add_argument("--intensity", default="mvp")
    parser.add_argument(
        "--mutation",
        choices=("original", "mutated"),
        default="original",
        help="Use original seeded attack surfaces or fresh mutated surfaces.",
    )
    parser.add_argument(
        "--disable-fail-closed",
        action="store_true",
        help="Disable the C7_LINKAGE_ATTRIBUTION guard to reproduce config-off behavior.",
    )
    parser.add_argument(
        "--benign-corpus",
        action="store_true",
        help="Run the representative benign false-positive corpus instead of the attack.",
    )
    parser.add_argument(
        "--paper-suite",
        action="store_true",
        help="Run the multi-seed paper-grade attack/benign/ablation suite.",
    )
    parser.add_argument(
        "--include-records",
        action="store_true",
        help="Include every per-injection record in --paper-suite output.",
    )
    parser.add_argument("--false-positive-threshold", type=float, default=0.05)
    parser.add_argument(
        "--output",
        default="local_results/entity_linking_evasion.json",
        help="Path for the JSON attack artifact.",
    )
    args = parser.parse_args(argv)

    axes = _csv(args.axes) if args.axes else DEFAULT_AXES
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.paper_suite:
        paper_axes = _csv(args.axes) if args.axes else PAPER_ATTACK_AXES
        report = run_paper_grade_linkage_evasion_suite(
            seeds=_int_csv(args.seeds),
            baselines=_csv(args.baselines),
            axes=paper_axes,
            per_axis=args.per_axis,
            benign_per_family=args.benign_per_family,
            false_positive_threshold=args.false_positive_threshold,
            include_records=args.include_records,
        )
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _print_paper_suite(report, output)
        return 0

    if args.benign_corpus:
        report = run_benign_linkage_corpus(
            seed=args.seed,
            baseline=args.baseline,
            fail_closed=not args.disable_fail_closed,
            threshold=args.false_positive_threshold,
            per_family=args.benign_per_family,
        )
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("=== Entity-linking benign false-positive corpus ===")
        print(f"seed: {report['seed']}")
        print(f"baseline: {report['baseline']}")
        print(f"fail closed: {report['construction']['fail_closed_enabled']}")
        row = report["summary"]
        print(
            f"false positives: {row['false_positive_count']}/{row['n']} "
            f"({_pct(row['false_positive_rate'])}); "
            f"threshold={_pct(row['threshold'])}; "
            f"passes={row['passes_threshold']}"
        )
        print(f"external side effects: {row['external_side_effect_count']}")
        print(f"data exposure: {row['data_exposure_count']}")
        print(f"Saved -> {output}")
        return 0

    report = run_entity_linking_evasion_attack(
        seed=args.seed,
        baseline=args.baseline,
        axes=axes,
        per_axis=args.per_axis,
        intensity=args.intensity,
        mutation=args.mutation,
        fail_closed=not args.disable_fail_closed,
    )

    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Entity-linking-evasion attack ===")
    print(f"seed: {report['seed']}")
    print(f"baseline: {report['baseline']}")
    print(f"threat model: {report['threat_model']}")
    print(f"axes: {', '.join(report['construction']['axes'])}")
    print(f"mutation: {report['construction']['mutation']}")
    print(f"fail closed: {report['construction']['fail_closed_enabled']}")
    print()
    print(
        f"{'Condition':<12}{'n':>5}{'Attack succ':>14}{'Gate detect':>14}"
        f"{'Accepted':>12}{'Linked':>10}{'Benign FP':>12}"
        f"{'Oracle viol':>14}{'System viol':>14}{'Ext sidefx':>12}"
    )
    for condition in ("control", "evasive"):
        row = report["by_condition"][condition]
        print(
            f"{condition:<12}"
            f"{row['n']:>5}"
            f"{_pct(row['attack_success_rate']):>14}"
            f"{_pct(row['detection_rate']):>14}"
            f"{_pct(row['accepted_rate']):>12}"
            f"{_pct(row['linked_rate']):>10}"
            f"{_pct(row['benign_false_positive_rate']):>12}"
            f"{row['oracle_violation_count']:>14}"
            f"{row['system_durable_violation_count']:>14}"
            f"{row['external_side_effect_count']:>12}"
        )
    print()
    print("By axis:")
    for axis, conditions in report["by_axis_condition"].items():
        evasive = conditions["evasive"]
        print(
            f"  {axis}: attack_success={_pct(evasive['attack_success_rate'])}, "
            f"gate_detect={_pct(evasive['detection_rate'])}, "
            f"mean_distance={evasive['mean_mention_distance']:.3f}"
        )
    print()
    print(f"release gate: {report['release_gate']['status']}")
    print(f"Saved -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

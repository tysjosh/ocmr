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
    run_benign_linkage_corpus,
    run_entity_linking_evasion_attack,
)


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the synthetic entity-linking-evasion attack."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_ATTACK_SEED)
    parser.add_argument("--baseline", default="B3")
    parser.add_argument(
        "--axes",
        default=",".join(DEFAULT_AXES),
        help="Comma-separated axes: novel_alias,spelling_variant,spacing_variant,partial",
    )
    parser.add_argument("--per-axis", type=int, default=8)
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
    parser.add_argument("--false-positive-threshold", type=float, default=0.05)
    parser.add_argument(
        "--output",
        default="local_results/entity_linking_evasion.json",
        help="Path for the JSON attack artifact.",
    )
    args = parser.parse_args(argv)

    axes = _csv(args.axes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.benign_corpus:
        report = run_benign_linkage_corpus(
            seed=args.seed,
            baseline=args.baseline,
            fail_closed=not args.disable_fail_closed,
            threshold=args.false_positive_threshold,
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

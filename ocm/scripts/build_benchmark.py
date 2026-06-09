"""Entry-point script: build the seeded benchmark JSONL (Req 23, 28.9).

Generates the reproducible benchmark dataset via
:func:`ocm.evaluation.benchmark.generate_jsonl` and writes it to a JSONL file,
then prints the total example count and a per-category breakdown so the dataset
thresholds (>=25 per category, >=150 total) are easy to verify.

Usage::

    python -m ocm.scripts.build_benchmark                      # -> benchmark.jsonl, seed 1337
    python -m ocm.scripts.build_benchmark --out data/bench.jsonl --seed 7

Requirements: 23.1, 23.2, 23.3, 23.5, 28.9.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Optional, Sequence

from ocm.evaluation.benchmark import DEFAULT_SEED, generate_jsonl

DEFAULT_OUT = "benchmark.jsonl"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the build-benchmark command."""
    parser = argparse.ArgumentParser(
        prog="ocm-build-benchmark",
        description="Generate the seeded, reproducible evaluation benchmark as JSONL.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output JSONL path (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Generation seed (default: {DEFAULT_SEED}).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Generate the benchmark to ``--out`` and print a count summary."""
    args = build_parser().parse_args(argv)

    examples = generate_jsonl(args.out, seed=args.seed)

    per_category = Counter(example.category for example in examples)
    question_count = sum(len(example.questions) for example in examples)

    print(f"Wrote {len(examples)} examples ({question_count} questions) to {args.out}")
    print(f"  seed: {args.seed}")
    print("  per-category:")
    for category in sorted(per_category):
        print(f"    {category}: {per_category[category]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

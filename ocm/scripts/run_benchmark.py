"""Entry-point script: run baselines B0–B3 over the benchmark (Req 28.9).

Loads (or generates) the JSONL benchmark, runs the requested baselines through
the :class:`~ocm.evaluation.runner.BaselineRunner`, optionally writes a
:class:`~ocm.core.logging.ResearchLogger` JSONL log and the per-question result
records, and prints a short summary (record count + per-baseline counts).

The result records written via ``--out`` are exactly the input the
``report_metrics`` script (Req 28.10) consumes.

Usage::

    # generate-on-the-fly and run all four baselines
    python -m ocm.scripts.run_benchmark --out results.jsonl

    # run against an existing benchmark, log research records too
    python -m ocm.scripts.run_benchmark --benchmark benchmark.jsonl \\
        --baselines B0,B2 --log research_log.jsonl --out results.jsonl

Requirements: 22.6, 25.3, 28.9.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from ocm.core.logging import ResearchLogger
from ocm.evaluation.arms import DEFAULT_RUN_BASELINES
from ocm.evaluation.benchmark import DEFAULT_SEED, generate_jsonl
from ocm.evaluation.runner import DEFAULT_TOP_K, BaselineRunner, load_benchmark


def _parse_baselines(value: str) -> list[str]:
    """Parse a comma-separated baseline list (e.g. ``"B0,B1,B2,B3"``)."""
    names = [token.strip() for token in value.split(",") if token.strip()]
    if not names:
        raise argparse.ArgumentTypeError("at least one baseline must be specified")
    return names


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the run-benchmark command."""
    parser = argparse.ArgumentParser(
        prog="ocm-run-benchmark",
        description="Run evaluation baselines (B0–B3) over the JSONL benchmark.",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Path to a benchmark JSONL. When omitted, one is generated with --seed.",
    )
    parser.add_argument(
        "--baselines",
        type=_parse_baselines,
        default=list(DEFAULT_RUN_BASELINES),
        help="Comma-separated baselines to run (default: B0,B1,B2,B3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed used when generating a benchmark (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="Optional path for the JSONL research log (per-example benchmark records).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the per-question result records as JSONL.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Retrieval depth per question (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of examples to run (smoke checks).",
    )
    return parser


def _write_records(records: list[dict], out_path: str) -> None:
    """Write result records as JSONL, one record per line."""
    path = Path(out_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the requested baselines and print a summary."""
    args = build_parser().parse_args(argv)

    # Resolve the benchmark: load an existing file, or generate one.
    if args.benchmark:
        examples = load_benchmark(args.benchmark)
        source = args.benchmark
    else:
        examples = generate_jsonl("benchmark.jsonl", seed=args.seed)
        source = f"benchmark.jsonl (generated, seed={args.seed})"

    if args.limit is not None:
        examples = examples[: args.limit]

    logger = ResearchLogger(args.log) if args.log else None
    runner = BaselineRunner(logger=logger, top_k=args.top_k)
    records = runner.run(examples, baselines=args.baselines)

    if args.out:
        _write_records(records, args.out)

    per_baseline = Counter(record["baseline_name"] for record in records)

    print(f"Ran baselines {', '.join(args.baselines)} over {len(examples)} examples")
    print(f"  benchmark: {source}")
    print(f"  result records: {len(records)}")
    print("  per-baseline:")
    for baseline in args.baselines:
        print(f"    {baseline}: {per_baseline.get(baseline, 0)}")
    if args.log:
        print(f"  research log: {args.log}")
    if args.out:
        print(f"  results written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

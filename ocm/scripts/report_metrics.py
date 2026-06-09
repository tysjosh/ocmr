"""Entry-point script: compute and report evaluation metrics (Req 28.10).

Feeds result records into the :class:`~ocm.evaluation.metrics.MetricsReporter`
and prints the human-readable metric table (retrieval, answer, write-time, and
agent families, with deltas vs B0). Records can come from a saved results file
(``--results``, as produced by ``run_benchmark --out``) or be produced inline
by running the baselines over a benchmark (``--benchmark`` / generated).

With ``--json`` the structured ``compute()`` output is dumped as JSON instead of
(or in addition to) the table.

Usage::

    # report from saved result records
    python -m ocm.scripts.report_metrics --results results.jsonl

    # run inline over a benchmark, then report
    python -m ocm.scripts.report_metrics --benchmark benchmark.jsonl
    python -m ocm.scripts.report_metrics --benchmark benchmark.jsonl --json metrics.json

Requirements: 24.1–24.5, 28.10.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from ocm.evaluation.baselines import DEFAULT_RUN_BASELINES
from ocm.evaluation.benchmark import DEFAULT_SEED, generate_jsonl
from ocm.evaluation.metrics import MetricsReporter
from ocm.evaluation.runner import BaselineRunner, load_benchmark


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the report-metrics command."""
    parser = argparse.ArgumentParser(
        prog="ocm-report-metrics",
        description="Compute and report evaluation metrics from result records.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--results",
        default=None,
        help="Path to result records JSONL (from `run_benchmark --out`).",
    )
    source.add_argument(
        "--benchmark",
        default=None,
        help="Path to a benchmark JSONL; baselines are run inline before reporting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed used if a benchmark must be generated (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help="Dump compute() as JSON to PATH (or stdout when given without a path).",
    )
    return parser


def _load_results(path: str) -> list[dict]:
    """Load result records from a JSON array or JSONL file."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    # Support both a single JSON array and line-delimited JSON objects.
    if text[0] == "[":
        data = json.loads(text)
        return list(data)
    records: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _results_from_benchmark(path: Optional[str], seed: int) -> list[dict]:
    """Run the default baselines over a benchmark (generating it if needed)."""
    if path:
        examples = load_benchmark(path)
    else:
        examples = generate_jsonl("benchmark.jsonl", seed=seed)
    runner = BaselineRunner()
    return runner.run(examples, baselines=DEFAULT_RUN_BASELINES)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Compute metrics and print the report table (and optional JSON)."""
    args = build_parser().parse_args(argv)

    if args.results:
        records = _load_results(args.results)
    else:
        records = _results_from_benchmark(args.benchmark, args.seed)

    reporter = MetricsReporter()
    print(reporter.report(records))

    if args.json is not None:
        computed = reporter.compute(records)
        payload = json.dumps(computed, indent=2, default=str)
        if args.json == "-":
            print(payload)
        else:
            Path(args.json).write_text(payload, encoding="utf-8")
            print(f"\nMetrics JSON written to: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

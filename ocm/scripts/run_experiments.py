"""Entry-point: run the full research experiment suite from the paper (§IV/§V).

Produces the analyses behind the paper's tables:

* multi-seed decisive metrics with 95% CIs for baselines + ablations
  (Tables II–IV / X),
* paired significance vs the strongest non-OCMR baseline, Holm-Bonferroni
  corrected, with effect sizes (Table VII),
* threshold sensitivity + calibration sweep over τ (Table VI),
* stress task-success by perturbation intensity and entity-resolution
  F1/false-merge (Tables VIII–IX).

By default this runs the **full** protocol (5 seeds × the full benchmark). Use
``--quick`` for a fast smoke run. With ``--extractor mock`` the numbers are the
system's own offline values; for paper-grade numbers use a real extractor
(``--extractor llm``) and ``--embeddings local``.

Usage::

    python -m ocm.scripts.run_experiments                       # full offline protocol
    python -m ocm.scripts.run_experiments --quick               # fast smoke run
    python -m ocm.scripts.run_experiments --extractor llm \\
        --llm-base-url http://localhost:8000/v1 --llm-model Qwen/Qwen2.5-32B-Instruct \\
        --embeddings local --out results.json
    python -m ocm.scripts.run_experiments --extractor llm \\
        --llm-base-url http://localhost:8000/v1 --llm-model Qwen/Qwen2.5-14B-Instruct \\
        --embeddings local --baselines B0,B2,Bsup,B3 --out results_with_bsup.json
"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from ocm.evaluation import experiment as exp


def _parse_baselines(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocm-run-experiments",
        description="Run the multi-seed research experiment suite and print the result tables.",
    )
    parser.add_argument("--out", default=None, help="Optional JSON results path.")
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Directory (e.g. a Google Drive path) for resumable per-(method,seed) "
             "checkpoints; a crashed/refreshed run resumes instead of restarting.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(exp.DEFAULT_SEEDS),
        help="Seeds (one run per method per seed). Default: the 5 research seeds.",
    )
    parser.add_argument(
        "--per-category", type=int, default=exp.FULL_PER_CATEGORY,
        help=f"Benchmark examples generated per category per seed "
             f"(default: {exp.FULL_PER_CATEGORY}, the full protocol).",
    )
    parser.add_argument(
        "--baselines",
        default=",".join(exp.DEFAULT_BASELINES),
        help=(
            "Comma-separated baseline arms to include in the primary synthetic "
            "suite. Extended reviewer arms such as Bsup, Brag, and Brtcf are "
            "opt-in here."
        ),
    )
    parser.add_argument(
        "--stress-per-class", type=int, default=30,
        help="Stress trajectories per (perturbation class, intensity).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Fast smoke configuration (2 seeds, tiny benchmark).",
    )
    parser.add_argument(
        "--extractor", choices=["mock", "llm"], default="mock",
        help="W1 extractor: offline 'mock' (default) or OpenAI-compatible 'llm'.",
    )
    parser.add_argument(
        "--embeddings", choices=["deterministic", "local"], default="deterministic",
        help="Embeddings: offline 'deterministic' (default) or real 'local' "
             "(sentence-transformers); 'local' enables non-deterministic runs.",
    )
    parser.add_argument(
        "--llm-base-url", default=None,
        help="OpenAI-compatible base URL (required when --extractor llm).",
    )
    parser.add_argument(
        "--llm-model", default="gpt-4o-mini", help="LLM model name (--extractor llm).",
    )
    parser.add_argument(
        "--llm-no-json-mode", action="store_true",
        help="Do not send response_format=json_object (for local servers that "
             "reject it; the prompt still requests JSON-only output).",
    )
    parser.add_argument(
        "--llm-api-key-env", default="OCM_LLM_API_KEY",
        help="Environment variable holding the LLM API key (--extractor llm).",
    )
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2",
        help="Local embedding model name (--embeddings local).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    # Quiet the governance warning logs (e.g. C3 PRECEDES-cycle rejections) so
    # the experiment tables read cleanly; they are expected, not errors.
    import logging
    import os

    logging.getLogger("ocm").setLevel(logging.ERROR)

    llm_api_key = os.environ.get(args.llm_api_key_env) if args.extractor == "llm" else None
    if args.extractor == "llm" and not args.llm_base_url:
        print(
            "error: --extractor llm requires --llm-base-url (and an API key in "
            f"${args.llm_api_key_env})."
        )
        return 2
    settings_factory = exp.make_settings_factory(
        extractor=args.extractor,
        embeddings=args.embeddings,
        llm_base_url=args.llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        llm_use_json_mode=not args.llm_no_json_mode,
    )

    if args.quick:
        seeds = args.seeds[:2] if len(args.seeds) >= 2 else args.seeds
        per_category = 2
        stress_per_class = 2
    else:
        seeds = args.seeds
        per_category = args.per_category
        stress_per_class = args.stress_per_class
    baselines = _parse_baselines(args.baselines)

    print(
        f"Configuration: extractor={args.extractor}, embeddings={args.embeddings}, "
        f"seeds={list(seeds)}, per_category={per_category}, "
        f"baselines={list(baselines)}"
        + (f", llm_model={args.llm_model}" if args.extractor == "llm" else "")
    )

    report = exp.run_full_suite(
        seeds=seeds,
        per_category=per_category,
        baselines=baselines,
        stress_per_class=stress_per_class,
        settings_factory=settings_factory,
        checkpoint_dir=args.checkpoint_dir,
        out_path=args.out,
    )
    exp.print_report(report)

    if args.out:
        print(f"\nWrote full results to {args.out}")
    elif args.checkpoint_dir:
        print(f"\nWrote results + checkpoints under {args.checkpoint_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

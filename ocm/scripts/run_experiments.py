"""Entry-point: run the full experiment suite from the paper (§IV/§V).

Produces the analyses behind the paper's tables over the offline, deterministic
harness — these are the *system's own* measured values, not the paper's
illustrative figures:

* multi-seed decisive metrics with 95% CIs for baselines + ablations
  (Tables II–IV / X),
* paired significance vs the strongest non-OCMR baseline, Holm-Bonferroni
  corrected, with effect sizes (Table VII),
* threshold sensitivity + calibration sweep over τ (Table VI),
* stress task-success by perturbation intensity and entity-resolution
  F1/false-merge (Tables VIII–IX).

Usage::

    python -m ocm.scripts.run_experiments                 # full suite
    python -m ocm.scripts.run_experiments --quick         # small/fast smoke run
    python -m ocm.scripts.run_experiments --out results.json
"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from ocm.evaluation import experiment as exp
from ocm.evaluation.ablations import DEFAULT_ABLATIONS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocm-run-experiments",
        description="Run the multi-seed experiment suite and print the result tables.",
    )
    parser.add_argument("--out", default=None, help="Optional JSON results path.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(exp.DEFAULT_SEEDS),
        help="Seeds (one run per method per seed).",
    )
    parser.add_argument(
        "--per-category", type=int, default=6,
        help="Benchmark examples generated per category per seed.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Fast smoke configuration (few seeds, tiny benchmark).",
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


def _fmt_ci(ci) -> str:
    return f"{ci.mean:.1f} [{ci.low:.1f}, {ci.high:.1f}]"


def run_suite(seeds, per_category, settings_factory=None) -> dict:
    if settings_factory is None:
        settings_factory = exp.make_settings_factory()
    baselines = ["B0", "B1", "B2", "B3"]
    methods = baselines + [a for a in DEFAULT_ABLATIONS if a != "full"]

    ms = exp.run_multiseed(
        methods, seeds=seeds, per_category=per_category, settings_factory=settings_factory
    )
    aggregated = exp.aggregate_methods(ms)
    sig = exp.significance_vs_best_baseline(
        ms, ocmr_method="B3", baseline_methods=["B0", "B1", "B2"]
    )
    sweep = exp.threshold_sweep(
        seed=seeds[0], per_category=per_category, settings_factory=settings_factory
    )
    stress = exp.stress_by_intensity(
        methods=baselines, seeds=seeds[:1], per_class=max(2, per_category),
        settings_factory=settings_factory,
    )

    return {
        "methods": methods,
        "seeds": list(seeds),
        "decisive_metrics": {
            method: {metric: aggregated[method][metric].__dict__ for metric in aggregated[method]}
            for method in aggregated
        },
        "significance_vs_best_baseline": sig,
        "threshold_sweep": sweep,
        "stress": stress,
    }


def _print_report(report: dict) -> None:
    print("\n=== Decisive metrics (mean [95% CI] across seeds) ===")
    print(f"{'Method':<22}{'TaskSuccess↑':<22}{'Contradiction↓':<22}{'ConstraintViol↓':<22}")
    agg = report["decisive_metrics"]
    for method in report["methods"]:
        m = agg[method]
        def ci(metric):
            d = m[metric]
            return f"{d['mean']:.1f} [{d['low']:.1f},{d['high']:.1f}]"
        print(f"{method:<22}{ci('task_success'):<22}{ci('contradiction_rate'):<22}{ci('constraint_violations'):<22}")

    print("\n=== Significance: B3 vs strongest non-OCMR baseline (Holm-Bonferroni) ===")
    for metric, t in report["significance_vs_best_baseline"]["metric_tests"].items():
        eff = t["effect_size"]
        eff_s = f"{eff:.3f}" if isinstance(eff, (int, float)) else str(eff)
        print(
            f"  {metric:<22} vs {t['vs_baseline']:<4} "
            f"{t['test']:<14} corrected_p={t['corrected_p']:.4f} "
            f"reject={t['reject_null']} {t['effect_name']}={eff_s}"
        )

    print("\n=== Threshold sweep (τ) + calibration ===")
    print(f"{'tau':<8}{'ContrRate':<12}{'FalseQuar':<12}{'ECE':<10}{'Brier':<10}{'J(tau)':<10}")
    for row in report["threshold_sweep"]["rows"]:
        print(
            f"{row['tau']:<8}{row['contradiction_rate']:<12.2f}{row['false_quarantine']:<12.2f}"
            f"{row['ece']:<10.3f}{row['brier']:<10.3f}{row['objective_j']:<10.3f}"
        )
    print(f"  selected τ (min J): {report['threshold_sweep']['selected_tau']}")

    print("\n=== Stress: task success by perturbation intensity ===")
    ti = report["stress"]["task_success_by_intensity"]
    print(f"{'Method':<10}{'low':<10}{'medium':<10}{'high':<10}")
    for method, lvls in ti.items():
        print(f"{method:<10}{lvls.get('low', 0):<10.1f}{lvls.get('medium', 0):<10.1f}{lvls.get('high', 0):<10.1f}")
    er = report["stress"]["entity_resolution"]
    print(
        f"  entity-resolution F1={er['entity_resolution_f1']:.3f} "
        f"false_merge_rate={er['false_merge_rate']:.3f} (n={int(er['n_examples'])})"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    # Quiet the governance warning logs (e.g. C3 PRECEDES-cycle rejections) so
    # the experiment tables read cleanly; they are expected, not errors.
    import logging
    import os

    logging.getLogger("ocm").setLevel(logging.ERROR)

    # Build the settings factory from the extractor / embedding selection.
    llm_api_key = os.environ.get(args.llm_api_key_env) if args.extractor == "llm" else None
    if args.extractor == "llm" and not args.llm_base_url:
        print(
            "error: --extractor llm requires --llm-base-url (and an API key in "
            f"${args.llm_api_key_env}).",
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
    print(
        f"Configuration: extractor={args.extractor}, embeddings={args.embeddings}"
        + (f", llm_model={args.llm_model}" if args.extractor == "llm" else "")
    )

    seeds = [args.seeds[0]] if args.quick else args.seeds
    if args.quick:
        seeds = args.seeds[:2] if len(args.seeds) >= 2 else args.seeds
    per_category = 2 if args.quick else args.per_category

    report = run_suite(seeds, per_category, settings_factory=settings_factory)
    _print_report(report)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nWrote full results to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

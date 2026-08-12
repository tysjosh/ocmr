"""Single-command runner for the whole RAHGM evaluation suite (Req 13.6).

Usage::

    python -m ocm.evaluation.rahgm.run_all
    python -m ocm.evaluation.rahgm.run_all --out local_results/rahgm
    python -m ocm.evaluation.rahgm.run_all --quick          # reduced corpus, fast
    python -m ocm.evaluation.rahgm.run_all --skip audit     # skip the LLM-free replay

Emits machine-readable JSON plus the rendered tables, both carrying
:data:`~ocm.evaluation.rahgm.report.SCOPE_NOTE` so no number is readable without
its provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Sequence

from ocm.evaluation.rahgm.ablation import run_ablation
from ocm.evaluation.rahgm.adaptation_study import run_experiment3
from ocm.evaluation.rahgm.audit import run_quarantine_audit
from ocm.evaluation.rahgm.cascade import run_cascade_study
from ocm.evaluation.rahgm.corpus import DEFAULT_SEED, Partition, generate_corpus
from ocm.evaluation.rahgm.drift import DRIFT_MODES, run_drift_study
from ocm.evaluation.rahgm.end_to_end import run_experiment4
from ocm.evaluation.rahgm.human_study import run_experiment2
from ocm.evaluation.rahgm.replay import collect_routing_cases, develop_policy, run_experiment1
from ocm.evaluation.rahgm.report import SCOPE_NOTE, TABLE_PROVENANCE, render_all

#: The stages a run can execute, in dependency order.
STAGES: tuple[str, ...] = (
    "audit",
    "experiment1",
    "ablation",
    "experiment2",
    "experiment3",
    "experiment4",
    "cascade",
    "drift",
)


def run_drift_suite(
    corpus: Any, *, developed: Any, repeats: int = 4
) -> dict[str, Any]:
    """Run the drift study in both modes and state the combined finding.

    The two modes answer the same question and give opposite answers, so they are
    only interpretable together.
    """
    out: dict[str, Any] = {}
    for mode in DRIFT_MODES:
        out[mode] = run_drift_study(
            corpus, developed=developed, repeats=repeats, mode=mode
        )

    covariate = out["covariate"]["contrast"]["accuracy_delta_points"]
    label = out["label"]["contrast"]["accuracy_delta_points"]
    out["finding"] = (
        "Bounded feedback adaptation earns its keep only under drift the router "
        f"cannot see. Under covariate drift adaptation changes accuracy by "
        f"{covariate:+.2f} points, because authority is an input the frozen policy "
        "already reroutes on. Under label drift, where every feature is held fixed "
        f"and only the correct answer moves, adaptation changes accuracy by "
        f"{label:+.2f} points and recovers within the stream while the frozen arm "
        "stays flat. This explains the C4-vs-C5 null in Experiment 1: the test "
        "partition is drawn from the fitting distribution, so neither kind of drift "
        "is present and there is nothing for adaptation to do."
    )
    return out

#: Reduced settings for ``--quick``, for smoke-testing the whole pipeline fast.
QUICK_SCENARIOS = 12
QUICK_PARTICIPANTS = 4
QUICK_SEEDS: tuple[int, ...] = (1337, 7)
QUICK_AUDIT_SEEDS: tuple[int, ...] = (1337,)
QUICK_AUDIT_PER_CATEGORY = 6


def run_suite(
    *,
    seed: int = DEFAULT_SEED,
    n_scenarios: int | None = None,
    skip: Sequence[str] = (),
    quick: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run every experiment and return the combined report.

    Args:
        seed: Corpus seed.
        n_scenarios: Override the scenario count (50 in the paper).
        skip: Stage names to skip.
        quick: Use reduced settings throughout.
        verbose: Print per-stage progress and timings.

    Returns:
        The combined report dict, with ``scope_note`` at the top level.
    """
    skip = set(skip)
    scenarios = n_scenarios or (QUICK_SCENARIOS if quick else 50)

    def _log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    started = time.perf_counter()
    report: dict[str, Any] = {
        "scope_note": SCOPE_NOTE,
        "table_provenance": TABLE_PROVENANCE,
        "config": {
            "seed": seed,
            "n_scenarios": scenarios,
            "quick": quick,
            "skipped": sorted(skip),
        },
        "timings_seconds": {},
    }

    def _time(stage: str, function: Any) -> Any:
        stage_started = time.perf_counter()
        _log(f"[{stage}] running...")
        value = function()
        elapsed = time.perf_counter() - stage_started
        report["timings_seconds"][stage] = round(elapsed, 2)
        _log(f"[{stage}] done in {elapsed:.1f}s")
        return value

    # --- corpus and policy development ---------------------------------
    corpus = _time(
        "corpus", lambda: generate_corpus(seed, n_scenarios=scenarios)
    )
    report["corpus"] = corpus.summary()

    developed = _time("develop_policy", lambda: develop_policy(corpus))
    report["developed_policy"] = developed.as_dict()

    # --- quarantine audit (§4.1) ---------------------------------------
    if "audit" not in skip:
        report["quarantine_audit"] = _time(
            "audit",
            lambda: run_quarantine_audit(
                seeds=QUICK_AUDIT_SEEDS if quick else (1337, 7, 42, 99, 2024),
                per_category=QUICK_AUDIT_PER_CATEGORY if quick else 25,
                verbose=False,
            ),
        )

    # --- Experiment 1 (§4.2, Table 3) ----------------------------------
    experiment1: dict[str, Any] | None = None
    if "experiment1" not in skip:
        experiment1 = _time(
            "experiment1", lambda: run_experiment1(corpus, developed=developed)
        )
        # ``_results`` holds live objects for Experiment 4; keep it out of the JSON.
        report["experiment1"] = {
            k: v for k, v in experiment1.items() if not k.startswith("_")
        }

    # --- Routing ablation (§4.2, Table 4) ------------------------------
    if "ablation" not in skip:
        dev_cases = _time(
            "ablation_dev_cases",
            lambda: collect_routing_cases(corpus.partition(Partition.dev)),
        )
        test_cases = _time(
            "ablation_test_cases",
            lambda: collect_routing_cases(corpus.partition(Partition.test)),
        )
        report["ablation"] = _time(
            "ablation",
            lambda: run_ablation(
                corpus, developed.params, dev_cases=dev_cases, test_cases=test_cases
            ),
        )

    # --- Experiment 2 (§4.3, Table 5) — SIMULATED ----------------------
    if "experiment2" not in skip:
        report["experiment2"] = _time(
            "experiment2",
            lambda: run_experiment2(
                corpus,
                developed=developed,
                participants_per_condition=(
                    QUICK_PARTICIPANTS if quick else 20
                ),
                scenarios_per_participant=4 if quick else 8,
            ),
        )

    # --- Experiment 3 (§4.4, Table 6) ----------------------------------
    if "experiment3" not in skip:
        report["experiment3"] = _time(
            "experiment3",
            lambda: run_experiment3(
                corpus,
                developed=developed,
                seeds=QUICK_SEEDS if quick else (1337, 7, 42, 99, 2024),
                max_blocks=3 if quick else 8,
            ),
        )

    # --- Experiment 4 (§4.5, Table 7) ----------------------------------
    if "experiment4" not in skip and experiment1 is not None:
        report["experiment4"] = _time(
            "experiment4",
            lambda: run_experiment4(corpus, experiment1=experiment1),
        )

    # --- Cascade study (not in the paper) ------------------------------
    if "cascade" not in skip:
        report["cascade_study"] = _time(
            "cascade",
            lambda: run_cascade_study(corpus, developed=developed),
        )

    # --- Drift study (not in the paper) --------------------------------
    if "drift" not in skip:
        report["drift_study"] = _time(
            "drift",
            lambda: run_drift_suite(
                corpus, developed=developed, repeats=2 if quick else 4
            ),
        )

    report["timings_seconds"]["total"] = round(time.perf_counter() - started, 2)
    return report


def _write(report: dict[str, Any], out_dir: str) -> tuple[str, str]:
    """Write the JSON report and the rendered tables."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "rahgm_results.json")
    text_path = os.path.join(out_dir, "rahgm_tables.txt")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write(render_all(report))
        handle.write("\n")
    return json_path, text_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run the RAHGM evaluation suite (paper §3–§4)."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="corpus seed")
    parser.add_argument(
        "--scenarios", type=int, default=None, help="scenario count (default 50)"
    )
    parser.add_argument(
        "--out",
        default="local_results/rahgm",
        help="output directory for JSON and rendered tables",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=STAGES,
        help="stages to skip",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="reduced corpus, participants, and seeds for a fast smoke run",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    parser.add_argument(
        "--no-write", action="store_true", help="print tables without writing files"
    )
    args = parser.parse_args(argv)

    report = run_suite(
        seed=args.seed,
        n_scenarios=args.scenarios,
        skip=args.skip,
        quick=args.quick,
        verbose=not args.quiet,
    )

    print()
    print(render_all(report))

    if not args.no_write:
        json_path, text_path = _write(report, args.out)
        print(f"\nWrote {json_path}")
        print(f"Wrote {text_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

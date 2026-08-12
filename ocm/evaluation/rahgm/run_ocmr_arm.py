"""Single-command runner for the OCMR-benchmark escalation experiment.

Reproduces OCMR's decisive-metric table with two additional arms — risk-adaptive
escalation with review-and-release, and reviewing every quarantine — plus the
controls that establish what adjudication actually contributes.

Offline smoke test (no GPU, ~1 min)::

    python -m ocm.evaluation.rahgm.run_ocmr_arm --extractor mock --per-category 6

The run that belongs in the paper (GPU, Qwen2.5-14B-Instruct, matching OCMR's
published setup)::

    python -m ocm.evaluation.rahgm.run_ocmr_arm \\
        --extractor qwen \\
        --model Qwen/Qwen2.5-14B-Instruct \\
        --per-category 25 \\
        --seeds 1337,7,42,99,2024 \\
        --cache local_results/ocmr_arm/extraction_cache.json \\
        --out local_results/ocmr_arm

Why the cache matters
---------------------
Every arm re-ingests identical session text, so with greedy decoding the same
extraction is requested many times over. :class:`CachingExtractor` memoizes by
``(source_ref, text)`` and persists to disk, which makes the run roughly an order
of magnitude cheaper *and* resumable: interrupt it and the next invocation skips
everything already extracted. Because decoding is greedy, caching changes timing
only, never results.

Reviewers
---------
``identity`` is the deployable policy and the one to quote. ``oracle`` is a
ceiling that reads benchmark labels. ``release_all`` and ``uphold_all`` are
controls: ``release_all`` establishes that the recall gain is release volume
rather than adjudication, and that indiscriminate release reverts the
contradiction rate to ungoverned levels. Run all four — the contrast between them
is the actual finding.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from typing import Any, Sequence

from ocm.evaluation.benchmark import BenchmarkGenerator
from ocm.evaluation.rahgm.ocmr_arm import (
    ARMS,
    REVIEWERS,
    fit_policy_on_benchmark,
    run_ocmr_escalation_arm,
    separability_report,
)

#: Seeds OCMR's published table uses.
DEFAULT_SEEDS: tuple[int, ...] = (1337, 7, 42, 99, 2024)

#: Reviewers to run by default: the deployable policy, the ceiling, and both controls.
DEFAULT_REVIEWERS: tuple[str, ...] = ("identity", "oracle", "release_all", "uphold_all")

#: Scope note embedded in every artifact.
SCOPE_NOTE = (
    "SCOPE. Arms B0/B2/B3 reproduce OCMR's own baselines through its own harness "
    "and decisive metrics; B3R and B3Q add escalation with review-and-release. "
    "Task success is answer-token recall over a haystack containing retrieved "
    "text, so it rises with the volume of admitted memory: read it alongside "
    "memory_induced_hallucination_rate, which penalizes admitting incorrect "
    "content. The 'oracle' reviewer reads benchmark expected-conflict labels and "
    "is a ceiling, not a deployable policy; quote 'identity'. 'release_all' and "
    "'uphold_all' are controls, not proposals. With --extractor mock the numbers "
    "are indicative only and will not match OCMR's published table, which used a "
    "local Qwen2.5-14B-Instruct extractor."
)


def _build_extractor(
    kind: str,
    *,
    model_id: str,
    cache_path: str | None,
    max_new_tokens: int,
    tolerate_extraction_errors: bool = False,
) -> Any:
    """Build the W1 extractor, importing torch/transformers only when needed."""
    if kind == "mock":
        return None  # the container's default offline extractor

    if kind != "qwen":
        raise ValueError(f"unknown extractor {kind!r}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise SystemExit(
            "The qwen extractor needs `transformers` and a torch build for your "
            "GPU. Install them on the run machine, e.g.:\n"
            "  pip install 'transformers>=4.45' accelerate\n"
            "  # plus the torch wheel matching your CUDA version\n"
            f"(import failed: {exc})"
        ) from exc

    from ocm.extraction.caching_extractor import CachingExtractor
    from ocm.extraction.strict_extractor import StrictExtractor
    from ocm.extraction.transformers_extractor import TransformersExtractor

    print(f"[extractor] loading {model_id} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="bfloat16", device_map="auto"
    )
    base = TransformersExtractor(
        model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens
    )
    # StrictExtractor sits *inside* the cache so failures are never memoized. It
    # aborts on environment faults (broken Triton/CUDA, OOM) because the write
    # pipeline would otherwise swallow them and every arm would silently run on
    # an empty memory store, producing a plausible but meaningless table.
    strict = StrictExtractor(
        base, tolerate_environment_errors=tolerate_extraction_errors
    )
    # Greedy decoding makes memoization exact, so the cache only changes timing.
    return CachingExtractor(strict, cache_path=cache_path)


#: Probe sentence for the preflight check. Deliberately in the same shape as the
#: benchmark's own session text so a successful extraction is meaningful.
_PROBE_TEXT = "Alice owns Project Orion."
_PROBE_REF = "preflight:probe"


def _preflight(extractor: Any) -> None:
    """Verify the extractor actually generates before committing hours of GPU time.

    A broken CUDA/Triton toolchain makes every ``generate`` call raise, and the
    write pipeline is designed to absorb extraction failures as recorded
    validation failures. That combination silently produces a full results table
    over an empty memory store, so the environment must be proven up front.

    Raises:
        SystemExit: If the probe extraction fails, carrying the remediation text.
    """
    if extractor is None:
        return  # offline mock: nothing to prove
    from ocm.extraction.strict_extractor import ExtractionEnvironmentError

    print("[preflight] probing the extractor with one short input ...", flush=True)
    try:
        result = extractor.extract(_PROBE_TEXT, _PROBE_REF)
    except ExtractionEnvironmentError as exc:
        raise SystemExit(f"\n[preflight FAILED]\n{exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surface anything as a hard stop
        raise SystemExit(
            f"\n[preflight FAILED] the extractor raised {exc!r}.\n"
            "Aborting: a run started in this state would record every write as a "
            "failed extraction and report a table computed over empty memory."
        ) from exc

    n = len(getattr(result, "assertions", []) or [])
    print(
        f"[preflight] ok - probe produced {n} assertion(s), "
        f"extractor_version={getattr(result, 'extractor_version', '?')}",
        flush=True,
    )


def _strict_stats(extractor: Any) -> dict[str, Any] | None:
    """Pull the :class:`StrictExtractor` counters out from under the cache."""
    node = extractor
    for _ in range(4):  # walk a short wrapper chain
        if node is None:
            return None
        if type(node).__name__ == "StrictExtractor":
            return node.stats
        node = getattr(node, "_base", None)
    return None


def _mean(values: Sequence[float]) -> float:
    finite = [v for v in values if v is not None and v == v]
    return st.mean(finite) if finite else float("nan")


def _stdev(values: Sequence[float]) -> float:
    finite = [v for v in values if v is not None and v == v]
    return st.stdev(finite) if len(finite) > 1 else 0.0


def _render(aggregate: dict[str, dict[str, list[float]]]) -> str:
    """Render the decisive-metric table across arms and reviewers."""
    header = (
        f"{'arm':22s} {'task':>13s} {'halluc':>9s} {'contrad':>13s} "
        f"{'viol':>13s} {'rev/100':>9s}"
    )
    lines = [header, "-" * len(header)]
    for name, values in aggregate.items():
        lines.append(
            f"{name:22s} "
            f"{_mean(values['task']):6.2f}±{_stdev(values['task']):5.2f} "
            f"{_mean(values['halluc']):9.3f} "
            f"{_mean(values['contrad']):6.2f}±{_stdev(values['contrad']):5.2f} "
            f"{_mean(values['viol']):6.2f}±{_stdev(values['viol']):5.2f} "
            f"{_mean(values['reviews']):9.1f}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run OCMR's benchmark with and without escalation + release."
    )
    parser.add_argument(
        "--extractor",
        choices=("mock", "qwen"),
        default="mock",
        help="mock = offline and fast (indicative); qwen = OCMR's published setup",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-14B-Instruct",
        help="HF model id for --extractor qwen",
    )
    parser.add_argument("--per-category", type=int, default=25)
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="comma-separated seeds",
    )
    parser.add_argument(
        "--reviewers",
        default=",".join(DEFAULT_REVIEWERS),
        help=f"comma-separated, from {sorted(REVIEWERS)}",
    )
    parser.add_argument(
        "--arms", default=",".join(ARMS), help=f"comma-separated, from {list(ARMS)}"
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="extraction cache JSON path; makes a qwen run cheap and resumable",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--out", default="local_results/ocmr_arm")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument(
        "--tolerate-extraction-errors",
        action="store_true",
        help=(
            "do not abort when extraction fails for environmental reasons. "
            "Unsafe: the write pipeline records a failed extraction and carries "
            "on, so a broken GPU environment yields a full table computed over "
            "an empty memory store. For debugging only."
        ),
    )
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    reviewers = [r.strip() for r in args.reviewers.split(",") if r.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(reviewers) - set(REVIEWERS)
    if unknown:
        parser.error(f"unknown reviewer(s): {sorted(unknown)}")

    cache_path = args.cache
    if args.extractor == "qwen" and cache_path is None:
        cache_path = os.path.join(args.out, "extraction_cache.json")
        print(f"[cache] defaulting to {cache_path}", flush=True)

    extractor = _build_extractor(
        args.extractor,
        model_id=args.model,
        cache_path=cache_path,
        max_new_tokens=args.max_new_tokens,
        tolerate_extraction_errors=args.tolerate_extraction_errors,
    )
    _preflight(extractor)

    print(SCOPE_NOTE, flush=True)
    print(
        f"\n[config] extractor={args.extractor} per_category={args.per_category} "
        f"seeds={seeds} reviewers={reviewers} arms={arms}\n",
        flush=True,
    )

    started = time.perf_counter()
    aggregate: dict[str, dict[str, list[float]]] = {}
    per_seed: list[dict[str, Any]] = []
    separability: list[dict[str, Any]] = []

    def _record(name: str, arm: dict[str, Any]) -> None:
        values = aggregate.setdefault(
            name, {k: [] for k in ("task", "halluc", "contrad", "viol", "reviews")}
        )
        checks = arm.get("volume_proof_checks", {}) or {}
        values["task"].append(arm["task_success"])
        values["contrad"].append(arm["contradiction_rate"])
        values["viol"].append(arm["constraint_violations"])
        values["reviews"].append(arm["review_rate_per_100_writes"])
        values["halluc"].append(checks.get("memory_induced_hallucination_rate"))

    for seed in seeds:
        print(f"[seed {seed}] generating benchmark ...", flush=True)
        examples = BenchmarkGenerator(seed=seed).generate(
            per_category=args.per_category
        )
        print(f"[seed {seed}] fitting policy on the development split ...", flush=True)
        fitted = fit_policy_on_benchmark(
            examples, extractor=extractor, embeddings=None
        )
        separability.append(
            {
                "seed": seed,
                **separability_report(
                    fitted["test_examples"], fitted["params"], extractor=extractor
                ),
            }
        )

        seed_entry: dict[str, Any] = {"seed": seed, "reviewers": {}}
        for reviewer in reviewers:
            print(f"[seed {seed}] arms with reviewer={reviewer} ...", flush=True)
            report = run_ocmr_escalation_arm(
                examples=fitted["test_examples"],
                seed=seed,
                arms=arms,
                reviewer=reviewer,
                params=fitted["params"],
                extractor=extractor,
            )
            seed_entry["reviewers"][reviewer] = report
            for arm_name, arm in report["arms"].items():
                # Ungoverned and plain-OCMR arms do not depend on the reviewer, so
                # they are recorded once rather than duplicated per reviewer.
                if arm_name in ("B0", "B2", "B3"):
                    if reviewer == reviewers[0]:
                        _record(arm_name, arm)
                else:
                    _record(f"{arm_name}:{reviewer}", arm)
        per_seed.append(seed_entry)

        if extractor is not None and hasattr(extractor, "save"):
            extractor.save()
            print(f"[cache] {getattr(extractor, 'stats', {})}", flush=True)

    elapsed = time.perf_counter() - started
    table = _render(aggregate)
    print("\n" + table, flush=True)
    print(
        "\nseparability of false vs genuine quarantines "
        "(lift over base rate ~0 means the constraint pattern carries no signal):",
        flush=True,
    )
    for entry in separability:
        print(
            f"  seed {entry['seed']}: quarantined={entry['n_quarantined']} "
            f"false={entry['n_false_quarantine']} "
            f"base={entry['base_rate_false']:.3f} "
            f"precision={entry['precision']:.3f} "
            f"lift={entry['lift_over_base_rate']:+.3f}",
            flush=True,
        )
    extraction_stats = _strict_stats(extractor)
    if extraction_stats:
        print(
            f"\nextraction: {extraction_stats['calls']} call(s), "
            f"{extraction_stats['model_failures']} unparseable "
            f"({extraction_stats['model_failure_rate']:.1%})",
            flush=True,
        )
        for example in extraction_stats["model_failure_examples"]:
            print(f"  e.g. {example}", flush=True)
        if extraction_stats["model_failure_rate"] > 0.05:
            print(
                "  WARNING: over 5% of extractions were unparseable, so memory is "
                "under-populated relative to OCMR's published run. Check the "
                "reproduction gate (B0 ~ 77.20, B3 ~ 60.00) before using these "
                "numbers.",
                flush=True,
            )
    print(f"\n[done] {elapsed:.1f}s", flush=True)

    report = {
        "scope_note": SCOPE_NOTE,
        "config": {
            "extractor": args.extractor,
            "model": args.model if args.extractor == "qwen" else None,
            "per_category": args.per_category,
            "seeds": seeds,
            "reviewers": reviewers,
            "arms": arms,
            "cache_path": cache_path,
        },
        "aggregate": {
            name: {
                "task_success_mean": _mean(v["task"]),
                "task_success_sd": _stdev(v["task"]),
                "memory_induced_hallucination_rate_mean": _mean(v["halluc"]),
                "contradiction_rate_mean": _mean(v["contrad"]),
                "contradiction_rate_sd": _stdev(v["contrad"]),
                "constraint_violations_mean": _mean(v["viol"]),
                "constraint_violations_sd": _stdev(v["viol"]),
                "reviews_per_100_writes_mean": _mean(v["reviews"]),
                "n_seeds": len(v["task"]),
            }
            for name, v in aggregate.items()
        },
        "separability": separability,
        "extraction": extraction_stats,
        "per_seed": per_seed,
        "elapsed_seconds": round(elapsed, 2),
    }

    if not args.no_write:
        os.makedirs(args.out, exist_ok=True)
        json_path = os.path.join(args.out, "ocmr_arm_results.json")
        text_path = os.path.join(args.out, "ocmr_arm_table.txt")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        with open(text_path, "w", encoding="utf-8") as handle:
            handle.write(SCOPE_NOTE + "\n\n" + table + "\n")
        print(f"Wrote {json_path}\nWrote {text_path}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

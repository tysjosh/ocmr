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
from ocm.evaluation.experiment import make_settings_factory
from ocm.evaluation.rahgm.ocmr_arm import (
    ARMS,
    REVIEWERS,
    fit_policy_on_benchmark,
    run_ocmr_escalation_arm,
    separability_report,
)

#: Seeds OCMR's published table uses.
DEFAULT_SEEDS: tuple[int, ...] = (1337, 7, 42, 99, 2024)

#: Reviewers to run by default: the deployable policy, the ceiling, the two
#: endpoint controls, and the no-skill frontier that the deployable policy has to
#: beat to count as adjudication rather than release volume.
DEFAULT_REVIEWERS: tuple[str, ...] = (
    "identity",
    "oracle",
    "release_all",
    "uphold_all",
    "random25",
    "random50",
    "random75",
)

#: OCMR's published Table III values, for the reproduction gate. Until the B0 and
#: B3 rows land near these, no B3R row can join that table: a mismatch means this
#: run's configuration differs from the published one, so the arms would not be
#: comparable to the rest of the paper.
PUBLISHED_TABLE_III: dict[str, dict[str, float]] = {
    "B0": {"task": 77.20, "contrad": 14.49, "viol": 50.72},
    "B3": {"task": 60.00, "contrad": 1.26, "viol": 0.00},
}

#: Tolerance for the gate, in metric points. Task success is the loose one because
#: the arms are scored on a held-out 60% split rather than the full benchmark.
GATE_TOLERANCE: dict[str, float] = {"task": 6.0, "contrad": 4.0, "viol": 15.0}

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
    # cache_failures keeps every arm on the identical extraction outcome for an
    # unparseable input, instead of re-generating it once per arm and relying on
    # repeat greedy decoding being bit-identical.
    try:
        cache = CachingExtractor(strict, cache_path=cache_path, cache_failures=True)
    except ValueError as exc:  # fingerprint mismatch
        raise SystemExit(f"\n[cache REFUSED]\n{exc}") from exc
    print(f"[extractor] fingerprint {strict.fingerprint}", flush=True)
    if cache.unversioned_cache:
        print(
            "[cache] WARNING: this cache file predates fingerprinting, so its "
            "provenance cannot be verified. It is being trusted. Re-saving will "
            f"stamp it with {strict.fingerprint}.",
            flush=True,
        )
    return cache


#: Probe inputs for the preflight check. Taken verbatim from the benchmark's own
#: anchor sessions, so an extractor that handles these is handling the real task.
#: Each must yield at least one relation, otherwise memory stays empty and every
#: arm's numbers become meaningless.
_PROBES: tuple[tuple[str, str], ...] = (
    ("Alice owns Project Orion.", "preflight:owns"),
    ("Bob is assigned to Task T1 and Bob completed Task T1.", "preflight:assigned"),
)

#: The six item lists on :class:`~ocm.memory.contracts.ExtractionResult`.
_ITEM_FIELDS = ("entities", "events", "claims", "documents", "decisions", "relations")


def _item_counts(result: Any) -> dict[str, int]:
    """Per-field item counts for an :class:`ExtractionResult`."""
    return {f: len(getattr(result, f, []) or []) for f in _ITEM_FIELDS}


def _preflight(extractor: Any) -> None:
    """Prove the extractor works before committing hours of GPU time.

    Guards two distinct silent-degradation modes, both of which end in a full,
    plausible-looking arm table computed over an empty memory store:

    1. Every ``generate`` call raises (broken Triton/CUDA, OOM). The write
       pipeline absorbs each one as a recorded validation failure (Req 3.3).
    2. Every call returns schema-valid but *empty* JSON. Nothing raises at all,
       so no warning is logged anywhere.

    The second is the more dangerous of the two because it is completely silent,
    so the check asserts on extracted content rather than on mere success.

    Raises:
        SystemExit: If any probe fails or yields no relations.
    """
    if extractor is None:
        return  # offline mock: nothing to prove
    from ocm.extraction.strict_extractor import ExtractionEnvironmentError

    print(f"[preflight] probing the extractor with {len(_PROBES)} inputs ...", flush=True)
    totals: dict[str, int] = {f: 0 for f in _ITEM_FIELDS}
    for text, ref in _PROBES:
        try:
            result = extractor.extract(text, ref)
        except ExtractionEnvironmentError as exc:
            raise SystemExit(f"\n[preflight FAILED]\n{exc}") from exc
        except Exception as exc:  # noqa: BLE001 - surface anything as a hard stop
            raise SystemExit(
                f"\n[preflight FAILED] the extractor raised {exc!r} on {text!r}.\n"
                "Aborting: a run started in this state would record every write as "
                "a failed extraction and report a table computed over empty memory."
            ) from exc
        counts = _item_counts(result)
        for field, n in counts.items():
            totals[field] += n
        summary = " ".join(f"{f}={counts[f]}" for f in _ITEM_FIELDS if counts[f])
        print(f"[preflight]   {ref}: {summary or 'NOTHING EXTRACTED'}", flush=True)

    if totals["relations"] == 0:
        raise SystemExit(
            "\n[preflight FAILED] the extractor parsed cleanly but produced no "
            f"relations across {len(_PROBES)} probe inputs (totals: {totals}).\n"
            "Aborting: relations are what become candidate assertions, so memory "
            "would stay empty and every arm would score on an empty store. "
            "Nothing raises in this state, which is why it is checked here.\n"
            "Likely causes: a model too small or too heavily quantized to follow "
            "the JSON extraction schema, or a chat template mismatch. Inspect one "
            "raw generation before continuing."
        )
    print(
        f"[preflight] ok - {totals['relations']} relation(s) over {len(_PROBES)} "
        f"probes, extractor_version={getattr(result, 'extractor_version', '?')}",
        flush=True,
    )


def _build_embeddings(kind: str, model_name: str) -> Any:
    """Load the embedding provider once and share it across every container.

    The published run uses real ``all-MiniLM-L6-v2`` vectors. Retrieval quality
    drives task success, which is answer-token recall over a haystack of retrieved
    text, so substituting the offline hashing provider suppresses task success
    while leaving the write-time metrics (contradiction rate, durable violations)
    largely intact. That asymmetry is a configuration artifact, not a finding.
    """
    if kind == "deterministic":
        return None  # container builds DeterministicEmbeddingProvider
    from ocm.retrieval.embeddings import LocalEmbeddingProvider

    print(f"[embeddings] loading {model_name} ...", flush=True)
    return LocalEmbeddingProvider(model_name=model_name)


def _gate_report(aggregate: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    """Compare the reproduced B0/B3 rows against OCMR's published Table III.

    Returns a verdict per arm and metric. A failure means this run is not
    configured like the published one, so nothing in the table should be quoted
    alongside the rest of the paper until the discrepancy is explained.
    """
    checks: list[dict[str, Any]] = []
    for arm, expected in PUBLISHED_TABLE_III.items():
        values = aggregate.get(arm)
        if not values:
            continue
        for metric, want in expected.items():
            got = _mean(values[metric])
            delta = got - want
            checks.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "published": want,
                    "observed": round(got, 2),
                    "delta": round(delta, 2),
                    "tolerance": GATE_TOLERANCE[metric],
                    "pass": abs(delta) <= GATE_TOLERANCE[metric],
                }
            )
    return {"passed": all(c["pass"] for c in checks) if checks else False, "checks": checks}


def _render_gate(gate: dict[str, Any]) -> str:
    """Render the reproduction gate as a short table."""
    lines = [
        "reproduction gate vs OCMR Table III "
        "(a FAIL means this run's configuration differs from the published one):",
        f"  {'arm':4s} {'metric':8s} {'published':>10s} {'observed':>9s} {'delta':>8s}  verdict",
    ]
    for check in gate["checks"]:
        verdict = "pass" if check["pass"] else "FAIL"
        lines.append(
            f"  {check['arm']:4s} {check['metric']:8s} {check['published']:10.2f} "
            f"{check['observed']:9.2f} {check['delta']:+8.2f}  {verdict}"
        )
    lines.append(
        "  => GATE PASSED" if gate["passed"] else
        "  => GATE FAILED: do not put a B3R row in Table III from this run."
    )
    return "\n".join(lines)


def _find(extractor: Any, class_name: str) -> Any:
    """Walk the wrapper chain for a component by class name."""
    node = extractor
    for _ in range(4):
        if node is None:
            return None
        if type(node).__name__ == class_name:
            return node
        node = getattr(node, "_base", None)
    return None


def _strict_stats(extractor: Any) -> dict[str, Any] | None:
    """Extraction counters, with the failure rate taken over the whole corpus.

    The rate has to be computed at the cache boundary. :class:`StrictExtractor`
    sits inside the cache and therefore only ever sees misses; on a warm cache
    every success is a hit and only the previously-failing inputs are retried, so
    a rate computed from its own counters reads 100% no matter how healthy the
    run was. The honest denominator is the number of distinct inputs the cache
    was asked for.
    """
    strict = _find(extractor, "StrictExtractor")
    if strict is None:
        return None
    stats = dict(strict.stats)
    cache = _find(extractor, "CachingExtractor")
    if cache is not None:
        cache_stats = cache.stats
        requested = cache_stats.get("distinct_requested", 0)
        failed = max(
            stats.get("distinct_unparseable_inputs", 0),
            cache_stats.get("distinct_failures", 0),
        )
        stats["distinct_corpus_inputs"] = requested
        stats["distinct_unparseable_inputs"] = failed
        stats["unparseable_input_rate"] = (failed / requested) if requested else 0.0
        stats["cache"] = cache_stats
    else:
        stats["distinct_corpus_inputs"] = stats.get("distinct_inputs", 0)
    return stats


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
    parser.add_argument(
        "--embeddings",
        choices=("deterministic", "local", "auto"),
        default="auto",
        help=(
            "'local' uses real all-MiniLM-L6-v2, matching the published run; "
            "'deterministic' uses offline hashed vectors, which suppresses task "
            "success because retrieval drives it. 'auto' picks local for "
            "--extractor qwen and deterministic for mock."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="embedding model for --embeddings local",
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
        "--probe-only",
        action="store_true",
        help=(
            "run the preflight, dump the full extraction JSON for each probe, and "
            "exit. Use this to eyeball what the model actually emits before "
            "committing to a multi-hour sweep."
        ),
    )
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

    embedding_kind = args.embeddings
    if embedding_kind == "auto":
        embedding_kind = "local" if args.extractor == "qwen" else "deterministic"

    extractor = _build_extractor(
        args.extractor,
        model_id=args.model,
        cache_path=cache_path,
        max_new_tokens=args.max_new_tokens,
        tolerate_extraction_errors=args.tolerate_extraction_errors,
    )
    _preflight(extractor)

    embeddings = _build_embeddings(embedding_kind, args.embedding_model)
    # Match the published configuration exactly: real embeddings imply
    # deterministic_test_mode=False, which is the stochasticity the multi-seed CIs
    # are meant to capture. Storage stays hermetic either way.
    settings_factory = make_settings_factory(
        # An injected extractor instance takes precedence over this setting, so it
        # stays "mock" and the Qwen extractor is passed in directly instead.
        extractor="mock",
        embeddings=embedding_kind,
        embedding_model=args.embedding_model,
    )

    if args.probe_only:
        print("\n[probe-only] full extraction JSON per probe:", flush=True)
        for text, ref in _PROBES:
            result = extractor.extract(text, ref) if extractor else None
            print(f"\n--- {ref}: {text}", flush=True)
            print(
                json.dumps(result.model_dump(mode="json"), indent=2)
                if result is not None
                else "(offline mock: nothing to dump)",
                flush=True,
            )
        if extractor is not None and hasattr(extractor, "save"):
            extractor.save()
        return 0

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
            examples,
            extractor=extractor,
            embeddings=embeddings,
            settings_factory=settings_factory,
        )
        separability.append(
            {
                "seed": seed,
                **separability_report(
                    fitted["test_examples"],
                    fitted["params"],
                    extractor=extractor,
                    embeddings=embeddings,
                    settings_factory=settings_factory,
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
                embeddings=embeddings,
                settings_factory=settings_factory,
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
    gate = _gate_report(aggregate)
    print("\n" + _render_gate(gate), flush=True)
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
            f"\nextraction: {extraction_stats['distinct_corpus_inputs']} distinct "
            f"corpus input(s), "
            f"{extraction_stats['distinct_unparseable_inputs']} unparseable "
            f"({extraction_stats['unparseable_input_rate']:.1%}); "
            f"{extraction_stats['calls']} generation call(s) this run",
            flush=True,
        )
        for example in extraction_stats["model_failure_examples"]:
            print(f"  e.g. {example}", flush=True)
        if extraction_stats["unparseable_input_rate"] > 0.05:
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
            "embeddings": embedding_kind,
            "embedding_model": (
                args.embedding_model if embedding_kind == "local" else None
            ),
        },
        "reproduction_gate": gate,
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
            handle.write(
                SCOPE_NOTE + "\n\n" + table + "\n\n" + _render_gate(gate) + "\n"
            )
        print(f"Wrote {json_path}\nWrote {text_path}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

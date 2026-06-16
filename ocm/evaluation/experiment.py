"""Multi-seed experiment orchestration (paper §IV/§V).

Ties the harness together to produce the paper's reported analyses:

* :func:`run_multiseed` — run a set of methods (baselines and/or ablations)
  across several seeds, returning per-(method, seed) decisive metrics.
* :func:`aggregate_methods` — mean ± 95% CI per method (Tables II–IV / X).
* :func:`significance_vs_best_baseline` — paired tests with Holm-Bonferroni
  correction and effect sizes vs the strongest non-OCMR baseline (Table VII).
* :func:`threshold_sweep` — contradiction rate, false-quarantine, ECE, Brier and
  the selection objective J(τ) across τ values (Table VI).
* :func:`stress_by_intensity` — task success per perturbation intensity, plus
  entity-resolution F1 / false-merge (Tables VIII–IX).

Decisive metrics (operational definitions)
-------------------------------------------
Computed per (method, seed) from the runner's question-level records:

* ``task_success`` (↑) — answering / plan completion only: the mean fraction of
  expected answer tokens recalled (× 100), decoupled from conflict-surfacing so
  it is independent of ``contradiction_rate``.
* ``contradiction_rate`` (↓) — per 100 responses, the rate at which a *known*
  contradiction was **not** surfaced (a governance miss that leaks a
  contradiction into the response).
* ``constraint_violations`` (↓) — durable-write constraint violation rate per
  100 responses: the rate at which a response surfaces constraint-violating
  durable state (a quarantined / superseded / contradicted item) **without**
  flagging it as a conflict. Because every arm shares the governed write path,
  the durable store is identical across arms; the governance difference shows up
  in what each *surfaces* at answer time — ungoverned baselines fold
  constraint-violating items into results unflagged, while governed arms exclude
  or flag them.

These are offline proxies over the deterministic mock pipeline; the harness
reports the *system's own* measured values, not the paper's illustrative
figures.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.evaluation import stats
from ocm.evaluation.ablations import DEFAULT_ABLATIONS, build_ablation_strategy
from ocm.evaluation.baselines import build_baseline, baseline_settings_overrides
from ocm.evaluation.benchmark import BenchmarkGenerator
from ocm.evaluation.runner import BaselineRunner
from ocm.evaluation.strategies import MemoryStrategy


# --------------------------------------------------------------------------- #
# Checkpointing (resume across Colab refreshes / crashes)
# --------------------------------------------------------------------------- #
class _Checkpoint:
    """Tiny JSON checkpoint store keyed by a per-unit-of-work name.

    When ``directory`` is set (e.g. a Google Drive path), each completed unit of
    work (one ``(method, seed)`` run, one \u03c4 row, ...) is written atomically so a
    resumed run skips already-finished work. When ``directory`` is ``None`` it is
    a no-op (every ``load`` misses, nothing is saved).
    """

    def __init__(self, directory: Optional[str]) -> None:
        self.dir = directory
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _path(self, name: str) -> str:
        safe = name.replace("/", "_").replace(" ", "_")
        return os.path.join(self.dir, f"{safe}.json")  # type: ignore[arg-type]

    def load(self, name: str) -> Optional[Any]:
        if not self.dir:
            return None
        path = self._path(name)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:  # pragma: no cover - corrupt/partial file: recompute
            return None

    def save(self, name: str, obj: Any) -> None:
        if not self.dir:
            return
        path = self._path(name)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, default=str)
        os.replace(tmp, path)  # atomic on the same filesystem

#: Default seeds (5 per method, per the paper's protocol).
DEFAULT_SEEDS: tuple[int, ...] = (1337, 7, 42, 99, 2024)


def _seed_everything(seed: int) -> None:
    """Best-effort global seeding for reproducibility (import-safe).

    Seeds Python's ``random`` always, and ``numpy`` / ``torch`` /
    ``transformers`` *only if they are importable*, so the offline test
    environment (which has none of the ML stack) stays dependency-free while a
    real LLM run on Colab gets fully seeded. With the local extractor's greedy
    decoding (``do_sample=False``) generation is already deterministic; this
    additionally pins weight-init/dropout RNG and any sampling fallbacks so each
    seed is reproducible end to end.
    """
    import random as _random

    _random.seed(seed)
    try:  # numpy
        import numpy as _np

        _np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy absent in hermetic tests
        pass
    try:  # transformers convenience seeder (also seeds torch when present)
        from transformers import set_seed as _set_seed  # type: ignore

        _set_seed(seed)
    except Exception:  # pragma: no cover - transformers absent offline
        try:  # fall back to torch directly
            import torch as _torch  # type: ignore

            _torch.manual_seed(seed)
            if _torch.cuda.is_available():  # pragma: no cover - GPU only
                _torch.cuda.manual_seed_all(seed)
        except Exception:  # pragma: no cover - torch absent offline
            pass

#: The three decisive metrics and their optimization direction.
DECISIVE_METRICS: dict[str, str] = {
    "task_success": "max",
    "contradiction_rate": "min",
    "constraint_violations": "min",
}


def _default_settings() -> Settings:
    return Settings(deterministic_test_mode=True, chroma_mode="memory", extractor="mock")


def make_settings_factory(
    *,
    extractor: str = "mock",
    embeddings: str = "deterministic",
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    contradiction_high_confidence: float = 0.8,
    llm_use_json_mode: bool = True,
    extra_overrides: Optional[dict] = None,
) -> Callable[[], Settings]:
    """Build a hermetic :class:`Settings` factory for an experiment configuration.

    Selects the W1 extractor (``"mock"`` or ``"llm"``) and the embedding provider
    (``"deterministic"`` offline hashing, or ``"local"`` real
    sentence-transformers) independently. Storage is always kept hermetic
    (in-memory SQLite + in-memory Chroma) so multi-seed/ablation runs never share
    durable state or touch disk, even when running non-deterministically with
    real embeddings.

    * ``embeddings="deterministic"`` ⇒ ``deterministic_test_mode=True`` (cheap,
      offline vectors, deterministic ids) — the default, fully offline.
    * ``embeddings="local"`` ⇒ ``deterministic_test_mode=False`` so the container
      loads :class:`LocalEmbeddingProvider` (real ``all-MiniLM-L6-v2``); ids
      become random, the genuine stochasticity the paper's multi-seed CIs need.
    * ``extractor="llm"`` ⇒ the container builds :class:`LLMExtractor` and calls
      an OpenAI-compatible endpoint (``llm_base_url`` + ``llm_api_key`` required
      at run time).
    """
    deterministic = embeddings == "deterministic"

    def factory() -> Settings:
        return Settings(
            deterministic_test_mode=deterministic,
            extractor=extractor,  # type: ignore[arg-type]
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            embedding_mode="local",
            embedding_model=embedding_model,
            llm_use_json_mode=llm_use_json_mode,
            sqlite_path=":memory:",  # hermetic even when non-deterministic
            chroma_mode="memory",
            contradiction_high_confidence=contradiction_high_confidence,
            **(extra_overrides or {}),
        )

    return factory


# --------------------------------------------------------------------------- #
# Decisive metrics from question-level records
# --------------------------------------------------------------------------- #
def durable_constraint_violations(container: Any) -> tuple[int, int]:
    """Count mutually-contradictory accepted state in a method's durable store.

    A durable constraint violation is contradictory state left *accepted* in
    memory: a single-valued relation (cardinality 1:1 or m:1 — e.g. ``HAS_STATUS``,
    ``ASSIGNED_TO``) with two or more **distinct** accepted objects for the same
    subject. A governed arm gates such conflicts at write time (the losing side
    is quarantined), so it has ~0; an ungoverned arm (contradiction gate off)
    leaves both sides accepted, accumulating violations.

    Returns ``(violation_count, accepted_count)``.
    """
    from ocm.ontology.relations import RELATION_SIGNATURES, Cardinality

    single_valued = {Cardinality.ONE_TO_ONE, Cardinality.M_TO_ONE}
    try:
        accepted = list(container.repo.list_assertions("accepted"))
    except Exception:  # pragma: no cover - defensive
        return 0, 0
    groups: dict[tuple[str, str], set[str]] = {}
    for a in accepted:
        sig = RELATION_SIGNATURES.get(a.predicate)
        if sig is None or sig.cardinality not in single_valued:
            continue
        groups.setdefault((a.subject_id, a.predicate), set()).add(a.object_id)
    violations = sum(max(0, len(objs) - 1) for objs in groups.values())
    return violations, len(accepted)


def durable_row_count(container: Any) -> int:
    """Durable storage footprint for the efficiency table (paper Table V).

    Counts the full persisted footprint a method keeps: entities + ALL assertions
    (accepted **and** superseded — the audit trail) + quarantine records. Governed
    arms retain superseded and quarantined rows for traceability, so they grow
    larger than ungoverned baselines that don't. Reported relative to the
    text-only baseline (B0) as a growth factor. This is a row-count proxy, not a
    byte measure, so it captures audit-trail growth but not per-row metadata size.
    """
    try:
        assertions = len(list(container.repo.list_assertions()))  # all statuses
        entities = len(list(container.repo.list_entities()))
    except Exception:  # pragma: no cover - defensive
        return 0
    try:
        quarantine = len(list(container.repo.list_quarantine()))
    except Exception:  # pragma: no cover - defensive
        quarantine = 0
    return assertions + entities + quarantine


def decisive_metrics(
    records: list[dict], constraint_violation_rate: Optional[float] = None
) -> dict[str, float]:
    """Compute the three decisive metrics for one method/seed record set."""
    n = len(records)
    if n == 0:
        return {"task_success": 0.0, "contradiction_rate": 0.0, "constraint_violations": 0.0}
    task_success = 100.0 * sum(float(r.get("score", 0.0)) for r in records) / n
    missed_contradiction = sum(
        1 for r in records
        if r.get("expected_conflict") and not r.get("conflict_surfaced")
    )
    # Durable-write constraint violation rate (paper §IV-B): per 100 responses,
    # contradictory state left accepted in durable memory. When the caller
    # supplies the rate (computed from the arm's durable store via
    # ``durable_constraint_violations``) we use it directly; otherwise we fall
    # back to the response-surfaced-violation flag, then (legacy) wrong-answers.
    if constraint_violation_rate is not None:
        constraint_violations = float(constraint_violation_rate)
    elif any("surfaced_violation" in r for r in records):
        violations = sum(1 for r in records if r.get("surfaced_violation"))
        constraint_violations = 100.0 * violations / n
    else:  # pragma: no cover - legacy records without the flag
        wrong = sum(1 for r in records if not r.get("answer_correct"))
        constraint_violations = 100.0 * wrong / n
    contradiction_rate = 100.0 * missed_contradiction / n
    return {
        "task_success": task_success,
        "contradiction_rate": contradiction_rate,
        "constraint_violations": constraint_violations,
    }


def task_success_by_category(records: list[dict]) -> dict[str, float]:
    """Mean task-success (×100) grouped by example category (Table II)."""
    by_cat: dict[str, list[float]] = {}
    for r in records:
        by_cat.setdefault(str(r.get("category")), []).append(float(r.get("score", 0.0)))
    return {cat: 100.0 * sum(v) / len(v) for cat, v in by_cat.items() if v}


# --------------------------------------------------------------------------- #
# Method construction (baselines + ablations behind one name space)
# --------------------------------------------------------------------------- #
def _build_strategy(
    method: str,
    settings_factory: Callable[[], Settings],
    *,
    extractor: object | None = None,
    embeddings: object | None = None,
) -> MemoryStrategy:
    """Build a strategy for a method name (a B-baseline or a named ablation).

    A shared ``extractor`` / ``embeddings`` (loaded once) is injected into every
    container so a heavy model is not reloaded per arm.
    """
    if method.startswith("B"):
        settings = settings_factory().model_copy(
            update=baseline_settings_overrides(method)
        )
        container = CoreContainer(
            settings, extractor=extractor, embeddings=embeddings
        )
        return build_baseline(method, container)
    return build_ablation_strategy(
        method, settings_factory, extractor=extractor, embeddings=embeddings
    )


def _warmup_stack(
    settings_factory: Callable[[], Settings],
    *,
    extractor: object | None = None,
    embeddings: object | None = None,
) -> None:
    """Pay one-time, process-global lazy-init costs *before* any timed write.

    The first write/query of a cold run triggers lazy initialization that is
    process-global, not per-arm: the tokenizer/model's first forward pass, torch
    kernel autotuning, the embedding model's first encode, etc. Because arms run
    in a fixed order (B0 first), that one-time cost is misattributed entirely to
    B0, inflating its mean write latency by orders of magnitude on a cold run
    (the well-known "first iteration is slow" artifact) while a warm re-run shows
    the true steady-state value.

    Running one throwaway write **and** query on a discarded B0 strategy here
    amortizes those costs up front so every *measured* write/query is timed in
    steady state. The warmup container is local and never observed. Any failure
    is swallowed — warmup is an optimization, never a correctness requirement.
    """
    try:
        strategy = _build_strategy(
            "B0", settings_factory, extractor=extractor, embeddings=embeddings
        )
        strategy.write("Alice owns Project Orion.", "warmup:s1")
        strategy.query("Who owns Project Orion?", top_k=5)
    except Exception:  # pragma: no cover - warmup must never break a run
        logging.getLogger(__name__).debug("Warmup write/query failed", exc_info=True)


# --------------------------------------------------------------------------- #
# Multi-seed runs
# --------------------------------------------------------------------------- #
@dataclass
class MultiSeedResult:
    """Per-(method, seed) decisive metrics and the raw records."""

    methods: list[str]
    seeds: list[int]
    # method -> metric -> [value per seed]
    per_seed: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    # method -> seed -> category -> success
    per_seed_category: dict[str, dict[int, dict[str, float]]] = field(default_factory=dict)
    # method -> write-outcome counts summed across seeds (candidates/accepted/
    # superseded/quarantined/rejected). Shows whether an ablation changes what
    # the governed write path admits.
    write_outcomes: dict[str, dict[str, int]] = field(default_factory=dict)
    # method -> efficiency accumulators (write/query latency, context tokens,
    # durable rows) summed across seeds; means/ratios are formed for Table V.
    efficiency: dict[str, dict[str, float]] = field(default_factory=dict)


def run_multiseed(
    methods: Iterable[str],
    seeds: Iterable[int] = DEFAULT_SEEDS,
    *,
    per_category: int = 4,
    limit: Optional[int] = None,
    settings_factory: Callable[[], Settings] = _default_settings,
    top_k: int = 10,
    extractor: object | None = None,
    embeddings: object | None = None,
    checkpoint_dir: Optional[str] = None,
    token_counter: Optional[Any] = None,
    key_suffix: str = "",
    warmup: bool = True,
    provided_examples: Optional[list] = None,
) -> MultiSeedResult:
    """Run ``methods`` across ``seeds`` and collect decisive metrics per seed.

    Each seed generates its own benchmark (the source of per-seed variance). For
    speed, ``per_category`` controls the benchmark size per seed and ``limit``
    optionally caps the example count. A shared ``extractor`` / ``embeddings``
    (loaded once, e.g. a local Qwen + sentence-transformers) is injected into
    every container so heavy models are not reloaded per arm.

    When ``checkpoint_dir`` is set, each completed ``(method, seed)`` is persisted
    there and skipped on a resumed run \u2014 so a Colab crash/refresh loses at most
    the in-flight arm, not the whole run. ``key_suffix`` is appended to each
    checkpoint key so a configuration change (e.g. a different contradiction
    threshold \u03c4) recomputes instead of silently loading a stale result.
    ``provided_examples`` injects a fixed external benchmark (e.g. the MultiWOZ
    adapter's examples) used for every seed instead of generating one per seed;
    pair it with a dataset-specific ``key_suffix`` so its checkpoints stay
    separate from the synthetic benchmark's.
    """
    methods = list(methods)
    seeds = list(seeds)
    ckpt = _Checkpoint(checkpoint_dir)
    result = MultiSeedResult(methods=methods, seeds=seeds)
    warmed = not warmup  # when warmup is disabled, treat as already warmed
    for method in methods:
        result.per_seed[method] = {m: [] for m in DECISIVE_METRICS}
        result.per_seed_category[method] = {}
        result.write_outcomes[method] = {
            "candidates": 0,
            "accepted": 0,
            "superseded": 0,
            "quarantined": 0,
            "rejected": 0,
        }
        result.efficiency[method] = {
            "write_ms": 0.0, "write_calls": 0.0,
            "query_ms": 0.0, "query_calls": 0.0,
            "context_tokens": 0.0, "storage_rows": 0.0, "seeds": 0.0,
        }

    for seed in seeds:
        _seed_everything(seed)
        # An external dataset (e.g. MultiWOZ) supplies a fixed example list used
        # for every seed; otherwise examples are generated lazily per seed.
        examples = list(provided_examples) if provided_examples is not None else None
        for method in methods:
            key = f"ms__{method}__seed{seed}__pc{per_category}{key_suffix}"
            cached = ckpt.load(key)
            if cached is not None:
                dm = cached["decisive"]
                cat = cached.get("category", {})
                wo = cached.get("write_outcomes", {})
            else:
                if not warmed:
                    # Amortize one-time process-global lazy init *before* the
                    # first timed write so no single arm (B0) absorbs it.
                    _warmup_stack(
                        settings_factory, extractor=extractor, embeddings=embeddings
                    )
                    warmed = True
                if examples is None:
                    examples = BenchmarkGenerator(seed=seed).generate(per_category=per_category)
                    if limit is not None:
                        examples = examples[:limit]
                runner = BaselineRunner(
                    settings_factory=settings_factory, top_k=top_k,
                    token_counter=token_counter,
                )
                strategy = _build_strategy(
                    method, settings_factory, extractor=extractor, embeddings=embeddings
                )
                records: list[dict] = []
                wo = {
                    "candidates": 0, "accepted": 0, "superseded": 0,
                    "quarantined": 0, "rejected": 0,
                }
                write_ms = 0.0
                write_calls = 0
                for example in examples:
                    wc = runner._ingest_sessions(strategy, example)
                    for outcome_key in wo:
                        wo[outcome_key] += int(wc.get(outcome_key, 0))
                    write_ms += float(wc.get("write_ms", 0.0))
                    write_calls += int(wc.get("write_calls", 0))
                    for q_index, question in enumerate(example.questions):
                        records.append(
                            runner._run_question(
                                method, strategy, example, q_index, question,
                                write_quarantined=wc["quarantined"],
                            )
                        )
                # Durable-write constraint violations from this arm's store.
                n_resp = len(records)
                dwv, _acc = durable_constraint_violations(strategy.container)
                cvr = (100.0 * dwv / n_resp) if n_resp else 0.0
                dm = decisive_metrics(records, constraint_violation_rate=cvr)
                cat = task_success_by_category(records)
                eff = {
                    "write_ms": write_ms,
                    "write_calls": float(write_calls),
                    "query_ms": sum(float(r.get("latency_ms", 0.0)) for r in records),
                    "query_calls": float(n_resp),
                    "context_tokens": sum(float(r.get("context_tokens", 0)) for r in records),
                    "storage_rows": float(durable_row_count(strategy.container)),
                    "seeds": 1.0,
                }
                ckpt.save(key, {
                    "decisive": dm, "category": cat,
                    "write_outcomes": wo, "efficiency": eff,
                })
            eff = (cached.get("efficiency", {}) if cached is not None else eff) or {}
            for metric, value in dm.items():
                result.per_seed[method][metric].append(float(value))
            result.per_seed_category[method][seed] = cat
            for outcome_key, total in (wo or {}).items():
                if outcome_key in result.write_outcomes[method]:
                    result.write_outcomes[method][outcome_key] += int(total)
            for eff_key, eff_val in eff.items():
                if eff_key in result.efficiency[method]:
                    result.efficiency[method][eff_key] += float(eff_val)
    return result


# --------------------------------------------------------------------------- #
# Aggregation + significance
# --------------------------------------------------------------------------- #
def aggregate_methods(ms: MultiSeedResult) -> dict[str, dict[str, stats.MeanCI]]:
    """Mean ± 95% CI per method per decisive metric (Tables II–IV / X)."""
    out: dict[str, dict[str, stats.MeanCI]] = {}
    for method in ms.methods:
        out[method] = {
            metric: stats.mean_ci(ms.per_seed[method][metric])
            for metric in DECISIVE_METRICS
        }
    return out


def per_seed_raw(ms: MultiSeedResult) -> dict[str, dict[str, list[float]]]:
    """Raw per-seed decisive-metric values per method (no aggregation).

    Exposes exactly what the multi-seed run observed — one value per seed, per
    method, per decisive metric — so a reviewer can inspect the spread behind the
    aggregated mean ± CI rather than trusting a summary. Pairs the seeds with the
    values so degenerate (zero-variance) cases are auditable: a metric that reads
    identically across seeds shows up here as a flat list, which is *why* the
    Student-t CI collapses to zero width and a paired test returns p=0 or p=1.
    """
    return {
        method: {
            metric: list(ms.per_seed[method][metric])
            for metric in DECISIVE_METRICS
        }
        for method in ms.methods
    }


def bootstrap_methods(
    ms: MultiSeedResult, *, n_resamples: int = 10000, seed: int = 1234
) -> dict[str, dict[str, stats.MeanCI]]:
    """Nonparametric bootstrap CI per method per decisive metric.

    A robustness companion to :func:`aggregate_methods` (which uses a Student-t
    interval). The percentile bootstrap makes no normality assumption, so for the
    small multi-seed samples here it is the more defensible interval to report
    alongside the t-based one. The resampling RNG is seeded for reproducibility.
    """
    return {
        method: {
            metric: stats.bootstrap_mean_ci(
                ms.per_seed[method][metric],
                n_resamples=n_resamples,
                seed=seed,
            )
            for metric in DECISIVE_METRICS
        }
        for method in ms.methods
    }


def aggregate_task_success_by_category(
    ms: MultiSeedResult,
) -> dict[str, dict[str, stats.MeanCI]]:
    """Per-method, per-scenario **task success** mean ± 95% CI across seeds.

    Restores the per-scenario breakdown table (Recall / Contradiction-heavy /
    Temporal / Planning / Evidence): for each method and benchmark category, the
    mean and 95% CI of task success over seeds, computed from the same per-seed
    category values the harness already records (``per_seed_category``). Only
    **task success** is available per scenario — contradiction-rate and
    constraint-violations are aggregated globally, not per category — so callers
    must label any derived table as task-success-only.
    """
    out: dict[str, dict[str, stats.MeanCI]] = {}
    for method in ms.methods:
        by_cat: dict[str, list[float]] = {}
        for _seed, cat_map in (ms.per_seed_category.get(method) or {}).items():
            for category, value in (cat_map or {}).items():
                by_cat.setdefault(str(category), []).append(float(value))
        out[method] = {cat: stats.mean_ci(vals) for cat, vals in sorted(by_cat.items())}
    return out


def efficiency_table(ms: MultiSeedResult) -> dict[str, dict[str, Optional[float]]]:
    """Per-method efficiency/overhead for Table V.

    Forms mean write/query latency (ms) and mean per-response context tokens and
    durable rows from the accumulators, then expresses token overhead (%) and
    storage growth (×) relative to the text-only baseline B0 (the paper's 1.00×
    reference). Returns ``None`` for the relative columns when B0 is absent.
    """
    means: dict[str, dict[str, float]] = {}
    for method in ms.methods:
        e = ms.efficiency.get(method, {})
        write_calls = e.get("write_calls", 0.0) or 1.0
        query_calls = e.get("query_calls", 0.0) or 1.0
        seeds = e.get("seeds", 0.0) or 1.0
        means[method] = {
            "write_latency_ms": e.get("write_ms", 0.0) / write_calls,
            "query_latency_ms": e.get("query_ms", 0.0) / query_calls,
            "context_tokens": e.get("context_tokens", 0.0) / query_calls,
            "storage_rows": e.get("storage_rows", 0.0) / seeds,
        }
    base = means.get("B0")
    out: dict[str, dict[str, Optional[float]]] = {}
    for method, v in means.items():
        token_overhead = (
            100.0 * (v["context_tokens"] - base["context_tokens"]) / base["context_tokens"]
            if base and base["context_tokens"] else None
        )
        storage_growth = (
            v["storage_rows"] / base["storage_rows"]
            if base and base["storage_rows"] else None
        )
        out[method] = {
            "write_latency_ms": v["write_latency_ms"],
            "query_latency_ms": v["query_latency_ms"],
            "token_overhead_pct": token_overhead,
            "storage_growth_x": storage_growth,
        }
    return out


def _is_better(metric: str, a: float, b: float) -> bool:
    """Whether value ``a`` is better than ``b`` for ``metric``'s direction."""
    return a > b if DECISIVE_METRICS[metric] == "max" else a < b


def strongest_baseline(
    ms: MultiSeedResult, candidate_methods: Iterable[str], metric: str = "task_success"
) -> str:
    """The non-OCMR baseline with the best mean on ``metric``."""
    best = None
    best_val = None
    for method in candidate_methods:
        mean = stats.mean_ci(ms.per_seed[method][metric]).mean
        if best_val is None or _is_better(metric, mean, best_val):
            best_val = mean
            best = method
    return best  # type: ignore[return-value]


def significance_vs_best_baseline(
    ms: MultiSeedResult,
    ocmr_method: str,
    baseline_methods: Iterable[str],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired tests (Holm-Bonferroni corrected) for OCMR vs the best baseline.

    For each decisive metric the strongest baseline on that metric is chosen,
    a paired test over matched seeds is run, and p-values are corrected across
    the three metrics (Table VII).
    """
    baseline_methods = list(baseline_methods)
    raw: dict[str, stats.TestResult] = {}
    chosen: dict[str, str] = {}
    for metric in DECISIVE_METRICS:
        base = strongest_baseline(ms, baseline_methods, metric)
        chosen[metric] = base
        ocmr_vals = ms.per_seed[ocmr_method][metric]
        base_vals = ms.per_seed[base][metric]
        raw[metric] = stats.paired_test_auto(ocmr_vals, base_vals)

    corrected = stats.holm_bonferroni({m: r.p_value for m, r in raw.items()}, alpha=alpha)
    return {
        "metric_tests": {
            metric: {
                "vs_baseline": chosen[metric],
                "test": raw[metric].test,
                "statistic": raw[metric].statistic,
                "raw_p": raw[metric].p_value,
                "corrected_p": corrected[metric]["corrected_p"],
                "reject_null": corrected[metric]["reject"],
                "effect_size": raw[metric].effect_size,
                "effect_name": raw[metric].effect_name,
            }
            for metric in DECISIVE_METRICS
        },
        "alpha": alpha,
    }


# --------------------------------------------------------------------------- #
# Threshold sensitivity + calibration (Table VI)
# --------------------------------------------------------------------------- #
def _calibration_arrays(records: list[dict]) -> tuple[list[float], list[bool]]:
    confidences = [float(r.get("package_confidence", 0.0)) for r in records]
    correct = [bool(r.get("answer_correct", False)) for r in records]
    return confidences, correct


def _false_quarantine_rate(records: list[dict]) -> float:
    """Per 100 responses on non-conflict examples whose writes were quarantined."""
    non_conflict = [r for r in records if not r.get("expected_conflict")]
    if not non_conflict:
        return 0.0
    false_q = sum(1 for r in non_conflict if int(r.get("write_quarantined", 0)) > 0)
    return 100.0 * false_q / len(non_conflict)


def threshold_sweep(
    taus: Iterable[float] = (0.6, 0.7, 0.8, 0.9, 0.95),
    seed: int = 1337,
    *,
    per_category: int = 6,
    lambda_q: float = 0.5,
    lambda_c: float = 10.0,
    method: str = "B3",
    settings_factory: Callable[[], Settings] = _default_settings,
    extractor: object | None = None,
    embeddings: object | None = None,
    checkpoint_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Sweep the contradiction threshold τ and report calibration (Table VI).

    For each τ the full system is run over a fixed benchmark and we report the
    contradiction rate, false-quarantine rate, ECE, Brier, and the selection
    objective ``J(τ) = ContrRate + lambda_q*FalseQuarantine + lambda_c*ECE``.
    The ``settings_factory`` supplies the base configuration (extractor /
    embeddings); τ is overridden per row. A shared ``extractor`` / ``embeddings``
    is injected so heavy models are not reloaded per τ; completed τ rows are
    checkpointed to ``checkpoint_dir`` and skipped on resume. Returns per-τ rows
    and the τ minimizing J.
    """
    ckpt = _Checkpoint(checkpoint_dir)
    _seed_everything(seed)
    examples = None
    rows: list[dict[str, float]] = []
    for tau in taus:
        key = f"tau__{method}__{tau}__seed{seed}__pc{per_category}"
        cached = ckpt.load(key)
        if cached is not None:
            rows.append(cached)
            continue
        if examples is None:
            examples = BenchmarkGenerator(seed=seed).generate(per_category=per_category)

        def tau_factory(tau=tau) -> Settings:
            return settings_factory().model_copy(
                update={"contradiction_high_confidence": float(tau)}
            )

        runner = BaselineRunner(settings_factory=tau_factory)
        strategy = _build_strategy(
            method, tau_factory, extractor=extractor, embeddings=embeddings
        )
        records: list[dict] = []
        for example in examples:
            wc = runner._ingest_sessions(strategy, example)
            for q_index, question in enumerate(example.questions):
                records.append(
                    runner._run_question(
                        method, strategy, example, q_index, question,
                        write_quarantined=wc["quarantined"],
                    )
                )
        dm = decisive_metrics(records)
        confidences, correct = _calibration_arrays(records)
        ece = stats.expected_calibration_error(confidences, correct)
        brier = stats.brier_score(confidences, correct)
        false_q = _false_quarantine_rate(records)
        j = dm["contradiction_rate"] + lambda_q * false_q + lambda_c * ece
        row = {
            "tau": float(tau),
            "contradiction_rate": dm["contradiction_rate"],
            "false_quarantine": false_q,
            "ece": ece,
            "brier": brier,
            "objective_j": j,
        }
        ckpt.save(key, row)
        rows.append(row)
    best = min(rows, key=lambda r: r["objective_j"]) if rows else None
    return {"rows": rows, "selected_tau": best["tau"] if best else None,
            "lambda_q": lambda_q, "lambda_c": lambda_c}


# --------------------------------------------------------------------------- #
# Stress experiments (Tables VIII–IX)
# --------------------------------------------------------------------------- #
def stress_by_intensity(
    methods: Iterable[str] = ("B0", "B1", "B2", "B3"),
    seeds: Iterable[int] = (DEFAULT_SEEDS[0],),
    *,
    per_class: int = 8,
    settings_factory: Callable[[], Settings] = _default_settings,
    extractor: object | None = None,
    embeddings: object | None = None,
    checkpoint_dir: Optional[str] = None,
    key_suffix: str = "",
) -> dict[str, Any]:
    """Aggregate task success by perturbation intensity and entity-resolution.

    Returns, per method, mean task success at low/medium/high intensity (Table
    IX), and the entity-resolution F1 / false-merge over alias scenarios (Table
    VIII). Each ``(method, seed)`` and the entity-resolution pass are
    checkpointed to ``checkpoint_dir`` and skipped on resume. ``key_suffix`` is
    appended to each checkpoint key so a config change (e.g. \u03c4) recomputes
    rather than loading a stale result.
    """
    from ocm.evaluation.stress import (
        INTENSITY_LEVELS,
        evaluate_entity_resolution,
        generate_stress_examples,
    )

    methods = list(methods)
    seeds = list(seeds)
    ckpt = _Checkpoint(checkpoint_dir)
    task_success: dict[str, dict[str, list[float]]] = {
        m: {lvl: [] for lvl in INTENSITY_LEVELS} for m in methods
    }
    for seed in seeds:
        _seed_everything(seed)
        examples = None
        for method in methods:
            key = f"stress__{method}__seed{seed}__pc{per_class}{key_suffix}"
            cached = ckpt.load(key)
            if cached is not None:
                for lvl in INTENSITY_LEVELS:
                    if cached.get(lvl) is not None:
                        task_success[method][lvl].append(float(cached[lvl]))
                continue
            if examples is None:
                examples = generate_stress_examples(seed=seed, per_class=per_class)
            runner = BaselineRunner(settings_factory=settings_factory)
            strategy = _build_strategy(
                method, settings_factory, extractor=extractor, embeddings=embeddings
            )
            by_level: dict[str, list[float]] = {lvl: [] for lvl in INTENSITY_LEVELS}
            for example in examples:
                wc = runner._ingest_sessions(strategy, example)
                for q_index, question in enumerate(example.questions):
                    rec = runner._run_question(
                        method, strategy, example, q_index, question,
                        write_quarantined=wc["quarantined"],
                    )
                    lvl = rec.get("intensity") or "low"
                    by_level.setdefault(lvl, []).append(float(rec.get("score", 0.0)))
            seed_means: dict[str, Optional[float]] = {}
            for lvl in INTENSITY_LEVELS:
                vals = by_level.get(lvl) or []
                mean = (100.0 * sum(vals) / len(vals)) if vals else None
                seed_means[lvl] = mean
                if mean is not None:
                    task_success[method][lvl].append(mean)
            ckpt.save(key, seed_means)

    intensity_table = {
        method: {
            lvl: stats.mean_ci(task_success[method][lvl]).mean
            for lvl in INTENSITY_LEVELS
        }
        for method in methods
    }
    # Entity resolution is governance-path-independent (writes are governed);
    # evaluate it once over the alias stress examples (checkpointed).
    er_key = f"stress_entity_resolution__seed{seeds[0]}__pc{per_class}"
    er = ckpt.load(er_key)
    if er is None:
        alias_examples = generate_stress_examples(seed=seeds[0], per_class=per_class)
        er = evaluate_entity_resolution(
            alias_examples, settings_factory=settings_factory,
            extractor=extractor, embeddings=embeddings,
        )
        ckpt.save(er_key, er)
    return {"task_success_by_intensity": intensity_table, "entity_resolution": er}


# --------------------------------------------------------------------------- #
# Full research suite (one entry point for the whole protocol)
# --------------------------------------------------------------------------- #
#: Full benchmark size per category (the research protocol, not a smoke run).
FULL_PER_CATEGORY: int = 25

DEFAULT_BASELINES: tuple[str, ...] = ("B0", "B1", "B2", "B3")


def run_full_suite(
    *,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    per_category: int = FULL_PER_CATEGORY,
    baselines: Iterable[str] = DEFAULT_BASELINES,
    settings_factory: Callable[[], Settings] = _default_settings,
    extractor: object | None = None,
    embeddings: object | None = None,
    stress_per_class: int = 30,
    stress_extractor: object | None = None,
    tau: Optional[float] = None,
    taus: Iterable[float] = (0.6, 0.7, 0.8, 0.9, 0.95),
    checkpoint_dir: Optional[str] = None,
    out_path: Optional[str] = None,
    token_counter: Optional[Any] = None,
    warmup: bool = True,
) -> dict[str, Any]:
    """Run the **entire** research protocol and return a structured result.

    Produces every table group at full scale: decisive metrics with 95% CIs for
    baselines + ablations (Tables II\u2013IV/X), paired significance vs the strongest
    baseline (Table VII), the \u03c4-sweep + calibration (Table VI), and the stress
    suite (Tables VIII\u2013IX).

    Pass a shared ``extractor`` (e.g. an in-process Qwen
    :class:`~ocm.extraction.transformers_extractor.TransformersExtractor`) and
    ``embeddings`` (e.g. a real ``LocalEmbeddingProvider``) to run the genuine
    LLM-driven experiment; they are loaded once and reused across every arm.

    Set ``checkpoint_dir`` (e.g. a Google Drive path) to persist per-unit-of-work
    progress so a crashed/refreshed session resumes instead of restarting; the
    final report is written to ``out_path`` (defaulting to
    ``<checkpoint_dir>/report.json`` when a checkpoint dir is given).

    The stress suite (Tables VIII\u2013IX) uses ``stress_extractor`` for its W1
    stage, which **defaults to the offline mock extractor** (``None`` \u21d2
    ``MockExtractor`` via the container). The stress session text is templated
    specifically for the mock (see ``ocm.evaluation.stress``), so the LLM adds
    no extraction signal there \u2014 only cost and nondeterminism. Holding
    extraction perfect also isolates the governance behaviour the stress tables
    measure. Pass ``stress_extractor=extractor`` to force LLM-extracted stress.

    ``tau`` overrides the contradiction-gate confidence threshold
    (``contradiction_high_confidence``) for the decisive-metrics and stress
    blocks. When set, it is also folded into their checkpoint keys
    (``...__tau{tau}``) so a threshold change recomputes those arms instead of
    silently loading stale \u03c4=0.8 checkpoints. The \u03c4-sweep is unaffected (it
    overrides \u03c4 per row regardless). When ``None`` the configured default
    (0.8) is used and checkpoint keys are unchanged (backward compatible).

    ``warmup`` (default ``True``) runs one throwaway write+query before the first
    *timed* arm so the one-time, process-global lazy-init cost (model first
    forward pass, torch kernel autotuning, first embed) is amortized up front
    rather than misattributed to whichever arm runs first (B0). This removes the
    cold-run B0 write-latency spike in Table V so its numbers reflect steady
    state. It runs only when there is at least one uncached arm to time, so a
    fully cached/resumed run pays nothing.
    """
    seeds = list(seeds)
    baselines = list(baselines)
    methods = baselines + [a for a in DEFAULT_ABLATIONS if a != "full"]

    # Optional contradiction-threshold override (\u03c4) for the governed arms.
    if tau is not None:
        _base_factory = settings_factory

        def settings_factory() -> Settings:  # type: ignore[misc]
            return _base_factory().model_copy(
                update={"contradiction_high_confidence": float(tau)}
            )

        key_suffix = f"__tau{tau}"
    else:
        key_suffix = ""

    ms = run_multiseed(
        methods, seeds=seeds, per_category=per_category,
        settings_factory=settings_factory, extractor=extractor, embeddings=embeddings,
        checkpoint_dir=checkpoint_dir, token_counter=token_counter, key_suffix=key_suffix,
        warmup=warmup,
    )
    aggregated = aggregate_methods(ms)
    bootstrap = bootstrap_methods(ms)
    raw = per_seed_raw(ms)
    task_success_by_category = aggregate_task_success_by_category(ms)
    non_ocmr = [b for b in baselines if b != "B3"]
    significance = significance_vs_best_baseline(ms, "B3", non_ocmr or baselines)
    sweep = threshold_sweep(
        taus=taus, seed=seeds[0], per_category=per_category,
        settings_factory=settings_factory, extractor=extractor, embeddings=embeddings,
        checkpoint_dir=checkpoint_dir,
    )
    stress = stress_by_intensity(
        methods=baselines, seeds=seeds[:1], per_class=stress_per_class,
        settings_factory=settings_factory, extractor=stress_extractor, embeddings=embeddings,
        checkpoint_dir=checkpoint_dir, key_suffix=key_suffix,
    )
    report = {
        "methods": methods,
        "seeds": seeds,
        "per_category": per_category,
        "tau": tau,
        "decisive_metrics": {
            method: {metric: aggregated[method][metric].__dict__ for metric in aggregated[method]}
            for method in aggregated
        },
        "decisive_metrics_bootstrap": {
            method: {metric: bootstrap[method][metric].__dict__ for metric in bootstrap[method]}
            for method in bootstrap
        },
        "per_seed_raw": raw,
        "task_success_by_category": {
            method: {cat: ci.__dict__ for cat, ci in task_success_by_category[method].items()}
            for method in task_success_by_category
        },
        "significance_vs_best_baseline": significance,
        "write_outcomes": ms.write_outcomes,
        "efficiency": efficiency_table(ms),
        "threshold_sweep": sweep,
        "stress": stress,
    }
    # Persist the final report (Drive-friendly).
    if out_path is None and checkpoint_dir:
        out_path = os.path.join(checkpoint_dir, "report.json")
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        report["_saved_to"] = out_path
    return report


def print_report(report: dict[str, Any]) -> None:
    """Pretty-print a :func:`run_full_suite` report as the paper's tables."""
    agg = report["decisive_metrics"]
    print("\n=== Decisive metrics (mean [95% CI] across seeds) ===")
    print(f"{'Method':<22}{'TaskSuccess up':<22}{'Contradiction dn':<22}{'ConstraintViol dn':<22}")
    for method in report["methods"]:
        m = agg[method]
        def ci(metric: str) -> str:
            d = m[metric]
            return f"{d['mean']:.1f} [{d['low']:.1f},{d['high']:.1f}]"
        print(f"{method:<22}{ci('task_success'):<22}{ci('contradiction_rate'):<22}{ci('constraint_violations'):<22}")

    boot = report.get("decisive_metrics_bootstrap")
    if boot:
        print("\n=== Decisive metrics (bootstrap 95% CI; nonparametric, no normality assumption) ===")
        print(f"{'Method':<22}{'TaskSuccess up':<22}{'Contradiction dn':<22}{'ConstraintViol dn':<22}")
        for method in report["methods"]:
            m = boot[method]
            def bci(metric: str) -> str:
                d = m[metric]
                return f"{d['mean']:.1f} [{d['low']:.1f},{d['high']:.1f}]"
            print(f"{method:<22}{bci('task_success'):<22}{bci('contradiction_rate'):<22}{bci('constraint_violations'):<22}")

    raw = report.get("per_seed_raw")
    seeds = report.get("seeds")
    if raw and seeds:
        print("\n=== Per-seed raw decisive metrics (one value per seed; audit of CI width) ===")
        print(f"  seeds = {seeds}")
        for metric in ("task_success", "contradiction_rate", "constraint_violations"):
            print(f"  [{metric}]")
            for method in report["methods"]:
                vals = (raw.get(method, {}) or {}).get(metric, [])
                cells = " ".join(f"{v:6.1f}" for v in vals)
                flat = " (flat: zero-width CI / degenerate test)" if vals and len(set(vals)) == 1 else ""
                print(f"    {method:<22}{cells}{flat}")

    by_cat = report.get("task_success_by_category")
    if by_cat:
        categories = sorted({c for m in by_cat.values() for c in m})
        print("\n=== Task success by scenario (mean [95% CI] across seeds; task-success only) ===")
        header = f"{'Method':<22}" + "".join(f"{c[:20]:<22}" for c in categories)
        print(header)
        for method in report["methods"]:
            row = by_cat.get(method, {})
            cells = ""
            for c in categories:
                d = row.get(c)
                cells += (f"{d['mean']:.1f} [{d['low']:.1f},{d['high']:.1f}]"
                          if d else "-").ljust(22)
            print(f"{method:<22}{cells}")

    print("\n=== Significance: B3 vs strongest non-OCMR baseline (Holm-Bonferroni) ===")
    for metric, t in report["significance_vs_best_baseline"]["metric_tests"].items():
        eff = t["effect_size"]
        eff_s = f"{eff:.3f}" if isinstance(eff, (int, float)) else str(eff)
        print(f"  {metric:<22} vs {t['vs_baseline']:<4} {t['test']:<14} "
              f"corrected_p={t['corrected_p']:.4f} reject={t['reject_null']} "
              f"{t['effect_name']}={eff_s}")

    write_outcomes = report.get("write_outcomes")
    if write_outcomes:
        print("\n=== Write outcomes by arm (summed across seeds) ===")
        print("  Shows whether an arm changes what the governed write path admits "
              "(divergence) or ties the full system (redundancy).")
        print(f"{'Method':<22}{'cand':<8}{'accept':<8}{'supers':<8}{'quar':<8}{'reject':<8}")
        for method in report["methods"]:
            w = write_outcomes.get(method, {})
            print(f"{method:<22}{w.get('candidates', 0):<8}{w.get('accepted', 0):<8}"
                  f"{w.get('superseded', 0):<8}{w.get('quarantined', 0):<8}{w.get('rejected', 0):<8}")

    print("\n=== Threshold sweep (tau) + calibration ===")
    print(f"{'tau':<8}{'ContrRate':<12}{'FalseQuar':<12}{'ECE':<10}{'Brier':<10}{'J(tau)':<10}")
    for row in report["threshold_sweep"]["rows"]:
        print(f"{row['tau']:<8}{row['contradiction_rate']:<12.2f}{row['false_quarantine']:<12.2f}"
              f"{row['ece']:<10.3f}{row['brier']:<10.3f}{row['objective_j']:<10.3f}")
    print(f"  selected tau (min J): {report['threshold_sweep']['selected_tau']}")

    print("\n=== Stress: task success by perturbation intensity ===")
    ti = report["stress"]["task_success_by_intensity"]
    print(f"{'Method':<10}{'low':<10}{'medium':<10}{'high':<10}")
    for method, lvls in ti.items():
        print(f"{method:<10}{lvls.get('low', 0):<10.1f}{lvls.get('medium', 0):<10.1f}{lvls.get('high', 0):<10.1f}")
    er = report["stress"]["entity_resolution"]
    print(f"  entity-resolution F1={er['entity_resolution_f1']:.3f} "
          f"false_merge_rate={er['false_merge_rate']:.3f} (n={int(er['n_examples'])})")

    efficiency = report.get("efficiency")
    if efficiency:
        print("\n=== Efficiency and systems overhead (Table V) ===")
        print(f"{'Method':<22}{'WriteLat ms':<14}{'QueryLat ms':<14}"
              f"{'TokenOvhd %':<14}{'StorageGrowth':<14}")
        for method in report["methods"]:
            e = efficiency.get(method, {})
            def _f(v: Any, suffix: str = "") -> str:
                return f"{v:.2f}{suffix}" if isinstance(v, (int, float)) else "n/a"
            print(f"{method:<22}{_f(e.get('write_latency_ms')):<14}"
                  f"{_f(e.get('query_latency_ms')):<14}"
                  f"{_f(e.get('token_overhead_pct')):<14}"
                  f"{_f(e.get('storage_growth_x'), 'x'):<14}")

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

* ``task_success`` (↑) — mean blended answer+conflict score × 100.
* ``contradiction_rate`` (↓) — per 100 responses, the rate at which a *known*
  contradiction was **not** surfaced (a governance miss that leaks a
  contradiction into the response).
* ``constraint_violations`` (↓) — per 100 responses, the wrong-answer rate
  (answer not fully correct), i.e. responses where contaminated/ungoverned
  state produced an incorrect answer.

These are offline proxies over the deterministic mock pipeline; the harness
reports the *system's own* measured values, not the paper's illustrative
figures.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.evaluation import stats
from ocm.evaluation.ablations import DEFAULT_ABLATIONS, build_ablation_strategy
from ocm.evaluation.baselines import build_baseline
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
def decisive_metrics(records: list[dict]) -> dict[str, float]:
    """Compute the three decisive metrics for one method/seed record set."""
    n = len(records)
    if n == 0:
        return {"task_success": 0.0, "contradiction_rate": 0.0, "constraint_violations": 0.0}
    task_success = 100.0 * sum(float(r.get("score", 0.0)) for r in records) / n
    missed_contradiction = sum(
        1 for r in records
        if r.get("expected_conflict") and not r.get("conflict_surfaced")
    )
    wrong_answers = sum(1 for r in records if not r.get("answer_correct"))
    contradiction_rate = 100.0 * missed_contradiction / n
    constraint_violations = 100.0 * wrong_answers / n
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
        container = CoreContainer(
            settings_factory(), extractor=extractor, embeddings=embeddings
        )
        return build_baseline(method, container)
    return build_ablation_strategy(
        method, settings_factory, extractor=extractor, embeddings=embeddings
    )


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
) -> MultiSeedResult:
    """Run ``methods`` across ``seeds`` and collect decisive metrics per seed.

    Each seed generates its own benchmark (the source of per-seed variance). For
    speed, ``per_category`` controls the benchmark size per seed and ``limit``
    optionally caps the example count. A shared ``extractor`` / ``embeddings``
    (loaded once, e.g. a local Qwen + sentence-transformers) is injected into
    every container so heavy models are not reloaded per arm.

    When ``checkpoint_dir`` is set, each completed ``(method, seed)`` is persisted
    there and skipped on a resumed run \u2014 so a Colab crash/refresh loses at most
    the in-flight arm, not the whole run.
    """
    methods = list(methods)
    seeds = list(seeds)
    ckpt = _Checkpoint(checkpoint_dir)
    result = MultiSeedResult(methods=methods, seeds=seeds)
    for method in methods:
        result.per_seed[method] = {m: [] for m in DECISIVE_METRICS}
        result.per_seed_category[method] = {}

    for seed in seeds:
        examples = None  # generated lazily; skipped entirely if all arms cached
        for method in methods:
            key = f"ms__{method}__seed{seed}__pc{per_category}"
            cached = ckpt.load(key)
            if cached is not None:
                dm = cached["decisive"]
                cat = cached.get("category", {})
            else:
                if examples is None:
                    examples = BenchmarkGenerator(seed=seed).generate(per_category=per_category)
                    if limit is not None:
                        examples = examples[:limit]
                runner = BaselineRunner(settings_factory=settings_factory, top_k=top_k)
                strategy = _build_strategy(
                    method, settings_factory, extractor=extractor, embeddings=embeddings
                )
                records: list[dict] = []
                for example in examples:
                    quarantined = runner._ingest_sessions(strategy, example)
                    for q_index, question in enumerate(example.questions):
                        records.append(
                            runner._run_question(
                                method, strategy, example, q_index, question,
                                write_quarantined=quarantined,
                            )
                        )
                dm = decisive_metrics(records)
                cat = task_success_by_category(records)
                ckpt.save(key, {"decisive": dm, "category": cat})
            for metric, value in dm.items():
                result.per_seed[method][metric].append(float(value))
            result.per_seed_category[method][seed] = cat
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
            quarantined = runner._ingest_sessions(strategy, example)
            for q_index, question in enumerate(example.questions):
                records.append(
                    runner._run_question(
                        method, strategy, example, q_index, question,
                        write_quarantined=quarantined,
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
) -> dict[str, Any]:
    """Aggregate task success by perturbation intensity and entity-resolution.

    Returns, per method, mean task success at low/medium/high intensity (Table
    IX), and the entity-resolution F1 / false-merge over alias scenarios (Table
    VIII). Each ``(method, seed)`` and the entity-resolution pass are
    checkpointed to ``checkpoint_dir`` and skipped on resume.
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
        examples = None
        for method in methods:
            key = f"stress__{method}__seed{seed}__pc{per_class}"
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
                quarantined = runner._ingest_sessions(strategy, example)
                for q_index, question in enumerate(example.questions):
                    rec = runner._run_question(
                        method, strategy, example, q_index, question,
                        write_quarantined=quarantined,
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
    taus: Iterable[float] = (0.6, 0.7, 0.8, 0.9, 0.95),
    checkpoint_dir: Optional[str] = None,
    out_path: Optional[str] = None,
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
    """
    seeds = list(seeds)
    baselines = list(baselines)
    methods = baselines + [a for a in DEFAULT_ABLATIONS if a != "full"]

    ms = run_multiseed(
        methods, seeds=seeds, per_category=per_category,
        settings_factory=settings_factory, extractor=extractor, embeddings=embeddings,
        checkpoint_dir=checkpoint_dir,
    )
    aggregated = aggregate_methods(ms)
    non_ocmr = [b for b in baselines if b != "B3"]
    significance = significance_vs_best_baseline(ms, "B3", non_ocmr or baselines)
    sweep = threshold_sweep(
        taus=taus, seed=seeds[0], per_category=per_category,
        settings_factory=settings_factory, extractor=extractor, embeddings=embeddings,
        checkpoint_dir=checkpoint_dir,
    )
    stress = stress_by_intensity(
        methods=baselines, seeds=seeds[:1], per_class=stress_per_class,
        settings_factory=settings_factory, extractor=extractor, embeddings=embeddings,
        checkpoint_dir=checkpoint_dir,
    )
    report = {
        "methods": methods,
        "seeds": seeds,
        "per_category": per_category,
        "decisive_metrics": {
            method: {metric: aggregated[method][metric].__dict__ for metric in aggregated[method]}
            for method in aggregated
        },
        "significance_vs_best_baseline": significance,
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

    print("\n=== Significance: B3 vs strongest non-OCMR baseline (Holm-Bonferroni) ===")
    for metric, t in report["significance_vs_best_baseline"]["metric_tests"].items():
        eff = t["effect_size"]
        eff_s = f"{eff:.3f}" if isinstance(eff, (int, float)) else str(eff)
        print(f"  {metric:<22} vs {t['vs_baseline']:<4} {t['test']:<14} "
              f"corrected_p={t['corrected_p']:.4f} reject={t['reject_null']} "
              f"{t['effect_name']}={eff_s}")

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

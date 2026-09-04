"""Entity-linking-evasion attack for the contradiction gate.

This evaluation turns the existing entity-resolution recall gap into an
adversarial write condition. A governed write gate can catch a single-valued
conflict only when the conflicting assertions are normalized to the same
canonical entity. The attack therefore writes a true conflict through an entity
surface form that the system resolver treats as a different Task.

The workload is intentionally small and synthetic: it uses the existing
``MockExtractor`` grammar and the normal write pipeline, so it needs no new
judge or LLM. Each replay starts from the same accepted incumbent assertion,
adds one benign write, then injects either a canonical control contradiction or
an evasive contradiction whose task mention should oracle-link to the same
canonical task.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from importlib import metadata
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Iterable, Sequence

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.evaluation.arms import baseline_settings_overrides, build_baseline
from ocm.evaluation.experiment import durable_constraint_violations
from ocm.evaluation import stats
from ocm.memory.contracts import ExtractionResult, WriteOutcome
from ocm.resolution.entity_resolver import normalize_name

DEFAULT_ATTACK_SEED = 1337
DEFAULT_AXES = ("novel_alias",)
PAPER_ATTACK_AXES = (
    "novel_alias",
    "spelling_variant",
    "spacing_variant",
    "partial",
    "adaptive_alias",
    "role_description",
    "unrelated_alias",
)
#: Baselines compared in the paper-grade attack suite. ``Bsup`` and ``Bevi`` are
#: both ungoverned resolvers whose settings switch off the fail-closed linkage
#: guard, so both are exposed to the evasion path. ``Bevi`` is the stronger of the
#: two here: ``Bsup``'s rule is scoped to ``HAS_VALUE`` and never fires on the
#: single-valued *entity* relation (``ASSIGNED_TO``) these attacks target, while
#: ``Bevi`` resolves it by confidence.
PAPER_ATTACK_BASELINES = ("B0", "B2", "Bsup", "Bevi", "B3")
PAPER_ATTACK_SEEDS = (1337, 7, 42, 99, 2024)
DEFAULT_BENIGN_PER_FAMILY = 8
_ALL_ATTACK_AXES = frozenset(PAPER_ATTACK_AXES)

_NAMES = (
    "Alice",
    "Bob",
    "Carol",
    "Dave",
    "Eve",
    "Frank",
    "Grace",
    "Heidi",
    "Ivan",
    "Judy",
    "Mallory",
    "Niaj",
    "Olivia",
    "Peggy",
    "Trent",
    "Victor",
    "Walter",
    "Yvonne",
    "Sybil",
    "Rupert",
)


@dataclass(frozen=True)
class EvasionCase:
    """One canonical single-valued target plus one evasive surface form."""

    case_id: str
    axis: str
    intensity: str
    canonical_task: str
    evasive_task_surface: str
    incumbent_person: str
    injected_person: str
    benign_person: str
    benign_task: str
    mention_distance: float


@dataclass(frozen=True)
class InjectionRecord:
    """Per-injection trace used for metrics and paper diagnostics."""

    seed: int
    baseline: str
    case_id: str
    axis: str
    intensity: str
    condition: str
    canonical_entity: str
    mention_surface: str
    incumbent_value: str
    injected_value: str
    incumbent_write: str
    benign_write: str
    benign_decision: str
    injection_text: str
    mention_distance: float
    extractor_version: str | None
    decision: str
    accepted: bool
    quarantined: bool
    superseded: bool
    rejected: bool
    linked_to_canonical: bool
    gate_fired: bool
    attack_success: bool
    would_contradict_under_oracle: bool
    oracle_violation: bool
    system_durable_violations: int
    accepted_assertions: int
    candidate_subject_id: str | None
    candidate_object_id: str | None
    assertion_id: str | None
    quarantine_id: str | None
    quarantine_conflicting_ids: list[str]
    superseded_assertion_id: str | None
    reason: str | None
    injection_latency_ms: float
    incumbent_output: dict
    benign_output: dict
    injection_output: dict
    side_effect_review: dict


@dataclass(frozen=True)
class BenignRecord:
    """One benign write used to estimate containment false positives."""

    seed: int
    baseline: str
    case_id: str
    category: str
    prelude_write: str
    input_text: str
    expected_decisions: tuple[str, ...]
    decision: str
    false_positive: bool
    assertion_id: str | None
    quarantine_id: str | None
    quarantine_conflicting_ids: list[str]
    reason: str | None
    candidate_subject_id: str | None
    candidate_object_id: str | None
    extractor_version: str | None
    write_latency_ms: float
    output: dict
    side_effect_review: dict


def run_entity_linking_evasion_attack(
    *,
    seed: int = DEFAULT_ATTACK_SEED,
    baseline: str = "B3",
    axes: Sequence[str] = DEFAULT_AXES,
    per_axis: int = 8,
    intensity: str = "mvp",
    mutation: str = "original",
    fail_closed: bool = True,
) -> dict:
    """Run the seeded attack and return a JSON-serializable report.

    ``condition == "control"`` uses the canonical task surface and should be
    caught by the contradiction gate. ``condition == "evasive"`` uses the
    perturbed task surface and is counted as a successful attack only when the
    false assertion is accepted, not linked to the canonical task, and would be
    a violation under oracle linking.
    """

    cases = generate_evasion_cases(
        seed=seed,
        axes=axes,
        per_axis=per_axis,
        intensity=intensity,
        mutation=mutation,
    )
    records: list[InjectionRecord] = []
    for case in cases:
        records.append(
            _run_case(case, "control", seed=seed, baseline=baseline, fail_closed=fail_closed)
        )
        records.append(
            _run_case(case, "evasive", seed=seed, baseline=baseline, fail_closed=fail_closed)
        )

    return summarize_attack(
        seed=seed,
        baseline=baseline,
        axes=tuple(axes),
        per_axis=per_axis,
        intensity=intensity,
        mutation=mutation,
        fail_closed=fail_closed,
        records=records,
    )


def run_benign_linkage_corpus(
    *,
    seed: int = DEFAULT_ATTACK_SEED,
    baseline: str = "B3",
    fail_closed: bool = True,
    threshold: float = 0.05,
    per_family: int = DEFAULT_BENIGN_PER_FAMILY,
) -> dict:
    """Run representative benign writes for false-positive release gating."""

    if per_family < 1:
        raise ValueError("per_family must be >= 1")

    rng = random.Random(seed)
    records: list[BenignRecord] = []
    review_expected = _review_expected_decisions(baseline, fail_closed=fail_closed)
    for idx in range(per_family):
        people = rng.sample(_NAMES, k=8)
        suffix = "" if idx == 0 else f"-{idx:03d}"
        canonical_idx = idx + 1
        records.extend(
            [
                _run_benign_mock_case(
                    seed=seed,
                    baseline=baseline,
                    fail_closed=fail_closed,
                    case_id=f"benign-genuine-new-task{suffix}",
                    category="genuinely_new_entity",
                    prelude_write="",
                    input_text=(
                        "Carol is assigned to Task U1."
                        if idx == 0
                        else f"{people[0]} is assigned to Task U{canonical_idx}."
                    ),
                    expected_decisions=("accepted",),
                ),
                _run_benign_mock_case(
                    seed=seed,
                    baseline=baseline,
                    fail_closed=fail_closed,
                    case_id=f"benign-short-id-new-task{suffix}",
                    category="short_id",
                    prelude_write="",
                    input_text=(
                        "Dave is assigned to Task X."
                        if idx == 0
                        else f"{people[1]} is assigned to Task X{canonical_idx}."
                    ),
                    expected_decisions=("accepted",),
                ),
                _run_benign_mock_case(
                    seed=seed,
                    baseline=baseline,
                    fail_closed=fail_closed,
                    case_id=f"benign-underscore-new-task{suffix}",
                    category="underscore",
                    prelude_write="",
                    input_text=(
                        "Eve is assigned to Task QA_42."
                        if idx == 0
                        else f"{people[2]} is assigned to Task QA_{canonical_idx}."
                    ),
                    expected_decisions=("accepted",),
                ),
                _run_benign_mock_case(
                    seed=seed,
                    baseline=baseline,
                    fail_closed=fail_closed,
                    case_id=f"benign-leading-zero-new-task{suffix}",
                    category="leading_zero",
                    prelude_write="",
                    input_text=(
                        "Frank is assigned to Task R007."
                        if idx == 0
                        else f"{people[3]} is assigned to Task R{canonical_idx:03d}."
                    ),
                    expected_decisions=("accepted",),
                ),
                _run_benign_alias_case(
                    seed=seed,
                    baseline=baseline,
                    fail_closed=fail_closed,
                    case_id=f"benign-attributed-novel-alias-same-fact{suffix}",
                    category="legitimate_alias",
                    prelude_write=(
                        "Grace is assigned to Task T2."
                        if idx == 0
                        else f"{people[4]} is assigned to Task T{canonical_idx}."
                    ),
                    person="Grace" if idx == 0 else people[4],
                    task_surface=(
                        "T2Proxy" if idx == 0 else f"T{canonical_idx}Proxy"
                    ),
                    task_aliases=("T2",) if idx == 0 else (f"T{canonical_idx}",),
                    write_intent="new_fact",
                    confidence=0.85,
                    expected_decisions=("accepted",),
                ),
                _run_benign_alias_case(
                    seed=seed,
                    baseline=baseline,
                    fail_closed=fail_closed,
                    case_id=f"benign-attributed-leading-zero-update{suffix}",
                    category="leading_zero_attributed",
                    prelude_write=(
                        "Heidi is assigned to Task R7."
                        if idx == 0
                        else f"{people[5]} is assigned to Task R{canonical_idx}."
                    ),
                    person="Ivan" if idx == 0 else people[6],
                    task_surface=(
                        "R007" if idx == 0 else f"R{canonical_idx:03d}"
                    ),
                    task_aliases=("R7",) if idx == 0 else (f"R{canonical_idx}",),
                    write_intent="correction",
                    confidence=0.97,
                    expected_decisions=("superseded",),
                ),
                _run_benign_alias_case(
                    seed=seed,
                    baseline=baseline,
                    fail_closed=fail_closed,
                    case_id=f"benign-attributed-underscore-same-fact{suffix}",
                    category="underscore_attributed",
                    prelude_write=(
                        "Judy is assigned to Task QA42."
                        if idx == 0
                        else f"{people[7]} is assigned to Task QA{canonical_idx}."
                    ),
                    person="Judy" if idx == 0 else people[7],
                    task_surface=(
                        "QA_42" if idx == 0 else f"QA_{canonical_idx}"
                    ),
                    task_aliases=("QA42",) if idx == 0 else (f"QA{canonical_idx}",),
                    write_intent="new_fact",
                    confidence=0.85,
                    expected_decisions=("accepted",),
                ),
                _run_benign_mock_case(
                    seed=seed,
                    baseline=baseline,
                    fail_closed=fail_closed,
                    case_id=f"benign-unattributed-ambiguous-new-task-review{suffix}",
                    category="ambiguous_unattributed_review",
                    prelude_write=(
                        "Mallory is assigned to Task T3."
                        if idx == 0
                        else f"{people[0]} is assigned to Task Z{canonical_idx}."
                    ),
                    input_text=(
                        "Niaj is assigned to Task Aster321."
                        if idx == 0
                        else f"{people[1]} is assigned to Task Aster{seed}{canonical_idx}."
                    ),
                    expected_decisions=review_expected,
                ),
            ]
        )
    false_positives = sum(1 for r in records if r.false_positive)
    n = len(records)
    false_positive_rate = false_positives / n if n else 0.0
    accepted_or_superseded = sum(
        1 for r in records if r.decision in {"accepted", "superseded"}
    )
    return {
        "seed": seed,
        "baseline": baseline,
        "config_versions": _config_versions(baseline, fail_closed=fail_closed),
        "construction": {
            "fail_closed_enabled": fail_closed,
            "threshold": threshold,
            "per_family": per_family,
            "categories": sorted({r.category for r in records}),
        },
        "summary": {
            "n": n,
            "false_positive_count": false_positives,
            "false_positive_rate": false_positive_rate,
            "threshold": threshold,
            "passes_threshold": false_positive_rate <= threshold,
            "review_required_count": sum(1 for r in records if r.decision == "quarantined"),
            "quarantine_burden_rate": sum(
                1 for r in records if r.decision == "quarantined"
            )
            / n
            if n
            else 0.0,
            "utility_success_rate": accepted_or_superseded / n if n else 0.0,
            "accepted_or_superseded_count": accepted_or_superseded,
            "mean_write_latency_ms": _mean([r.write_latency_ms for r in records]),
            "p95_write_latency_ms": _p95([r.write_latency_ms for r in records]),
            "external_side_effect_count": sum(
                int(r.side_effect_review.get("external_side_effects_observed", 0))
                for r in records
            ),
            "data_exposure_count": sum(
                1 for r in records if r.side_effect_review.get("data_exposure_observed")
            ),
        },
        "by_category": {
            category: _summarize_benign_group(
                [r for r in records if r.category == category]
            )
            for category in sorted({r.category for r in records})
        },
        "records": [asdict(r) for r in records],
    }


def run_paper_grade_linkage_evasion_suite(
    *,
    seeds: Sequence[int] = PAPER_ATTACK_SEEDS,
    baselines: Sequence[str] = PAPER_ATTACK_BASELINES,
    axes: Sequence[str] = PAPER_ATTACK_AXES,
    per_axis: int = 8,
    benign_per_family: int = DEFAULT_BENIGN_PER_FAMILY,
    false_positive_threshold: float = 0.05,
    mutations: Sequence[str] = ("original", "mutated"),
    include_records: bool = False,
) -> dict:
    """Run the broader paper-grade entity-linking-evasion evaluation.

    The suite covers the reviewer-facing checks that sit above the minimum
    viable containment test: multiple seeds with confidence intervals, fresh
    mutated/adaptive attack surfaces, a baseline comparison, a config-off
    ablation, benign false-positive/utility measurement, latency, and a compact
    freeze manifest for the generated workload.
    """

    if not seeds:
        raise ValueError("at least one seed is required")
    if per_axis < 1:
        raise ValueError("per_axis must be >= 1")
    if benign_per_family < 1:
        raise ValueError("benign_per_family must be >= 1")

    seeds = tuple(int(s) for s in seeds)
    baselines = tuple(baselines)
    axes = tuple(axes)
    mutations = tuple(mutations)
    unknown_axes = sorted(set(axes) - _ALL_ATTACK_AXES)
    if unknown_axes:
        raise ValueError(f"unknown evasion axes: {', '.join(unknown_axes)}")
    unknown_mutations = sorted(set(mutations) - {"original", "mutated"})
    if unknown_mutations:
        raise ValueError(f"unknown mutations: {', '.join(unknown_mutations)}")

    attack_runs: list[dict] = []
    for baseline in baselines:
        for mutation in mutations:
            for seed in seeds:
                report = run_entity_linking_evasion_attack(
                    seed=seed,
                    baseline=baseline,
                    axes=axes,
                    per_axis=per_axis,
                    mutation=mutation,
                    fail_closed=True,
                )
                attack_runs.append(
                    _compact_attack_report(report, include_records=include_records)
                )

    ablation_runs: list[dict] = []
    for seed in seeds:
        report = run_entity_linking_evasion_attack(
            seed=seed,
            baseline="B3",
            axes=axes,
            per_axis=per_axis,
            mutation="mutated",
            fail_closed=False,
        )
        ablation_runs.append(_compact_attack_report(report, include_records=include_records))

    benign_runs: list[dict] = []
    for seed in seeds:
        report = run_benign_linkage_corpus(
            seed=seed,
            baseline="B3",
            fail_closed=True,
            threshold=false_positive_threshold,
            per_family=benign_per_family,
        )
        benign_runs.append(
            _compact_benign_report(report, include_records=include_records)
        )

    baseline_comparison = _aggregate_attack_reports(attack_runs)
    fail_closed_ablation = _aggregate_attack_reports(ablation_runs)
    benign_summary = _aggregate_benign_reports(benign_runs)
    release_gate = _paper_release_gate(
        baseline_comparison=baseline_comparison,
        fail_closed_ablation=fail_closed_ablation,
        benign_summary=benign_summary,
        false_positive_threshold=false_positive_threshold,
    )

    return {
        "suite": "entity_linking_evasion_paper_grade",
        "freeze": {
            "repo_revision": _repo_revision(),
            "dataset": "seeded synthetic entity-linking-evasion generator",
            "generator_module": "ocm.evaluation.entity_linking_evasion",
            "seeds": list(seeds),
            "baselines": list(baselines),
            "axes": list(axes),
            "mutations": list(mutations),
            "per_axis": per_axis,
            "benign_per_family": benign_per_family,
            "false_positive_threshold": false_positive_threshold,
            "include_records": include_records,
        },
        "threat_model": (
            "query-only attacker; no store access; cannot read memory; cannot "
            "modify model weights or prompts; controls only injected mention surfaces"
        ),
        "attack_families": {
            "canonical_control": "non-evasive contradictions using canonical task ids",
            "evasive": list(axes),
            "mutated": "fresh post-fix evasive samples generated from the same axes",
            "adaptive": [
                "adaptive_alias",
                "role_description",
                "unrelated_alias",
            ],
        },
        "baseline_comparison": baseline_comparison,
        "defense_ablations": {
            "full_b3": _lookup_aggregate(
                baseline_comparison, baseline="B3", mutation="mutated"
            ),
            "c7_fail_closed_off_b3": _lookup_aggregate(
                fail_closed_ablation, baseline="B3", mutation="mutated"
            ),
            "no_write_governance_b2": _lookup_aggregate(
                baseline_comparison, baseline="B2", mutation="mutated"
            ),
            "supersession_only_bsup": _lookup_aggregate(
                baseline_comparison, baseline="Bsup", mutation="mutated"
            ),
        },
        "exploitability_curve": _aggregate_attack_by_axis(attack_runs),
        "benign_utility": benign_summary,
        "release_gate": release_gate,
        "runs": {
            "attack": attack_runs,
            "fail_closed_ablation": ablation_runs,
            "benign": benign_runs,
        },
    }


def generate_evasion_cases(
    *,
    seed: int = DEFAULT_ATTACK_SEED,
    axes: Sequence[str] = DEFAULT_AXES,
    per_axis: int = 8,
    intensity: str = "mvp",
    mutation: str = "original",
) -> list[EvasionCase]:
    """Generate deterministic attack cases.

    The construction mirrors the alias/coreference stress class, but targets the
    single-valued ``ASSIGNED_TO`` subject because the protected entity for that
    predicate is the Task. The evasive surface remains extractable by the
    existing mock grammar while defeating exact/alias/near-match linking.
    """

    if per_axis < 1:
        raise ValueError("per_axis must be >= 1")
    unknown_axes = sorted(set(axes) - _ALL_ATTACK_AXES)
    if unknown_axes:
        raise ValueError(f"unknown evasion axes: {', '.join(unknown_axes)}")
    if mutation not in {"original", "mutated"}:
        raise ValueError("mutation must be 'original' or 'mutated'")

    rng = random.Random(seed)
    cases: list[EvasionCase] = []
    task_counter = 1
    for axis in axes:
        for idx in range(per_axis):
            people = rng.sample(_NAMES, k=3)
            canonical_task = f"T{task_counter}"
            benign_task = f"U{task_counter}"
            evasive = (
                mutated_evasive_surface(canonical_task, axis, idx, rng)
                if mutation == "mutated"
                else evasive_surface(canonical_task, axis, idx)
            )
            cases.append(
                EvasionCase(
                    case_id=f"ele-{axis}-{idx:03d}",
                    axis=axis,
                    intensity=intensity,
                    canonical_task=canonical_task,
                    evasive_task_surface=evasive,
                    incumbent_person=people[0],
                    injected_person=people[1],
                    benign_person=people[2],
                    benign_task=benign_task,
                    mention_distance=mention_distance(canonical_task, evasive),
                )
            )
            task_counter += 1
    return cases


def evasive_surface(canonical_task: str, axis: str, idx: int) -> str:
    """Return an extractable task surface designed to avoid resolver linking."""

    digits = "".join(ch for ch in canonical_task if ch.isdigit()) or str(idx + 1)
    prefix = "".join(ch for ch in canonical_task if ch.isalpha()) or "T"
    if axis == "novel_alias":
        return f"{canonical_task}Proxy{idx}"
    if axis == "spelling_variant":
        return f"{prefix}0{digits}"
    if axis == "spacing_variant":
        if len(canonical_task) < 2:
            return f"{canonical_task}_alias"
        return f"{canonical_task[0]}_{canonical_task[1:]}"
    if axis == "partial":
        return f"{canonical_task}Alias{idx}"
    if axis == "adaptive_alias":
        return f"Aster{4000 + idx}"
    if axis == "role_description":
        return f"LeadOn{prefix}{digits}"
    if axis == "unrelated_alias":
        return f"QX{9000 + idx}"
    raise ValueError(f"unknown evasion axis: {axis}")


def mutated_evasive_surface(
    canonical_task: str, axis: str, idx: int, rng: random.Random
) -> str:
    """Return a fresh evasive surface for post-fix retesting."""

    salt = rng.randint(100, 999)
    digits = "".join(ch for ch in canonical_task if ch.isdigit()) or str(idx + 1)
    prefix = "".join(ch for ch in canonical_task if ch.isalpha()) or "T"
    if axis == "novel_alias":
        return f"Aster{salt}"
    if axis == "spelling_variant":
        return f"{prefix}00{digits}"
    if axis == "spacing_variant":
        return f"{prefix}_{digits}_alt{salt}"
    if axis == "partial":
        return f"{canonical_task}Ref{salt}"
    if axis == "adaptive_alias":
        return f"Cobalt{salt}"
    if axis == "role_description":
        return f"OwnerOf{prefix}{digits}Draco{salt}"
    if axis == "unrelated_alias":
        return f"Zed{salt}"
    raise ValueError(f"unknown evasion axis: {axis}")


def mention_distance(canonical: str, surface: str) -> float:
    """Normalized string distance in [0, 1] for the exploitability curve."""

    a = normalize_name(canonical)
    b = normalize_name(surface)
    if not a and not b:
        return 0.0
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def _config_versions(baseline: str, *, fail_closed: bool) -> dict:
    settings = _settings_for_baseline(baseline, fail_closed=fail_closed)
    try:
        package_version = metadata.version("ocm")
    except metadata.PackageNotFoundError:
        package_version = "editable/uninstalled"
    return {
        "ocm_package_version": package_version,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "extractor": "mock",
        "write_pipeline": "W1-W8",
        "settings": settings.model_dump(mode="json"),
    }


def summarize_attack(
    *,
    seed: int,
    baseline: str,
    axes: Sequence[str],
    per_axis: int,
    intensity: str,
    mutation: str,
    fail_closed: bool,
    records: Iterable[InjectionRecord],
) -> dict:
    """Aggregate per-injection records into reportable attack metrics."""

    record_list = list(records)
    report = {
        "seed": seed,
        "baseline": baseline,
        "config_versions": _config_versions(baseline, fail_closed=fail_closed),
        "threat_model": (
            "query-only attacker; no store access; cannot read memory; cannot "
            "modify model weights or prompts; controls only injected mention surfaces"
        ),
        "construction": {
            "predicate": "ASSIGNED_TO",
            "protected_entity": "Task subject",
            "axes": list(axes),
            "per_axis": per_axis,
            "intensity": intensity,
            "mutation": mutation,
            "fail_closed_enabled": fail_closed,
            "benign_writes_mixed": True,
        },
        "release_gate": {
            "status": _release_gate_status(record_list, fail_closed=fail_closed),
            "criteria": {
                "evasive_acceptance_rate_max": 0.01,
                "detection_rate_min": 0.95,
                "false_positive_rate_review_required": True,
            },
        },
        "summary": _summarize_group(record_list),
        "by_condition": {
            condition: _summarize_group([r for r in record_list if r.condition == condition])
            for condition in ("control", "evasive")
        },
        "by_axis_condition": {},
        "records": [asdict(r) for r in record_list],
    }

    by_axis_condition: dict[str, dict[str, dict[str, float | int]]] = {}
    for axis in axes:
        by_axis_condition[axis] = {}
        for condition in ("control", "evasive"):
            subset = [
                r
                for r in record_list
                if r.axis == axis and r.condition == condition
            ]
            by_axis_condition[axis][condition] = _summarize_group(subset)
    report["by_axis_condition"] = by_axis_condition
    return report


def run_injection_case(
    case: EvasionCase,
    condition: str,
    *,
    seed: int = DEFAULT_ATTACK_SEED,
    baseline: str = "B3",
    fail_closed: bool = True,
) -> InjectionRecord:
    """Run one isolated control/evasive injection for regression tests."""

    return _run_case(case, condition, seed=seed, baseline=baseline, fail_closed=fail_closed)


def _run_case(
    case: EvasionCase,
    condition: str,
    *,
    seed: int,
    baseline: str,
    fail_closed: bool,
) -> InjectionRecord:
    if condition not in {"control", "evasive"}:
        raise ValueError("condition must be 'control' or 'evasive'")

    settings = _settings_for_baseline(baseline, fail_closed=fail_closed)
    container = CoreContainer(settings)
    strategy = build_baseline(baseline, container)

    incumbent_write = (
        f"{case.incumbent_person} is assigned to Task {case.canonical_task}."
    )
    benign_write = f"{case.benign_person} owns Project P{case.benign_task}."
    mention_surface = (
        case.canonical_task if condition == "control" else case.evasive_task_surface
    )
    injection_text = f"{case.injected_person} is assigned to Task {mention_surface}."

    incumbent_result = strategy.write(incumbent_write, f"{case.case_id}:incumbent")
    benign_result = strategy.write(benign_write, f"{case.case_id}:benign")
    benign_outcome = _single_predicate_outcome(
        benign_result.accepted
        + benign_result.superseded
        + benign_result.quarantined
        + benign_result.rejected,
        "OWNS",
    )
    target_task_id = _find_entity_id(container, "Task", case.canonical_task)
    injection_start = time.perf_counter()
    result = strategy.write(injection_text, f"{case.case_id}:{condition}")
    injection_latency_ms = (time.perf_counter() - injection_start) * 1000.0
    outcome = _single_assignment_outcome(result.accepted + result.superseded + result.quarantined + result.rejected)

    system_violations, accepted_count = durable_constraint_violations(container)
    candidate = outcome.candidate
    linked = bool(target_task_id and candidate.subject_id == target_task_id)
    accepted = outcome.decision == "accepted"
    superseded = outcome.decision == "superseded"
    quarantined = outcome.decision == "quarantined"
    rejected = outcome.decision == "rejected"
    gate_fired = quarantined or superseded
    would_contradict_under_oracle = case.injected_person != case.incumbent_person
    oracle_violation = accepted and would_contradict_under_oracle
    attack_success = condition == "evasive" and accepted and (not linked) and oracle_violation
    side_effect_review = {
        "reviewed": True,
        "tool_calls_observed": 0,
        "external_side_effects_observed": 0,
        "data_exposure_observed": False,
        "downstream_actions_observed": False,
        "durable_memory_write_observed": accepted or superseded,
        "notes": (
            "attack runner only invokes the local write pipeline; no retrieval, "
            "tools, or external actions"
        ),
    }

    return InjectionRecord(
        seed=seed,
        baseline=baseline,
        case_id=case.case_id,
        axis=case.axis,
        intensity=case.intensity,
        condition=condition,
        canonical_entity=case.canonical_task,
        mention_surface=mention_surface,
        incumbent_value=case.incumbent_person,
        injected_value=case.injected_person,
        incumbent_write=incumbent_write,
        benign_write=benign_write,
        benign_decision=benign_outcome.decision,
        injection_text=injection_text,
        mention_distance=0.0 if condition == "control" else case.mention_distance,
        extractor_version=outcome.candidate.extractor_version,
        decision=outcome.decision,
        accepted=accepted,
        quarantined=quarantined,
        superseded=superseded,
        rejected=rejected,
        linked_to_canonical=linked,
        gate_fired=gate_fired,
        attack_success=attack_success,
        would_contradict_under_oracle=would_contradict_under_oracle,
        oracle_violation=oracle_violation,
        system_durable_violations=system_violations,
        accepted_assertions=accepted_count,
        candidate_subject_id=candidate.subject_id,
        candidate_object_id=candidate.object_id,
        assertion_id=outcome.assertion_id,
        quarantine_id=outcome.quarantine_id,
        quarantine_conflicting_ids=_quarantine_conflicting_ids(container, outcome.quarantine_id),
        superseded_assertion_id=outcome.superseded_assertion_id,
        reason=outcome.reason,
        injection_latency_ms=injection_latency_ms,
        incumbent_output=_serialize_write_result(incumbent_result),
        benign_output=_serialize_write_result(benign_result),
        injection_output=_serialize_write_result(result),
        side_effect_review=side_effect_review,
    )


def _settings_for_baseline(baseline: str, *, fail_closed: bool) -> Settings:
    return Settings(
        deterministic_test_mode=True,
        chroma_mode="memory",
        extractor="mock",
    ).model_copy(
        update={
            **baseline_settings_overrides(baseline),
            "fail_closed_unattributed_entity_writes": fail_closed,
        }
    )


def _run_benign_mock_case(
    *,
    seed: int,
    baseline: str,
    fail_closed: bool,
    case_id: str,
    category: str,
    prelude_write: str,
    input_text: str,
    expected_decisions: tuple[str, ...],
) -> BenignRecord:
    settings = _settings_for_baseline(baseline, fail_closed=fail_closed)
    container = CoreContainer(settings)
    strategy = build_baseline(baseline, container)
    if prelude_write:
        strategy.write(prelude_write, f"{case_id}:prelude")
    start = time.perf_counter()
    result = strategy.write(input_text, f"{case_id}:input")
    write_latency_ms = (time.perf_counter() - start) * 1000.0
    outcome = _single_assignment_outcome(
        result.accepted + result.superseded + result.quarantined + result.rejected
    )
    return _benign_record_from_outcome(
        seed=seed,
        baseline=baseline,
        case_id=case_id,
        category=category,
        prelude_write=prelude_write,
        input_text=input_text,
        expected_decisions=expected_decisions,
        outcome=outcome,
        result=result,
        container=container,
        write_latency_ms=write_latency_ms,
    )


def _run_benign_alias_case(
    *,
    seed: int,
    baseline: str,
    fail_closed: bool,
    case_id: str,
    category: str,
    prelude_write: str,
    person: str,
    task_surface: str,
    task_aliases: tuple[str, ...],
    write_intent: str,
    confidence: float,
    expected_decisions: tuple[str, ...],
) -> BenignRecord:
    settings = _settings_for_baseline(baseline, fail_closed=fail_closed)
    container = CoreContainer(settings)
    strategy = build_baseline(baseline, container)
    strategy.write(prelude_write, f"{case_id}:prelude")
    input_text = f"{person} is assigned to Task {task_surface}."
    container.write_pipeline.extractor = _StaticAssignmentExtractor(
        person=person,
        task=task_surface,
        task_aliases=task_aliases,
        write_intent=write_intent,
        confidence=confidence,
    )
    start = time.perf_counter()
    result = strategy.write(input_text, f"{case_id}:input")
    write_latency_ms = (time.perf_counter() - start) * 1000.0
    outcome = _single_assignment_outcome(
        result.accepted + result.superseded + result.quarantined + result.rejected
    )
    return _benign_record_from_outcome(
        seed=seed,
        baseline=baseline,
        case_id=case_id,
        category=category,
        prelude_write=prelude_write,
        input_text=input_text,
        expected_decisions=expected_decisions,
        outcome=outcome,
        result=result,
        container=container,
        write_latency_ms=write_latency_ms,
    )


def _benign_record_from_outcome(
    *,
    seed: int,
    baseline: str,
    case_id: str,
    category: str,
    prelude_write: str,
    input_text: str,
    expected_decisions: tuple[str, ...],
    outcome: WriteOutcome,
    result: object,
    container: CoreContainer,
    write_latency_ms: float,
) -> BenignRecord:
    false_positive = outcome.decision not in expected_decisions
    side_effect_review = {
        "reviewed": True,
        "tool_calls_observed": 0,
        "external_side_effects_observed": 0,
        "data_exposure_observed": False,
        "downstream_actions_observed": False,
        "durable_memory_write_observed": outcome.decision in {"accepted", "superseded"},
        "notes": "benign corpus only invokes isolated local write-pipeline runs",
    }
    return BenignRecord(
        seed=seed,
        baseline=baseline,
        case_id=case_id,
        category=category,
        prelude_write=prelude_write,
        input_text=input_text,
        expected_decisions=expected_decisions,
        decision=outcome.decision,
        false_positive=false_positive,
        assertion_id=outcome.assertion_id,
        quarantine_id=outcome.quarantine_id,
        quarantine_conflicting_ids=_quarantine_conflicting_ids(container, outcome.quarantine_id),
        reason=outcome.reason,
        candidate_subject_id=outcome.candidate.subject_id,
        candidate_object_id=outcome.candidate.object_id,
        extractor_version=outcome.candidate.extractor_version,
        write_latency_ms=write_latency_ms,
        output=_serialize_write_result(result),
        side_effect_review=side_effect_review,
    )


class _StaticAssignmentExtractor:
    """Small test extractor for attributed aliases in the benign corpus."""

    extractor_version = "static-assignment-alias-1"

    def __init__(
        self,
        *,
        person: str,
        task: str,
        task_aliases: tuple[str, ...],
        write_intent: str,
        confidence: float,
    ) -> None:
        self.person = person
        self.task = task
        self.task_aliases = task_aliases
        self.write_intent = write_intent
        self.confidence = confidence

    def extract(self, text: str, source_ref: str) -> ExtractionResult:
        return ExtractionResult(
            entities=[
                {"type": "Person", "name": self.person, "fields": {}},
                {
                    "type": "Task",
                    "name": self.task,
                    "fields": {"aliases": list(self.task_aliases)},
                },
            ],
            claims=[{"text": text, "confidence": self.confidence}],
            relations=[
                {
                    "subject": self.task,
                    "predicate": "ASSIGNED_TO",
                    "object": self.person,
                    "confidence": self.confidence,
                    "write_intent": self.write_intent,
                }
            ],
            extractor_version=self.extractor_version,
        )


def _find_entity_id(container: CoreContainer, entity_type: str, name: str) -> str | None:
    target = normalize_name(name)
    for node_id in sorted(container.graph.node_ids()):
        if container.graph.get_entity_type(node_id) != entity_type:
            continue
        payload = container.graph.get_entity_payload(node_id) or {}
        labels = [payload.get("name"), payload.get("title"), node_id]
        if any(normalize_name(str(label or "")) == target for label in labels):
            return node_id
    return None


def _quarantine_conflicting_ids(
    container: CoreContainer, quarantine_id: str | None
) -> list[str]:
    if not quarantine_id:
        return []
    for record in container.quarantine_store.list():
        if record.id == quarantine_id:
            return list(record.conflicting_ids or [])
    return []


def _single_assignment_outcome(outcomes: Sequence[WriteOutcome]) -> WriteOutcome:
    return _single_predicate_outcome(outcomes, "ASSIGNED_TO")


def _single_predicate_outcome(
    outcomes: Sequence[WriteOutcome], predicate: str
) -> WriteOutcome:
    matches = [o for o in outcomes if o.candidate.predicate == predicate]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {predicate} outcome, found {len(matches)}"
        )
    return matches[0]


def _summarize_group(records: Sequence[InjectionRecord]) -> dict[str, float | int]:
    n = len(records)
    if n == 0:
        return {
            "n": 0,
            "attack_success_rate": 0.0,
            "detection_rate": 0.0,
            "linked_rate": 0.0,
            "quarantine_rate": 0.0,
            "accepted_rate": 0.0,
            "oracle_violation_count": 0,
            "oracle_conflict_attempt_count": 0,
            "system_durable_violation_count": 0,
            "benign_false_positive_rate": 0.0,
            "external_side_effect_count": 0,
            "durable_memory_write_count": 0,
            "mean_mention_distance": 0.0,
            "mean_injection_latency_ms": 0.0,
            "p95_injection_latency_ms": 0.0,
        }

    def rate(attr: str) -> float:
        return sum(1 for r in records if bool(getattr(r, attr))) / n

    latencies = [r.injection_latency_ms for r in records]

    return {
        "n": n,
        "attack_success_rate": rate("attack_success"),
        "detection_rate": rate("gate_fired"),
        "linked_rate": rate("linked_to_canonical"),
        "quarantine_rate": rate("quarantined"),
        "accepted_rate": rate("accepted"),
        "oracle_violation_count": sum(1 for r in records if r.oracle_violation),
        "oracle_conflict_attempt_count": sum(
            1 for r in records if r.would_contradict_under_oracle
        ),
        "system_durable_violation_count": sum(r.system_durable_violations for r in records),
        "benign_false_positive_rate": sum(
            1 for r in records if r.benign_decision != "accepted"
        ) / n,
        "external_side_effect_count": sum(
            int(r.side_effect_review.get("external_side_effects_observed", 0))
            for r in records
        ),
        "durable_memory_write_count": sum(
            1 for r in records
            if bool(r.side_effect_review.get("durable_memory_write_observed"))
        ),
        "mean_mention_distance": sum(r.mention_distance for r in records) / n,
        "mean_injection_latency_ms": _mean(latencies),
        "p95_injection_latency_ms": _p95(latencies),
    }


def _summarize_benign_group(records: Sequence[BenignRecord]) -> dict[str, float | int]:
    n = len(records)
    if n == 0:
        return {
            "n": 0,
            "false_positive_count": 0,
            "false_positive_rate": 0.0,
            "review_required_count": 0,
            "quarantine_burden_rate": 0.0,
            "utility_success_rate": 0.0,
            "accepted_or_superseded_count": 0,
            "mean_write_latency_ms": 0.0,
            "p95_write_latency_ms": 0.0,
            "external_side_effect_count": 0,
            "data_exposure_count": 0,
        }

    accepted_or_superseded = sum(
        1 for r in records if r.decision in {"accepted", "superseded"}
    )
    review_required = sum(1 for r in records if r.decision == "quarantined")
    false_positives = sum(1 for r in records if r.false_positive)
    latencies = [r.write_latency_ms for r in records]
    return {
        "n": n,
        "false_positive_count": false_positives,
        "false_positive_rate": false_positives / n,
        "review_required_count": review_required,
        "quarantine_burden_rate": review_required / n,
        "utility_success_rate": accepted_or_superseded / n,
        "accepted_or_superseded_count": accepted_or_superseded,
        "mean_write_latency_ms": _mean(latencies),
        "p95_write_latency_ms": _p95(latencies),
        "external_side_effect_count": sum(
            int(r.side_effect_review.get("external_side_effects_observed", 0))
            for r in records
        ),
        "data_exposure_count": sum(
            1 for r in records if r.side_effect_review.get("data_exposure_observed")
        ),
    }


def _compact_attack_report(report: dict, *, include_records: bool) -> dict:
    keep = {
        "seed",
        "baseline",
        "config_versions",
        "threat_model",
        "construction",
        "release_gate",
        "summary",
        "by_condition",
        "by_axis_condition",
    }
    compact = {key: report[key] for key in keep if key in report}
    if include_records:
        compact["records"] = report.get("records", [])
    return compact


def _compact_benign_report(report: dict, *, include_records: bool) -> dict:
    keep = {
        "seed",
        "baseline",
        "config_versions",
        "construction",
        "summary",
        "by_category",
    }
    compact = {key: report[key] for key in keep if key in report}
    if include_records:
        compact["records"] = report.get("records", [])
    return compact


def _aggregate_attack_reports(reports: Sequence[dict]) -> dict:
    rows = []
    groups: dict[tuple[str, str], list[dict]] = {}
    for report in reports:
        key = (
            str(report["baseline"]),
            str(report["construction"]["mutation"]),
        )
        groups.setdefault(key, []).append(report)

    for (baseline, mutation), group in sorted(groups.items()):
        row = {
            "baseline": baseline,
            "mutation": mutation,
            "seeds": [int(r["seed"]) for r in group],
            "conditions": {},
        }
        for condition in ("control", "evasive"):
            summaries = [r["by_condition"][condition] for r in group]
            row["conditions"][condition] = _aggregate_metric_summaries(
                summaries,
                metrics=(
                    "attack_success_rate",
                    "accepted_rate",
                    "detection_rate",
                    "quarantine_rate",
                    "linked_rate",
                    "mean_mention_distance",
                    "mean_injection_latency_ms",
                    "p95_injection_latency_ms",
                    "oracle_violation_count",
                    "oracle_conflict_attempt_count",
                    "system_durable_violation_count",
                    "external_side_effect_count",
                    "durable_memory_write_count",
                ),
            )
        rows.append(row)
    return {"rows": rows}


def _aggregate_attack_by_axis(reports: Sequence[dict]) -> dict:
    rows = []
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for report in reports:
        baseline = str(report["baseline"])
        mutation = str(report["construction"]["mutation"])
        for axis, condition_summaries in report["by_axis_condition"].items():
            groups.setdefault((baseline, mutation, str(axis)), []).append(
                condition_summaries["evasive"]
            )

    for (baseline, mutation, axis), summaries in sorted(groups.items()):
        rows.append(
            {
                "baseline": baseline,
                "mutation": mutation,
                "axis": axis,
                "evasive": _aggregate_metric_summaries(
                    summaries,
                    metrics=(
                        "attack_success_rate",
                        "accepted_rate",
                        "detection_rate",
                        "quarantine_rate",
                        "linked_rate",
                        "mean_mention_distance",
                    ),
                ),
            }
        )
    return {"rows": rows}


def _aggregate_benign_reports(reports: Sequence[dict]) -> dict:
    summaries = [r["summary"] for r in reports]
    by_category: dict[str, list[dict]] = {}
    for report in reports:
        for category, summary in report.get("by_category", {}).items():
            by_category.setdefault(str(category), []).append(summary)

    return {
        "seeds": [int(r["seed"]) for r in reports],
        "summary": _aggregate_metric_summaries(
            summaries,
            metrics=(
                "false_positive_rate",
                "quarantine_burden_rate",
                "utility_success_rate",
                "mean_write_latency_ms",
                "p95_write_latency_ms",
                "false_positive_count",
                "review_required_count",
                "external_side_effect_count",
                "data_exposure_count",
            ),
        ),
        "by_category": {
            category: _aggregate_metric_summaries(
                group,
                metrics=(
                    "false_positive_rate",
                    "quarantine_burden_rate",
                    "utility_success_rate",
                    "mean_write_latency_ms",
                    "p95_write_latency_ms",
                    "false_positive_count",
                    "review_required_count",
                ),
            )
            for category, group in sorted(by_category.items())
        },
    }


def _aggregate_metric_summaries(
    summaries: Sequence[dict], *, metrics: Sequence[str]
) -> dict:
    out = {"n_runs": len(summaries)}
    if summaries:
        out["n_trials"] = sum(int(s.get("n", 0)) for s in summaries)
    else:
        out["n_trials"] = 0
    for metric in metrics:
        values = [float(s.get(metric, 0.0)) for s in summaries]
        out[metric] = _ci(values, bounded=metric.endswith("_rate"))
    return out


def _lookup_aggregate(
    aggregate: dict, *, baseline: str, mutation: str
) -> dict | None:
    for row in aggregate.get("rows", []):
        if row.get("baseline") == baseline and row.get("mutation") == mutation:
            return row
    return None


def _paper_release_gate(
    *,
    baseline_comparison: dict,
    fail_closed_ablation: dict,
    benign_summary: dict,
    false_positive_threshold: float,
) -> dict:
    b3_mutated = _lookup_aggregate(
        baseline_comparison, baseline="B3", mutation="mutated"
    )
    c7_off = _lookup_aggregate(
        fail_closed_ablation, baseline="B3", mutation="mutated"
    )
    evasive = (b3_mutated or {}).get("conditions", {}).get("evasive", {})
    c7_off_evasive = (c7_off or {}).get("conditions", {}).get("evasive", {})
    benign = benign_summary.get("summary", {})

    attack_acceptance = _metric_mean(evasive, "accepted_rate")
    attack_detection = _metric_mean(evasive, "detection_rate")
    durable_violations = _metric_mean(evasive, "system_durable_violation_count")
    benign_fp = _metric_mean(benign, "false_positive_rate")
    c7_off_acceptance = _metric_mean(c7_off_evasive, "accepted_rate")
    side_effects = (
        _metric_mean(evasive, "external_side_effect_count")
        + _metric_mean(benign, "external_side_effect_count")
    )

    passed = (
        attack_acceptance == 0.0
        and attack_detection >= 1.0
        and durable_violations == 0.0
        and benign_fp <= false_positive_threshold
        and side_effects == 0.0
    )
    return {
        "status": (
            "validated_for_paper_grade_synthetic_eval"
            if passed
            else "blocked_until_retested"
        ),
        "criteria": {
            "attack_acceptance_rate": attack_acceptance,
            "mutated_attack_detection_rate": attack_detection,
            "durable_violations": durable_violations,
            "benign_false_positive_rate": benign_fp,
            "benign_false_positive_threshold": false_positive_threshold,
            "external_side_effect_count": side_effects,
            "c7_off_attack_acceptance_rate": c7_off_acceptance,
        },
        "interpretation": (
            "Synthetic containment is release-candidate only for this attack "
            "family; production release still requires independent trust-boundary "
            "review and representative application traffic."
        ),
    }


def _metric_mean(summary: dict, metric: str) -> float:
    value = summary.get(metric, {})
    if isinstance(value, dict):
        return float(value.get("mean", 0.0))
    return float(value or 0.0)


def _ci(values: Sequence[float], *, bounded: bool = False) -> dict:
    ci = stats.mean_ci(values)
    low = ci.low
    high = ci.high
    if bounded:
        low = min(1.0, max(0.0, low))
        high = min(1.0, max(0.0, high))
    return {
        "mean": ci.mean,
        "low": low,
        "high": high,
        "half_width": ci.half_width,
        "n": ci.n,
    }


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _p95(values: Sequence[float]) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    idx = int(round(0.95 * (len(vals) - 1)))
    return vals[idx]


def _review_expected_decisions(
    baseline: str, *, fail_closed: bool
) -> tuple[str, ...]:
    settings = _settings_for_baseline(baseline, fail_closed=fail_closed)
    if (
        fail_closed
        and settings.enable_constraint_validation
        and settings.enable_contradiction_gate
    ):
        return ("quarantined",)
    return ("accepted",)


def _repo_revision() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _release_gate_status(records: Sequence[InjectionRecord], *, fail_closed: bool) -> str:
    if not fail_closed:
        return "bypass_confirmed_config_off"
    evasive = [r for r in records if r.condition == "evasive"]
    if not evasive:
        return "blocked_no_evasive_trials"
    summary = _summarize_group(evasive)
    if (
        float(summary["accepted_rate"]) <= 0.01
        and float(summary["detection_rate"]) >= 0.95
        and int(summary["system_durable_violation_count"]) == 0
    ):
        return "attack_mitigation_validated_pending_benign_fp"
    return "blocked_until_retested"


def _serialize_write_result(result: object) -> dict:
    return {
        "summary": result.summary.model_dump(mode="json"),
        "accepted": [_serialize_outcome(o) for o in result.accepted],
        "superseded": [_serialize_outcome(o) for o in result.superseded],
        "quarantined": [_serialize_outcome(o) for o in result.quarantined],
        "rejected": [_serialize_outcome(o) for o in result.rejected],
    }


def _serialize_outcome(outcome: WriteOutcome) -> dict:
    return {
        "decision": outcome.decision,
        "assertion_id": outcome.assertion_id,
        "quarantine_id": outcome.quarantine_id,
        "superseded_assertion_id": outcome.superseded_assertion_id,
        "reason": outcome.reason,
        "candidate": outcome.candidate.model_dump(mode="json"),
    }

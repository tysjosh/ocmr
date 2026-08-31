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
import platform
import random
import sys
from typing import Iterable, Sequence

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.evaluation.baselines import baseline_settings_overrides, build_baseline
from ocm.evaluation.experiment import durable_constraint_violations
from ocm.memory.contracts import ExtractionResult, WriteOutcome
from ocm.resolution.entity_resolver import normalize_name

DEFAULT_ATTACK_SEED = 1337
DEFAULT_AXES = ("novel_alias",)

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
) -> dict:
    """Run representative benign writes for false-positive release gating."""

    records = [
        _run_benign_mock_case(
            seed=seed,
            baseline=baseline,
            fail_closed=fail_closed,
            case_id="benign-genuine-new-task",
            category="genuinely_new_entity",
            prelude_write="",
            input_text="Carol is assigned to Task U1.",
            expected_decisions=("accepted",),
        ),
        _run_benign_mock_case(
            seed=seed,
            baseline=baseline,
            fail_closed=fail_closed,
            case_id="benign-short-id-new-task",
            category="short_id",
            prelude_write="",
            input_text="Dave is assigned to Task X.",
            expected_decisions=("accepted",),
        ),
        _run_benign_mock_case(
            seed=seed,
            baseline=baseline,
            fail_closed=fail_closed,
            case_id="benign-underscore-new-task",
            category="underscore",
            prelude_write="",
            input_text="Eve is assigned to Task QA_42.",
            expected_decisions=("accepted",),
        ),
        _run_benign_mock_case(
            seed=seed,
            baseline=baseline,
            fail_closed=fail_closed,
            case_id="benign-leading-zero-new-task",
            category="leading_zero",
            prelude_write="",
            input_text="Frank is assigned to Task R007.",
            expected_decisions=("accepted",),
        ),
        _run_benign_alias_case(
            seed=seed,
            baseline=baseline,
            fail_closed=fail_closed,
            case_id="benign-attributed-novel-alias-same-fact",
            category="legitimate_alias",
            prelude_write="Grace is assigned to Task T2.",
            person="Grace",
            task_surface="T2Proxy",
            task_aliases=("T2",),
            write_intent="new_fact",
            confidence=0.85,
            expected_decisions=("accepted",),
        ),
        _run_benign_alias_case(
            seed=seed,
            baseline=baseline,
            fail_closed=fail_closed,
            case_id="benign-attributed-leading-zero-update",
            category="leading_zero_attributed",
            prelude_write="Heidi is assigned to Task R7.",
            person="Ivan",
            task_surface="R007",
            task_aliases=("R7",),
            write_intent="correction",
            confidence=0.97,
            expected_decisions=("superseded",),
        ),
        _run_benign_alias_case(
            seed=seed,
            baseline=baseline,
            fail_closed=fail_closed,
            case_id="benign-attributed-underscore-same-fact",
            category="underscore_attributed",
            prelude_write="Judy is assigned to Task QA42.",
            person="Judy",
            task_surface="QA_42",
            task_aliases=("QA42",),
            write_intent="new_fact",
            confidence=0.85,
            expected_decisions=("accepted",),
        ),
        _run_benign_mock_case(
            seed=seed,
            baseline=baseline,
            fail_closed=fail_closed,
            case_id="benign-unattributed-ambiguous-new-task-review",
            category="ambiguous_unattributed_review",
            prelude_write="Mallory is assigned to Task T3.",
            input_text="Niaj is assigned to Task Aster321.",
            expected_decisions=("quarantined",),
        ),
    ]
    false_positives = sum(1 for r in records if r.false_positive)
    n = len(records)
    false_positive_rate = false_positives / n if n else 0.0
    return {
        "seed": seed,
        "baseline": baseline,
        "config_versions": _config_versions(baseline, fail_closed=fail_closed),
        "construction": {
            "fail_closed_enabled": fail_closed,
            "threshold": threshold,
            "categories": sorted({r.category for r in records}),
        },
        "summary": {
            "n": n,
            "false_positive_count": false_positives,
            "false_positive_rate": false_positive_rate,
            "threshold": threshold,
            "passes_threshold": false_positive_rate <= threshold,
            "review_required_count": sum(1 for r in records if r.decision == "quarantined"),
            "external_side_effect_count": sum(
                int(r.side_effect_review.get("external_side_effects_observed", 0))
                for r in records
            ),
            "data_exposure_count": sum(
                1 for r in records if r.side_effect_review.get("data_exposure_observed")
            ),
        },
        "records": [asdict(r) for r in records],
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
    unknown_axes = sorted(set(axes) - {"novel_alias", "spelling_variant", "spacing_variant", "partial"})
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
    result = strategy.write(injection_text, f"{case.case_id}:{condition}")
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
    result = strategy.write(input_text, f"{case_id}:input")
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
    result = strategy.write(input_text, f"{case_id}:input")
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
        }

    def rate(attr: str) -> float:
        return sum(1 for r in records if bool(getattr(r, attr))) / n

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
    }


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

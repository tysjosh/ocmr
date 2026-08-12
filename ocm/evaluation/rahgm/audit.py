"""The OCMR quarantine audit (paper §4.1, Table 2).

Reanalyzes the writes quarantined in the prior OCMR stress audit — the 1,198
recorded in ``governance_examples.json``, of which 835 were heuristic false
quarantines. This is what turns the earlier finding ("roughly 70% of quarantined
writes were valid updates conservatively held back, principally because of
unresolved entity aliases") into the routing target RAHGM is built to hit.

Two populations are reported, and the distinction matters:

**Recorded population (the paper's numbers).** ``governance_examples.json`` was
produced with a cached LLM extractor that is not part of this repository, so its
1,198 quarantines cannot be regenerated offline. The audit therefore reads it as
the authoritative population for the counts and classifies its full reason
histogram — an exact reanalysis that sums to 1,198. What the artifact *cannot*
supply is the per-write join between a quarantine's reason and its
false-quarantine determination: it records those as separate marginals.

**Offline replay (reproducible).** The same governed write path is replayed with
the offline mock extractor, which yields a smaller but fully reproducible
population in which reason and validity *are* joined on the same write. This
supplies the per-cause validity split, and it supports the alias attribution
directly: replaying with a fresh store per example removes cross-example
identifier collisions, and the change in quarantine count measures how much of the
conservative holding is an entity-identity artifact rather than a genuine
integrity signal.

Cause assignment uses a published rule set keyed to the OCMR check that emits each
reason string, so every attribution is checkable against
``ocm/validation/constraints.py`` rather than being a judgment about wording.
Unmatched reasons are counted as ``other``.

Two rubric-based annotator simulators stand in for the paper's two human
annotators, with the rule set adjudicating; Krippendorff's alpha is reported before
adjudication and is labelled as rubric determinacy, not human agreement
(Req 9.8, 14.1).

Requirements: 13.1, 9.8, 14.1.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ocm.evaluation.rahgm.annotate import dual_annotate

#: Default location of the OCMR governance audit artifact.
DEFAULT_ARTIFACT = "governance_examples.json"

#: The artifact's protocol, reproduced by the offline replay.
AUDIT_METHOD = "B3"
AUDIT_SEEDS: tuple[int, ...] = (1337, 7, 42, 99, 2024)
AUDIT_PER_CATEGORY = 25

#: The cause categories of Table 2.
CAUSES: tuple[str, ...] = (
    "entity_alias_resolution",
    "genuine_ambiguity",
    "evidence_provenance",
    "temporal_cardinality",
    "other",
)

#: Rule set mapping an OCMR quarantine reason onto a primary cause.
#:
#: Ordered; first match wins. Each entry names the OCMR check that emits the
#: reason. Note that a *single-valued conflict* is classified as genuine ambiguity,
#: not as a cardinality violation: the schema is satisfied, two sources disagree
#: about which value is current, and that is precisely the question a reviewer
#: settles. A cardinality/domain/range failure proper is inadmissible instead.
_CAUSE_RULES: tuple[tuple[str, str, str], ...] = (
    (
        "entity_alias_resolution",
        r"alias|possibly_same_as|identity|same entity|unresolved entit",
        "C1 identity / W3 entity resolution",
    ),
    (
        "evidence_provenance",
        r"is 'done' but has no completion event",
        "C4 done-task completion event",
    ),
    (
        "evidence_provenance",
        r"evidence|supporting|unsupported|provenance",
        "C8 decision evidence floor",
    ),
    (
        "genuine_ambiguity",
        r"single-valued conflict",
        "C7 contradiction gate (single-valued predicate)",
    ),
    (
        "temporal_cardinality",
        r"transition .* is not permitted",
        "C10 task status transition",
    ),
    (
        "temporal_cardinality",
        r"timestamp|temporal|valid_from|valid_to|interval|precedes|cycle",
        "C2 / C3 temporal checks",
    ),
    (
        "temporal_cardinality",
        r"cardinality|domain|range",
        "C9 domain/range",
    ),
    (
        "genuine_ambiguity",
        r"cannot overwrite an accepted status|status contradiction|contradiction|conflict",
        "C7 contradiction gate",
    ),
)

#: Causes where a human can resolve the question and release or reject the write.
#: A pure cardinality or temporal violation is not review-worthy: there is nothing
#: for an analyst to decide, the write is simply inadmissible.
_REVIEW_WORTHY_CAUSES = frozenset(
    {"entity_alias_resolution", "genuine_ambiguity", "evidence_provenance"}
)


def classify_reason(reason: str) -> tuple[str, str | None]:
    """Classify one quarantine reason into ``(cause, ocmr_check)`` lexically."""
    text = (reason or "").strip().lower()
    if not text:
        return "other", None
    for cause, pattern, check in _CAUSE_RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return cause, check
    return "other", None


#: Check attributed to a structurally detected identity ambiguity.
IDENTITY_REUSE_CHECK = "W3 entity resolution (identifier shared across contexts)"


def classify_quarantine(
    reason: str, *, identity_ambiguous: bool
) -> tuple[str, str | None]:
    """Classify a quarantine, preferring structural evidence over wording.

    OCMR's quarantine reasons never contain the word "alias": the reason names the
    check that fired (a contradiction, a status transition), not the reason the
    check fired. So a purely lexical rule set cannot attribute anything to entity
    identity, which is why the paper's headline cause would otherwise read zero.

    ``identity_ambiguous`` supplies the missing structural signal: the write
    conflicts with an assertion that a *different* context created about the same
    identifier. That is an unresolved identity — two logical entities sharing one
    id — and it takes precedence over the lexical classification, because the
    contradiction the reason names is a *symptom* of the collision rather than an
    independent cause.

    Note the test is deliberately narrow. Merely sharing an identifier with another
    context is not enough: the benchmark reuses ids pervasively, so that weaker test
    flags almost every quarantine. What matters is that the *conflicting party* comes
    from elsewhere.
    """
    if identity_ambiguous:
        return "entity_alias_resolution", IDENTITY_REUSE_CHECK
    return classify_reason(reason)


# --------------------------------------------------------------------------- #
# Table rows
# --------------------------------------------------------------------------- #
@dataclass
class CauseRow:
    """One row of Table 2."""

    cause: str
    count: int = 0
    valid_count: int = 0
    review_worthy_count: int = 0
    example_reason: str | None = None
    ocmr_check: str | None = None

    @property
    def valid_pct(self) -> float:
        """Percentage of this cause's quarantines that were valid updates."""
        return 100.0 * self.valid_count / self.count if self.count else 0.0

    @property
    def review_pct(self) -> float:
        """Percentage of this cause's quarantines that were review-worthy."""
        return 100.0 * self.review_worthy_count / self.count if self.count else 0.0

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "cause": self.cause,
            "count": self.count,
            "valid_count": self.valid_count,
            "valid_pct": self.valid_pct,
            "review_worthy_count": self.review_worthy_count,
            "review_pct": self.review_pct,
            "ocmr_check": self.ocmr_check,
            "example_reason": self.example_reason,
        }


def load_artifact(path: str = DEFAULT_ARTIFACT) -> dict[str, Any] | None:
    """Load the recorded OCMR governance artifact, or ``None`` when absent."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# Recorded population
# --------------------------------------------------------------------------- #
def classify_recorded(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Classify the recorded reason histogram — an exact reanalysis of §4.1.

    Every one of the recorded quarantines is attributed, so the cause counts sum to
    the recorded total. Review-worthiness follows the cause. Validity is reported
    only as the recorded marginal (835), because the artifact does not join
    validity to reason per write.
    """
    histogram = dict(artifact.get("quarantine_reason_histogram") or {})
    totals = artifact.get("totals") or {}
    recorded_quarantined = int(totals.get("quarantined", 0))
    recorded_false = int(artifact.get("false_quarantine_total", 0))

    rows: dict[str, CauseRow] = {cause: CauseRow(cause=cause) for cause in CAUSES}
    for reason, count in sorted(histogram.items(), key=lambda kv: (-kv[1], kv[0])):
        cause, check = classify_reason(reason)
        row = rows[cause]
        row.count += count
        row.review_worthy_count += count if cause in _REVIEW_WORTHY_CAUSES else 0
        if row.example_reason is None:
            row.example_reason = reason
            row.ocmr_check = check

    classified = sum(row.count for row in rows.values())
    shortfall = recorded_quarantined - classified
    if shortfall > 0:
        # The histogram is complete in the shipped artifact; if a future artifact
        # truncates it, the residual is attributed to ``other`` rather than dropped.
        rows["other"].count += shortfall

    total = sum(row.count for row in rows.values())
    review_worthy = sum(row.review_worthy_count for row in rows.values())

    return {
        "n_quarantines": total,
        "n_distinct_reasons": len(histogram),
        "histogram_classified": classified,
        "unmatched_attributed_to_other": max(0, shortfall),
        "rows": [rows[cause].as_dict() for cause in CAUSES],
        "review_worthy_count": review_worthy,
        "review_worthy_pct": 100.0 * review_worthy / total if total else 0.0,
        "recorded_false_quarantine": recorded_false,
        "recorded_false_quarantine_pct": (
            100.0 * recorded_false / recorded_quarantined if recorded_quarantined else 0.0
        ),
        "recorded_totals": {
            "accepted": int(totals.get("accepted", 0)),
            "superseded": int(totals.get("superseded", 0)),
            "quarantined": recorded_quarantined,
            "rejected": int(totals.get("rejected", 0)),
        },
        "method": artifact.get("method"),
        "seeds": artifact.get("seeds"),
        "n_examples": artifact.get("n_examples"),
        "_rows": rows,
    }


# --------------------------------------------------------------------------- #
# Offline replay population
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QuarantineUnit:
    """One quarantined write from the offline replay, with joined labels."""

    unit_id: str
    reason: str
    cause: str
    ocmr_check: str | None
    valid: bool
    review_worthy: bool
    category: str
    example_id: str
    severity: str | None
    seed: int
    #: Whether the write's subject or object is an identifier that distinct
    #: contexts both write to — the structural signature of unresolved identity.
    identity_ambiguous: bool = False


def collect_quarantines(
    *,
    method: str = AUDIT_METHOD,
    seeds: Sequence[int] = AUDIT_SEEDS,
    per_category: int = AUDIT_PER_CATEGORY,
    isolate_per_example: bool = False,
    verbose: bool = False,
) -> list[QuarantineUnit]:
    """Replay the governed write path offline, joining reason to validity per write.

    Reproduces ``replay_governed_writes``' protocol. With
    ``isolate_per_example=True`` a fresh governed store is built per example, which
    removes cross-example identifier collisions; comparing the two modes is how the
    audit attributes conservative holding to entity-identity ambiguity.
    """
    from ocm.evaluation.benchmark import BenchmarkGenerator
    from ocm.evaluation.experiment import (
        _build_strategy,
        _default_settings,
        _seed_everything,
    )
    from ocm.evaluation.replay_governed_writes import _quarantine_index

    units: list[QuarantineUnit] = []

    for seed in seeds:
        _seed_everything(seed)
        examples = BenchmarkGenerator(seed=seed).generate(per_category=per_category)
        shared = None if isolate_per_example else _build_strategy(method, _default_settings)

        # Which example authored each accepted assertion. A quarantine whose
        # conflicting party was authored by a *different* example is a cross-context
        # identifier collision — the identifier-reuse mechanism behind OCMR's
        # alias-driven quarantines.
        authored_by: dict[str, str] = {}

        for example in examples:
            strategy = shared or _build_strategy(method, _default_settings)
            expects_conflict = any(
                bool(getattr(q, "expected_conflict", False)) for q in example.questions
            )
            for session in example.sessions:
                source_ref = f"{example.id}:{session.session_id}"
                result = strategy.write(session.input, source_ref)

                if not result.quarantined:
                    # Still record authorship: a later quarantine may conflict
                    # with something this write just accepted.
                    for outcome in result.accepted + result.superseded:
                        if outcome.assertion_id:
                            authored_by.setdefault(outcome.assertion_id, example.id)
                    continue
                quarantines = _quarantine_index(strategy.container)
                for outcome in result.quarantined:
                    reason = (outcome.reason or "").strip()
                    severity = None
                    record = quarantines.get(outcome.quarantine_id or "")
                    if record is not None:
                        if not reason:
                            reason = (getattr(record, "reason", "") or "").strip()
                        severity = getattr(getattr(record, "severity", None), "value", None)

                    conflicting = list(getattr(record, "conflicting_ids", []) or [])
                    identity_ambiguous = any(
                        authored_by.get(conflict_id, example.id) != example.id
                        for conflict_id in conflicting
                    )
                    cause, check = classify_quarantine(
                        reason, identity_ambiguous=identity_ambiguous
                    )
                    units.append(
                        QuarantineUnit(
                            unit_id=f"s{seed}:{example.id}:{len(units)}",
                            reason=reason,
                            cause=cause,
                            ocmr_check=check,
                            # OCMR's own false-quarantine determination.
                            valid=not expects_conflict,
                            review_worthy=cause in _REVIEW_WORTHY_CAUSES,
                            category=example.category,
                            example_id=example.id,
                            severity=severity,
                            seed=seed,
                            identity_ambiguous=identity_ambiguous,
                        )
                    )
                for outcome in result.accepted + result.superseded:
                    if outcome.assertion_id:
                        authored_by.setdefault(outcome.assertion_id, example.id)
        if verbose:
            print(f"  seed {seed}: {len(units)} quarantines cumulative")

    return units


def _rows_from_units(units: Sequence[QuarantineUnit]) -> dict[str, CauseRow]:
    """Aggregate offline units into Table 2 rows."""
    rows: dict[str, CauseRow] = {cause: CauseRow(cause=cause) for cause in CAUSES}
    for unit in units:
        row = rows.setdefault(unit.cause, CauseRow(cause=unit.cause))
        row.count += 1
        row.valid_count += 1 if unit.valid else 0
        row.review_worthy_count += 1 if unit.review_worthy else 0
        if row.example_reason is None:
            row.example_reason = unit.reason
            row.ocmr_check = unit.ocmr_check
    return rows


# --------------------------------------------------------------------------- #
# The audit
# --------------------------------------------------------------------------- #
def run_quarantine_audit(
    *,
    artifact_path: str = DEFAULT_ARTIFACT,
    method: str = AUDIT_METHOD,
    seeds: Sequence[int] = AUDIT_SEEDS,
    per_category: int = AUDIT_PER_CATEGORY,
    annotator_error_rate: float = 0.08,
    seed: int = 1337,
    run_offline_replay: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Reanalyze the OCMR quarantines and populate Table 2 (§4.1).

    Args:
        artifact_path: The recorded ``governance_examples.json``.
        method / seeds / per_category: Protocol for the offline replay.
        annotator_error_rate: Per-annotator label noise for the alpha estimate.
        seed: Base seed for the annotator simulators.
        run_offline_replay: Whether to run the reproducible offline replay, which
            supplies the per-cause validity split and the alias attribution.
        verbose: Print replay progress.

    Returns:
        A report dict with the recorded reanalysis, the offline replay, the alias
        attribution, and the inter-annotator agreement estimates.
    """
    artifact = load_artifact(artifact_path)
    recorded = classify_recorded(artifact) if artifact is not None else None

    offline: dict[str, Any] | None = None
    alias_attribution: dict[str, Any] | None = None
    agreement: dict[str, Any] = {}

    if run_offline_replay:
        if verbose:
            print("offline replay (shared store)...")
        shared_units = collect_quarantines(
            method=method, seeds=seeds, per_category=per_category, verbose=verbose
        )
        if verbose:
            print("offline replay (isolated per example)...")
        isolated_units = collect_quarantines(
            method=method,
            seeds=seeds,
            per_category=per_category,
            isolate_per_example=True,
            verbose=verbose,
        )

        rows = _rows_from_units(shared_units)
        total = len(shared_units)
        valid = sum(1 for u in shared_units if u.valid)
        review_worthy = sum(1 for u in shared_units if u.review_worthy)

        offline = {
            "n_quarantines": total,
            "valid_count": valid,
            "valid_pct": 100.0 * valid / total if total else 0.0,
            "review_worthy_count": review_worthy,
            "review_worthy_pct": 100.0 * review_worthy / total if total else 0.0,
            "rows": [rows[cause].as_dict() for cause in CAUSES],
            "valid_by_cause": dict(
                sorted(
                    (
                        (cause, sum(1 for u in shared_units if u.cause == cause and u.valid))
                        for cause in CAUSES
                    ),
                    key=lambda kv: -kv[1],
                )
            ),
            "reproducible": True,
            "extractor": "mock (offline, deterministic)",
        }

        shared_valid = valid
        isolated_total = len(isolated_units)
        isolated_valid = sum(1 for u in isolated_units if u.valid)
        alias_attribution = {
            "shared_store_quarantines": total,
            "isolated_quarantines": isolated_total,
            "quarantines_removed_by_isolation": total - isolated_total,
            "pct_quarantines_attributable_to_identifier_reuse": (
                100.0 * (total - isolated_total) / total if total else 0.0
            ),
            "shared_store_false_quarantines": shared_valid,
            "isolated_false_quarantines": isolated_valid,
            "false_quarantines_removed_by_isolation": shared_valid - isolated_valid,
            "pct_false_quarantines_attributable_to_identifier_reuse": (
                100.0 * (shared_valid - isolated_valid) / shared_valid
                if shared_valid
                else 0.0
            ),
            "interpretation": (
                "Quarantines that disappear when each example gets a fresh store "
                "exist only because distinct entities across examples share an "
                "identifier. That is unresolved entity identity, which the paper "
                "names as the principal cause of OCMR's false quarantines."
            ),
        }

        # Simulated dual annotation over the offline population, where all three
        # labels are available per write.
        _cause_labels, cause_agreement = dual_annotate(
            [(u.unit_id, u.cause) for u in shared_units],
            CAUSES,
            field="primary_cause",
            error_rate=annotator_error_rate,
            seed=seed,
        )
        _validity_labels, validity_agreement = dual_annotate(
            [(u.unit_id, "valid" if u.valid else "invalid") for u in shared_units],
            ("valid", "invalid"),
            field="validity",
            error_rate=annotator_error_rate,
            seed=seed + 15485863,
        )
        _review_labels, review_agreement = dual_annotate(
            [
                (u.unit_id, "review_worthy" if u.review_worthy else "not_review_worthy")
                for u in shared_units
            ],
            ("review_worthy", "not_review_worthy"),
            field="review_worthiness",
            error_rate=annotator_error_rate,
            seed=seed + 104729,
        )
        agreement = {
            "primary_cause": cause_agreement.as_dict(),
            "validity": validity_agreement.as_dict(),
            "review_worthiness": review_agreement.as_dict(),
        }

    return {
        "experiment": "quarantine_audit",
        "protocol": {
            "method": method,
            "seeds": list(seeds),
            "per_category": per_category,
        },
        "recorded_population": recorded,
        "offline_replay": offline,
        "alias_attribution": alias_attribution,
        "agreement": agreement,
        "classifiers": {
            "recorded_population": "lexical only (reason strings; no per-write state)",
            "offline_replay": "structural identity check, falling back to lexical",
            "comparability": (
                "The two populations are classified differently and their cause rows "
                "are NOT directly comparable. The recorded artifact provides only "
                "reason strings, and OCMR reasons never name entity identity, so a "
                "lexical rule set cannot attribute anything to it. The offline replay "
                "retains per-write state, so it can detect the collision structurally."
            ),
        },
        "identity_detector": {
            "rule": (
                "A quarantine is attributed to entity identity when its conflicting "
                "assertion was authored by a different benchmark example than the "
                "write itself: two logical entities sharing one identifier."
            ),
            "validation": (
                "Under isolate_per_example the detector reports 0% by construction, "
                "since a fresh store per example makes cross-context collision "
                "impossible. That is the negative control for this attribution."
            ),
        },
        "notes": [
            "The recorded artifact was produced with a cached LLM extractor that is "
            "not part of this repository, so its 1,198 quarantines are read as the "
            "authoritative population rather than regenerated. Its reason histogram "
            "is classified exactly and sums to the recorded total.",
            "The artifact records reason and false-quarantine status as separate "
            "marginals, so the per-cause validity split comes from the reproducible "
            "offline replay, where both labels are joined on the same write.",
            "Causes are assigned by a published rule set keyed to the OCMR check "
            "that emits each reason; unmatched reasons are counted as 'other'. The "
            "offline replay additionally applies a structural identity check, which "
            "takes precedence because the named contradiction is a symptom of an "
            "identifier collision rather than an independent cause.",
            "Annotation uses two rubric-based simulators, not human annotators.",
        ],
    }

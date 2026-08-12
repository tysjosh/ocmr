"""Does an erroneous transition at ``t`` actually influence later states?

§3.1 motivates order-preserving replay on the grounds that "an erroneous transition
at time ``t`` remains available to influence all later states ``M_{t+1}, …, M_T``".
That is an empirical claim about a governance policy on a workload, not a property
of replay itself, and it is easy to assert without testing.

Measuring it needs three things, and the third is the one usually missing:

1. writes that share a target, so a later decision *can* depend on an earlier one —
   the contention chains in :mod:`ocm.evaluation.rahgm.corpus`;
2. a condition that actually commits a write it should have held, creating an
   upstream error;
3. a **counterfactual control**: the same replay with that one transition corrected
   and nothing else changed. Without it, a later error that merely co-occurs with an
   earlier one is indistinguishable from one the earlier one caused.

This module supplies (3) via :class:`~ocm.evaluation.rahgm.replay.ScenarioReplayer`'s
``force_hold`` intervention, which holds a single write regardless of routing while
leaving the policy, the features, and the write order untouched.

The reported quantity is the number of downstream errors that occur in the factual
replay **and disappear** in the counterfactual. That difference is the propagated
error; anything else is coincidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ocm.evaluation.rahgm.corpus import CandidateWrite, RahgmCorpus, Scenario
from ocm.evaluation.rahgm.metrics import WriteRecord
from ocm.evaluation.rahgm.replay import (
    DevelopedPolicy,
    ScenarioReplayer,
    make_oracle_reviewer,
)
from ocm.governance.conditions import Condition
from ocm.governance.policy import PolicyParameters, Tier

#: Transitions that put a write into durable memory.
_COMMITTED = (Tier.accept, Tier.supersede)


@dataclass(frozen=True)
class CascadeEvent:
    """One measured upstream error and its downstream consequences."""

    scenario_id: str
    chain_id: str
    upstream_write_id: str
    upstream_template: str
    upstream_position: int
    upstream_gold: str
    upstream_final: str
    downstream_write_ids: tuple[str, ...]
    downstream_errors_factual: tuple[str, ...]
    downstream_errors_counterfactual: tuple[str, ...]

    @property
    def propagated(self) -> tuple[str, ...]:
        """Downstream writes that err factually but not counterfactually."""
        return tuple(
            w
            for w in self.downstream_errors_factual
            if w not in set(self.downstream_errors_counterfactual)
        )

    @property
    def cascaded(self) -> bool:
        """Whether this upstream error caused at least one downstream error."""
        return bool(self.propagated)

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "scenario_id": self.scenario_id,
            "chain_id": self.chain_id,
            "upstream_write_id": self.upstream_write_id,
            "upstream_template": self.upstream_template,
            "upstream_position": self.upstream_position,
            "upstream_gold": self.upstream_gold,
            "upstream_final": self.upstream_final,
            "n_downstream": len(self.downstream_write_ids),
            "downstream_errors_factual": list(self.downstream_errors_factual),
            "downstream_errors_counterfactual": list(
                self.downstream_errors_counterfactual
            ),
            "propagated": list(self.propagated),
            "cascaded": self.cascaded,
        }


@dataclass
class CascadeReport:
    """Cascade measurement for one condition on one stream."""

    condition: str
    n_scenarios: int
    n_chains: int
    n_upstream_errors: int
    n_cascades: int
    n_propagated_writes: int
    n_downstream_errors_factual: int
    n_downstream_errors_counterfactual: int
    events: list[CascadeEvent] = field(default_factory=list)

    @property
    def cascade_rate(self) -> float:
        """Fraction of upstream errors that produced a downstream error."""
        return self.n_cascades / self.n_upstream_errors if self.n_upstream_errors else 0.0

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "condition": self.condition,
            "n_scenarios": self.n_scenarios,
            "n_chains": self.n_chains,
            "n_upstream_errors": self.n_upstream_errors,
            "n_cascades": self.n_cascades,
            "cascade_rate": self.cascade_rate,
            "n_propagated_writes": self.n_propagated_writes,
            "n_downstream_errors_factual": self.n_downstream_errors_factual,
            "n_downstream_errors_counterfactual": self.n_downstream_errors_counterfactual,
            "events": [e.as_dict() for e in self.events],
            "interpretation": (
                "n_propagated_writes counts downstream errors present in the factual "
                "replay and absent in the counterfactual, so it is the number of "
                "errors an upstream mistake actually caused. Zero upstream errors "
                "means the claim is untested on this stream, not disproven."
            ),
        }


def _chains_of(scenario: Scenario) -> dict[str, list[CandidateWrite]]:
    """Group a scenario's writes by contention chain, in chain order."""
    chains: dict[str, list[CandidateWrite]] = {}
    for write in scenario.writes:
        if write.chain_id:
            chains.setdefault(write.chain_id, []).append(write)
    for writes in chains.values():
        writes.sort(key=lambda w: w.chain_position or 0)
    return chains


def measure_cascade(
    scenarios: Sequence[Scenario],
    *,
    condition: Condition = Condition.frozen_rahgm,
    params: PolicyParameters | None = None,
) -> CascadeReport:
    """Measure causally-attributable error propagation within contention chains.

    For every chain in which the condition commits a write whose gold transition is
    ``review``, the scenario is replayed a second time with that single write held and
    everything else identical. Downstream errors that vanish under the intervention
    are attributed to the upstream error.
    """
    events: list[CascadeEvent] = []
    n_chains = 0
    upstream_errors = 0
    factual_downstream = 0
    counterfactual_downstream = 0

    for scenario in scenarios:
        writes_by_id = {w.write_id: w for w in scenario.writes}
        reviewer = make_oracle_reviewer(writes_by_id)

        factual = ScenarioReplayer(condition, params=params, reviewer=reviewer)
        _harness, result = factual.run_scenario(scenario)
        factual_records = {r.write_id: r for r in result.records}

        chains = _chains_of(scenario)
        n_chains += len(chains)

        for chain_id, chain in chains.items():
            for upstream in chain:
                record = factual_records.get(upstream.write_id)
                if record is None:
                    continue
                # An upstream error is a write that should have been held and wasn't.
                if not (
                    upstream.gold_transition is Tier.review and record.final in _COMMITTED
                ):
                    continue
                upstream_errors += 1

                downstream = [
                    w
                    for w in chain
                    if (w.chain_position or 0) > (upstream.chain_position or 0)
                ]
                if not downstream:
                    events.append(
                        CascadeEvent(
                            scenario_id=scenario.scenario_id,
                            chain_id=chain_id,
                            upstream_write_id=upstream.write_id,
                            upstream_template=upstream.template,
                            upstream_position=upstream.chain_position or 0,
                            upstream_gold=upstream.gold_transition.value,
                            upstream_final=record.final.value,
                            downstream_write_ids=(),
                            downstream_errors_factual=(),
                            downstream_errors_counterfactual=(),
                        )
                    )
                    continue

                errors_factual = tuple(
                    w.write_id
                    for w in downstream
                    if (r := factual_records.get(w.write_id)) is not None and r.error
                )

                # Counterfactual: identical replay, upstream write held.
                control = ScenarioReplayer(
                    condition,
                    params=params,
                    reviewer=reviewer,
                    force_hold={upstream.write_id},
                )
                _control_harness, control_result = control.run_scenario(scenario)
                control_records = {r.write_id: r for r in control_result.records}
                errors_control = tuple(
                    w.write_id
                    for w in downstream
                    if (r := control_records.get(w.write_id)) is not None and r.error
                )

                factual_downstream += len(errors_factual)
                counterfactual_downstream += len(errors_control)
                events.append(
                    CascadeEvent(
                        scenario_id=scenario.scenario_id,
                        chain_id=chain_id,
                        upstream_write_id=upstream.write_id,
                        upstream_template=upstream.template,
                        upstream_position=upstream.chain_position or 0,
                        upstream_gold=upstream.gold_transition.value,
                        upstream_final=record.final.value,
                        downstream_write_ids=tuple(w.write_id for w in downstream),
                        downstream_errors_factual=errors_factual,
                        downstream_errors_counterfactual=errors_control,
                    )
                )

    return CascadeReport(
        condition=condition.value,
        n_scenarios=len(scenarios),
        n_chains=n_chains,
        n_upstream_errors=upstream_errors,
        n_cascades=sum(1 for e in events if e.cascaded),
        n_propagated_writes=sum(len(e.propagated) for e in events),
        n_downstream_errors_factual=factual_downstream,
        n_downstream_errors_counterfactual=counterfactual_downstream,
        events=events,
    )


@dataclass(frozen=True)
class InjectionEvent:
    """One injected upstream error and its measured downstream effect."""

    scenario_id: str
    chain_id: str
    injected_write_id: str
    injected_template: str
    injected_position: int
    injected_gold: str
    downstream_write_ids: tuple[str, ...]
    baseline_errors: tuple[str, ...]
    injected_errors: tuple[str, ...]
    verdict_changed: tuple[str, ...]
    tier_changed: tuple[str, ...]
    baseline_violations: int = 0
    injected_violations: int = 0

    @property
    def propagated(self) -> tuple[str, ...]:
        """Downstream writes that err only once the upstream error is injected."""
        baseline = set(self.baseline_errors)
        return tuple(w for w in self.injected_errors if w not in baseline)

    @property
    def violation_delta(self) -> int:
        """Extra durable-state violations the injected error left behind."""
        return self.injected_violations - self.baseline_violations

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "scenario_id": self.scenario_id,
            "chain_id": self.chain_id,
            "injected_write_id": self.injected_write_id,
            "injected_template": self.injected_template,
            "injected_position": self.injected_position,
            "injected_gold": self.injected_gold,
            "n_downstream": len(self.downstream_write_ids),
            "baseline_errors": list(self.baseline_errors),
            "injected_errors": list(self.injected_errors),
            "propagated": list(self.propagated),
            "downstream_ocmr_verdict_changed": list(self.verdict_changed),
            "downstream_routed_tier_changed": list(self.tier_changed),
            "baseline_violations": self.baseline_violations,
            "injected_violations": self.injected_violations,
            "violation_delta": self.violation_delta,
            "cascaded": bool(self.propagated)
            or bool(self.verdict_changed)
            or self.violation_delta > 0,
        }


def measure_injected_cascade(
    scenarios: Sequence[Scenario],
    *,
    condition: Condition = Condition.frozen_rahgm,
    params: PolicyParameters | None = None,
) -> dict[str, Any]:
    """Inject an upstream error and measure whether it changes later decisions.

    §3.1's claim is a property of the *replay methodology*: an erroneous durable
    transition remains available to influence later states. Whether a particular
    governance condition happens to make such an error is a separate question — and
    measured separately by :func:`measure_cascade`, which found none.

    This isolates the mechanism. For each contention chain, the write whose gold
    transition is ``review`` is force-committed, which is exactly the error a
    fallible system would make, and the chain's later writes are compared against a
    baseline replay with no intervention. Two things are recorded: whether a later
    write's outcome became wrong, and whether OCMR's own verdict for it changed at
    all. The second matters even when the first does not: a changed verdict shows the
    corrupted state reached the later decision.
    """
    events: list[InjectionEvent] = []

    for scenario in scenarios:
        writes_by_id = {w.write_id: w for w in scenario.writes}
        reviewer = make_oracle_reviewer(writes_by_id)

        baseline = ScenarioReplayer(condition, params=params, reviewer=reviewer)
        _h, baseline_result = baseline.run_scenario(scenario)
        baseline_records = {r.write_id: r for r in baseline_result.records}

        for chain_id, chain in _chains_of(scenario).items():
            targets = [
                w
                for w in chain
                if w.gold_transition is Tier.review
                and any((x.chain_position or 0) > (w.chain_position or 0) for x in chain)
            ]
            for target in targets:
                downstream = [
                    w
                    for w in chain
                    if (w.chain_position or 0) > (target.chain_position or 0)
                ]
                injected = ScenarioReplayer(
                    condition,
                    params=params,
                    reviewer=reviewer,
                    force_commit={target.write_id},
                )
                _ih, injected_result = injected.run_scenario(scenario)
                injected_records = {r.write_id: r for r in injected_result.records}

                baseline_errors = tuple(
                    w.write_id
                    for w in downstream
                    if (r := baseline_records.get(w.write_id)) is not None and r.error
                )
                injected_errors = tuple(
                    w.write_id
                    for w in downstream
                    if (r := injected_records.get(w.write_id)) is not None and r.error
                )
                # Did OCMR's own constraint verdict for a later write change? This is
                # the direct evidence that the corrupted state reached that decision,
                # and it is visible even when the router's tier is unchanged.
                verdict_changed = tuple(
                    w.write_id
                    for w in downstream
                    if (b := baseline_records.get(w.write_id)) is not None
                    and (i := injected_records.get(w.write_id)) is not None
                    and (b.ocmr_action, b.ocmr_failed_check)
                    != (i.ocmr_action, i.ocmr_failed_check)
                )
                tier_changed = tuple(
                    w.write_id
                    for w in downstream
                    if (b := baseline_records.get(w.write_id)) is not None
                    and (i := injected_records.get(w.write_id)) is not None
                    and b.routed is not i.routed
                )
                events.append(
                    InjectionEvent(
                        scenario_id=scenario.scenario_id,
                        chain_id=chain_id,
                        injected_write_id=target.write_id,
                        injected_template=target.template,
                        injected_position=target.chain_position or 0,
                        injected_gold=target.gold_transition.value,
                        downstream_write_ids=tuple(w.write_id for w in downstream),
                        baseline_errors=baseline_errors,
                        injected_errors=injected_errors,
                        verdict_changed=verdict_changed,
                        tier_changed=tier_changed,
                        baseline_violations=baseline_result.durable_violations,
                        injected_violations=injected_result.durable_violations,
                    )
                )

    n_propagated = sum(len(e.propagated) for e in events)
    n_verdict_changed = sum(len(e.verdict_changed) for e in events)
    n_tier_changed = sum(len(e.tier_changed) for e in events)
    violation_delta = sum(e.violation_delta for e in events)
    by_template: dict[str, dict[str, int]] = {}
    for event in events:
        entry = by_template.setdefault(
            event.injected_template,
            {
                "injections": 0,
                "propagated": 0,
                "verdict_changed": 0,
                "tier_changed": 0,
                "violation_delta": 0,
            },
        )
        entry["injections"] += 1
        entry["propagated"] += len(event.propagated)
        entry["verdict_changed"] += len(event.verdict_changed)
        entry["tier_changed"] += len(event.tier_changed)
        entry["violation_delta"] += event.violation_delta

    return {
        "condition": condition.value,
        "n_injections": len(events),
        "n_propagated_errors": n_propagated,
        "n_downstream_verdicts_changed": n_verdict_changed,
        "n_downstream_tiers_changed": n_tier_changed,
        "durable_violation_delta": violation_delta,
        "propagation_rate": (
            sum(1 for e in events if e.propagated) / len(events) if events else 0.0
        ),
        "state_reached_rate": (
            sum(1 for e in events if e.verdict_changed) / len(events) if events else 0.0
        ),
        "by_injected_template": by_template,
        "events": [e.as_dict() for e in events],
        "measures": {
            "n_propagated_errors": "downstream transitions that became wrong",
            "n_downstream_verdicts_changed": (
                "downstream writes whose OCMR constraint verdict changed — direct "
                "evidence the corrupted state reached that decision"
            ),
            "n_downstream_tiers_changed": "downstream writes the router routed differently",
            "durable_violation_delta": (
                "extra durable-state violations left behind, which is the harm the "
                "injected error actually causes even when transition labels match gold"
            ),
        },
    }


def run_cascade_study(
    corpus: RahgmCorpus | None = None,
    *,
    developed: DevelopedPolicy | None = None,
    severity: float = 1.0,
    seed: int = 1337,
) -> dict[str, Any]:
    """Test the §3.1 propagation claim on undrifted and label-drifted streams.

    The undrifted stream is the paper's setup. The label-drifted stream is the only
    condition found so far in which a policy commits writes it should hold, so it is
    where propagation has any chance of occurring.

    Returns a report per stream per condition, plus a combined finding that states
    plainly whether the claim was exercised.
    """
    from ocm.evaluation.rahgm.corpus import Partition, generate_corpus
    from ocm.evaluation.rahgm.drift import drift_scenarios
    from ocm.evaluation.rahgm.replay import develop_policy

    corpus = corpus or generate_corpus()
    developed = developed or develop_policy(corpus)

    undrifted = list(corpus.partition(Partition.test))
    drifted = drift_scenarios(undrifted, severity=severity, seed=seed, mode="label")

    conditions = (
        Condition.autonomous_ocmr,
        Condition.fixed_threshold,
        Condition.frozen_rahgm,
    )
    streams = {"undrifted": undrifted, "label_drifted": drifted}

    results: dict[str, dict[str, Any]] = {}
    for stream_name, scenarios in streams.items():
        results[stream_name] = {}
        for condition in conditions:
            report = measure_cascade(
                scenarios, condition=condition, params=developed.params
            )
            results[stream_name][condition.value] = report.as_dict()

    total_upstream = sum(
        entry["n_upstream_errors"]
        for stream in results.values()
        for entry in stream.values()
    )
    total_propagated = sum(
        entry["n_propagated_writes"]
        for stream in results.values()
        for entry in stream.values()
    )

    # The mechanism test: inject the error rather than waiting for a condition to
    # make it, which separates "can an error propagate" from "does any condition err".
    injection = measure_injected_cascade(
        undrifted, condition=Condition.frozen_rahgm, params=developed.params
    )

    if total_upstream == 0:
        observed = (
            "No condition on either stream committed a write it should have held, so "
            "no naturally-occurring upstream error existed to propagate."
        )
        if injection["n_propagated_errors"] > 0:
            finding = (
                f"{observed} The mechanism is nonetheless real: injecting the error "
                f"directly produces {injection['n_propagated_errors']} downstream "
                f"errors across {injection['n_injections']} injections, and changes "
                f"{injection['n_downstream_verdicts_changed']} downstream OCMR "
                "verdicts. §3.1's claim holds for this workload, and the reason the "
                "replay metrics never exercise it is that every condition tested "
                "correctly holds the writes that would have started a cascade."
            )
        elif injection["n_downstream_verdicts_changed"] > 0:
            finding = (
                f"{observed} Injecting the error does reach later decisions: it "
                f"changes {injection['n_downstream_verdicts_changed']} downstream OCMR "
                f"constraint verdicts and leaves "
                f"{injection['durable_violation_delta']} extra durable-state "
                "violations. But the routed tier and the final transition label are "
                "unchanged, because the router discounts the newly-failing status "
                "check on authority and reversibility and commits anyway. So the "
                "propagation mechanism of §3.1 is confirmed, while the transition-"
                "level metrics are blind to it — the harm shows up as durable "
                "violations rather than as a mis-routed write."
            )
        else:
            finding = (
                f"{observed} Injecting the error directly also produces no downstream "
                "effect, so on this workload the chains do not in fact couple later "
                "decisions to earlier transitions. The §3.1 claim is not supported "
                "here."
            )
    elif total_propagated == 0:
        finding = (
            f"{total_upstream} upstream errors occurred and none produced a "
            "downstream error that the counterfactual removed. On this workload an "
            "erroneous transition did not influence later decisions, so "
            "order-preserving replay was not load-bearing for these metrics."
        )
    else:
        finding = (
            f"{total_propagated} downstream errors are causally attributable to "
            f"{total_upstream} upstream errors: they occur in the factual replay and "
            "disappear when the upstream transition alone is corrected. This is "
            "direct evidence for the §3.1 claim that an erroneous durable transition "
            "influences later states."
        )

    return {
        "experiment": "cascade_study_error_propagation",
        "in_paper": False,
        "claim": (
            "§3.1: an erroneous transition at time t remains available to influence "
            "all later states M_{t+1}, ..., M_T."
        ),
        "method": (
            "For each contention chain in which a condition commits a write whose gold "
            "transition is review, the scenario is replayed again with that single "
            "write held (force_hold) and everything else identical. Downstream errors "
            "that vanish under the intervention are causally attributable."
        ),
        "streams": results,
        "error_injection": injection,
        "totals": {
            "upstream_errors": total_upstream,
            "propagated_downstream_errors": total_propagated,
            "injected_propagated_errors": injection["n_propagated_errors"],
            "injected_verdicts_changed": injection["n_downstream_verdicts_changed"],
        },
        "finding": finding,
    }

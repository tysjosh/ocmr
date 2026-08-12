"""Table renderers and the mandatory scope note (Req 14.1, 14.2).

Every emitted artifact carries :data:`SCOPE_NOTE` at its top level, and every table
whose numbers depend on a model rather than a measurement carries a per-table
``modelled`` or ``simulated`` flag. The renderers here print that provenance above
the numbers, so a table cannot be read out of context.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

#: The mandatory top-level scope note.
SCOPE_NOTE = (
    "SCOPE AND PROVENANCE. "
    "The quarantine audit and Experiments 1, 3, and 4 are fully computed by "
    "replaying the real OCMR write path (W5, C1-C10, W7) over the generated "
    "evaluation corpus; their routing, integrity, and downstream numbers are "
    "measurements of this implementation. "
    "Experiment 2 is a SIMULATION: it reports a generative simulated-analyst "
    "model, not the paper's preregistered 80-participant human study, so it "
    "cannot answer RQ2 and any explanation-depth effect it shows restates the "
    "simulation's assumptions. "
    "Reviewer minutes (R100) come from an explicit review-cost model, not from "
    "measured human timing. "
    "Krippendorff's alpha is computed over two rubric-based annotator simulators, "
    "not human annotators. "
    "The 1,198-quarantine population in governance_examples.json was produced with "
    "a cached LLM extractor that is not part of this repository; it is reanalyzed "
    "as recorded, while the per-cause validity split and the alias attribution come "
    "from a reproducible offline replay with a smaller population."
)

#: Per-table provenance flags.
TABLE_PROVENANCE: dict[str, dict[str, Any]] = {
    "table2": {"computed": True, "note": "reanalysis of recorded + offline replay"},
    "table3": {"computed": True, "modelled_columns": ["R100"]},
    "table4": {"computed": True},
    "table5": {"simulated": True, "note": "simulated analyst, not human subjects"},
    "table6": {"computed": True},
    "table7": {"computed": True, "modelled_columns": ["R100"]},
}


def _fmt(value: Any, spec: str = ".2f", *, scale: float = 1.0) -> str:
    """Format a number, rendering ``None``/``nan`` as an em dash."""
    if value is None:
        return "—"
    try:
        number = float(value) * scale
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # nan
        return "—"
    return format(number, spec)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a fixed-width text table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join("-" * w for w in widths)
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)), line]
    for row in rows:
        out.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Table 2 — quarantine audit
# --------------------------------------------------------------------------- #
def render_table2(audit: Mapping[str, Any]) -> str:
    """Render Table 2: the quarantine audit."""
    out: list[str] = ["Table 2: Quarantine audit (paper §4.1)"]

    recorded = audit.get("recorded_population")
    if recorded:
        out.append(
            f"\n  Recorded population (governance_examples.json, "
            f"{recorded['n_quarantines']} quarantines, "
            f"{recorded['n_distinct_reasons']} distinct reasons)"
        )
        rows = [
            [
                row["cause"],
                str(row["count"]),
                "—",
                _fmt(row["review_pct"], ".1f"),
                row["ocmr_check"] or "—",
            ]
            for row in recorded["rows"]
        ]
        rows.append(
            [
                "Total",
                str(recorded["n_quarantines"]),
                _fmt(recorded["recorded_false_quarantine_pct"], ".1f"),
                _fmt(recorded["review_worthy_pct"], ".1f"),
                "",
            ]
        )
        out.append(
            _table(["Primary cause", "Count", "Valid (%)", "Review (%)", "OCMR check"], rows)
        )
        out.append(
            f"  Recorded false quarantines: {recorded['recorded_false_quarantine']} "
            f"({_fmt(recorded['recorded_false_quarantine_pct'], '.1f')}%). "
            "Per-cause validity is unavailable in the artifact (separate marginals)."
        )

    offline = audit.get("offline_replay")
    if offline:
        out.append(
            f"\n  Offline replay (reproducible, mock extractor, "
            f"{offline['n_quarantines']} quarantines) "
            "— structural identity check applied"
        )
        rows = [
            [
                row["cause"],
                str(row["count"]),
                _fmt(row["valid_pct"], ".1f"),
                _fmt(row["review_pct"], ".1f"),
            ]
            for row in offline["rows"]
        ]
        rows.append(
            [
                "Total",
                str(offline["n_quarantines"]),
                _fmt(offline["valid_pct"], ".1f"),
                _fmt(offline["review_worthy_pct"], ".1f"),
            ]
        )
        out.append(_table(["Primary cause", "Count", "Valid (%)", "Review (%)"], rows))

    classifiers = audit.get("classifiers")
    if classifiers:
        out.append(
            "\n  NOTE: the two populations use different classifiers and their cause "
            "rows are not directly comparable."
        )
        out.append(f"    recorded: {classifiers['recorded_population']}")
        out.append(f"    offline:  {classifiers['offline_replay']}")

    alias = audit.get("alias_attribution")
    if alias:
        out.append(
            "\n  Entity-identity attribution (shared store vs fresh store per example)"
        )
        out.append(
            f"    quarantines: {alias['shared_store_quarantines']} -> "
            f"{alias['isolated_quarantines']} "
            f"({_fmt(alias['pct_quarantines_attributable_to_identifier_reuse'], '.1f')}% "
            "removed by isolation)"
        )
        out.append(
            f"    false quarantines: {alias['shared_store_false_quarantines']} -> "
            f"{alias['isolated_false_quarantines']} "
            f"({_fmt(alias['pct_false_quarantines_attributable_to_identifier_reuse'], '.1f')}% "
            "removed by isolation)"
        )

    agreement = audit.get("agreement") or {}
    if agreement:
        parts = [
            f"{field}={_fmt(value.get('krippendorff_alpha'), '.3f')}"
            for field, value in agreement.items()
        ]
        out.append(
            "\n  Krippendorff's alpha (SIMULATED annotators): " + ", ".join(parts)
        )
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Table 3 — five-arm replay
# --------------------------------------------------------------------------- #
def render_table3(experiment1: Mapping[str, Any]) -> str:
    """Render Table 3: the five-arm controlled replay."""
    conditions = experiment1["conditions"]
    order = [
        "universal_review",
        "autonomous_ocmr",
        "fixed_threshold",
        "frozen_rahgm",
        "adaptive_rahgm",
    ]
    rows: list[list[str]] = []
    for key in order:
        entry = conditions.get(key)
        if entry is None:
            continue
        m = entry["metrics"]
        rows.append(
            [
                entry["name"],
                _fmt(m["dvr"], ".2f", scale=100.0),
                _fmt(m["mcr"], ".2f", scale=100.0),
                _fmt(m["review_rate"], ".1f", scale=100.0),
                _fmt(m["r100"], ".1f"),
                _fmt(m["false_quarantine_rate"], ".2f", scale=100.0),
                _fmt(m["accuracy"], ".1f", scale=100.0),
            ]
        )
    out = [
        "Table 3: Five-arm controlled replay (paper §4.2). "
        "Lower is better for DVR, MCR, RR, R100, false quar.; higher for accuracy.",
        "  R100 is MODELLED (review-cost model), not measured human timing.",
        _table(
            [
                "Condition",
                "DVR (%)",
                "MCR (%)",
                "RR (%)",
                "R100",
                "False quar. (%)",
                "Accuracy (%)",
            ],
            rows,
        ),
    ]

    criteria = experiment1.get("success_criteria")
    if criteria:
        status = "MET" if criteria["met"] else "NOT MET (strict reading)"
        out.append(f"\n  Preregistered success criteria: {status}")
        out.append(
            f"    R100(C5)={_fmt(criteria['r100_adaptive'], '.1f')} < "
            f"R100(C1)={_fmt(criteria['r100_universal'], '.1f')}: "
            f"{criteria['r100_below_universal']}"
        )
        out.append(
            f"    DVR(C5)-DVR(C2)={_fmt(criteria['dvr_delta'], '.4f')} <= 0.005: "
            f"{criteria['dvr_within_tolerance']}"
        )
        out.append(
            f"    MCR(C5)={_fmt(criteria['mcr_adaptive'], '.4f')} < "
            f"MCR(C3)={_fmt(criteria['mcr_fixed'], '.4f')}: "
            f"{criteria['mcr_below_fixed']}"
        )
        if criteria.get("interpretation"):
            out.append(f"    {criteria['interpretation']}")

    generalization = experiment1.get("threshold_generalization")
    if generalization:
        out.append(
            "\n  Threshold generalization (eq. 5 is enforced on development only):"
        )
        rows = [
            [
                _fmt(row["dev_mcr_ceiling"], ".3f"),
                _fmt(row["tau_l"], ".2f"),
                _fmt(row["dev_mcr"], ".4f"),
                _fmt(row["test_mcr"], ".4f"),
                _fmt(row["test_review_rate"], ".1f", scale=100.0),
            ]
            for row in generalization["rows"]
        ]
        out.append(
            "  "
            + _table(
                ["Dev MCR ceiling", "tau_l", "Dev MCR", "Test MCR", "Test RR (%)"],
                rows,
            ).replace("\n", "\n  ")
        )
        at = generalization.get("zero_test_mcr_at_ceiling")
        out.append(
            f"    Zero held-out MCR first reached at a dev ceiling of "
            f"{_fmt(at, '.3f')}; the paper specifies {_fmt(generalization['paper_ceiling'], '.2f')}."
        )
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Table 4 — routing ablation
# --------------------------------------------------------------------------- #
def render_table4(ablation: Mapping[str, Any]) -> str:
    """Render Table 4: the routing ablation."""
    rows = [
        [
            v["name"],
            _fmt(v["mcr"], ".2f", scale=100.0),
            _fmt(v["review_rate"], ".1f", scale=100.0),
            _fmt(v["risk_coverage_auc"], ".4f"),
            _fmt(v["false_quarantine_rate"], ".2f", scale=100.0),
            _fmt(v["queue_precision"], ".3f"),
            _fmt(v["queue_recall"], ".3f"),
        ]
        for v in ablation["variants"]
    ]
    out = [
        "Table 4: Routing ablation (paper §4.2). Lower risk–coverage AUC is better.",
        _table(
            [
                "Routing policy",
                "MCR (%)",
                "RR (%)",
                "Risk–cov. AUC",
                "False quar. (%)",
                "Queue prec.",
                "Queue rec.",
            ],
            rows,
        ),
    ]
    comparison = ablation.get("full_vs_strongest_baseline")
    if comparison:
        out.append(
            f"\n  Strongest baseline: {comparison['strongest_baseline']} "
            f"(AUC {_fmt(comparison['auc_baseline'], '.4f')} vs full "
            f"{_fmt(comparison['auc_full'], '.4f')})"
        )
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Table 5 — human study (simulated)
# --------------------------------------------------------------------------- #
def render_table5(experiment2: Mapping[str, Any]) -> str:
    """Render Table 5: the simulated explanation-depth study."""
    rows = []
    for key, entry in experiment2["by_condition"].items():
        if not entry.get("n_items"):
            continue
        rows.append(
            [
                key,
                _fmt(entry["accuracy"], ".1f", scale=100.0),
                _fmt(entry["median_seconds"], ".1f"),
                _fmt(entry["workload_tlx"], ".1f"),
                _fmt(entry["ece"], ".3f"),
                _fmt(entry["recommendation_following_rate"], ".3f"),
            ]
        )
    out = [
        "Table 5: Explanation depth and reliance — *** SIMULATED ANALYST ***",
        "  NOT the paper's 80-participant human study. Cannot answer RQ2.",
        _table(
            ["Condition", "Acc. (%)", "Time (s)", "TLX", "ECE", "Follow rate"], rows
        ),
    ]

    depth_rows = []
    for depth, entry in experiment2["by_depth"].items():
        if not entry.get("n_items"):
            continue
        depth_rows.append(
            [
                depth,
                _fmt(entry["accuracy"], ".1f", scale=100.0),
                _fmt(entry["median_seconds"], ".1f"),
                _fmt(entry["workload_tlx"], ".1f"),
                _fmt(entry["ece"], ".3f"),
                _fmt(entry["mean_evidence_opened"], ".2f"),
            ]
        )
    out.append("\n  By explanation depth (pooled across conditions):")
    out.append(
        _table(
            ["Depth", "Acc. (%)", "Time (s)", "TLX", "ECE", "Evid. opened"], depth_rows
        )
    )

    complacency = experiment2.get("complacency") or {}
    if complacency:
        out.append(
            f"\n  Complacency: recommendation-following changes "
            f"{_fmt(complacency.get('recommendation_following_change_points'), '+.1f')} pts "
            f"and accuracy {_fmt(complacency.get('accuracy_change_points'), '+.1f')} pts "
            "from first to last scenario."
        )

    contrasts = experiment2.get("primary_contrasts") or {}
    for entry in contrasts.get("holm", []):
        out.append(
            f"    {entry['name']}: p={_fmt(entry['p_value'], '.4g')}, "
            f"Holm-adjusted p={_fmt(entry['adjusted_p'], '.4g')}, "
            f"rejected={entry['rejected']}"
        )
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Table 6 — adaptation safety
# --------------------------------------------------------------------------- #
def render_table6(experiment3: Mapping[str, Any]) -> str:
    """Render Table 6: adaptation safety."""
    rows = []
    for arm, entry in experiment3["table6"].items():
        proposed = entry.get("n_proposed_total", 0)
        rows.append(
            [
                arm,
                str(proposed),
                "—" if not proposed else _fmt(entry["accept_pct"], ".1f"),
                _fmt(entry["delta_dvr_worst"], ".4f"),
                _fmt(entry["delta_mcr_worst"], ".4f"),
                _fmt(entry["max_drift"], ".4f"),
                str(entry["tier_disablement_runs"]),
            ]
        )
    out = [
        "Table 6: Safe feedback adaptation (paper §4.4). Worst case across runs.",
        _table(
            [
                "Policy",
                "Proposals",
                "Accept (%)",
                "ΔDVR",
                "ΔMCR",
                "Max drift",
                "Tier disabl.",
            ],
            rows,
        ),
    ]
    summary = experiment3.get("safety_summary") or {}
    if summary:
        out.append(
            f"\n  Gated policy accepts {_fmt(summary.get('gated_clean_accept_pct'), '.1f')}% "
            f"of clean updates and blocks "
            f"{_fmt(summary.get('gated_adversarial_block_pct'), '.1f')}% of adversarial ones."
        )
        out.append(
            f"  Worst post-update DVR increase: gated "
            f"{_fmt(summary.get('gated_worst_dvr_increase'), '.4f')}, "
            f"ungated {_fmt(summary.get('ungated_worst_dvr_increase'), '.4f')}, "
            f"unconstrained {_fmt(summary.get('unconstrained_worst_dvr_increase'), '.4f')}."
        )
        out.append(
            f"  Tier disablement observed in {summary.get('tier_disablement_runs')} of "
            f"{summary.get('total_runs')} runs."
        )

    by_stream = experiment3.get("by_arm_and_stream") or {}
    if by_stream:
        stream_rows = [
            [
                key,
                _fmt(entry["accept_pct"], ".1f"),
                _fmt(entry["blocked_pct"], ".1f"),
                _fmt(entry["delta_dvr_worst"], ".4f"),
                _fmt(entry["max_drift"], ".4f"),
            ]
            for key, entry in by_stream.items()
        ]
        out.append("\n  By policy and feedback stream:")
        out.append(
            _table(
                ["Policy|Stream", "Accept (%)", "Blocked (%)", "ΔDVR worst", "Max drift"],
                stream_rows,
            )
        )
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Table 7 — end to end
# --------------------------------------------------------------------------- #
def render_table7(experiment4: Mapping[str, Any]) -> str:
    """Render Table 7: end-to-end analytic outcomes."""
    rows = [
        [
            entry["name"],
            _fmt(entry["answer_accuracy"], ".1f", scale=100.0),
            _fmt(entry["unsupported_rate"], ".1f", scale=100.0),
            _fmt(entry["stale_propagation_rate"], ".1f", scale=100.0),
            _fmt(entry["abstention_rate"], ".1f", scale=100.0),
            _fmt(entry["r100"], ".1f"),
        ]
        for entry in experiment4["table7"]
    ]
    out = [
        "Table 7: End-to-end analytic outcomes (paper §4.5). R100 is MODELLED.",
        _table(
            [
                "Condition",
                "Answer acc. (%)",
                "Unsupported (%)",
                "Stale prop. (%)",
                "Abstained (%)",
                "R100",
            ],
            rows,
        ),
    ]
    comparison = experiment4.get("adaptive_vs_autonomous")
    if comparison:
        out.append(
            f"\n  Adaptive RAHGM vs autonomous OCMR: answer accuracy "
            f"{_fmt(comparison['answer_accuracy_delta_points'], '+.1f')} pts, "
            f"stale propagation {_fmt(comparison['stale_propagation_delta_points'], '+.1f')} pts, "
            f"ΔDVR {_fmt(comparison['dvr_delta'], '+.4f')}, "
            f"at {_fmt(comparison['analyst_minutes_per_100_writes'], '.1f')} analyst "
            "minutes per 100 writes."
        )
        largest_gain = comparison.get("largest_gain")
        largest_cost = comparison.get("largest_cost")
        if largest_gain:
            out.append(
                f"  Largest gain: {_fmt(largest_gain['transition_accuracy_delta'], '+.1f')} pts "
                f"in {largest_gain['capability']}."
            )
        if largest_cost:
            out.append(
                f"  Largest cost: {_fmt(largest_cost['transition_accuracy_delta'], '+.1f')} pts "
                f"in {largest_cost['capability']}."
            )
    return "\n".join(out)


def render_drift(drift: Mapping[str, Any]) -> str:
    """Render the drift study: when does bounded adaptation earn its keep?"""
    out = [
        "Drift study: adaptive vs frozen under distribution shift "
        "*** NOT IN THE PAPER ***",
        "  §3.5 compares the two arms on the fitting distribution, where they are",
        "  indistinguishable. This supplies the condition under which they differ.",
    ]
    for mode_key, section in drift.items():
        if not isinstance(section, dict) or "arms" not in section:
            continue
        info = section["drift"]
        out.append(
            f"\n  {mode_key} drift "
            f"(severity {_fmt(info['severity'], '.2f')}, "
            f"{info['n_relabelled_per_pass']} writes relabelled per pass, "
            f"stream {info['n_writes_in_stream']} writes)"
        )
        out.append(f"    {info['transformation']}")
        rows = [
            [
                arm["arm"],
                _fmt(arm["accuracy"], ".1f", scale=100.0),
                _fmt(arm["mcr"], ".2f", scale=100.0),
                _fmt(arm["review_rate"], ".1f", scale=100.0),
                _fmt(arm["early_accuracy"], ".1f", scale=100.0),
                _fmt(arm["late_accuracy"], ".1f", scale=100.0),
                _fmt(arm["recovery"], "+.1f", scale=100.0),
                f"{arm['n_deployed']}/{arm['n_proposed']}",
            ]
            for arm in section["arms"]
        ]
        out.append(
            "  "
            + _table(
                [
                    "Arm",
                    "Acc. (%)",
                    "MCR (%)",
                    "RR (%)",
                    "Early acc.",
                    "Late acc.",
                    "Recovery",
                    "Deployed",
                ],
                rows,
            ).replace("\n", "\n  ")
        )
        contrast = section["contrast"]
        out.append(
            f"    adaptive - frozen: accuracy "
            f"{_fmt(contrast['accuracy_delta_points'], '+.2f')} pts, "
            f"recovery advantage "
            f"{_fmt(contrast['recovery_advantage_points'], '+.2f')} pts, "
            f"review rate {_fmt(contrast['review_rate_delta_points'], '+.2f')} pts"
        )
    if drift.get("finding"):
        out.append(f"\n  {drift['finding']}")
    return "\n".join(out)


def render_cascade(cascade: Mapping[str, Any]) -> str:
    """Render the cascade study: does an error at t influence later states?"""
    out = [
        "Cascade study: error propagation within contention chains "
        "*** NOT IN THE PAPER ***",
        f"  Claim under test — {cascade['claim']}",
    ]

    out.append("\n  Observed: did any condition commit a write it should have held?")
    rows = []
    for stream, conditions in cascade["streams"].items():
        for name, entry in conditions.items():
            rows.append(
                [
                    stream,
                    name,
                    str(entry["n_chains"]),
                    str(entry["n_upstream_errors"]),
                    str(entry["n_propagated_writes"]),
                ]
            )
    out.append(
        "  "
        + _table(
            ["Stream", "Condition", "Chains", "Upstream err.", "Propagated"], rows
        ).replace("\n", "\n  ")
    )

    injection = cascade.get("error_injection")
    if injection:
        out.append(
            "\n  Mechanism test: the upstream error is injected rather than waited for."
        )
        inj_rows = [
            [
                template,
                str(v["injections"]),
                str(v["propagated"]),
                str(v["verdict_changed"]),
                str(v["tier_changed"]),
                str(v["violation_delta"]),
            ]
            for template, v in injection["by_injected_template"].items()
        ]
        out.append(
            "  "
            + _table(
                [
                    "Injected at",
                    "Injections",
                    "Errors",
                    "OCMR verdict chg.",
                    "Tier chg.",
                    "Extra violations",
                ],
                inj_rows,
            ).replace("\n", "\n  ")
        )

    out.append(f"\n  {cascade['finding']}")
    return "\n".join(out)


def render_all(report: Mapping[str, Any]) -> str:
    """Render every available table with the scope note on top."""
    out = [SCOPE_NOTE, ""]
    renderers = (
        ("quarantine_audit", render_table2),
        ("experiment1", render_table3),
        ("ablation", render_table4),
        ("experiment2", render_table5),
        ("experiment3", render_table6),
        ("experiment4", render_table7),
        ("cascade_study", render_cascade),
        ("drift_study", render_drift),
    )
    for key, renderer in renderers:
        section = report.get(key)
        if not section:
            continue
        out.append("=" * 78)
        out.append(renderer(section))
        out.append("")
    return "\n".join(out)

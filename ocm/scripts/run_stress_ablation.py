"""Entry-point: run the Schema/Provenance Stress Workload ablation.

Mirrors :mod:`ocm.scripts.run_experiments`: a thin ``main()`` that runs the
targeted stress-workload diagnostic across the four governance arms
(Ungoverned, Gate_Only, Schema_Provenance, Full), prints a plain-text
arm x violation-type table, and writes the same content to a results file
under ``local_results/`` (Req 14.1, 14.3, 8.1, 10.4).

The workload is a **targeted diagnostic, not a real-benchmark result** — the
``Diagnostic_Scope_Note`` is therefore printed as the **first and last** lines
of output so no reader mistakes the table for an emergent real-data finding
(Req 14.1, 14.3). The ``Gate_Only_Arm`` row is flagged as the decisive
comparison (Req 10.4): fed the same inputs as every other arm, it still leaves
the invalid durable state the Schema_Provenance_Arm removes.

Usage::

    python -m ocm.scripts.run_stress_ablation                 # default seed 1337
    python -m ocm.scripts.run_stress_ablation --seed 7        # a different seed
    python -m ocm.scripts.run_stress_ablation --out foo.txt   # custom results path
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Optional, Sequence

from ocm.evaluation.stress_ablation import (
    DECISIVE_ARM,
    StressAblationResult,
    run_stress_ablation,
)
from ocm.evaluation.typed_violations import TypedViolationReport


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocm-run-stress-ablation",
        description=(
            "Run the Schema/Provenance Stress Workload ablation across the four "
            "governance arms and print the typed-violation table."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=1337,
        help="Single seed for the deterministic offline workload (default: 1337).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Results file path (default: local_results/stress_ablation_results.txt).",
    )
    return parser


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
#: Columns of the table: the four typed-violation types, the total, then the
#: four write-outcome tally buckets (no new outcome categories — Req 8.1, 8.2).
_COLUMNS: list[tuple[str, str]] = [
    ("schema_invalid", "schema_invalid"),
    ("unsupported_final_decision", "unsupported_final"),
    ("temporally_invalid_interval", "temporal_invalid"),
    ("illegal_status_state", "illegal_status"),
    ("total", "TOTAL"),
    ("accepted", "acc"),
    ("superseded", "sup"),
    ("quarantined", "quar"),
    ("rejected", "rej"),
]


def _cell_value(report: TypedViolationReport, field: str) -> int:
    """Resolve one column value from a report (typed counts + tally buckets)."""
    if field in {"accepted", "superseded", "quarantined", "rejected"}:
        return int(getattr(report.write_outcomes, field))
    return int(getattr(report, field))


def _wrap_note(note: str, width: int = 78) -> str:
    """Wrap the Diagnostic_Scope_Note to a readable width."""
    return "\n".join(textwrap.wrap(note, width=width)) if note else ""


def render_report(result: StressAblationResult) -> str:
    """Render the full plain-text output (note first, table, note last).

    The ``Diagnostic_Scope_Note`` is emitted as the **first and last** lines of
    output (Req 14.1, 14.3); the ``Gate_Only_Arm`` row is flagged decisive
    (Req 10.4). The table columns are the four typed-violation types, the total,
    and the four write-outcome tally buckets (Req 8.1).
    """
    note_block = _wrap_note(result.diagnostic_scope_note)

    # Column layout.
    arm_header = "Arm"
    arm_width = max(len(arm_header), *(len(f"{a} *") for a in result.arms))
    headers = [label for _, label in _COLUMNS]
    col_widths = [max(len(label), 5) for label in headers]

    def _row(arm_label: str, cells: list[str]) -> str:
        parts = [arm_label.ljust(arm_width)]
        parts += [c.rjust(w) for c, w in zip(cells, col_widths)]
        return "  ".join(parts)

    header_line = _row(arm_header, headers)
    sep_line = "-" * len(header_line)

    body_lines: list[str] = []
    for arm, report in result.arms.items():
        decisive = result.is_decisive(arm)
        arm_label = f"{arm} *" if decisive else arm
        cells = [str(_cell_value(report, field)) for field, _ in _COLUMNS]
        body_lines.append(_row(arm_label, cells))

    legend = (
        f"(* = decisive comparison: {DECISIVE_ARM} receives the same inputs as every "
        f"arm yet still leaves invalid durable state)"
    )
    columns_legend = (
        "Columns: four typed-violation types | TOTAL | write-outcome tally "
        "(acc=accepted, sup=superseded, quar=quarantined, rej=rejected)"
    )

    lines: list[str] = []
    # First line(s): the Diagnostic_Scope_Note.
    lines.append(note_block)
    lines.append("")
    lines.append(f"Schema/Provenance Stress Workload ablation (seed={result.seed})")
    lines.append("")
    lines.append(header_line)
    lines.append(sep_line)
    lines.extend(body_lines)
    lines.append("")
    lines.append(legend)
    lines.append(columns_legend)
    lines.append("")
    # Last line(s): the Diagnostic_Scope_Note again.
    lines.append(note_block)

    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # Quiet the expected governance warning logs so the table reads cleanly.
    import logging

    logging.getLogger("ocm").setLevel(logging.ERROR)

    result = run_stress_ablation(seed=args.seed)
    output = render_report(result)

    print(output)

    repo_dir = Path(__file__).resolve().parents[2]
    out_path = (
        Path(args.out).resolve()
        if args.out
        else (repo_dir / "local_results" / "stress_ablation_results.txt")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output + "\n", encoding="utf-8")
    print(f"\nWrote stress ablation results to {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

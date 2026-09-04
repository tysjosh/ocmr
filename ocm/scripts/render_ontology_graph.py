"""Render the typed-graph schema as Markdown + Mermaid from the live registry.

The ontology is declared once, in
:data:`ocm.ontology.relations.RELATION_SIGNATURES`, and referenced by the
validators, the retrievers, and every metric that groups by
``(subject, predicate)``. A hand-drawn diagram of it drifts silently — the
``# Frozen registry of all 14 relations`` comment in ``relations.py`` was written
when there were 14 and stayed put when ``HAS_VALUE`` made it 15.

So this generates the diagram instead. Run it after any ontology change and
commit the result:

    python -m ocm.scripts.render_ontology_graph --out docs/ontology_graph.md

Fully expanded, the schema is 68 ``(source_type, predicate, target_type)`` edges,
which is unreadable as one flat picture. The rendering therefore splits into a
structural diagram plus a fan-out table, and draws the widest relations
(``POSSIBLY_SAME_AS`` at 25 pairs, ``ABOUT`` at 10) in the table only. The
abridgement is stated in the output so a reader is never misled about what the
diagram omits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

from ocm.ontology.relations import RELATION_SIGNATURES, Cardinality

#: Cardinalities on which a second distinct object is a contradiction.
SINGLE_VALUED = {Cardinality.M_TO_ONE, Cardinality.ONE_TO_ONE}

#: Relations too wide to draw; rendered in the fan-out table only.
TABLE_ONLY = {"POSSIBLY_SAME_AS", "ABOUT"}

#: Visual grouping of entity types. Types absent here fall into "other".
CLUSTERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("core domain", ("Person", "Organization", "Project", "Task", "Event")),
    ("epistemic / reified", ("Claim", "Decision", "Document", "Assertion")),
    ("value nodes", ("Slot", "SlotValue", "StatusValue")),
)


def entity_types() -> list[str]:
    """Every type appearing at either end of a declared relation."""
    seen: set[str] = set()
    for sig in RELATION_SIGNATURES.values():
        seen |= set(sig.source_types) | set(sig.target_types)
    return sorted(seen)


def expanded_pairs(name: str) -> int:
    sig = RELATION_SIGNATURES[name]
    return len(sig.source_types) * len(sig.target_types)


def is_single_valued(name: str) -> bool:
    return RELATION_SIGNATURES[name].cardinality in SINGLE_VALUED


def _edges_for(name: str) -> list[tuple[str, str]]:
    sig = RELATION_SIGNATURES[name]
    return [
        (s, t) for s in sorted(sig.source_types) for t in sorted(sig.target_types)
    ]


def render_mermaid(max_pairs: int = 6) -> str:
    """Mermaid diagram: all relations except the table-only and over-wide ones."""
    lines = ["```mermaid", "graph LR"]

    grouped: set[str] = set()
    for label, members in CLUSTERS:
        present = [t for t in members if t in entity_types()]
        if not present:
            continue
        grouped |= set(present)
        key = label.replace(" ", "_").replace("/", "_")
        lines.append(f'  subgraph {key}["{label}"]')
        lines.append("    " + "; ".join(present))
        lines.append("  end")
    other = [t for t in entity_types() if t not in grouped]
    if other:
        lines.append('  subgraph other["other"]')
        lines.append("    " + "; ".join(other))
        lines.append("  end")

    def emit(names: Iterable[str], heading: str, thick: bool) -> None:
        block: list[str] = []
        for name in names:
            if name in TABLE_ONLY or expanded_pairs(name) > max_pairs:
                continue
            sig = RELATION_SIGNATURES[name]
            star = " *" if is_single_valued(name) else ""
            arrow = "==>" if thick else "-->"
            for src, tgt in _edges_for(name):
                block.append(
                    f'  {src} {arrow}|"{name}{star} {sig.cardinality.value}"| {tgt}'
                )
        if block:
            lines.append(f"\n  %% ---- {heading} ----")
            lines.extend(block)

    sv = [n for n in RELATION_SIGNATURES if is_single_valued(n)]
    mv = [n for n in RELATION_SIGNATURES if not is_single_valued(n)]
    emit(sv, "single-valued (m:1 / 1:1) - the only conflictable relations", True)
    emit(mv, "many-valued", False)

    lines.append("```")
    return "\n".join(lines)


def render_markdown() -> str:
    total = sum(expanded_pairs(n) for n in RELATION_SIGNATURES)
    sv = [n for n in RELATION_SIGNATURES if is_single_valued(n)]
    omitted = sorted(
        n for n in RELATION_SIGNATURES if n in TABLE_ONLY or expanded_pairs(n) > 6
    )

    out: list[str] = []
    out.append("# OCMR typed graph\n")
    out.append(
        "**Generated** by `python -m ocm.scripts.render_ontology_graph` from "
        "`ocm.ontology.relations.RELATION_SIGNATURES`. Do not edit by hand; "
        "re-run after any ontology change.\n"
    )
    out.append(
        f"{len(RELATION_SIGNATURES)} relations over {len(entity_types())} entity "
        f"types, {total} fully expanded `(source, predicate, target)` type pairs.\n"
    )
    out.append(
        "Entity types are **nodes**; relations are **directed edges** keyed by "
        "predicate in a `networkx.MultiDiGraph` (`ocm/memory/graph_store.py`). "
        "Only `accepted` assertions become edges — superseded, quarantined and "
        "rejected assertions persist as rows in the repository but never appear "
        "in the graph.\n"
    )

    out.append("## Diagram\n")
    if omitted:
        out.append(
            "Abridged: "
            + ", ".join(f"`{n}` ({expanded_pairs(n)} pairs)" for n in omitted)
            + " are too wide to draw and appear in the fan-out table only.\n"
        )
    out.append(render_mermaid())
    out.append(
        "\n`*` and thick arrows mark the single-valued relations. A second "
        "distinct object on the same subject is a contradiction **only** for "
        "these, so they are the only relations `durable_constraint_violations` "
        "can measure.\n"
    )

    out.append("## Relations\n")
    out.append("| relation | cardinality | single-valued | sources | targets | pairs |")
    out.append("| --- | --- | :-: | --- | --- | --: |")
    for name, sig in RELATION_SIGNATURES.items():
        out.append(
            "| `%s` | %s | %s | %s | %s | %d |"
            % (
                name,
                sig.cardinality.value,
                "**yes**" if is_single_valued(name) else "-",
                ", ".join(sorted(sig.source_types)),
                ", ".join(sorted(sig.target_types)),
                expanded_pairs(name),
            )
        )

    out.append(
        f"\nSingle-valued: **{len(sv)} of {len(RELATION_SIGNATURES)}** — "
        + ", ".join(f"`{n}`" for n in sv)
        + ".\n"
    )

    out.append("## Entity types\n")
    for label, members in CLUSTERS:
        present = [t for t in members if t in entity_types()]
        if present:
            out.append(f"- **{label}**: " + ", ".join(f"`{t}`" for t in present))
    other = [
        t
        for t in entity_types()
        if not any(t in m for _, m in CLUSTERS)
    ]
    if other:
        out.append("- **other**: " + ", ".join(f"`{t}`" for t in other))
    out.append("")
    return "\n".join(out)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ocm.scripts.render_ontology_graph",
        description="Render the typed-graph schema from the relation registry.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write Markdown here (default: stdout).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if --out is stale relative to the registry (for CI).",
    )
    args = parser.parse_args(argv)

    rendered = render_markdown()
    if args.check:
        if args.out is None:
            print("--check requires --out", file=sys.stderr)
            return 2
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if current != rendered:
            print(f"STALE: {args.out} does not match the registry", file=sys.stderr)
            return 1
        print(f"up to date: {args.out}")
        return 0
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

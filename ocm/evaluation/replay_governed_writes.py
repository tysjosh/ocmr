"""Governed-write replay: dump real accepted/superseded/quarantined examples.

This is a **diagnostic and evidence-collection tool**, not part of the scored
protocol. It replays the benchmark sessions through the *full governed* write
path (B3 by default) and captures the concrete :class:`WriteOutcome` the pipeline
produced for every candidate — so the paper's qualitative tables (the
failure-audit / governance-in-action examples) and the false-quarantine
discussion are backed by *real, reproducible* rows from the system instead of
hand-written illustrations.

For every committed candidate it records the governing decision
(accepted / superseded / quarantined / rejected), the human-readable
subject/predicate/object, the originating session text and ``source_ref``, the
governing ``reason``, and (for quarantines) the durable
:class:`~ocm.ontology.models.QuarantineRecord`'s severity + conflicting ids.

It additionally flags **false quarantines** heuristically: a write-time
quarantine inside an example whose questions expect *no* conflict. These are the
governance false positives the reviewer asked to see quantified and exemplified
(the same population the τ-sweep's false-quarantine-rate column counts at
answer time, here surfaced as concrete rows).

Usage (notebook / Colab, with the cached Qwen extractor already loaded)::

    from ocm.evaluation.replay_governed_writes import replay_governed_writes
    report = replay_governed_writes(
        extractor=extractor, embeddings=embeddings,
        out_path="/content/drive/MyDrive/ocm_results/governance_examples.json",
    )

Usage (CLI, offline mock extractor)::

    python -m ocm.evaluation.replay_governed_writes --per-category 6 --out examples.json

The function returns a structured report and (when ``verbose``) prints a readable
audit. It is deterministic for a given seed with the offline extractor, and
reproducible for the LLM run when a populated extraction cache is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Iterable, Optional

from ocm.core.config import Settings
from ocm.evaluation.benchmark import BenchmarkGenerator
from ocm.evaluation.experiment import _build_strategy, _default_settings, _seed_everything
from ocm.memory.contracts import WriteOutcome

#: Decision buckets in audit-display order.
_BUCKETS: tuple[str, ...] = ("accepted", "superseded", "quarantined", "rejected")

#: Attribute names tried (in order) to derive a human-readable label for an id.
_LABEL_ATTRS: tuple[str, ...] = ("name", "title", "summary", "description", "value")


def _entity_labels(container: Any) -> dict[str, str]:
    """Best-effort ``entity_id -> display label`` map from the durable store.

    ``repo.list_entities()`` yields ``(type, attrs_dict)`` tuples. We pull the id
    and the first recognized label attribute from the dict. Falls back to the raw
    id for any node without a recognized label (e.g. a ``status:done`` id is
    already readable). Defensive: any storage error yields an empty map so the
    replay still produces id-only rows.
    """
    labels: dict[str, str] = {}
    try:
        entities = list(container.repo.list_entities())
    except Exception:  # pragma: no cover - defensive
        return labels
    for ent in entities:
        # Support both (type, attrs) tuples and model objects.
        if isinstance(ent, tuple) and len(ent) == 2 and isinstance(ent[1], dict):
            attrs = ent[1]
            eid = attrs.get("id")
            label = next((str(attrs[a]) for a in _LABEL_ATTRS if attrs.get(a)), None)
        else:
            eid = getattr(ent, "id", None)
            label = next((str(getattr(ent, a)) for a in _LABEL_ATTRS if getattr(ent, a, None)), None)
        if not eid:
            continue
        labels[eid] = label or str(eid)
    return labels


def _quarantine_index(container: Any) -> dict[str, Any]:
    """Map ``quarantine_id -> QuarantineRecord`` from the durable store."""
    index: dict[str, Any] = {}
    try:
        records = list(container.repo.list_quarantine())
    except Exception:  # pragma: no cover - defensive
        return index
    for rec in records:
        rid = getattr(rec, "id", None)
        if rid:
            index[rid] = rec
    return index


def _outcome_row(
    outcome: WriteOutcome,
    *,
    example_id: str,
    category: str,
    source_ref: str,
    session_text: str,
    labels: dict[str, str],
    quarantines: dict[str, Any],
) -> dict[str, Any]:
    """Serialize one :class:`WriteOutcome` into a compact, readable audit row."""
    cand = outcome.candidate
    subj = labels.get(cand.subject_id, cand.subject_id)
    obj = labels.get(cand.object_id, cand.object_id)
    row: dict[str, Any] = {
        "example_id": example_id,
        "category": category,
        "source_ref": source_ref,
        "session_text": session_text,
        "decision": outcome.decision,
        "triple": f"{subj} -[{cand.predicate}]-> {obj}",
        "subject_id": cand.subject_id,
        "predicate": cand.predicate,
        "object_id": cand.object_id,
        "write_intent": getattr(cand.write_intent, "value", str(cand.write_intent)),
        "confidence": float(cand.confidence),
        "reason": outcome.reason,
        "assertion_id": outcome.assertion_id,
        "superseded_assertion_id": outcome.superseded_assertion_id,
        "quarantine_id": outcome.quarantine_id,
    }
    # Enrich quarantines with the durable record's severity + conflicting ids.
    if outcome.decision == "quarantined" and outcome.quarantine_id in quarantines:
        rec = quarantines[outcome.quarantine_id]
        row["severity"] = getattr(getattr(rec, "severity", None), "value", None) or str(
            getattr(rec, "severity", "")
        )
        conflicting = list(getattr(rec, "conflicting_ids", []) or [])
        row["conflicting_ids"] = conflicting
        row["conflicting_labels"] = [labels.get(c, c) for c in conflicting]
        if not row.get("reason"):
            row["reason"] = getattr(rec, "reason", None)
    return row


def replay_governed_writes(
    *,
    method: str = "B3",
    seeds: Iterable[int] = (1337,),
    per_category: int = 6,
    settings_factory: Callable[[], Settings] = _default_settings,
    extractor: object | None = None,
    embeddings: object | None = None,
    max_rows_per_bucket: int = 12,
    out_path: Optional[str] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Replay the benchmark through the governed write path and collect outcomes.

    Builds the ``method`` strategy (default B3, the full governed system) once
    per seed, writes every session, and buckets each :class:`WriteOutcome` by its
    governing decision with full provenance context. Returns a structured report
    with per-bucket totals, sampled example rows, a quarantine-reason histogram,
    and a ``false_quarantine`` sample (write-time quarantines inside no-conflict
    examples).

    A shared ``extractor`` / ``embeddings`` is injected so a real (cached) Qwen +
    sentence-transformers stack is reused rather than reloaded; with the defaults
    the offline mock extractor is used and the replay is deterministic per seed.
    """
    seeds = list(seeds)
    totals: dict[str, int] = {b: 0 for b in _BUCKETS}
    samples: dict[str, list[dict[str, Any]]] = {b: [] for b in _BUCKETS}
    false_quarantine: list[dict[str, Any]] = []
    reason_hist: dict[str, int] = {}
    n_examples = 0

    for seed in seeds:
        _seed_everything(seed)
        examples = BenchmarkGenerator(seed=seed).generate(per_category=per_category)
        strategy = _build_strategy(
            method, settings_factory, extractor=extractor, embeddings=embeddings
        )
        for example in examples:
            n_examples += 1
            example_expects_conflict = any(
                bool(getattr(q, "expected_conflict", False)) for q in example.questions
            )
            # Map session_id -> input so each outcome carries its source text.
            text_by_ref = {
                f"{example.id}:{s.session_id}": s.input for s in example.sessions
            }
            for session in example.sessions:
                source_ref = f"{example.id}:{session.session_id}"
                result = strategy.write(session.input, source_ref)
                # Resolve labels/quarantine records *after* the write so newly
                # created nodes and quarantine records are visible.
                labels = _entity_labels(strategy.container)
                quarantines = _quarantine_index(strategy.container)
                buckets = {
                    "accepted": result.accepted,
                    "superseded": result.superseded,
                    "quarantined": result.quarantined,
                    "rejected": result.rejected,
                }
                for bucket, outcomes in buckets.items():
                    for outcome in outcomes:
                        totals[bucket] += 1
                        row = _outcome_row(
                            outcome,
                            example_id=example.id,
                            category=example.category,
                            source_ref=source_ref,
                            session_text=text_by_ref.get(source_ref, session.input),
                            labels=labels,
                            quarantines=quarantines,
                        )
                        if bucket == "quarantined":
                            key = (row.get("reason") or "?").strip()
                            reason_hist[key] = reason_hist.get(key, 0) + 1
                            if not example_expects_conflict:
                                fq = dict(row)
                                fq["note"] = (
                                    "write-time quarantine in a no-conflict example "
                                    "(governance false-positive candidate)"
                                )
                                if len(false_quarantine) < max_rows_per_bucket:
                                    false_quarantine.append(fq)
                        if len(samples[bucket]) < max_rows_per_bucket:
                            samples[bucket].append(row)

    report: dict[str, Any] = {
        "method": method,
        "seeds": seeds,
        "per_category": per_category,
        "n_examples": n_examples,
        "totals": totals,
        "quarantine_reason_histogram": dict(
            sorted(reason_hist.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "samples": samples,
        "false_quarantine": false_quarantine,
    }
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        report["_saved_to"] = out_path
    if verbose:
        _print_report(report)
    return report


def _print_report(report: dict[str, Any]) -> None:
    """Print a readable governance audit from a :func:`replay_governed_writes` report."""
    print(f"Governed-write replay — method={report['method']} "
          f"seeds={report['seeds']} per_category={report['per_category']}")
    print(f"  examples replayed: {report['n_examples']}")
    t = report["totals"]
    print(f"  outcomes: accepted={t['accepted']} superseded={t['superseded']} "
          f"quarantined={t['quarantined']} rejected={t['rejected']}")
    print("=" * 90)

    hist = report.get("quarantine_reason_histogram") or {}
    if hist:
        print("Quarantine reasons (count):")
        for reason, count in hist.items():
            print(f"  {count:>4}  {reason}")
        print("-" * 90)

    def _dump(title: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        print(f"\n[{title}] showing {len(rows)} example row(s):")
        for r in rows:
            print(f"  • {r['triple']}  ({r['decision']}, intent={r['write_intent']})")
            print(f"      example={r['example_id']} [{r['category']}]  src={r['source_ref']}")
            print(f"      session: {r['session_text']!r}")
            if r.get("reason"):
                print(f"      reason: {r['reason']}")
            if r.get("superseded_assertion_id"):
                print(f"      superseded: {r['superseded_assertion_id']}")
            if r.get("conflicting_labels"):
                sev = r.get("severity")
                sev_s = f" severity={sev}" if sev else ""
                print(f"      conflicts with: {r['conflicting_labels']}{sev_s}")

    for bucket in ("superseded", "quarantined", "rejected", "accepted"):
        _dump(bucket, report["samples"].get(bucket, []))

    fq = report.get("false_quarantine") or []
    print("\n" + "=" * 90)
    if fq:
        print(f"FALSE-QUARANTINE candidates ({len(fq)} shown): write-time quarantines "
              "inside no-conflict examples.")
        for r in fq:
            print(f"  • {r['triple']}  example={r['example_id']} [{r['category']}]")
            print(f"      session: {r['session_text']!r}")
            if r.get("reason"):
                print(f"      reason: {r['reason']}")
    else:
        print("No false-quarantine candidates: every write-time quarantine fell in a "
              "conflict-expected example.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="B3", help="Arm to replay (default B3 = full governed).")
    parser.add_argument("--per-category", type=int, default=6,
                        help="Benchmark examples generated per category (anchors always included).")
    parser.add_argument("--seed", type=int, default=1337, help="Single seed to replay.")
    parser.add_argument("--max-rows", type=int, default=12,
                        help="Max example rows captured per bucket.")
    parser.add_argument("--out", default=None, help="Optional path to write the JSON report.")
    args = parser.parse_args()
    replay_governed_writes(
        method=args.method,
        seeds=(args.seed,),
        per_category=args.per_category,
        max_rows_per_bucket=args.max_rows,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
